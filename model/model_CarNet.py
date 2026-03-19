import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CarNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_rays):
        super().__init__()

        self.n_rays = n_rays
        state_dim = input_dim - n_rays  # remaining inputs (speed, heading, etc.)

        # ---- Ray encoder ----
        # Treat rays as a 1D signal and pass through a small conv net
        self.ray_net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # output shape: [batch, 16 * n_rays]
        )
        ray_out_dim = 16 * n_rays

        # ---- State encoder ----
        self.state_net = nn.Sequential(nn.Linear(state_dim, hidden_dim // 2), nn.ReLU())

        # ---- Control network ----
        self.fc = nn.Sequential(
            nn.Linear(ray_out_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),  # outputs steering [-1,1] and acceleration [-1,1]
        )

    def forward(self, x):
        # Split rays vs. state features
        rays = x[:, : self.n_rays].unsqueeze(1)  # (batch, 1, n_rays)
        state = x[:, self.n_rays :]  # (batch, state_dim)

        r = self.ray_net(rays)
        s = self.state_net(state)

        combined = torch.cat([r, s], dim=1)
        return self.fc(combined)

class CarState:
    def __init__(self, n_cars, n_rays, device):
        """Initialize car state for multiple cars.

        Args:
            n_cars: Number of cars in the batch
            n_rays: Number of rays per car for collision detection
            device: Torch device (cuda/mps/cpu)
        """
        self.n_cars = n_cars
        self.pos = torch.zeros((n_cars, 2), dtype=torch.float32, device=device)
        self.prev_pos = self.pos.clone()
        self.vel = torch.zeros((n_cars, 2), dtype=torch.float32, device=device)
        self.angle = torch.zeros(
            n_cars, dtype=torch.float32, device=device
        )  # Car heading angles

        self.speed = torch.zeros(n_cars, dtype=torch.float32, device=device)
        self.active = torch.ones(n_cars, dtype=torch.bool, device=device)
        self.collision = torch.zeros(n_cars, dtype=torch.bool, device=device)

        # Rays
        self.n_rays = n_rays
        u = torch.linspace(0, 1, n_rays, device=device)
        angles_normalized = 2 * u - 1  # [-1, 1]
        # Then apply a power to concentrate at center
        angles_normalized = torch.sign(angles_normalized) * (angles_normalized.abs() ** 1.4)
        self.ray_offsets = (math.pi / 2) * angles_normalized
        #print(f"Ray offsets (radians): {self.ray_offsets.cpu().numpy()}")

        self.gates_passed = torch.zeros(n_cars, dtype=torch.int, device=device)

        self.prev_steer = torch.zeros(n_cars, dtype=torch.float32, device=device)
        self.steer_smooth_alpha = 0.0  # Steering smoothing factor (0 = no smoothing)

        self.max_speed = 140.0
        self.accel_rate = 5.0
        self.friction = 0.02
        self.drift_factor = 0.5  # Velocity lag: 0 = instant, 1 = no effect
        self.wheelbase = 15.0
        self.max_steering_angle = math.pi / 4
        self.max_grip_accel = 15.0
        self.max_steering_rate = 0.1  # Max steering change per frame (prevents instant left-right switching) 

    def reset(self, track, start_idx):
        """Reset car positions and velocities for a new episode.

        Args:
            track: Track object
            start_idx: Starting gate index
        """
        device = self.pos.device
        start_gate = track.gates[start_idx]
        start_center = torch.tensor(
            [(start_gate[0] + start_gate[2]) / 2, (start_gate[1] + start_gate[3]) / 2],
            dtype=torch.float32,
            device=device,
        )

        # --- Gate direction ---
        gate_vec = torch.tensor(
            [start_gate[2] - start_gate[0], start_gate[3] - start_gate[1]],
            dtype=torch.float32,
            device=device,
        )
        gate_dir = gate_vec / gate_vec.norm()  # unit vector along gate
        perp_dir = torch.tensor(
            [-gate_dir[1], gate_dir[0]], device=device
        )  # perpendicular

        # --- Random offsets ---
        along_offset = ( 
            (torch.rand(self.n_cars, device=device) - 0.5) * gate_vec.norm() * 0.2
        )  # +/-20% along gate
        perp_offset = (
            torch.rand(self.n_cars, device=device) - 0.5
        ) * 10.0  # +/-20 px perpendicular

        # --- Start positions ---
        start_pos = (
            start_center.unsqueeze(0)
            + along_offset.unsqueeze(1) * gate_dir
            + perp_offset.unsqueeze(1) * perp_dir
        )

        # --- Start angles ---
        base_angle = torch.atan2(gate_vec[1], gate_vec[0]) + math.pi / 2
        start_angle = torch.full((self.n_cars,), base_angle, device=device)
        start_angle += (
            torch.rand(self.n_cars, device=device) - 0.5
        ) * 0.25  # +/-0.2 rad random spread

        self.pos = start_pos
        self.angle = start_angle
        self.active.fill_(True)
        self.prev_steer.zero_()

        start_speed = 5.0
        self.speed.fill_(start_speed)
        self.vel = (
            torch.stack([torch.cos(self.angle), torch.sin(self.angle)], dim=1)
            * start_speed
        )

    @property
    def ray_angles(self):
        """
        Compute the absolute angles of rays for all cars.
        Returns: (N,R) tensor
        """
        return self.angle.unsqueeze(1) + self.ray_offsets.unsqueeze(0)

    def physics_update(self, steering, accel, dt=0.1, steer_smooth_alpha=None):
        """Update car physics: steering, acceleration, grip, drift, and position.

        Args:
            steering: Input steering [-1, 1] from neural network
            accel: Input acceleration [-1, 1] from neural network
            dt: Timestep in seconds (default 0.1)
            steer_smooth_alpha: Unused (kept for compatibility)
        """
        # Cache physics parameters (local access is faster than self. lookups)
        accel_rate = self.accel_rate
        max_speed = self.max_speed
        friction = self.friction
        drift_factor = self.drift_factor
        wheelbase = self.wheelbase
        max_steering_angle = self.max_steering_angle
        max_steering_rate = self.max_steering_rate
        max_grip_accel = self.max_grip_accel

        self.prev_pos = self.pos.clone()

        # ============================================================================
        # STEERING: Rate limiting + speed-dependent response
        # ============================================================================
        # Prevent instantaneous steering reversals (steering inertia)
        steering_delta = torch.clamp(steering - self.prev_steer, -max_steering_rate, max_steering_rate)
        steering = self.prev_steer + steering_delta
        self.prev_steer = steering

        # At high speeds, reduce steering authority (can't turn as sharply)
        speed_normalized = self.speed / max_speed
        steering_reduction = 1.0 - 0.8 * speed_normalized  # 40% steering at max speed
        effective_max_steering_angle = max_steering_angle * steering_reduction
        steering_angle = torch.clamp(steering, -1.0, 1.0) * effective_max_steering_angle

        # ============================================================================
        # SPEED: Acceleration + friction
        # ============================================================================
        self.speed = self.speed + accel * accel_rate * dt
        self.speed = self.speed * (1.0 - friction * dt)
        self.speed = torch.clamp(self.speed, -max_speed, max_speed)

        # ============================================================================
        # TURNING: Ackermann steering kinematics
        # ============================================================================
        # Turning radius from steering angle: r = wheelbase / tan(angle)
        turning_radius = wheelbase / torch.tan(steering_angle.abs())

        # Angular velocity: ω = v / r (how fast the car rotates)
        steering_sign = torch.sign(steering_angle)
        steering_sign = torch.where(steering_sign == 0, torch.ones_like(steering_sign), steering_sign)
        angular_velocity = (self.speed / (turning_radius + 1e-6)) * steering_sign

        # Update heading angle
        self.angle = self.angle + angular_velocity * dt

        # Current velocity direction
        heading = torch.stack([torch.cos(self.angle), torch.sin(self.angle)], dim=1)
        ideal_velocity = heading * self.speed.unsqueeze(1)

        # ============================================================================
        # GRIP & DRIFTING: Apply sliding when centrifugal force exceeds grip
        # ============================================================================
        # Required centripetal acceleration to maintain turn at current speed
        centripetal_accel = (self.speed ** 2) / (turning_radius.abs() + 1e-6)

        # How much we exceed available grip (0 = no slip, >1 = heavy slip)
        slip_ratio = torch.clamp(centripetal_accel / max_grip_accel, 0.0, 2.0)
        slip_mask = slip_ratio > 1.0
        fullsteer_mask = slip_ratio <= 1.0

        # Speed loss from drifting (lateral friction during oversteer)
        slip_excess = (slip_ratio - 1.0).clamp(min=0.0)
        drift_friction_factor = 1.0  # Tunable: higher = more speed loss while drifting
        self.speed[slip_mask] = self.speed[slip_mask] * (1.0 - slip_excess[slip_mask] * drift_friction_factor * dt)

        # Recompute ideal velocity with the new (lower) speed from drift friction
        ideal_velocity = heading * self.speed.unsqueeze(1)

        # Update velocity:
        # - Within grip: velocity snaps to ideal heading instantly
        # - While slipping: velocity lags behind heading (smooth drift effect)
        self.vel[slip_mask] = self.vel[slip_mask]*(1 - drift_factor*dt) + ideal_velocity[slip_mask]*drift_factor*dt
        self.vel[fullsteer_mask] = ideal_velocity[fullsteer_mask]

        # ============================================================================
        # POSITION & SYNC
        # ============================================================================
        self.pos += self.vel * dt

        # Keep self.speed synced with actual velocity magnitude (after all blending)
        self.speed = torch.norm(self.vel, dim=1)

    def check_collisions(self, track):
        """Update active status based on collisions with track walls.

        Args:
            track: Track object with distance_field
        """
        active_mask = self.active

        # positions of active cars only
        pos_active = self.pos[active_mask]

        px = pos_active[:, 0].long().clamp(0, track.w - 1)
        py = pos_active[:, 1].long().clamp(0, track.h - 1)

        dist = track.distance_field[py, px]

        collided = (dist < track.car_radius).to(self.active.device)

        # deactivate only those active cars that collided
        full_collided = torch.zeros_like(self.active, dtype=torch.bool)
        full_collided[active_mask] = collided
        self.active = self.active & (~full_collided)

 
