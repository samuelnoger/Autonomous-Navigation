import torch
import torch.nn as nn
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

        # Track previous slip state for velocity projection on grip recovery
        self.was_slipping = torch.zeros(n_cars, dtype=torch.bool, device=device)

        # Driving direction per car: 1 for forward, -1 for reverse
        self.direction = torch.ones(n_cars, dtype=torch.int64, device=device)

        self.max_speed = 140.0
        self.accel_rate = 5.0
        self.friction = 0.02
        self.drift_factor = 0.5  # Velocity lag: 0 = instant, 1 = no effect
        self.wheelbase = 15.0
        self.max_steering_angle = math.pi / 4
        self.max_grip_accel = 20.0
        self.max_steering_rate = 0.1  # Max steering change per frame (prevents instant left-right switching)
        self.enter_threshold = 1.0
        self.exit_threshold = 0.75

    def reset(self, track, start_idx, epoch):
        """Reset car positions and velocities for a new episode.

        Args:
            epoch: Current training epochs
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

        # --- Assign directions: alternate based on epoch for simulation, randomize for training ---
        n = self.n_cars
        if epoch is not None:
            # Simulation mode: alternate direction each epoch
            dir_value = 1 if epoch % 2 == 0 else -1
            self.direction.fill_(dir_value)
        else:
            # Training mode: balanced half forward / half reverse
            half = n // 2
            dir_tensor = torch.ones(n, dtype=torch.int, device=device)
            dir_tensor[half: half * 2] = -1
            if n % 2 == 1:
                # randomize leftover car direction
                dir_tensor[-1] = -1 if torch.rand(1, device=device) < 0.5 else 1
            self.direction = dir_tensor

        self.pos = start_pos
        # Flip angle by 180° for reverse cars
        self.angle = start_angle + math.pi * (1 - self.direction) / 2
        self.active.fill_(True)
        self.prev_steer.zero_()
        self.was_slipping.zero_()

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
        max_grip_accel = self.max_grip_accel

        self.prev_pos = self.pos.clone()
        # ============================================================================
        # STEERING: Rate limiting + speed-dependent response
        # ============================================================================
        steering = steering * (1.0 - steer_smooth_alpha*dt) + self.prev_steer * steer_smooth_alpha*dt
        self.prev_steer = steering.clone()

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
        slip_mask = slip_ratio > self.enter_threshold
        fullsteer_mask = slip_ratio <= self.enter_threshold
        
        slip_mask[self.was_slipping] = slip_ratio[self.was_slipping] > self.exit_threshold
        fullsteer_mask[self.was_slipping] = slip_ratio[self.was_slipping] <= self.exit_threshold

        # Speed loss from drifting (lateral friction during oversteer)
        slip_excess = (slip_ratio - 1.0).clamp(min=0.0)
        drift_friction_factor = 7.5  # Tunable: higher = more speed loss while drifting
        self.speed[slip_mask] = self.speed[slip_mask] * (1.0 - slip_excess[slip_mask] * drift_friction_factor * dt)

        # Recompute ideal velocity with the new (lower) speed from drift friction
        ideal_velocity = heading * self.speed.unsqueeze(1)

        # ============================================================================
        # VELOCITY PROJECTION: When recovering from drift to grip
        # ============================================================================
        # Detect cars that just recovered grip (were slipping, now have grip)
        # Must do this BEFORE updating self.vel, so we project the actual drifted velocity
        just_recovered_grip = self.was_slipping & fullsteer_mask

        if just_recovered_grip.any():
            # Project drifting velocity onto heading direction
            # heading is a unit vector, so we get the component of velocity pointing forward
            vel_drifted = self.vel[just_recovered_grip]
            heading_recovered = heading[just_recovered_grip]

            # Forward component: dot product of velocity with heading direction
            # If sideways drift (perpendicular to heading): forward_component ≈ 0
            # If aligned drift: forward_component ≈ ||vel||
            vel_norm = torch.norm(vel_drifted, dim=1, keepdim=True)
            forward_component = (vel_drifted * heading_recovered).sum(dim=1, keepdim=True) / (vel_norm + 1e-6)
            forward_component = torch.clamp(forward_component, 0.0, 1.0)

            # Apply grip recovery penalty: new_velocity = heading * vel_norm * forward_component * penalty
            grip_recovery_penalty = 0.75
            ideal_velocity[just_recovered_grip] = heading_recovered * vel_norm * forward_component * grip_recovery_penalty

        # Update velocity:
        # - Within grip: velocity snaps to ideal heading instantly
        # - While slipping: velocity lags behind heading (smooth drift effect)
        self.vel[slip_mask] = self.vel[slip_mask]*(1 - drift_factor*dt) + ideal_velocity[slip_mask]*drift_factor*dt
        self.vel[fullsteer_mask] = ideal_velocity[fullsteer_mask]

        # Update slip state for next frame
        self.was_slipping = slip_mask.clone()

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

 
