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


class GRUCarNet(nn.Module):
    """
    GRU-based architecture for temporal control sequences.

    Advantages over LSTM:
    - Simpler (3 gates vs 4 gates) → more stable training
    - Fewer parameters → less prone to divergence
    - Better gradient flow → less likely to crash suddenly
    - Still learns temporal patterns and remembers recent actions

    Best for: Racing where steering/acceleration must be coordinated over time
    without the instability of LSTM.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_rays, gru_layers=1):
        super().__init__()

        self.n_rays = n_rays
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        state_dim = input_dim - n_rays

        # ---- Ray encoder ----
        self.ray_net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        ray_out_dim = 16 * n_rays

        # ---- State encoder ----
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # ---- GRU for temporal processing ----
        gru_input_dim = ray_out_dim + hidden_dim // 2
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True
        )

        # ---- Output head ----
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def init_hidden(self, batch_size, device):
        """Initialize GRU hidden state."""
        return torch.zeros(self.gru_layers, batch_size, self.hidden_dim, device=device)

    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, input_dim) input features
            hidden: (gru_layers, batch, hidden_dim) or None

        Returns:
            outputs: (batch, output_dim) steering and acceleration
            hidden: updated hidden state
        """
        # Split rays vs. state
        rays = x[:, :self.n_rays].unsqueeze(1)  # (batch, 1, n_rays)
        state = x[:, self.n_rays:]  # (batch, state_dim)

        # Encode rays and state
        r = self.ray_net(rays)  # (batch, ray_out_dim)
        s = self.state_net(state)  # (batch, hidden_dim//2)
        combined = torch.cat([r, s], dim=1)  # (batch, gru_input_dim)

        # Add sequence dimension for GRU
        combined = combined.unsqueeze(1)  # (batch, 1, gru_input_dim)

        # GRU forward pass
        gru_out, hidden = self.gru(combined, hidden)  # gru_out: (batch, 1, hidden_dim)
        gru_out = gru_out.squeeze(1)  # (batch, hidden_dim)

        # Output head
        outputs = self.fc(gru_out)  # (batch, output_dim)

        return outputs, hidden


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
        self.steer_smooth_alpha = 0.0

        self.max_speed = 140.0
        self.accel_rate = 20.0
        self.steering_rate = 1.0
        self.drift_factor = 0.1
        self.friction = 0.05
        self.cornering_factor = 0.25

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

    def step_physics(self, steering, accel, dt=0.1, steer_smooth_alpha=None):
        
        accel_rate = self.accel_rate
        steering_rate = self.steering_rate
        max_speed = self.max_speed
        drift_factor = self.drift_factor
        friction = self.friction
        cornering_factor = self.cornering_factor

        alpha = self.steer_smooth_alpha if steer_smooth_alpha is None else steer_smooth_alpha
        if alpha > 0.0:
            steering = alpha * self.prev_steer + (1.0 - alpha) * steering
            self.prev_steer = steering

        self.prev_pos = self.pos.clone()

        # Update angle
        self.angle = self.angle + steering * steering_rate * dt

        # Update speed with acceleration then apply friction
        self.speed = self.speed + accel * accel_rate * dt
        self.speed = self.speed * (1.0 - friction * dt)

        # Centripetal speed loss: proportional to steering magnitude
        turn_magnitude = steering.abs()
        self.speed = self.speed * (1.0 - cornering_factor * turn_magnitude * dt)

        self.speed = torch.clamp(self.speed, -max_speed, max_speed)

        # Compute heading [batch_size, 2]
        heading = torch.stack([torch.cos(self.angle), torch.sin(self.angle)], dim=1)

        # Current velocity
        vel = self.vel.clone()

        # Drift: blend current velocity toward heading * speed
        ideal_velocity = heading * self.speed.unsqueeze(
            1
        )  # expand speed to [batch_size, 1]
        self.vel = vel * (1 - drift_factor) + ideal_velocity * drift_factor

        # Update position
        self.pos += self.vel * dt

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

 
