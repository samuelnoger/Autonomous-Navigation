import torch
import torch.nn as nn
import math
import time


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
    def __init__(self, n_cars, n_rays, device, car_length=8.0, car_width=4.0, start_spacing=1.5):
        """Initialize car state for multiple cars.

        Args:
            n_cars: Number of cars in the batch
            n_rays: Number of rays per car for collision detection
            device: Torch device (cuda/mps/cpu)
        """
        self.n_cars = n_cars
        t0 = time.time()
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

        self.max_speed = 80.0
        self.accel_rate = 7.5
        self.breaking_rate = 20.0
        self.friction = 0.02
        self.drift_factor = 0.25  # Velocity lag: 0 = instant, 1 = no effect
        self.wheelbase = 7.5
        self.max_steering_angle = math.pi / 4
        self.max_grip_accel = 20.0
        self.max_steering_rate = 1.0  
        self.enter_threshold = 1.0
        self.exit_threshold = 0.05
        # grouping id for batch-local interactions. Default single group 0
        self.batch = torch.zeros(n_cars, dtype=torch.long, device=device)
        # Physical dimensions (pixels)
        self.car_length = float(car_length)
        self.car_width = float(car_width)
        # Radius used for circle-approx collision tests
        self.car_radius = float(max(self.car_length, self.car_width) * 0.5)
        # Start spacing multiplier (longitudinal spacing = car_length * start_spacing)
        self.start_spacing = float(start_spacing)
        # Number of columns is fixed to two for F1-style starts
        # Per-step impact recording and slowdown factor (1.0 = no slowdown)
        self.last_impacts = torch.zeros(n_cars, device=device)
        self.contact_slowdown = torch.ones(n_cars, device=device)
        # Progress / overtaking bookkeeping
        self.prev_progress = torch.zeros(n_cars, device=device)
        self.prev_rank = torch.zeros(n_cars, dtype=torch.long, device=device)
        # init timing print removed

    def reset(self, track, start_idx, epoch, force_direction=None, start_mode=None, group_size=None):
        """Reset car positions and velocities for a new episode.

        Args:
            epoch: Current training epochs
            track: Track object
            start_idx: Starting gate index
        """
        device = self.pos.device
        t0 = time.time()

        # Gate center and orientation
        start_gate = track.gates[start_idx]
        start_center = torch.tensor(
            [(start_gate[0] + start_gate[2]) / 2, (start_gate[1] + start_gate[3]) / 2],
            dtype=torch.float32,
            device=device,
        )
        gate_vec = torch.tensor(
            [start_gate[2] - start_gate[0], start_gate[3] - start_gate[1]],
            dtype=torch.float32,
            device=device,
        )
        gate_dir = gate_vec / (gate_vec.norm() + 1e-9)
        perp_dir = torch.tensor([-gate_dir[1], gate_dir[0]], device=device)

        # choose start mode / group size
        if start_mode is None:
            start_mode = getattr(self, 'start_mode', 'random')
        if group_size is None:
            group_size = getattr(self, 'group_size', None)

        n = self.n_cars
        idxs = torch.arange(n, device=device)

        # base angle with small per-car jitter
        base_angle = torch.atan2(gate_vec[1], gate_vec[0]) + math.pi / 2
        start_angle = torch.full((n,), base_angle, device=device)
        start_angle += (torch.rand(n, device=device) - 0.5) * 0.25

        # default simple random offsets (fast, on-device)
        along_offset = (torch.rand(n, device=device) - 0.5) * gate_vec.norm() * 0.2
        perp_offset = (torch.rand(n, device=device) - 0.5) * 10.0
        start_pos = start_center.unsqueeze(0) + along_offset.unsqueeze(1) * gate_dir + perp_offset.unsqueeze(1) * perp_dir

        # F1 / grouped starts: vectorized placement into two columns per group
        if start_mode == 'f1' or (group_size is not None and group_size > 0):
            if group_size is None or group_size <= 0:
                group_size = max(4, min(6, n))
            groups = idxs // group_size
            self.batch = groups

            # position within group
            pos_in_group = idxs - groups * group_size
            col = pos_in_group % 2
            row_idx = pos_in_group // 2

            # compute rows per group to center the grid
            counts = torch.bincount(groups, minlength=(groups.max().item() + 1)).to(device=device)
            n_rows = (counts + 1) // 2
            n_rows_per = n_rows[groups]

            car_length = self.car_length
            spacing_long = car_length * max(2.5, self.start_spacing)
            spacing_lat = self.car_width

            row_offset = (row_idx.to(torch.float32) - (n_rows_per.to(torch.float32) - 1.0) / 2.0) * spacing_long
            lateral = (-0.5 + col.to(torch.float32)) * spacing_lat

            start_pos = start_center.unsqueeze(0) + row_offset.unsqueeze(1) * perp_dir + lateral.unsqueeze(1) * gate_dir

            # small deterministic angle offset for side-by-side symmetry (safe integer-based)
            sign = torch.where((pos_in_group % 2) == 0, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))
            start_angle = start_angle + sign * 0.02

        # assign driving directions (half forward, half reverse) vectorized
        if force_direction is not None:
            # honor explicit override from caller (e.g., Trainer.collect_rollout)
            self.direction = torch.full((n,), int(force_direction), dtype=torch.int, device=device)
        else:
            half = n // 2
            dir_tensor = torch.ones(n, dtype=torch.int, device=device)
            dir_tensor[half: half * 2] = -1
            if n % 2 == 1:
                dir_tensor[-1] = -1 if torch.rand(1, device=device) < 0.5 else 1
            self.direction = dir_tensor

        # ensure batch id exists for non-grouped starts
        if not hasattr(self, 'batch') or self.batch.numel() != n:
            self.batch = torch.zeros(n, dtype=torch.long, device=device)

        # write positions/angles and standard resets (fast, on-device)
        self.pos = start_pos
        self.angle = start_angle + math.pi * (1 - self.direction) / 2
        self.active.fill_(True)
        self.prev_steer.zero_()
        self.was_slipping.zero_()

        start_speed = 0.0
        self.speed.fill_(start_speed)
        self.vel = torch.stack([torch.cos(self.angle), torch.sin(self.angle)], dim=1) * start_speed

        # reset per-step contact bookkeeping and progress/rank
        try:
            self.last_impacts.zero_()
            self.contact_slowdown.fill_(1.0)
            self.prev_progress.zero_()
            self.prev_rank.zero_()
        except Exception:
            pass

        # reset timing print removed

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
        breaking_rate = self.breaking_rate

        self.prev_pos = self.pos.clone()
        # ============================================================================
        # STEERING: Rate limiting + speed-dependent response
        # ============================================================================
        steering = steering * (1.0 - steer_smooth_alpha*dt) + self.prev_steer * steer_smooth_alpha*dt
        self.prev_steer = steering.clone()

        # At high speeds, reduce steering authority (can't turn as sharply)
        speed_normalized = self.speed / max_speed
        steering_reduction = 1.0 - 0.95*speed_normalized 
        effective_max_steering_angle = max_steering_angle * steering_reduction
        steering_angle = torch.clamp(steering, -1.0, 1.0) * effective_max_steering_angle

        # ============================================================================
        # SPEED: Acceleration + friction
        # ============================================================================
        # apply slowdown to requested acceleration when in contact
        if hasattr(self, 'contact_slowdown'):
            try:
                accel = accel * self.contact_slowdown
            except Exception:
                pass

        accel_mask = (accel >= 0) | (self.speed <= 0)
        brake_mask = (accel < 0) & (self.speed > 0)
        self.speed[accel_mask] = self.speed[accel_mask] + accel[accel_mask] * accel_rate * dt
        self.speed[brake_mask] = self.speed[brake_mask] + accel[brake_mask] * breaking_rate * dt

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
        drift_friction_factor = 10  # Tunable: higher = more speed loss while drifting
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
        # record per-car impact magnitudes this step (sum of absolute impulses)
        self.last_impacts = torch.zeros(self.n_cars, device=self.pos.device)

        # --- Border collisions: resolve penetration using distance-field gradient ---
        active_idx_all = torch.nonzero(active_mask, as_tuple=False).flatten()
        if active_idx_all.numel() > 0:
            pos_active = self.pos[active_idx_all]

            # try bilinear sampling; fallback to nearest lookup
            try:
                dist = track._sample_distance_field_bilinear(pos_active)
            except Exception:
                px = pos_active[:, 0].long().clamp(0, track.w - 1)
                py = pos_active[:, 1].long().clamp(0, track.h - 1)
                dist = track.distance_field[py, px]

            dist = dist.to(self.pos.device)

            penetrating = dist < self.car_radius

            # reset recorded impacts for this step
            try:
                self.last_impacts.zero_()
                self.contact_slowdown.fill_(1.0)
            except Exception:
                pass

            if penetrating.any():
                idx_pen = active_idx_all[penetrating]
                pos_pen = self.pos[idx_pen]
                vel_pen = self.vel[idx_pen]

                # finite-difference gradient estimate
                eps = 1.0
                offs_xp = pos_pen + torch.tensor([eps, 0.0], device=pos_pen.device)
                offs_xm = pos_pen + torch.tensor([-eps, 0.0], device=pos_pen.device)
                offs_yp = pos_pen + torch.tensor([0.0, eps], device=pos_pen.device)
                offs_ym = pos_pen + torch.tensor([0.0, -eps], device=pos_pen.device)

                try:
                    d_xp = track._sample_distance_field_bilinear(offs_xp)
                    d_xm = track._sample_distance_field_bilinear(offs_xm)
                    d_yp = track._sample_distance_field_bilinear(offs_yp)
                    d_ym = track._sample_distance_field_bilinear(offs_ym)
                except Exception:
                    # nearest lookup fallback
                    def _nns(p):
                        px = p[:, 0].long().clamp(0, track.w - 1)
                        py = p[:, 1].long().clamp(0, track.h - 1)
                        return track.distance_field[py, px]

                    d_xp = _nns(offs_xp)
                    d_xm = _nns(offs_xm)
                    d_yp = _nns(offs_yp)
                    d_ym = _nns(offs_ym)

                grad_x = (d_xp - d_xm) / (2.0 * eps)
                grad_y = (d_yp - d_ym) / (2.0 * eps)
                grad = torch.stack([grad_x, grad_y], dim=1)
                grad_norm = torch.norm(grad, dim=1, keepdim=True)
                n = grad / (grad_norm + 1e-6)

                pen = (self.car_radius - dist[penetrating]).unsqueeze(1)

                # positional correction
                pos_correction = 0.9
                self.pos[idx_pen] = self.pos[idx_pen] + n * (pen * pos_correction)

                # reflect velocity for components moving into wall
                restitution = 0.1
                rel_v = vel_pen
                rel_norm = (rel_v * n).sum(dim=1)
                moving_in = rel_norm < 0
                if moving_in.any():
                    rv = rel_norm[moving_in]
                    n_mv = n[moving_in]
                    idx_mv = idx_pen[moving_in]
                    j_impulse = -(1.0 + restitution) * rv
                    self.vel[idx_mv] = self.vel[idx_mv] + (j_impulse.unsqueeze(1) * n_mv)

                    # mark heavy impacts as crash
                    impact_thresh = 10.0
                    crashed = j_impulse.abs() > impact_thresh
                    if crashed.any():
                        to_crash = idx_mv[crashed]
                        self.active[to_crash] = False
                    # accumulate impact magnitudes for these indices and set slowdown
                    self.last_impacts[idx_mv] += j_impulse.abs()
                    try:
                        speed_scale = torch.clamp(1.0 - (self.last_impacts[idx_mv] / 20.0), min=0.2)
                        self.contact_slowdown[idx_mv] = speed_scale
                        self.vel[idx_mv] = self.vel[idx_mv] * speed_scale.unsqueeze(1)
                    except Exception:
                        pass

        # --- Car-to-car collisions ---
        # Resolve collisions only within the same batch/group so cars in different
        # interaction groups do not collide (supports large total N split into groups).
        from utils.my_utils import resolve_pairwise_collisions
        unique_batches = torch.unique(self.batch)
        for b in unique_batches:
            mask = (self.batch == b) & self.active
            idxs = torch.nonzero(mask, as_tuple=False).flatten()
            if idxs.numel() <= 1:
                continue

            pos_subset = self.pos[idxs]
            vel_subset = self.vel[idxs]
            device = pos_subset.device
            radii = torch.full((pos_subset.shape[0],), self.car_radius, device=device)

            new_pos, new_vel, impacts = resolve_pairwise_collisions(
                pos_subset.clone(), vel_subset.clone(), radii=radii, masses=None, restitution=0.25
            )

            # write back resolved positions/velocities for this group
            self.pos[idxs] = new_pos
            self.vel[idxs] = new_vel

            # If impact is large, mark those cars as inactive (they 'crashed')
            impact_thresh = 5.0
            crashed = impacts > impact_thresh
            if crashed.any():
                to_crash = idxs[crashed]
                self.active[to_crash] = False
            # store impact magnitudes for this group's indices and set slowdown
            try:
                self.last_impacts[idxs] = impacts
                speed_scale = torch.clamp(1.0 - (impacts / 20.0), min=0.2)
                self.contact_slowdown[idxs] = speed_scale
                # apply immediate speed reduction proportional to impacts
                self.vel[idxs] = self.vel[idxs] * speed_scale.unsqueeze(1)
                # keep speed magnitude consistent
                self.speed[idxs] = torch.norm(self.vel[idxs], dim=1)
            except Exception:
                pass

 
