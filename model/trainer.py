"""Trainer class for racing neural network models."""
import torch
from torch.distributions import Normal
from tqdm import tqdm, trange

from model import CarState
from utils import save_checkpoint, get_inputs, RewardPlotter


class Trainer:
    """Trainer for racing car neural networks using policy gradient learning."""

    def __init__(
        self,
        model,
        track,
        optimizer,
        scheduler=None,
        device='cpu',
        checkpoint_path="checkpoints/last_ckpt.pth",
        max_ray_dist=500.0,
        steer_smooth_alpha=0.0,
        steer_noise=0.02,
        accel_noise=0.05,
        start_mode=None,
        group_size=None,
        car_length=8.0,
        car_width=4.0,
        start_spacing=1.5,
        update_frequency: int = 0,
        min_alive_per_group: int = 2,
        progress_reward_weight: float = 1.0,
        overtake_reward: float = 1.0,
        # Optional explicit reward rate overrides (if None, track-specific defaults apply)
        speed_reward_rate: float = None,
        collision_penalty_rate: float = None,
        gate_pass_reward_rate: float = None,
        wall_penalty_rate: float = None,
        direction_reward_rate: float = None,
        alive_reward_rate: float = None,
        # Optional args namespace (preferred): if provided, values are read from it
        args=None,
    ):
        """Initialize the trainer.

        Args:
            model: Neural network model (CarNet, GRU, LSTM).
            track: Track object for training.
            optimizer: Torch optimizer (e.g., Adam).
            scheduler: Optional learning rate scheduler.
            device: Device to train on (cpu/mps/cuda).
            checkpoint_path: Path to save checkpoints.
            max_ray_dist: Maximum ray casting distance.
            steer_smooth_alpha: Steering smoothing factor.
        """
        self.model = model
        self.track = track
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.max_ray_dist = max_ray_dist
        self.steer_smooth_alpha = steer_smooth_alpha
        self.steer_noise = steer_noise
        self.accel_noise = accel_noise
        self.start_mode = start_mode
        self.group_size = group_size
        # Car physical parameters used for spawn spacing and collisions - initialize
        # early so `args` processing can safely reference them.
        self.car_length = float(car_length)
        self.car_width = float(car_width)
        self.start_spacing = float(start_spacing)
        # Update frequency (0 = update at episode/rollout end)
        self.update_frequency = int(update_frequency)
        # Early-end condition: if every group has <= min_alive_per_group active cars
        self.min_alive_per_group = int(min_alive_per_group)
        # New reward weights
        self.progress_reward_weight = float(progress_reward_weight)
        self.overtake_reward = float(overtake_reward)
        # Debugging switch for overtaking counts (can be set via args.debug_overtake)
        self.debug_overtake = False
        # One-shot timing probe to measure per-step component costs
        self.debug_timing = False
        # Reward rates (can be configured via args or explicit kwargs)
        # If left as None, compute_rewards will fall back to legacy defaults
        self.speed_reward_rate = None if speed_reward_rate is None else float(speed_reward_rate)
        self.collision_penalty_rate = None if collision_penalty_rate is None else float(collision_penalty_rate)
        self.gate_pass_reward_rate = None if gate_pass_reward_rate is None else float(gate_pass_reward_rate)
        self.wall_penalty_rate = None if wall_penalty_rate is None else float(wall_penalty_rate)
        self.direction_reward_rate = None if direction_reward_rate is None else float(direction_reward_rate)
        self.alive_reward_rate = None if alive_reward_rate is None else float(alive_reward_rate)

        # If an argparse Namespace `args` is provided, prefer its values when present
        if args is not None:
            # prefer explicit args.* when provided; leave None otherwise
            self.speed_reward_rate = getattr(args, 'speed_reward_rate', self.speed_reward_rate)
            self.collision_penalty_rate = getattr(args, 'collision_penalty_rate', self.collision_penalty_rate)
            self.gate_pass_reward_rate = getattr(args, 'gate_pass_reward_rate', self.gate_pass_reward_rate)
            self.wall_penalty_rate = getattr(args, 'wall_penalty_rate', self.wall_penalty_rate)
            self.direction_reward_rate = getattr(args, 'direction_reward_rate', self.direction_reward_rate)
            self.alive_reward_rate = getattr(args, 'alive_reward_rate', self.alive_reward_rate)
            # core Trainer hyperparameters from args (if provided)
            self.max_ray_dist = getattr(args, 'max_ray_dist', self.max_ray_dist)
            self.steer_smooth_alpha = getattr(args, 'steer_smooth_alpha', self.steer_smooth_alpha)
            self.steer_noise = getattr(args, 'steer_noise', self.steer_noise)
            self.accel_noise = getattr(args, 'accel_noise', self.accel_noise)
            self.start_mode = getattr(args, 'car_start_mode', self.start_mode)
            self.group_size = getattr(args, 'group_size', self.group_size)
            self.car_length = float(getattr(args, 'car_length', self.car_length))
            self.car_width = float(getattr(args, 'car_width', self.car_width))
            self.start_spacing = float(getattr(args, 'start_spacing', self.start_spacing))
            self.update_frequency = int(getattr(args, 'update_frequency', self.update_frequency))
            self.min_alive_per_group = int(getattr(args, 'min_alive_per_group', self.min_alive_per_group))
            self.progress_reward_weight = float(getattr(args, 'progress_reward_weight', self.progress_reward_weight))
            self.overtake_reward = float(getattr(args, 'overtake_reward', self.overtake_reward))
            self.debug_overtake = getattr(args, 'debug_overtake', self.debug_overtake)
            self.debug_timing = getattr(args, 'debug_timing', self.debug_timing)

        # Initialize reward plotter
        self.plotter = RewardPlotter()

        # Precompute track info
        self.gates_tensor = track.gates.clone().to(device)
        self.track_start_idx = track.gates.shape[0] - 1
        self.track_name = track.track_name

    def get_inputs(self, cars, gate_indices):
        """Get neural network inputs for all cars.

        Computes ray distances, heading error, speed, curvature lookahead, etc.

        Args:
            cars: CarState object.
            gate_indices: Current gate indices for all cars.

        Returns:
            inputs: Model input tensor (N, input_dim).
            ray_dists: Ray distances for reward computation (N, n_rays).
        """
        
        return get_inputs(cars,self.track, self.gates_tensor, gate_indices, self.max_ray_dist)

    def compute_rewards(
        self, cars, gate_indices, last_gate_indices, ray_dists, prev_dist, step, n_steps
    ):
        """Compute step rewards for all cars.

        Includes speed rewards, gate passing, collision penalties, wall proximity penalties,
        direction rewards, and alive rewards.

        Args:
            cars: CarState object.
            gate_indices: Current gate indices.
            last_gate_indices: Previous gate indices.
            ray_dists: Ray distances from sensors.
            prev_dist: Previous distance to gate (unused but kept for interface).
            step: Current step number.
            n_steps: Total steps per epoch.

        Returns:
            step_rewards: Dictionary of reward components.
            prev_dist: Updated distance to gate.
        """
        pos = cars.pos
        speed = cars.speed

        # Reward rates (tunable hyperparameters). Values come from Trainer attrs
        # when set, otherwise fall back to legacy defaults. Track-specific
        # defaults are applied only when the corresponding Trainer attr is None.
        speed_reward_rate = (
            self.speed_reward_rate if self.speed_reward_rate is not None else 0.005
        )
        collision_penalty_rate = (
            self.collision_penalty_rate if self.collision_penalty_rate is not None else 60.0
        )
        gate_pass_reward_rate = (
            self.gate_pass_reward_rate if self.gate_pass_reward_rate is not None else 2.0
        )
        wall_penalty_rate = (
            self.wall_penalty_rate if self.wall_penalty_rate is not None else 0.0
        )
        direction_reward_rate = (
            self.direction_reward_rate if self.direction_reward_rate is not None else 0.0
        )
        alive_reward_rate = (
            self.alive_reward_rate if self.alive_reward_rate is not None else 0.0
        )

        # Track-specific sensible defaults (apply only when user didn't override)
        if self.track.track_name == "simple":
            collision_penalty_rate = 50.0
            alive_reward_rate = 0.025
            gate_pass_reward_rate = 10.0
            direction_reward_rate = 0.02

        if self.track.track_name == "square":
            collision_penalty_rate = 50.0
            gate_pass_reward_rate = 5.0
            direction_reward_rate = 0.02

        # -----------------------------
        # Distance to next gate
        # -----------------------------
        gate_coords = self.gates_tensor[gate_indices]
        x1, y1 = gate_coords[:, 0], gate_coords[:, 1]
        x2, y2 = gate_coords[:, 2], gate_coords[:, 3]

        line_vec = torch.stack([x2 - x1, y2 - y1], dim=1)
        p_vec = pos - torch.stack([x1, y1], dim=1)
        line_len2 = (line_vec**2).sum(dim=1)
        u = torch.clamp((p_vec * line_vec).sum(dim=1) / line_len2, 0.0, 1.0)
        closest = torch.stack([x1, y1], dim=1) + u.unsqueeze(1) * line_vec
        dist = (pos - closest).norm(dim=1)

        # -----------------------------
        # Reward components
        # -----------------------------

        # Speed reward: encourage faster speeds
        speed_reward = speed_reward_rate * speed * cars.active

        # Gate passing: reward cars that passed through the gate this step
        passed_mask = (
            (dist < 10.0)
            & (gate_indices == (last_gate_indices + cars.direction) % self.track.gates.shape[0])
            & cars.active
        )

        gate_reward = torch.zeros_like(speed)
        with torch.no_grad():
            cars.gates_passed[passed_mask] += cars.direction[passed_mask]
            gate_reward[passed_mask] = gate_pass_reward_rate
            last_gate_indices[passed_mask] = gate_indices[passed_mask]
            gate_indices[passed_mask] = (
                gate_indices[passed_mask] + cars.direction[passed_mask]
            ) % self.gates_tensor.shape[0]

        # -------- Progress and overtaking rewards --------
        # Continuous progress metric: gate index + projection fraction u
        device = pos.device
        progress = gate_indices.to(torch.float32) + u

        # Use stored previous progress if available; initialize to current
        # progress to avoid spurious overtakes at startup.
        prev_prog = getattr(cars, 'prev_progress', None)
        if prev_prog is None or prev_prog.shape[0] != progress.shape[0]:
            prev_prog = progress.clone()

        delta_progress = (progress - prev_prog) * cars.active.float()
        try:
            cars.prev_progress = progress.detach().clone()
        except Exception:
            pass

        progress_reward = delta_progress * self.progress_reward_weight

        # -------- Vectorized overtaking reward (O(N^2) memory) --------
        # Create pairwise differences and masks for same-group active cars.
        N = progress.shape[0]
        if N > 1:
            diff_now = progress.unsqueeze(1) - progress.unsqueeze(0)
            diff_prev = prev_prog.unsqueeze(1) - prev_prog.unsqueeze(0)

            same_group = (cars.batch.unsqueeze(1) == cars.batch.unsqueeze(0))
            active_mask = cars.active.unsqueeze(1) & cars.active.unsqueeze(0)
            diag = torch.eye(N, dtype=torch.bool, device=device)
            mask = same_group & active_mask & (~diag)

            overtake_matrix = (diff_prev < 0) & (diff_now > 0) & mask
            overtaken_matrix = (diff_prev > 0) & (diff_now < 0) & mask

            overtakes = overtake_matrix.sum(dim=1).to(torch.float32)
            overtaken = overtaken_matrix.sum(dim=1).to(torch.float32)

            overtake_reward = (overtakes - overtaken) * float(self.overtake_reward) * cars.active.float()

            # Optional debug logging (enable by setting args.debug_overtake=True)
            if getattr(self, 'debug_overtake', False):
                try:
                    print(f"[Trainer] overtakes_sum={overtakes.sum().item()}, overtaken_sum={overtaken.sum().item()}")
                except Exception:
                    pass
        else:
            overtake_reward = torch.zeros_like(progress)

        # Collision penalty
        prev_active = cars.active.clone()
        cars.check_collisions(self.track)

        if self.track.track_name == "simple":
            collision_penalty = (
                -(~cars.active & prev_active).float()
                * (n_steps - step)
                / n_steps
                * collision_penalty_rate
            )
        else:
            collision_penalty = (
                -(~cars.active & prev_active).float()
                * (n_steps - 0.6 * step)
                / n_steps
                * collision_penalty_rate
            )

        # (No continuous contact reward: contact effects are applied as
        # direct speed/acceleration slowdowns in CarState.check_collisions)

        # Wall penalty: penalize being too close to walls
        track_width = self.track.track_width
        min_dist = ray_dists.min(dim=1).values
        safe_margin = 0.2 * track_width
        scaled_min_dist = min_dist / safe_margin
        wall_penalty = (
            - wall_penalty_rate * (1.0 - torch.clamp(scaled_min_dist, 0, 1)) ** 2 * cars.active
        )
        #print("Min scaled distance:", scaled_min_dist.min())
        #print("Wall penalty rate:", wall_penalty_rate)
        #print("Wall penalty mean:", wall_penalty.mean().item())

        # Direction reward: encourage heading toward next gate
        next_gate_centers = (
            self.gates_tensor[gate_indices, 0:2] + self.gates_tensor[gate_indices, 2:4]
        ) / 2
        gate_vec = next_gate_centers - cars.pos
        gate_dir = gate_vec / (gate_vec.norm(dim=1, keepdim=True) + 1e-6)

        vel = cars.vel
        vel_norm = vel / (vel.norm(dim=1, keepdim=True) + 1e-6)

        direction_reward = (vel_norm * gate_dir).sum(dim=1) * cars.active * direction_reward_rate

        # Alive reward: small reward for each timestep the car is active
        alive_reward = alive_reward_rate * cars.active

        # Aggregate step reward
        step_rewards = {
            "speed": speed_reward,
            "gate": gate_reward,
            "collision": collision_penalty,
            "progress": progress_reward,
            "overtake": overtake_reward,
            "wall": wall_penalty,
            "direction": direction_reward,
            "alive": alive_reward,
        }

        prev_dist = dist.detach()

        return step_rewards, prev_dist

    def train_step(
        self, cars, gate_indices, last_gate_indices, prev_dist, step, n_steps
    ):
        """Execute a single training step.

        Args:
            cars: CarState object.
            gate_indices: Current gate indices.
            last_gate_indices: Previous gate indices.
            prev_dist: Previous distance to gate.
            step: Current step number.
            n_steps: Total steps per epoch.
            hidden: Hidden state for RNN models.

        Returns:
            log_prob: Log probability for policy gradient.
            reward: Total reward for this step.
            step_rewards: Dictionary of reward components.
            prev_dist: Updated distance to gate.
            hidden: Updated hidden state (or None).
        """
        # Get model inputs
        t0_step = None
        if getattr(self, 'debug_timing', False) and step == 0:
            import time as _time
            t0_step = _time.time()
        inputs, ray_dists = self.get_inputs(cars, gate_indices)

        # Forward pass
        outputs = self.model(inputs)

        # Sample actions from policy
        steer_mean = outputs[:, 0]
        accel_mean = outputs[:, 1]

        steer_dist = Normal(steer_mean, self.steer_noise)
        accel_dist = Normal(accel_mean, self.accel_noise)

        steer = torch.clamp(steer_dist.sample(), -1, 1)
        accel = torch.clamp(accel_dist.sample(), -1, 1)

        # Compute log probabilities for policy gradient
        log_prob = steer_dist.log_prob(steer) + accel_dist.log_prob(accel)

        # Physics update
        # Zero-out actions for inactive cars so they don't move anymore and don't interfere
        if hasattr(cars, 'active'):
            active_mask = cars.active
            if active_mask.numel() == steer.numel():
                steer = steer * active_mask.float()
                accel = accel * active_mask.float()

        with torch.no_grad():
            t_phys0 = _time.time() if getattr(self, 'debug_timing', False) and step == 0 else None
            cars.physics_update(steer, accel, dt=0.1, steer_smooth_alpha=self.steer_smooth_alpha)
            if getattr(self, 'debug_timing', False) and step == 0:
                t_phys = _time.time() - t_phys0

        # Compute rewards
        if getattr(self, 'debug_timing', False) and step == 0:
            t_rew0 = _time.time()
        step_rewards, prev_dist = self.compute_rewards(
            cars, gate_indices, last_gate_indices, ray_dists, prev_dist, step, n_steps
        )
        if getattr(self, 'debug_timing', False) and step == 0:
            t_rew = _time.time() - t_rew0

        # Sum all reward components
        reward = (
            step_rewards["speed"]
            + step_rewards["gate"]
            + step_rewards["collision"]
            + step_rewards["wall"]
            + step_rewards["direction"]
            + step_rewards["alive"]
            + step_rewards["progress"]
            + step_rewards["overtake"]
        )

        # debug_timing prints removed

        return log_prob, reward, step_rewards, prev_dist

    
    def collect_rollout(self, cars, n_steps, force_direction, epoch, print_summary=True):
        """Run a single rollout of length `n_steps` with all cars set to `force_direction`.

        Returns:
            log_sum: Tensor (n_cars,) sum of log-probs across steps for each car
            returns: Tensor (n_cars,) undiscounted returns per car
            mean_rewards: dict of mean rewards per component (scalars)
            episode_reward: scalar mean episode reward across cars
        """
        # Force-reset cars to the requested direction. Pass start_mode/group_size if set.
        import time as _time
        t_collect_start = _time.time()
        # collect_rollout start debug print removed
        cars.reset(track=self.track, start_idx=self.track_start_idx, epoch=None, force_direction=force_direction, start_mode=self.start_mode, group_size=self.group_size)

        # Gate tracking
        gate_indices = torch.full((cars.n_cars,), self.track_start_idx, dtype=torch.long, device=self.device)
        gate_indices = (gate_indices + cars.direction) % self.gates_tensor.shape[0]
        last_gate_indices = (gate_indices.clone() - cars.direction) % self.gates_tensor.shape[0]

        # Hidden state
        if hasattr(self.model, 'init_hidden'):
            hidden = self.model.init_hidden(cars.n_cars, self.device)
        else:
            hidden = None

        n_cars_local = cars.n_cars
        rewards_buffer = {
            "speed": torch.zeros(n_steps, n_cars_local, device=self.device),
            "gate": torch.zeros(n_steps, n_cars_local, device=self.device),
            "collision": torch.zeros(n_steps, n_cars_local, device=self.device),
            "progress": torch.zeros(n_steps, n_cars_local, device=self.device),
            "overtake": torch.zeros(n_steps, n_cars_local, device=self.device),
            "wall": torch.zeros(n_steps, n_cars_local, device=self.device),
            "direction": torch.zeros(n_steps, n_cars_local, device=self.device),
            "alive": torch.zeros(n_steps, n_cars_local, device=self.device),
        }

        # Per-step diagnostics for wall penalty: count of non-zero entries and mean
        step_wall_nonzero = torch.zeros(n_steps, dtype=torch.long, device=self.device)
        step_wall_mean = torch.zeros(n_steps, device=self.device)

        log_probs = torch.zeros(n_steps, n_cars_local, device=self.device)
        step_rewards_total = torch.zeros(n_steps, n_cars_local, device=self.device)
        prev_dist = torch.zeros(n_cars_local, device=self.device)

        # Use a step-level progress bar for visibility when running epochs
    
        step_bar = tqdm(range(n_steps), desc=f"Epoch {epoch}, dir={force_direction}", leave=False, position=1)
        # chunked buffers for on-the-fly updates
        chunk_log_probs = []
        chunk_rewards = []
        # track per-chunk mean rewards for debugging/plotting (average across cars)
        chunk_episode_rewards = []
        # track per-chunk component means (raw averages across steps and cars)
        chunk_comp_means = []

        for step in step_bar:
            # Early termination: if no cars active, stop
            if not cars.active.any():
                break
            if step == 0:
                t_step0 = _time.time()
            log_prob, reward, step_rewards, prev_dist = self.train_step(
                cars, gate_indices, last_gate_indices, prev_dist, step, n_steps
            )

            # Store log_probs. If we're doing chunked on-the-fly updates
            # (`update_frequency>0`) we store a detached copy for later
            # aggregation to avoid holding the computational graph across
            # the entire episode. If not chunking, keep the tensor attached
            # so `fit()` can backprop over the full episode.
            if self.update_frequency > 0:
                log_probs[step] = log_prob.detach()
            else:
                log_probs[step] = log_prob
            step_rewards_total[step] = reward

            for k in step_rewards:
                rewards_buffer[k][step] = step_rewards[k]


            # accumulate for chunked update
            chunk_log_probs.append(log_prob)
            chunk_rewards.append(reward)

            # record first-step duration immediately after first step
            if step == 0:
                try:
                    first_step_duration = _time.time() - t_step0
                except Exception:
                    first_step_duration = None

            # check early-end condition: every group has <= min_alive_per_group active cars
            # When `group_size` is 1 (independent cars), skip this group-based early termination
            try:
                if getattr(self, 'group_size', None) is None or int(getattr(self, 'group_size', 0)) <= 1:
                    # For independent cars, only end when no cars are active (handled above)
                    pass
                else:
                    unique_batches = torch.unique(cars.batch)
                    batch_ok = True
                    for b in unique_batches:
                        alive_count = ((cars.batch == b) & cars.active).sum().item()
                        if alive_count > self.min_alive_per_group:
                            batch_ok = False
                            break
                    if batch_ok:
                        # all groups have <= min_alive_per_group alive -> end rollout
                        break
            except Exception:
                pass

            # perform update when enough steps collected (if configured)
            if self.update_frequency > 0 and ((step + 1) % self.update_frequency == 0):
                # stack chunk tensors
                chunk_log = torch.stack(chunk_log_probs, dim=0)  # (chunk, N)
                chunk_log_sum = chunk_log.sum(dim=0)
                chunk_rewards_t = torch.stack(chunk_rewards, dim=0)
                returns_chunk = chunk_rewards_t.sum(dim=0)

                # record mean reward for this chunk (average per car)
                try:
                    chunk_episode_rewards.append(returns_chunk.mean().item())
                except Exception:
                    pass

                # record raw per-component means for this chunk (no normalization)
                try:
                    start_idx = step + 1 - self.update_frequency
                    end_idx = step + 1
                    # Per-chunk mean: sum across steps for each car, then mean across cars
                    comp_means = {
                        k: rewards_buffer[k][start_idx:end_idx].sum(dim=0).mean().item()
                        for k in rewards_buffer
                    }
                    chunk_comp_means.append((start_idx, end_idx, comp_means))
                except Exception:
                    pass

                # normalize returns across cars in chunk
                returns_norm = (returns_chunk - returns_chunk.mean()) / (returns_chunk.std() + 1e-6)

                loss = -(chunk_log_sum * returns_norm).mean()

                # per-chunk immediate prints removed; chunk means collected earlier

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # clear chunk buffers
                chunk_log_probs = []
                chunk_rewards = []

            if hasattr(step_bar, "set_postfix") and (step % 20 == 0):
                try:
                    active_count = cars.active.sum().item()
                    step_bar.set_postfix(active_cars=f"{active_count}/{n_cars_local}")
                except Exception:
                    pass

        # If any leftover chunk (when update_frequency>0) perform final update
        if len(chunk_log_probs) > 0:
            chunk_log = torch.stack(chunk_log_probs, dim=0)
            chunk_log_sum = chunk_log.sum(dim=0)
            chunk_rewards_t = torch.stack(chunk_rewards, dim=0)
            returns_chunk = chunk_rewards_t.sum(dim=0)
            try:
                chunk_episode_rewards.append(returns_chunk.mean().item())
            except Exception:
                pass
            # record raw per-component means for this final leftover chunk
            try:
                # estimate start index for the leftover chunk based on the
                # actual number of executed steps (use `step` from loop)
                chunk_len = int(chunk_log.shape[0])
                if 'step' in locals():
                    start_idx = max(0, (step + 1) - chunk_len)
                else:
                    start_idx = max(0, n_steps - chunk_len)
                end_idx = start_idx + chunk_len
                # Per-chunk mean: sum across steps for each car, then mean across cars
                comp_means = {
                    k: rewards_buffer[k][start_idx:end_idx].sum(dim=0).mean().item()
                    for k in rewards_buffer
                }
                chunk_comp_means.append((start_idx, end_idx, comp_means))
            except Exception:
                pass
            # per-chunk final immediate print removed; final aggregation printed at end
            returns_norm = (returns_chunk - returns_chunk.mean()) / (returns_chunk.std() + 1e-6)
            loss = -(chunk_log_sum * returns_norm).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Summaries
        # collect_rollout timing summary removed
        # Full-episode returns per car (sum over steps)
        returns = step_rewards_total.sum(dim=0)  # (n_cars,)
        # Episode reward: mean total reward across cars for the full episode
        episode_reward = returns.mean().item()
        # Mean rewards per component: sum across steps per car, then mean across cars
        mean_rewards = {k: rewards_buffer[k].sum(dim=0).mean().item() for k in rewards_buffer}

        # Sum log-probs over time per car
        log_sum = log_probs.sum(dim=0)

        # Print a single summary line comparing full-episode means vs chunk-aggregated means
        try:
            # aggregate means across chunks if available
            agg = None
            if len(chunk_comp_means) > 0:
                agg = {k: 0.0 for k in rewards_buffer.keys()}
                for _s, _e, comp in chunk_comp_means:
                    for k, v in comp.items():
                        agg[k] += v
                n_chunks = float(len(chunk_comp_means))
                for k in list(agg.keys()):
                    agg[k] = agg[k] / n_chunks

            # Format both dicts for concise printing
            mean_str = ", ".join([f"{k}:{mean_rewards[k]:.4f}" for k in mean_rewards.keys()])
            if agg is None:
                agg_str = "None"
            else:
                agg_str = ", ".join([f"{k}:{agg[k]:.4f}" for k in agg.keys()])

            if print_summary:
                try:
                    tqdm.write(f"[Trainer SUMMARY] mean_rewards: {mean_str} | chunk_agg: {agg_str}")
                except Exception:
                    print(f"[Trainer SUMMARY] mean_rewards: {mean_str} | chunk_agg: {agg_str}")

        except Exception:
            pass

        

        return log_sum, returns, mean_rewards, episode_reward

    def fit(self, n_cars, n_rays, n_epochs, n_steps, start_epoch=0, best_reward=-float("inf"), both_directions=False):
        """Main training loop.

        Args:
            n_cars: Number of cars per batch.
            n_rays: Number of rays per car.
            n_epochs: Total number of epochs.
            n_steps: Steps per epoch.
            start_epoch: Epoch to start from (for resuming).
            best_reward: Best reward so far (for tracking).

        Returns:
            best_reward: Best reward achieved during training.
        """
        current_lr = self.optimizer.param_groups[0]["lr"]

        # Initialize car states with configured physical parameters
        cars = CarState(
            n_cars,
            n_rays,
            device=self.device,
            car_length=self.car_length,
            car_width=self.car_width,
            start_spacing=self.start_spacing,
        )

        epoch_bar = trange(
            start_epoch,
            n_epochs,
            leave=False,
            position=0,
            initial=start_epoch,
            total=n_epochs,
            desc="Training",
        )

        for epoch in epoch_bar:

            if both_directions:
                # Use the helper to collect forward and reverse rollouts
                logs_fwd, returns_fwd, mean_fwd, ep_fwd = self.collect_rollout(cars, n_steps, force_direction=1, epoch=epoch, print_summary=False)
                logs_rev, returns_rev, mean_rev, ep_rev = self.collect_rollout(cars, n_steps, force_direction=-1, epoch=epoch, print_summary=False)

                # Normalize per-rollout then combine
                r_fwd = (returns_fwd - returns_fwd.mean()) / (returns_fwd.std() + 1e-6)
                r_rev = (returns_rev - returns_rev.mean()) / (returns_rev.std() + 1e-6)

                combined_logs = torch.cat([logs_fwd, logs_rev], dim=0)
                combined_returns = torch.cat([r_fwd, r_rev], dim=0)

                loss = -(combined_logs * combined_returns).mean()

                # Aggregate metrics
                episode_reward = 0.5 * (ep_fwd + ep_rev)
                mean_rewards = {}
                for k in mean_fwd.keys():
                    mean_rewards[k] = 0.5 * (mean_fwd[k] + mean_rev[k])

            else:
                logs, returns, mean_rewards, episode_reward = self.collect_rollout(cars, n_steps, force_direction=1, epoch=epoch, print_summary=False)
                returns = (returns - returns.mean()) / (returns.std() + 1e-6)
                loss = -(logs * returns).mean()

            
            # Safety: if the computed `loss` does not require grad (e.g. because
            # log-probs were detached due to chunked in-rollout updates), avoid
            # calling backward on it. Provide a helpful diagnostic message.
            if not getattr(loss, 'requires_grad', True):
                # Skipping epoch backward because loss.requires_grad is False (no print)
                #print(f"[Trainer] Warning: skipping backward because loss does not require grad.")
                pass
            else:
                if self.update_frequency == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                else:
                    # chunked updates handled inside rollout; no-op here
                    pass

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step(episode_reward)

            if episode_reward > best_reward:
                best_reward = episode_reward

            # Plot raw episode reward (keep normalized returns for loss only)
            self.plotter.add_reward(episode_reward)

            if self.optimizer.param_groups[0]["lr"] != current_lr:
                epoch_bar.write(f"Epoch {epoch} [{self.track_name}] | LR changed from {current_lr:.2e} to {self.optimizer.param_groups[0]['lr']:.2e}")
                current_lr = self.optimizer.param_groups[0]["lr"]

            # Single-line epoch summary (one print per epoch)
            try:
                epoch_bar.write(
                    f"Epoch {epoch} [{self.track_name}] | Total reward: {episode_reward:.2f} | "
                    f"Speed:{mean_rewards['speed']:.3f}, Gate:{mean_rewards['gate']:.3f}, "
                    f"Collision:{mean_rewards['collision']:.3f}, Wall:{mean_rewards['wall']:.3f}, "
                    f"Direction:{mean_rewards['direction']:.3f}, Alive:{mean_rewards['alive']:.3f} | Best:{best_reward:.2f} | LR:{current_lr:.2e}"
                )
            except Exception:
                pass

            save_checkpoint(
                epoch,
                self.model,
                self.optimizer,
                self.checkpoint_path,
                total_epochs=n_epochs,
                scheduler=self.scheduler,
                best_reward=best_reward,
                track_name=self.track_name,
            )


        return best_reward
