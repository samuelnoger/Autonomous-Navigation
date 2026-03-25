"""Trainer class for racing neural network models."""
import torch
from torch.distributions import Normal
from tqdm import tqdm, trange

from model import CarState
from utils import save_checkpoint, RewardPlotter


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
        # Compute next gate centers
        next_gate_centers = (
            self.gates_tensor[gate_indices, 0:2] + self.gates_tensor[gate_indices, 2:4]
        ) / 2

        # ---- Ray distances ----
        ray_angles = cars.ray_angles  # (N, R)
        ray_dists_original = self.track.get_lines_along_rays(
            cars.pos, ray_angles, self.max_ray_dist
        )  # (N, R)
        ray_dists = ray_dists_original / self.max_ray_dist

        # ---- Heading error to next gate ----
        delta = next_gate_centers - cars.pos  # (N, 2)
        target_angle = torch.atan2(delta[:, 1], delta[:, 0])
        heading_error = target_angle - cars.angle
        heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
        sin_error = torch.sin(heading_error)
        cos_error = torch.cos(heading_error)

        # ---- Speed input (normalize) ----
        speed_input = cars.speed.unsqueeze(1) / 140.0

        # ---- Lookahead curvature (gates 1, 2, 3 ahead) ----
        curvatures = []
        n_gates = self.gates_tensor.shape[0]
        for lookahead in [1, 2, 3]:
            curr_idx = gate_indices
            next_idx = (gate_indices + lookahead * cars.direction) % n_gates

            # Extract gate vectors and compute direction vectors
            curr_gate = self.gates_tensor[curr_idx]  # (N, 4)
            next_gate = self.gates_tensor[next_idx]  # (N, 4)

            curr_gx = curr_gate[:, 2] - curr_gate[:, 0]
            curr_gy = curr_gate[:, 3] - curr_gate[:, 1]
            curr_norm = torch.hypot(curr_gx, curr_gy)
            curr_dir_x = -curr_gy / (curr_norm + 1e-6)
            curr_dir_y = curr_gx / (curr_norm + 1e-6)

            next_gx = next_gate[:, 2] - next_gate[:, 0]
            next_gy = next_gate[:, 3] - next_gate[:, 1]
            next_norm = torch.hypot(next_gx, next_gy)
            next_dir_x = -next_gy / (next_norm + 1e-6)
            next_dir_y = next_gx / (next_norm + 1e-6)

            # Curvature: angle between current and next gate directions
            cos_curv = curr_dir_x * next_dir_x + curr_dir_y * next_dir_y
            sin_curv = (curr_dir_x * next_dir_y - curr_dir_y * next_dir_x) #* cars.direction

            curvatures.append(cos_curv.unsqueeze(1))
            curvatures.append(sin_curv.unsqueeze(1))

        # ---- Concatenate all features ----
        feature_list = [
            ray_dists,  # n_rays features
            speed_input,  # 1 feature
            sin_error.unsqueeze(1),  # 1 feature
            cos_error.unsqueeze(1),  # 1 feature
        ]
        feature_list.extend(curvatures)  # 6 features (3 gates * 2 for sin/cos)

        inputs = torch.cat(feature_list, dim=1)
        return inputs, ray_dists_original

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

        # Reward rates (tunable hyperparameters)
        speed_reward_rate = 0.005
        collision_penalty_rate = 20.0
        gate_pass_reward_rate = 2.0
        wall_penalty_rate = 1.0
        direction_reward_rate = 0.01
        alive_reward_rate = 0.01

        if self.track.track_name == "simple":
            collision_penalty_rate = 50.0
            wall_penalty_rate = 10.0
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

        # Wall penalty: penalize being too close to walls
        track_width = self.track.track_width
        min_dist = ray_dists.min(dim=1).values
        safe_margin = 0.2 * track_width
        scaled_min_dist = min_dist / safe_margin
        wall_penalty = (
            -wall_penalty_rate * (1.0 - torch.clamp(scaled_min_dist, 0, 1)) ** 2 * cars.active
        )

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
            "wall": wall_penalty,
            "direction": direction_reward,
            "alive": alive_reward,
        }

        prev_dist = dist.detach()

        return step_rewards, prev_dist

    def train_step(
        self, cars, gate_indices, last_gate_indices, prev_dist, step, n_steps, hidden=None
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
        inputs, ray_dists = self.get_inputs(cars, gate_indices)

        # Forward pass
        if hidden is not None:
            outputs, hidden = self.model(inputs, hidden)
            # Detach hidden state to prevent backprop through entire episode
            if isinstance(hidden, tuple):
                hidden = (hidden[0].detach(), hidden[1].detach())
            else:
                hidden = hidden.detach()
        else:
            outputs = self.model(inputs)

        # Sample actions from policy
        steer_mean = outputs[:, 0]
        accel_mean = outputs[:, 1]

        steer_dist = Normal(steer_mean, 0.1)
        accel_dist = Normal(accel_mean, 0.15)

        steer = torch.clamp(steer_dist.sample(), -1, 1)
        accel = torch.clamp(accel_dist.sample(), -1, 1)

        # Compute log probabilities for policy gradient
        log_prob = steer_dist.log_prob(steer) + accel_dist.log_prob(accel)

        # Physics update
        with torch.no_grad():
            cars.physics_update(steer, accel, dt=0.1, steer_smooth_alpha=self.steer_smooth_alpha)

        # Compute rewards
        step_rewards, prev_dist = self.compute_rewards(
            cars, gate_indices, last_gate_indices, ray_dists, prev_dist, step, n_steps
        )

        # Sum all reward components
        reward = (
            step_rewards["speed"]
            + step_rewards["gate"]
            + step_rewards["collision"]
            + step_rewards["wall"]
            + step_rewards["direction"]
            + step_rewards["alive"]
        )

        return log_prob, reward, step_rewards, prev_dist, hidden

    def train_epoch(self, cars, n_steps, epoch, gate_indices, last_gate_indices):
        """Train for one epoch.

        Args:
            cars: CarState object.
            n_steps: Number of steps per epoch.
            epoch: Current epoch number.
            gate_indices: Gate indices tensor.
            last_gate_indices: Last gate indices tensor.

        Returns:
            episode_reward: Total reward for this epoch.
            mean_rewards: Dictionary of mean rewards by component.
        """
        # Initialize RNN hidden state if needed
        if hasattr(self.model, 'init_hidden'):
            hidden = self.model.init_hidden(cars.n_cars, self.device)
        else:
            hidden = None

        # Initialize buffers
        n_cars = cars.n_cars
        rewards_buffer = {
            "speed": torch.zeros(n_steps, n_cars, device=self.device),
            "gate": torch.zeros(n_steps, n_cars, device=self.device),
            "collision": torch.zeros(n_steps, n_cars, device=self.device),
            "wall": torch.zeros(n_steps, n_cars, device=self.device),
            "direction": torch.zeros(n_steps, n_cars, device=self.device),
            "alive": torch.zeros(n_steps, n_cars, device=self.device),
        }

        log_probs = torch.zeros(n_steps, n_cars, device=self.device)
        step_rewards_total = torch.zeros(n_steps, n_cars, device=self.device)

        prev_dist = torch.zeros(n_cars, device=self.device)

        step_bar = tqdm(
            range(n_steps), desc=f"Epoch {epoch} [{self.track_name}]", leave=False, position=1
        )

        for step in step_bar:
            if not cars.active.any():
                break  # All cars crashed

            # Training step
            log_prob, reward, step_rewards, prev_dist, hidden = self.train_step(
                cars, gate_indices, last_gate_indices, prev_dist, step, n_steps, hidden
            )

            # Store results
            log_probs[step] = log_prob
            step_rewards_total[step] = reward

            for k in step_rewards:
                rewards_buffer[k][step] = step_rewards[k]

            # Update progress bar
            if step % 20 == 0:
                active_count = cars.active.sum().item()
                step_bar.set_postfix(active_cars=f"{active_count}/{n_cars}")

        # Compute returns and normalize
        returns = step_rewards_total.sum(dim=0)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        # Policy gradient loss
        loss = -(log_probs.sum(dim=0) * returns).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute episode reward
        episode_reward = step_rewards_total.sum(dim=0).mean().item()

        # Compute mean rewards per component
        mean_rewards = {}
        for key, buf in rewards_buffer.items():
            mean_rewards[key] = buf.sum(dim=0).mean().item()

        return episode_reward, mean_rewards

    def collect_rollout(self, cars, n_steps, force_direction):
        """Run a single rollout of length `n_steps` with all cars set to `force_direction`.

        Returns:
            log_sum: Tensor (n_cars,) sum of log-probs across steps for each car
            returns: Tensor (n_cars,) undiscounted returns per car
            mean_rewards: dict of mean rewards per component (scalars)
            episode_reward: scalar mean episode reward across cars
        """
        # Force-reset cars to the requested direction
        cars.reset(track=self.track, start_idx=self.track_start_idx, epoch=None, force_direction=force_direction)

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
            "wall": torch.zeros(n_steps, n_cars_local, device=self.device),
            "direction": torch.zeros(n_steps, n_cars_local, device=self.device),
            "alive": torch.zeros(n_steps, n_cars_local, device=self.device),
        }

        log_probs = torch.zeros(n_steps, n_cars_local, device=self.device)
        step_rewards_total = torch.zeros(n_steps, n_cars_local, device=self.device)
        prev_dist = torch.zeros(n_cars_local, device=self.device)

        # Use a step-level progress bar for visibility when running epochs
    
        step_bar = tqdm(range(n_steps), desc=f"Rollout dir={force_direction}", leave=False, position=1)

        for step in step_bar:
            if not cars.active.any():
                break

            log_prob, reward, step_rewards, prev_dist, hidden = self.train_step(
                cars, gate_indices, last_gate_indices, prev_dist, step, n_steps, hidden
            )

            log_probs[step] = log_prob
            step_rewards_total[step] = reward

            for k in step_rewards:
                rewards_buffer[k][step] = step_rewards[k]

            if hasattr(step_bar, "set_postfix") and (step % 20 == 0):
                try:
                    active_count = cars.active.sum().item()
                    step_bar.set_postfix(active_cars=f"{active_count}/{n_cars_local}")
                except Exception:
                    pass

        # Summaries
        returns = step_rewards_total.sum(dim=0)
        episode_reward = returns.mean().item()
        mean_rewards = {k: buf.sum(dim=0).mean().item() for k, buf in rewards_buffer.items()}

        # Sum log-probs over time per car
        log_sum = log_probs.sum(dim=0)

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
        # Initialize car states
        cars = CarState(n_cars, n_rays, device=self.device)

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
                logs_fwd, returns_fwd, mean_fwd, ep_fwd = self.collect_rollout(cars, n_steps, force_direction=1)
                logs_rev, returns_rev, mean_rev, ep_rev = self.collect_rollout(cars, n_steps, force_direction=-1)

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
                logs, returns, mean_rewards, episode_reward = self.collect_rollout(cars, n_steps, force_direction=1)
                returns = (returns - returns.mean()) / (returns.std() + 1e-6)
                loss = -(logs * returns).mean()

            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step(episode_reward)

            if episode_reward > best_reward:
                best_reward = episode_reward

            # Plot raw episode reward (keep normalized returns for loss only)
            norm_episode = combined_returns.mean().item()
            self.plotter.add_reward(episode_reward)

            if epoch % 10 == 0 and epoch != 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                epoch_bar.write(
                    f"Epoch {epoch} [{self.track_name}] | Total reward: {episode_reward:.2f} | "
                    f"Speed:{mean_rewards['speed']:.2f}, Dir.:{mean_rewards['direction']:.2f}, "
                    f"Gate:{mean_rewards['gate']:.2f}, Coll.:{mean_rewards['collision']:.2f}, "
                    f"Wall:{mean_rewards['wall']:.2f}, Alive:{mean_rewards['alive']:.2f} | "
                    f"Best:{best_reward:.2f} | LR:{current_lr:.2e}"
                )

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
