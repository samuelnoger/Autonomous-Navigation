"""
Training script for racing neural network models.

This module implements the main training loop for vehicle control models using
policy gradient learning with vectorized training across multiple cars.
"""

import torch
import torch.optim as optim
from torch.distributions import Normal
import os
import random
import json

from tqdm import tqdm, trange

# Relative imports from parent directory (Race/)
from track import Track
from model import CarState, CarNet, GRUCarNet, LSTMCarNet
from utils import compute_step_reward, get_inputs, args_nn, RewardPlotter

# Default path for saving training checkpoints
CHECKPOINT_PATH = "last_ckpt.pth"


def train_model(
    model,
    tracks,
    optimizer,
    n_cars=64,
    n_rays=9,
    n_epochs=50,
    n_steps=800,
    lr=1e-3,
    device=None,
    start_epoch=0,
    start_idx=0,
    checkpoint_path=CHECKPOINT_PATH,
    scheduler=None,
    best_reward=-float("inf"),
    max_ray_dist=500,
    steer_smooth_alpha=0.0,
):
    """Vectorized training loop for multiple cars using policy gradient learning.

    Uses CarState for efficient vectorized physics simulation with multiple cars.
    At each step, the model predicts steer and acceleration for all cars simultaneously.
    Rewards are computed for each car based on speed, gate progress, collisions, etc.
    Policy gradients are accumulated and backpropagated after each epoch.

    Args:
        model: CarNet (or variants) already moved to device.
        tracks: Track object or list of Track objects for multi-track training.
        optimizer: Torch optimizer (e.g., Adam) for model parameters.
        n_cars: Number of cars per batch (vectorized training).
        n_rays: Number of rays per car for obstacle detection.
        n_epochs: Total number of training epochs.
        n_steps: Maximum simulation steps per epoch.
        lr: Learning rate (unused here, optimizer provided separately).
        device: torch.device to use for training.
        start_epoch: Epoch to resume from (for checkpoint restoring).
        start_idx: Starting gate index for each car.
        checkpoint_path: Where to save model checkpoints.
        scheduler: Optional LR scheduler (e.g., ReduceLROnPlateau).
        best_reward: Best mean episode reward seen so far (for tracking).
        max_ray_dist: Maximum distance for ray casting.
        steer_smooth_alpha: Smoothing factor for steering (0.0 = no smoothing).
    """
    device = device or torch.device("cpu")

    # Initialize reward plotter in a separate process to keep GUI responsive
    plotter = RewardPlotter()

    # Handle both single track and multi-track training setups
    if not isinstance(tracks, list):
        tracks = [tracks]

    # Initialize vectorized car states for parallel training
    cars = CarState(n_cars, n_rays, device=device)

    gate_idx_start = start_idx

    # Initialize gate tracking tensors for all cars
    # Each car tracks which gate it's currently aiming for and has passed
    gate_indices = torch.full(
        (n_cars,), gate_idx_start, dtype=torch.long, device=device
    )
    last_gate_indices = torch.full(
        (n_cars,), gate_idx_start - 1, dtype=torch.long, device=device
    )
    gate_times = torch.zeros(n_cars, device=device)
    prev_dist = torch.zeros(n_cars, device=device)

    epoch_bar = trange(
        start_epoch,
        n_epochs,
        leave=True,
        position=2,
        initial=start_epoch,
        total=n_epochs,
        desc=f"Training",
    )

    try:
        for epoch in epoch_bar:
            # Randomly select a track for this epoch (supports multi-track training)
            track = random.choice(tracks)
            gates_tensor = track.gates.clone().to(device)
            track_start_idx = track.gates.shape[0] - 1
            gate_idx_start = track_start_idx
            track_name = track.track_name

            # Reset all cars to starting position for this epoch
            cars.reset(track=track, start_idx=track_start_idx)

            # Reset gate tracking indices for all cars
            gate_indices.fill_(gate_idx_start)
            last_gate_indices.fill_(gate_idx_start - 1)
            gate_times.zero_()
            prev_dist.zero_()

            # Initialize RNN hidden state for this epoch (GRU/LSTM models only)
            if hasattr(model, 'init_hidden'):
                hidden = model.init_hidden(n_cars, device)
            else:
                hidden = None

            step_bar = tqdm(range(n_steps), desc=f"Epoch {epoch} [{track_name}]", leave=False, position=1)

            # Initialize buffers to store rewards for each reward component across all steps
            rewards_buffer = {
                "speed": torch.zeros(n_steps, n_cars, device=device),
                "gate": torch.zeros(n_steps, n_cars, device=device),
                "collision": torch.zeros(n_steps, n_cars, device=device),
                "wall": torch.zeros(n_steps, n_cars, device=device),
                "direction": torch.zeros(n_steps, n_cars, device=device),
                "alive": torch.zeros(n_steps, n_cars, device=device),
            }

            # Buffers for policy gradients and total rewards per step
            log_probs = torch.zeros(n_steps, n_cars, device=device)
            step_rewards_total = torch.zeros(n_steps, n_cars, device=device)

            for step in step_bar:
                if not cars.active.any():
                    break  # Early exit if all cars have crashed

                # Compute the center point of the next gate each car is aiming for
                next_gate_centers = (
                    gates_tensor[gate_indices, 0:2] + gates_tensor[gate_indices, 2:4]
                ) / 2

                # Get ray distances and model inputs from the track and car states
                inputs, ray_dists = get_inputs(
                    track,
                    cars,
                    next_gate_centers=next_gate_centers,
                    max_ray_dist=max_ray_dist,
                    gates_tensor=gates_tensor,
                    gate_indices=gate_indices
                )

                # Forward pass through the model
                if hidden is not None:
                    # RNN model: pass hidden state and get updated hidden state
                    outputs, hidden = model(inputs, hidden)
                    # Detach hidden state to prevent backprop through entire episode
                    if isinstance(hidden, tuple):
                        # LSTM returns (h, c) tuple
                        hidden = (hidden[0].detach(), hidden[1].detach())
                    else:
                        # GRU returns single tensor
                        hidden = hidden.detach()
                else:
                    # Feedforward model: no hidden state
                    outputs = model(inputs)

                # Extract mean values for steer and acceleration from model output
                steer_mean = outputs[:, 0]
                accel_mean = outputs[:, 1]

                # Create normal distributions for stochastic action sampling
                steer_dist = Normal(steer_mean, 0.1)
                accel_dist = Normal(accel_mean, 0.15)  # std reduced from 0.3 for more consistent learning

                steer = torch.clamp(steer_dist.sample(), -1, 1)
                accel = torch.clamp(accel_dist.sample(), -1, 1)

                # Compute log probabilities for policy gradient
                log_prob = steer_dist.log_prob(steer) + accel_dist.log_prob(accel)
                log_probs[step] = log_prob

                # Physics update: compute new positions based on actions
                with torch.no_grad():
                    cars.physics_update(steer, accel, dt=0.1, steer_smooth_alpha=steer_smooth_alpha)

                # Compute reward components for this step
                step_rewards, prev_dist = compute_step_reward(
                    cars,
                    track,
                    gates_tensor,
                    gate_indices,
                    last_gate_indices,
                    ray_dists,
                    prev_dist,
                    step,
                    n_steps,
                    max_ray_dist
                )

                # Sum all reward components for total reward this step
                reward = (
                    step_rewards["speed"]
                    + step_rewards["gate"]
                    + step_rewards["collision"]
                    + step_rewards["wall"]
                    + step_rewards["direction"]
                    + step_rewards["alive"]
                )

                step_rewards_total[step] = reward

                # Store individual reward components in buffer for logging
                for k in step_rewards:
                    rewards_buffer[k][step] = step_rewards[k]

                if step % 20 == 0:
                    active_count = cars.active.sum().item()
                    step_bar.set_postfix(active_cars=f"{active_count}/{n_cars}")

            # Compute and Normalize returns to reduce variance in policy gradient estimates
            returns = step_rewards_total.sum(dim=0)
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)

            # Policy gradient loss: maximize expected return weighted by log probability
            loss = -(log_probs.sum(dim=0) * returns).mean()

            # Backpropagation: reset gradients, compute gradients, update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute mean reward for this epoch across all cars
            episode_reward = step_rewards_total.sum(dim=0).mean().item()

            # Update learning rate scheduler if provided
            if scheduler is not None:
                scheduler.step(episode_reward)

            # Compute mean reward for each component across all cars and steps
            mean_rewards = {}
            for key, buf in rewards_buffer.items():
                mean_rewards[key] = buf.sum(dim=0).mean().item()

            # Track current learning rate
            current_lr = optimizer.param_groups[0]["lr"]

            # Update best reward tracking
            if episode_reward > best_reward:
                best_reward = episode_reward

            # Send reward to plotter for real-time visualization
            plotter.add_reward(episode_reward)

            # Log metrics every 10 epochs
            if epoch % 10 == 0 and epoch != 0:
                epoch_bar.write(
                    f"Epoch {epoch} [{track_name}] | Total reward: {episode_reward:.2f} | "
                    f"Speed:{mean_rewards['speed']:.2f}, Dir.:{mean_rewards['direction']:.2f}, Gate:{mean_rewards['gate']:.2f}, "
                    f"Coll.:{mean_rewards['collision']:.2f}, Wall:{mean_rewards['wall']:.2f}, "
                    f"Alive:{mean_rewards['alive']:.2f} | Best:{best_reward:.2f} | LR:{current_lr:.2e}"
                )

            # Save checkpoint every epoch
            _save_checkpoint(
                epoch,
                model,
                optimizer,
                checkpoint_path,
                total_epochs=n_epochs,
                scheduler=scheduler,
                best_reward=best_reward,
            )
    finally:
        plotter.close()

    # Final checkpoint save
    _save_checkpoint(
        epoch,
        model,
        optimizer,
        checkpoint_path,
        total_epochs=n_epochs,
        scheduler=scheduler,
        best_reward=best_reward,
    )


def _save_checkpoint(
    epoch,
    model,
    optimizer,
    checkpoint_path=CHECKPOINT_PATH,
    total_epochs=None,
    scheduler=None,
    best_reward=-float("inf"),
):
    """Save model and optimizer state for checkpoint resuming.

    Saves model weights, optimizer state, learning rate scheduler state,
    and training metadata needed to resume training from this epoch.

    Args:
        epoch: Current epoch number.
        model: Model with state to save.
        optimizer: Optimizer with state to save.
        checkpoint_path: Where to save the checkpoint.
        total_epochs: Total number of epochs planned for training.
        scheduler: Optional LR scheduler with state to save.
        best_reward: Best reward achieved so far.
    """
    torch.save(
        {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "current_lr": optimizer.param_groups[0]["lr"],
            "best_reward": best_reward,
        },
        checkpoint_path,
    )


def main():
    """Initialize and train the racing model with command-line argument support."""
    # Parse command-line arguments
    parser = args_nn()
    args = parser.parse_args()

    # Save configuration to JSON for reproducibility
    with open("config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Print training setup information
    if args.multi_track:
        print(f"Config saved: Multi-track training (square + square_narrow + redbull_ring)")
    else:
        print(f"Config saved: Single-track training ({args.track})")
    print(f"Config file: config.json\n")

    # Disable anomaly detection for faster training
    torch.autograd.set_detect_anomaly(False)

    screen_width, screen_height = 1000, 600

    # Extract training hyperparameters from arguments
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    n_epochs = args.n_epochs
    n_steps = args.n_steps
    n_cars = args.n_cars
    n_rays = args.n_rays
    max_ray_dist = args.max_ray_dist
    lr = args.lr

    # Select device for training (GPU/Metal/CPU)
    if args.device:
        device = torch.device(args.device)
    else:
        device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )

    # Load track(s) for training
    if args.multi_track:
        tracks = [
            Track("square", screen_width, screen_height, device=device, ray_method=args.ray_method),
            Track("square_narrow", screen_width, screen_height, device=device, ray_method=args.ray_method),
            Track("redbull_ring", screen_width, screen_height, device=device, ray_method=args.ray_method)
        ]
        print(f"Multi-track training: square + square_narrow + redbull_ring")
    else:
        track = Track(args.track, screen_width, screen_height, device=device, ray_method=args.ray_method)
        tracks = [track]
        print(f"Single-track training: {args.track}")

    print(f"Using device: {device}")
    print(f"Configuration: n_cars={n_cars}, n_rays={n_rays}, n_epochs={n_epochs}, n_steps={n_steps}, lr={lr}")

    # Initialize model based on model type argument
    if args.model == "lstm":
        model = LSTMCarNet(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_rays=n_rays, lstm_layers=1
        ).to(device)
        print(f"Using LSTMCarNet with 1 LSTM layer")
    elif args.model == "gru":
        model = GRUCarNet(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_rays=n_rays, gru_layers=1
        ).to(device)
        print(f"Using GRUCarNet with 1 GRU layer")
    elif args.model == "carnet":
        model = CarNet(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_rays=n_rays
        ).to(device)
        print(f"Using CarNet")
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Set up checkpoint directory and path with model name
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_name = model.__class__.__name__

    # Construct checkpoint path using model name to avoid conflicts
    if os.path.dirname(args.checkpoint) == "":
        # No directory in checkpoint path, use checkpoints/ directory
        checkpoint_base = os.path.splitext(args.checkpoint)[0]
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_base}_{model_name}.pth")
    else:
        # User provided full path, use as-is
        checkpoint_path = args.checkpoint

    print(f"Checkpoint will be saved to: {checkpoint_path}")

    # Initialize learning rate scheduler if not disabled
    scheduler = None
    if not args.disable_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )

    # Track training state for resuming from checkpoints
    start_epoch = 0
    best_reward = -float("inf")

    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Restore model and optimizer state
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        # Restore scheduler state if applicable
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            if hasattr(scheduler, "min_lrs"):
                scheduler.min_lrs = [args.min_lr for _ in scheduler.optimizer.param_groups]
                print(f"Scheduler min_lr set to {args.min_lr} after checkpoint load.")

        saved_epoch = checkpoint["epoch"]

        # Handle different resume modes
        if args.start_mode == "start_new":
            # Start new training but keep loaded weights
            start_epoch = 0
            best_reward = -float("inf")
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr
            print("Start mode is 'start_new': starting from epoch 0 (weights/optimizer loaded from checkpoint).")
            print(f"Learning rate reset to {args.lr}.")
            if scheduler is not None and hasattr(scheduler, "best"):
                scheduler.best = -float("inf")
            if scheduler is not None and hasattr(scheduler, "num_bad_epochs"):
                scheduler.num_bad_epochs = 0
        else:
            # Resume training from where it left off
            start_epoch = saved_epoch + 1
            best_reward = checkpoint.get("best_reward", -float("inf"))
            if "current_lr" in checkpoint and not args.force_lr:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = checkpoint["current_lr"]
            elif args.force_lr:
                # Override checkpoint LR with command-line LR
                for param_group in optimizer.param_groups:
                    param_group["lr"] = args.lr
                print(f"Force LR enabled: using lr={args.lr} instead of checkpoint value.")

        resumed_lr = optimizer.param_groups[0]["lr"]
        print(f"Resuming training from epoch {start_epoch} with learning rate {resumed_lr:.2e}...")
        if args.start_mode == "resume" and best_reward > -float("inf"):
            print(f"Best reward so far: {best_reward:.2f}")

    else:
        print("No checkpoint found, starting fresh.")

    # Run the training loop
    try:
        train_model(
            model,
            tracks,
            optimizer=optimizer,
            n_cars=n_cars,
            n_rays=n_rays,
            n_epochs=n_epochs,
            n_steps=n_steps,
            device=device,
            start_epoch=start_epoch,
            start_idx=0,  # Will be set per-track inside train_model
            checkpoint_path=checkpoint_path,
            scheduler=scheduler,
            best_reward=best_reward,
            max_ray_dist=max_ray_dist,
            steer_smooth_alpha=args.steer_smooth_alpha,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Checkpoint saved.")

if __name__ == "__main__":
    main()
