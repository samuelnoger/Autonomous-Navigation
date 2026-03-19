import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm, trange
from track import Track
from torch.distributions import Normal
import os
import sys

# Model imports (make sure model.py and model_variants.py are in the same directory and importable)
from model import CarState
from model import CarNet
from model import GRUCarNet 
from model_variants import LSTMCarNet

# Utils imports (make sure utils.py is in the same directory and importable)
from utils import compute_step_reward
from utils import get_inputs
from utils import args_nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from plot_rewards import RewardPlotter

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
    """Vectorized training loop for multiple cars using CarState and get_inputs.

    Args:
        model: CarNet already moved to device
        tracks: Track object or list of Track objects for multi-track training
        n_cars: number of cars per batch
        n_rays: number of rays per car
        n_epochs: training epochs
        n_steps: simulation steps per epoch
        lr: learning rate (unused, optimizer provided separately)
        device: torch.device
        start_epoch: epoch to resume from
        start_idx: starting gate index
        checkpoint_path: path to save checkpoints to
        scheduler: optional LR scheduler (e.g. ReduceLROnPlateau)
        best_reward: best mean episode reward seen so far
    """
    import random
    device = device or torch.device("cpu")

    # Initialize reward plotter in a separate process to keep GUI responsive.
    plotter = RewardPlotter()

    # Handle both single track and multi-track
    if not isinstance(tracks, list):
        tracks = [tracks]

    # ---- Initialize car states ----
    cars = CarState(n_cars, n_rays, device=device)

    gate_idx_start = start_idx

    # ---- Gate tracking ----
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
            # ---- Select track (for multi-track training) ----
            track = random.choice(tracks)
            gates_tensor = track.gates.clone().to(device)
            track_start_idx = track.gates.shape[0] - 1
            gate_idx_start = track_start_idx
            track_name = track.track_name

            # Print track selection for debugging
            #epoch_bar.write(f"Selected track for epoch {epoch}: {track_name}")

            # ---- Reset cars each epoch ----
            cars.reset(track=track, start_idx=track_start_idx)

            gate_indices.fill_(gate_idx_start)
            last_gate_indices.fill_(gate_idx_start - 1)
            gate_times.zero_()
            prev_dist.zero_()

            # ---- Initialize LSTM hidden state for this epoch ----
            if hasattr(model, 'init_hidden'):
                hidden = model.init_hidden(n_cars, device)
            else:
                hidden = None

            step_bar = tqdm(range(n_steps), desc=f"Epoch {epoch} [{track_name}]", leave=False, position=1)

            rewards_buffer = {
                "speed": torch.zeros(n_steps, n_cars, device=device),
                "gate": torch.zeros(n_steps, n_cars, device=device),
                "collision": torch.zeros(n_steps, n_cars, device=device),
                "wall": torch.zeros(n_steps, n_cars, device=device),
                "direction": torch.zeros(n_steps, n_cars, device=device),
                "alive": torch.zeros(n_steps, n_cars, device=device),
            }

            log_probs = torch.zeros(n_steps, n_cars, device=device)
            step_rewards_total = torch.zeros(n_steps, n_cars, device=device)

            for step in step_bar:
                if not cars.active.any():
                    break  # all cars inactive

                # ---- Compute NN inputs ----
                next_gate_centers = (
                    gates_tensor[gate_indices, 0:2] + gates_tensor[gate_indices, 2:4]
                ) / 2

                inputs, ray_dists = get_inputs(
                    track,
                    cars,
                    next_gate_centers=next_gate_centers,
                    max_ray_dist=max_ray_dist,
                    gates_tensor=gates_tensor,
                    gate_indices=gate_indices
                )

                # ---- Forward pass ----
                if hidden is not None:
                    # GRU/LSTM model: pass hidden state and get new hidden state
                    outputs, hidden = model(inputs, hidden)
                    # Detach hidden state to prevent backprop through entire episode
                    if isinstance(hidden, tuple):
                        # LSTM returns (h, c) tuple
                        hidden = (hidden[0].detach(), hidden[1].detach())
                    else:
                        # GRU returns single tensor
                        hidden = hidden.detach()
                else:
                    # Regular feedforward model
                    outputs = model(inputs)

                steer_mean = outputs[:, 0]
                accel_mean = outputs[:, 1]

                steer_dist = Normal(steer_mean, 0.1)
                accel_dist = Normal(accel_mean, 0.15)  # Reduced from 0.3 to 0.15 for more consistent learning

                steer = torch.clamp(steer_dist.sample(), -1, 1)
                accel = torch.clamp(accel_dist.sample(), -1, 1)

                log_prob = steer_dist.log_prob(steer) + accel_dist.log_prob(accel)
                log_probs[step] = log_prob

                # Exploration: 10% random actions
                #epsilon = 0.1
                #rand_mask = torch.rand_like(steer) < epsilon
                #steer[rand_mask] = (torch.rand_like(steer[rand_mask]) * 2 - 1) * 0.3

                # ---- Update physics ----
                with torch.no_grad():
                    cars.step_physics(steer, accel, dt=0.1, steer_smooth_alpha=steer_smooth_alpha)

                # ---- Compute rewards ----
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

                reward = (
                    step_rewards["speed"]
                    + step_rewards["gate"]
                    + step_rewards["collision"]
                    + step_rewards["wall"]
                    + step_rewards["direction"]
                    + step_rewards["alive"]
                )

                step_rewards_total[step] = reward

                # Store in buffer
                for k, v in step_rewards.items():
                    rewards_buffer[k][step] = v

                active_count = cars.active.sum().item()  # number of active cars

                if step % 10 == 0:
                    step_bar.set_postfix(active_cars=f"{active_count}/{n_cars}")

            # Compute returns (cumulative rewards)
            returns = step_rewards_total.sum(dim=0)
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)

            # Policy gradient loss
            loss = -(log_probs.sum(dim=0) * returns).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            episode_reward = step_rewards_total.sum(dim=0).mean().item()

            if scheduler is not None:
                scheduler.step(episode_reward)

            # Logging and checkpointing
            mean_rewards = {}
            for key, buf in rewards_buffer.items():
                mean_rewards[key] = buf.sum(dim=0).mean().item()

            current_lr = optimizer.param_groups[0]["lr"]

            if episode_reward > best_reward:
                best_reward = episode_reward

            # Add reward to plotter (every epoch)
            plotter.add_reward(episode_reward)

            if epoch % 10 == 0 and epoch != 0:
                epoch_bar.write(
                    f"Epoch {epoch} [{track_name}] | Total reward: {episode_reward:.2f} | "
                    f"Speed:{mean_rewards['speed']:.2f}, Dir.:{mean_rewards['direction']:.2f}, Gate:{mean_rewards['gate']:.2f}, "
                    f"Coll.:{mean_rewards['collision']:.2f}, Wall:{mean_rewards['wall']:.2f}, "
                    f"Alive:{mean_rewards['alive']:.2f} | Best:{best_reward:.2f} | LR:{current_lr:.2e}"
                )

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

    # Final checkpoint
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
    """Save model and optimizer checkpoint."""
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
    parser = args_nn()

    args = parser.parse_args()

    # Save arguments to config.json for reproducibility
    import json
    with open("config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Show config
    if args.multi_track:
        print(f"Config saved: Multi-track training (square + square_narrow + redbull_ring)")
    else:
        print(f"Config saved: Single-track training ({args.track})")
    print(f"Config file: config.json\n")

    torch.autograd.set_detect_anomaly(False)

    screen_width, screen_height = 1000, 600

    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    n_epochs = args.n_epochs
    n_steps = args.n_steps
    n_cars = args.n_cars
    n_rays = args.n_rays
    max_ray_dist = args.max_ray_dist
    lr = args.lr


    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )

    # Load track(s)
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

    # Load model and optimizer
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create checkpoints directory and set checkpoint path with model name
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_name = model.__class__.__name__  # e.g., "LSTMCarNet", "CarNet", "SeparateHeadsCarNet"

    # If args.checkpoint is just a filename (no directory), use checkpoints/ directory
    if os.path.dirname(args.checkpoint) == "":
        # Extract the base name pattern (e.g., "last_ckpt" from "last_ckpt.pth")
        checkpoint_base = os.path.splitext(args.checkpoint)[0]
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_base}_{model_name}.pth")
    else:
        # User provided a full path, use it as-is
        checkpoint_path = args.checkpoint

    print(f"Checkpoint will be saved to: {checkpoint_path}")

    scheduler = None
    if not args.disable_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )

    start_epoch = 0
    best_reward = -float("inf")

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            if hasattr(scheduler, "min_lrs"):
                scheduler.min_lrs = [args.min_lr for _ in scheduler.optimizer.param_groups]
                print(f"Scheduler min_lr set to {args.min_lr} after checkpoint load.")

        saved_epoch = checkpoint["epoch"]

        if args.start_mode == "start_new":
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
            start_epoch = saved_epoch + 1
            best_reward = checkpoint.get("best_reward", -float("inf"))
            if "current_lr" in checkpoint and not args.force_lr:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = checkpoint["current_lr"]
            elif args.force_lr:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = args.lr
                print(f"Force LR enabled: using lr={args.lr} instead of checkpoint value.")

        resumed_lr = optimizer.param_groups[0]["lr"]
        print(f"Resuming training from epoch {start_epoch} with learning rate {resumed_lr:.2e}...")
        if args.start_mode == "resume" and best_reward > -float("inf"):
            print(f"Best reward so far: {best_reward:.2f}")

    else:
        print("No checkpoint found, starting fresh.")

    # Training loop
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
            start_idx=0,  # will be set per-track inside train_model
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
