"""Entry point for training racing neural network models."""

import os
import json
import torch
import time
import torch.optim as optim
from torch.profiler import profile, ProfilerActivity
import sys

from track import Track
from model import CarNet, Trainer
from utils import args_nn, load_checkpoint_tolerant


def load_checkpoint_state(checkpoint_path, model, optimizer, scheduler, device):
    """Load checkpoint and return metadata.

    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model to load state into.
        optimizer: Optimizer to load state into.
        scheduler: Optional scheduler to load state into.
        device: Device to load on.

    Returns:
        Tuple of (saved_epoch, best_reward, track_name)
    """
    # Use tolerant loader which copies matching parameters and partially copies
    # parameters when shapes differ (e.g. input-dimension changed). It will also
    # attempt to restore optimizer and scheduler state when available.
    ckpt = load_checkpoint_tolerant(checkpoint_path, model, optimizer=optimizer, scheduler=scheduler, device=device, verbose=True)
    if isinstance(ckpt, dict):
        saved_epoch = ckpt.get("epoch", 0)
        best_reward = ckpt.get("best_reward", -float("inf"))
        track_name = ckpt.get("track_name")
    else:
        saved_epoch = 0
        best_reward = -float("inf")
        track_name = None
    return saved_epoch, best_reward, track_name


def main():
    parser = args_nn()
    args = parser.parse_args()

    # Save configuration for reproducibility
    with open("config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Config saved: Training on track '{args.track}'")
    print("Config file: config.json\n")

    # Disable anomaly detection for performance
    torch.autograd.set_detect_anomaly(False)

    # Screen for Track creation
    screen_width, screen_height = 1000, 600

    # Hyperparameters from args
    n_steps = args.n_steps
    n_cars = args.n_cars
    n_rays = args.n_rays
    lr = args.lr

    # Device selection
    device = (
        torch.device(args.device)
        if args.device
        else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    )

    # Build schedule from --tracks/--epochs or single track
    tracks_list = args.tracks if args.tracks else [args.track]
    epochs_list = args.epochs if args.epochs else [args.n_epochs] * len(tracks_list)
    if len(tracks_list) != len(epochs_list):
        raise ValueError("Number of tracks and epochs must match")
    schedule = list(zip(tracks_list, epochs_list))

    print(f"Training schedule: {schedule}")
    print(f"Using device: {device}")

    # Model and optimizer
    model = CarNet(input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, n_rays=n_rays).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Checkpoint path: allow either a full .pth path or a base directory.
    # If args.checkpoint ends with '.pth' treat as full path; otherwise
    # treat it as a base directory and save under base_dir/{track_name}/last.pth
    def build_checkpoint_path(arg_checkpoint, track_name):
        if arg_checkpoint is None or arg_checkpoint == "":
            base_dir = "checkpoints"
        elif arg_checkpoint.endswith('.pth'):
            return arg_checkpoint
        else:
            base_dir = arg_checkpoint

        # ensure directory exists
        path = os.path.join(base_dir, track_name, "last.pth")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    # Build initial checkpoint path using first track in schedule
    checkpoint_path = build_checkpoint_path(args.checkpoint, tracks_list[0])
    print(f"Checkpoint will be saved to: {checkpoint_path}")

    # Scheduler
    scheduler = None
    if not args.disable_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr
        )

    # Optionally load checkpoint
    best_reward = -float("inf")
    resume_phase = 0
    resume_epoch = 0

    # Determine which checkpoint to load for the initial phase
    initial_checkpoint_path = None
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        initial_checkpoint_path = args.resume_checkpoint
    elif os.path.exists(checkpoint_path):
        initial_checkpoint_path = checkpoint_path

    if initial_checkpoint_path:
        print(f"Loading checkpoint from {initial_checkpoint_path}...")
        checkpoint = load_checkpoint_tolerant(initial_checkpoint_path, model, optimizer=optimizer, scheduler=scheduler, device=device, verbose=True)
        # checkpoint may be dict or other return; guard accesses
        if isinstance(checkpoint, dict):
            saved_epoch = checkpoint.get("epoch", 0)
            ckpt_track = checkpoint.get("track_name")
        else:
            saved_epoch = 0
            ckpt_track = None
        if args.start_mode == "start_new":
            best_reward = -float("inf")
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr
            if scheduler is not None:
                if hasattr(scheduler, "best"):
                    scheduler.best = -float("inf")
                if hasattr(scheduler, "num_bad_epochs"):
                    scheduler.num_bad_epochs = 0
        else:
            # Resume training using the checkpoint metadata. We resume model
            # and optimizer state; we also determine which schedule phase and
            # epoch to continue from if the checkpoint recorded a track name.
            best_reward = checkpoint.get("best_reward", -float("inf"))
            if ckpt_track is not None and ckpt_track in tracks_list:
                resume_phase = tracks_list.index(ckpt_track)
                resume_epoch = saved_epoch + 1
            else:
                resume_phase = 0
                resume_epoch = saved_epoch + 1
            if "current_lr" in checkpoint and not args.force_lr:
                for pg in optimizer.param_groups:
                    pg["lr"] = checkpoint.get("current_lr", pg.get("lr", args.lr))
            elif args.force_lr:
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr
        resumed_lr = optimizer.param_groups[0]["lr"]
        print(f"Loaded checkpoint (track={ckpt_track}, epoch={saved_epoch}). Resuming from epoch {resume_epoch} with learning rate {resumed_lr:.2e}...")
    else:
        print("No checkpoint found, starting fresh.")

    # Create first Track and Trainer
    first_track_name = schedule[0][0]
    t0_track = time.time()
    first_track = Track(first_track_name, screen_width, screen_height, device=device, ray_method=args.ray_method)
    # Startup track construction timing print removed
    trainer = Trainer(
        model=model,
        track=first_track,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_path=checkpoint_path,
        args=args,
    )
    # Startup model/trainer creation print removed

    # Run scheduled phases. If resuming from a checkpoint we start at
    # `resume_phase` and for that phase begin at `resume_epoch`.
    try:
        for i in range(resume_phase, len(schedule)):
            track_name, phase_epochs = schedule[i]
            if i == 0 and track_name == first_track_name:
                track = first_track
            else:
                track = Track(track_name, screen_width, screen_height, device=device, ray_method=args.ray_method)

            print(f"\n=== Phase: training on '{track_name}' for {phase_epochs} epochs ===")

            trainer.track = track
            trainer.gates_tensor = track.gates.clone().to(device)
            trainer.track_start_idx = track.gates.shape[0] - 1
            trainer.track_name = track.track_name

            # Update per-phase checkpoint path (store per-track under base dir)
            trainer.checkpoint_path = build_checkpoint_path(args.checkpoint, track_name)

            # Load track-specific checkpoint when switching to a new track
            # (only if this is not the first phase being initially loaded)
            if i > 0:
                if args.resume_checkpoint:
                    # Use manually specified checkpoint
                    if os.path.exists(args.resume_checkpoint):
                        saved_epoch, best_reward, _ = load_checkpoint_state(
                            args.resume_checkpoint, model, optimizer, scheduler, device
                        )
                        start_epoch_for_phase = 0
                        print(f"Loaded manual checkpoint from {args.resume_checkpoint}")
                    else:
                        print(f"Warning: specified resume_checkpoint {args.resume_checkpoint} not found, continuing without loading")
                        start_epoch_for_phase = 0
                elif os.path.exists(trainer.checkpoint_path):
                    # Load track-specific checkpoint if it exists
                    saved_epoch, best_reward, _ = load_checkpoint_state(
                        trainer.checkpoint_path, model, optimizer, scheduler, device
                    )
                    start_epoch_for_phase = 0
                else:
                    # No checkpoint for this track, continue with current weights
                    print(f"No checkpoint found for track '{track_name}', continuing with current model weights")
                    start_epoch_for_phase = 0
            else:
                # First phase: use epoch from initial checkpoint loading
                start_epoch_for_phase = resume_epoch

            for pg in optimizer.param_groups:
                pg["lr"] = lr
            if scheduler is not None:
                if hasattr(scheduler, "best"):
                    scheduler.best = -float("inf")
                if hasattr(scheduler, "num_bad_epochs"):
                    scheduler.num_bad_epochs = 0
                if hasattr(scheduler, "cooldown_counter"):
                    scheduler.cooldown_counter = 0
                if hasattr(scheduler, "last_epoch"):
                    scheduler.last_epoch = -1

            trainer.optimizer = optimizer
            trainer.scheduler = scheduler

            trainer.fit(
                n_cars=n_cars,
                n_rays=n_rays,
                n_epochs=phase_epochs,
                n_steps=n_steps,
                start_epoch=start_epoch_for_phase,
                best_reward=best_reward,
                both_directions=False
            )
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Checkpoint saved.")
    finally:
        save_path = getattr(args, "save_plot", None)
        if save_path:
            try:
                trainer.plotter.save(save_path)
                print(f"Saved reward plot to {save_path}")
            except Exception as e:
                print(f"Warning: failed to save reward plot: {e}")
        try:
            trainer.plotter.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
