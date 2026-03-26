"""Entry point for training racing neural network models."""

import os
import json
import torch
import torch.optim as optim
from torch.profiler import profile, ProfilerActivity
import sys

from track import Track
from model import CarNet, Trainer
from utils import args_nn


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

    # Checkpoint path
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    if os.path.dirname(args.checkpoint) == "":
        checkpoint_base = os.path.splitext(args.checkpoint)[0]
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_base}_{model.__class__.__name__}.pth")
    else:
        checkpoint_path = args.checkpoint
    print(f"Checkpoint will be saved to: {checkpoint_path}")

    # Scheduler
    scheduler = None
    if not args.disable_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr
        )

    # Optionally load checkpoint
    start_epoch = 0
    best_reward = -float("inf")
    resume_phase = 0
    resume_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state", {}))
        try:
            optimizer.load_state_dict(checkpoint.get("optimizer_state", {}))
        except Exception:
            pass
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            try:
                scheduler.load_state_dict(checkpoint.get("scheduler_state"))
            except Exception:
                pass
        saved_epoch = checkpoint.get("epoch", 0)
        ckpt_track = checkpoint.get("track_name")
        if args.start_mode == "start_new":
            start_epoch = 0
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
            start_epoch = saved_epoch + 1
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
    first_track = Track(first_track_name, screen_width, screen_height, device=device, ray_method=args.ray_method)
    trainer = Trainer(
        model=model,
        track=first_track,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_path=checkpoint_path,
        max_ray_dist=args.max_ray_dist,
        steer_smooth_alpha=args.steer_smooth_alpha,
    )

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
            # Determine start epoch for this phase
            if i == resume_phase:
                start_epoch_for_phase = resume_epoch
            else:
                start_epoch_for_phase = 0

            trainer.fit(
                n_cars=n_cars,
                n_rays=n_rays,
                n_epochs=phase_epochs,
                n_steps=n_steps,
                start_epoch=start_epoch_for_phase,
                best_reward=best_reward,
                both_directions=True
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
