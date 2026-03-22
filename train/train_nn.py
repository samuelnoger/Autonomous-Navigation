"""Entry point for training racing neural network models."""

import torch
import torch.optim as optim
import os
import json

from track import Track
from model import CarNet, GRUCarNet, LSTMCarNet, Trainer
from utils import args_nn


def main():
    """Initialize and train the racing model with command-line argument support."""
    # Parse command-line arguments
    parser = args_nn()
    args = parser.parse_args()

    # Save configuration to JSON for reproducibility
    with open("config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Config saved: Training on track '{args.track}'")
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

    # Select device for training
    if args.device:
        device = torch.device(args.device)
    else:
        device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )

    # Load track for training
    track = Track(args.track, screen_width, screen_height, device=device, ray_method=args.ray_method)
    print(f"Training on track: {args.track}")

    print(f"Using device: {device}")
    print(f"Configuration: n_cars={n_cars}, n_epochs={n_epochs}, n_steps={n_steps}, lr={lr}")

    model = CarNet(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_rays=n_rays
    ).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Set up checkpoint directory and path
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_name = model.__class__.__name__

    if os.path.dirname(args.checkpoint) == "":
        checkpoint_base = os.path.splitext(args.checkpoint)[0]
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_base}_{model_name}.pth")
    else:
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

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            if hasattr(scheduler, "min_lrs"):
                scheduler.min_lrs = [args.min_lr for _ in scheduler.optimizer.param_groups]

        saved_epoch = checkpoint["epoch"]

        if args.start_mode == "start_new":
            start_epoch = 0
            best_reward = -float("inf")
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr
            print("Start mode is 'start_new': starting from epoch 0 (weights/optimizer loaded).")
            print(f"Learning rate reset to {args.lr}.")
            if scheduler is not None:
                if hasattr(scheduler, "best"):
                    scheduler.best = -float("inf")
                if hasattr(scheduler, "num_bad_epochs"):
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
        print(f"Resuming from epoch {start_epoch} with learning rate {resumed_lr:.2e}...")
        if best_reward > -float("inf"):
            print(f"Best reward so far: {best_reward:.2f}")

    else:
        print("No checkpoint found, starting fresh.")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        track=track,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_path=checkpoint_path,
        max_ray_dist=max_ray_dist,
        steer_smooth_alpha=args.steer_smooth_alpha,
    )

    # Run the training loop
    try:
        trainer.fit(
            n_cars=n_cars,
            n_rays=n_rays,
            n_epochs=n_epochs,
            n_steps=n_steps,
            start_epoch=start_epoch,
            best_reward=best_reward,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Checkpoint saved.")


if __name__ == "__main__":
    main()
