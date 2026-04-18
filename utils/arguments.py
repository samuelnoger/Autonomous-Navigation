import argparse

import argparse


def args_nn():
    parser = argparse.ArgumentParser(description="Train neural network model")

    # Environment / dataset
    parser.add_argument("--track", type=str, default="simple", help="Track name or path to geojson")
    parser.add_argument("--tracks", nargs="+", type=str, default=None, help="List of track names to train sequentially")
    parser.add_argument("--epochs", nargs="+", type=int, default=None, help="List of epoch counts matching --tracks")

    # Run / device
    parser.add_argument("--device", type=str, default=None, help="Device (mps/cpu/cuda), auto-detect if not specified")
    parser.add_argument("--checkpoint", type=str, default="checkpoints", help="Checkpoint file or base directory")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Optional checkpoint path to load when switching to a new track")

    # Batch / model size
    parser.add_argument("--n_cars", type=int, default=512, help="Number of cars per batch")
    parser.add_argument("--n_rays", type=int, default=17, help="Number of rays per car")
    parser.add_argument("--input_dim", type=int, default=39, help="Input dimension size (n_rays + speed + curvature + distances)")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size")
    parser.add_argument("--output_dim", type=int, default=2, help="Output dimension size")

    # Training schedule
    parser.add_argument("--n_epochs", type=int, default=500, help="Number of epochs to train (used when --epochs not provided)")
    parser.add_argument("--n_steps", type=int, default=1000, help="Steps per epoch")

    # Optimizer / LR schedule
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="ReduceLROnPlateau factor")
    parser.add_argument("--lr_patience", type=int, default=25, help="Epochs without improvement before reducing LR")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--disable_lr_scheduler", action="store_true", help="Disable LR scheduler")
    parser.add_argument("--force_lr", action="store_true", help="Force learning rate to --lr even when resuming")

    # Sim / physics / spawn
    parser.add_argument("--ray_method", type=str, default="line", choices=["sphere", "line"], help="Ray casting method: 'sphere' or 'line'")
    parser.add_argument("--start_mode", type=str, default="start_new", choices=["continue", "start_new"], help="'continue' to continue from last epoch, 'start_new' to start from epoch 0 even if checkpoint exists.")
    parser.add_argument("--car_start_mode", type=str, default="random", choices=["random", "f1"], help="Car spawn layout: 'random' or 'f1'")
    parser.add_argument("--group_size", type=int, default=8, help="Number of cars per interaction group (used with --car_start_mode=f1)")
    parser.add_argument("--car_length", type=float, default=8.0, help="Car length in pixels (used for spawn spacing and collision)")
    parser.add_argument("--car_width", type=float, default=4.0, help="Car width in pixels (used for spawn spacing and collision)")
    parser.add_argument("--start_spacing", type=float, default=1.5, help="Longitudinal spacing multiplier at start")
    parser.add_argument("--max_ray_dist", type=float, default=500.0, help="Maximum distance for ray inputs (for normalization)")

    # Update / chunking / early end
    parser.add_argument("--update_frequency", type=int, default=100, help="If >0, perform optimizer updates every this many steps during rollouts (0=update at episode end)")
    parser.add_argument("--min_alive_per_group", type=int, default=2, help="Early end condition: if every group has <= this many active cars, end the rollout")

    # Rewards / exploration
    parser.add_argument("--progress_reward_weight", type=float, default=1.0, help="Weight for per-step delta-progress reward")
    parser.add_argument("--overtake_reward", type=float, default=5.0, help="Reward magnitude per position gained when overtaking")
    parser.add_argument("--speed_reward_rate", type=float, default=0.0025, help="Per-step speed reward rate (multiplies speed)")
    parser.add_argument("--collision_penalty_rate", type=float, default=60.0, help="Penalty applied on collision / crash")
    parser.add_argument("--gate_pass_reward_rate", type=float, default=2.0, help="Reward for passing a gate")
    parser.add_argument("--wall_penalty_rate", type=float, default=5.0, help="Penalty rate for proximity to walls")
    parser.add_argument("--direction_reward_rate", type=float, default=0.0, help="Reward rate for heading toward next gate")
    parser.add_argument("--alive_reward_rate", type=float, default=0.02, help="Small per-step reward for being active")

    # Noise / smoothing
    parser.add_argument("--steer_smooth_alpha", type=float, default=0.8, help="Steering smoothing factor (0 disables smoothing)")
    parser.add_argument("--steer_noise", type=float, default=0.01, help="Stddev of Gaussian noise added to steering output during training")
    parser.add_argument("--accel_noise", type=float, default=0.04, help="Stddev of Gaussian noise added to acceleration output during training")

    # Output / misc
    parser.add_argument("--save_plot", type=str, default="reward_plot.png", help="Path to save the final reward plot image after training; set empty to disable saving.")

    return parser