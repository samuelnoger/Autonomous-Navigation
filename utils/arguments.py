import argparse

def args_nn():
    parser = argparse.ArgumentParser(description="Train racing neural network model")
    parser.add_argument("--n_cars", type=int, default=512, help="Number of cars per batch")
    parser.add_argument("--n_rays", type=int, default=15, help="Number of rays per car")
    parser.add_argument("--n_epochs", type=int, default=500, help="Number of epochs to train")
    parser.add_argument("--n_steps", type=int, default=1000, help="Steps per epoch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--input_dim", type=int, default=26, help="Input dimension size (n_rays[15] + speed[1] + heading[2] + dist_to_gate[1] + min_ray[1] + curvature[6] = 26)")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size")
    parser.add_argument("--output_dim", type=int, default=2, help="Output dimension size")
    parser.add_argument("--track", type=str, default="simple", help="Track name or path to geojson")
    parser.add_argument("--multi_track", action="store_true", help="Train on multiple tracks (simple and square_narrow)")
    parser.add_argument("--device",type=str,default=None,help="Device (mps/cpu/cuda), auto-detect if not specified",)
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file path")
    parser.add_argument("--lr_factor",type=float,default=0.5,help="ReduceLROnPlateau factor (new_lr = lr * factor)")
    parser.add_argument("--lr_patience",type=int,default=20,help="Epochs without reward improvement before reducing LR",)
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--disable_lr_scheduler",action="store_true",help="Disable automatic learning-rate reduction on plateaus",)
    parser.add_argument("--force_lr", action="store_true", help="Force learning rate to --lr even when resuming from checkpoint")
    parser.add_argument("--ray_method", type=str, default="sphere", choices=["sphere", "line"], help="Ray casting method: 'sphere' (distance field) or 'line' (segment intersection)")
    parser.add_argument("--start_mode", type=str, default="continue", choices=["continue", "restart"], help="'continue' to continue from last epoch, 'start_new' to start from epoch 0 even if checkpoint exists.")
    parser.add_argument("--max_ray_dist", type=float, default=200.0, help="Maximum distance for ray inputs (for normalization)")
    parser.add_argument("--steer_smooth_alpha", type=float, default=0.0, help="Steering smoothing factor (0 disables smoothing)")
    parser.add_argument("--model", type=str, default="gru", choices=["gru", "lstm", "carnet"], help="Model type to train: 'gru', 'lstm', or 'carnet'")

    return parser