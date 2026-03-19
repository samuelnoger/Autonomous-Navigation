#!/bin/bash

# Configuration variables
N_CARS=512
N_RAYS=17
N_EPOCHS=400
N_STEPS=1500
LR=5e-4
INPUT_DIM=26  # For 17 rays: 17 + 1(speed) + 2(heading) + 6(curvature)
HIDDEN_DIM=128  # Increased from 64 to handle 17 rays (ray_out_dim = 16*17 = 272)
OUTPUT_DIM=2
TRACK="redbull_ring"  # "simple", "square", "square_narrow" or "redbull_ring"
DEVICE="mps"  # or "cpu", "cuda", or leave empty for auto-detection
CHECKPOINT="last_ckpt.pth"  # Will be saved as checkpoints/last_ckpt_{ModelName}.pth
LR_FACTOR=0.5
LR_PATIENCE=25
MIN_LR=1e-6
DISABLE_LR_SCHEDULER=false  # set to true to disable the scheduler
MULTI_TRACK=false  # set to true to train on both simple and square_narrow tracks
RAY_METHOD="line"  # or "line" for segment intersection
START_MODE="start_new"  # "resume" to continue from last epoch, "start_new" to start from epoch 0
MAX_RAY_DIST=500  # maximum distance for rays
STEER_SMOOTH_ALPHA=0.6  # 0 disables smoothing
MODEL_TYPE="carnet"  # "gru", "lstm", "carnet"

# Activate virtual environment (if needed)
# source env/bin/activate

# Run training with specified arguments
python train_nn.py \
    --n_cars "$N_CARS" \
    --n_rays "$N_RAYS" \
    --n_epochs "$N_EPOCHS" \
    --n_steps "$N_STEPS" \
    --lr "$LR" \
    --input_dim "$INPUT_DIM" \
    --hidden_dim "$HIDDEN_DIM" \
    --output_dim "$OUTPUT_DIM" \
    --track "$TRACK" \
    --device "$DEVICE" \
    --checkpoint "$CHECKPOINT" \
    --lr_factor "$LR_FACTOR" \
    --lr_patience "$LR_PATIENCE" \
    --min_lr "$MIN_LR" \
    --ray_method "$RAY_METHOD" \
    --max_ray_dist "$MAX_RAY_DIST" \
    --steer_smooth_alpha "$STEER_SMOOTH_ALPHA" \
    --start_mode "$START_MODE" \
    --model "$MODEL_TYPE" \
    $( [ "$DISABLE_LR_SCHEDULER" = true ] && echo "--disable_lr_scheduler" ) \
    $( [ "$MULTI_TRACK" = true ] && echo "--multi_track" ) \
    --force_lr
    
