#!/bin/bash

# Configuration variables
N_CARS=512
N_EPOCHS=300
N_STEPS=1000
LR=1e-3
INPUT_DIM=26  # For 17 rays: 17 + 1(speed) + 2(heading) + 6(curvature)
HIDDEN_DIM=128  # Increased from 64 to handle 17 rays (ray_out_dim = 16*17 = 272)
OUTPUT_DIM=2
TRACK="square_narrow"  # "simple", "square", "square_narrow" or "redbull_ring"
DEVICE="mps"  # or "cpu", "cuda", or leave empty for auto-detection
CHECKPOINT="last_ckpt.pth"  # Will be saved as checkpoints/last_ckpt_{ModelName}.pth
MULTI_TRACK=false  # set to true to train on both simple and square_narrow tracks
START_MODE="continue"  # "continue" to continue from last epoch, "start_new" to start from epoch 0

# Activate virtual environment (if needed)
# source env/bin/activate

# Run training with specified arguments
python3 -m train.train_nn \
    --n_cars "$N_CARS" \
    --n_epochs "$N_EPOCHS" \
    --n_steps "$N_STEPS" \
    --lr "$LR" \
    --input_dim "$INPUT_DIM" \
    --hidden_dim "$HIDDEN_DIM" \
    --output_dim "$OUTPUT_DIM" \
    --track "$TRACK" \
    --device "$DEVICE" \
    --checkpoint "$CHECKPOINT" \
    --start_mode "$START_MODE" \
    --force_lr