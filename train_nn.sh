#!/bin/bash

# Configuration variables
N_CARS=512
N_EPOCHS=500
N_STEPS=1000
LR=5e-4
LR_PATIENCE=25
INPUT_DIM=24  # For 17 rays: 17 + 1(speed) + 2(heading) + 6(curvature)
HIDDEN_DIM=128  # Increased from 64 to handle 17 rays (ray_out_dim = 16*17 = 272)
OUTPUT_DIM=2
TRACK="redbull_ring"  # fallback single track: "simple", "square", "square_narrow" or "redbull_ring"
DEVICE="mps"  # or "cpu", "cuda", or leave empty for auto-detection
CHECKPOINT="last_ckpt.pth"  # Will be saved as checkpoints/last_ckpt_{ModelName}.pth
START_MODE="start_new"  # "continue" to continue from last epoch, "start_new" to start from epoch 0
TRACKS=("simple" "square" "square_narrow" "triangle")  # List of tracks to train on
EPOCHS=(40 40 30 20)

# Run training with specified arguments
python3 -m train.train_nn \
    --n_cars "$N_CARS" \
    --n_epochs "$N_EPOCHS" \
    --n_steps "$N_STEPS" \
    --lr "$LR" \
    --lr_patience "$LR_PATIENCE" \
    --input_dim "$INPUT_DIM" \
    --hidden_dim "$HIDDEN_DIM" \
    --output_dim "$OUTPUT_DIM" \
    --track "$TRACK" \
    --device "$DEVICE" \
    --checkpoint "$CHECKPOINT" \
    --start_mode "$START_MODE" \
    --tracks "${TRACKS[@]}" \
    --epochs "${EPOCHS[@]}" \
    --force_lr \