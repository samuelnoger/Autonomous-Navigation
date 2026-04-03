#!/bin/bash

# Configuration variables
N_CARS=512
N_EPOCHS=300
N_STEPS=1600
LR=5e-4
LR_PATIENCE=40
INPUT_DIM=27
OUTPUT_DIM=2
TRACK="triangle"  
DEVICE="cpu"  
CHECKPOINT="/Users/samuelnoger/Programms/Drive_NN/checkpoints/"
RESUME_CHECKPOINT="/Users/samuelnoger/Programms/Drive_NN/checkpoints/us_track/last.pth" 
START_MODE="start_new"
TRACKS=("triangle")
EPOCHS=(100)  

# Run training with specified arguments
python3 -m train.train_nn \
    --n_cars "$N_CARS" \
    --n_epochs "$N_EPOCHS" \
    --n_steps "$N_STEPS" \
    --lr "$LR" \
    --lr_patience "$LR_PATIENCE" \
    --input_dim "$INPUT_DIM" \
    --output_dim "$OUTPUT_DIM" \
    --track "$TRACK" \
    --device "$DEVICE" \
    --checkpoint "$CHECKPOINT" \
    --start_mode "$START_MODE" \
    --resume_checkpoint "$RESUME_CHECKPOINT" \
    --tracks "${TRACKS[@]}" \
    --epochs "${EPOCHS[@]}" \
    --force_lr