#!/bin/bash

# Configuration variables
N_CARS=512
N_STEPS=1000
LR=5e-4
LR_PATIENCE=50
INPUT_DIM=39
OUTPUT_DIM=2
STEER_NOISE=0.01
ACCEL_NOISE=0.02
TRACK="triangle"  
DEVICE="cpu"  
CAR_START_MODE="f1"
GROUP_SIZE=6
UPDATE_FREQUENCY=1000
CHECKPOINT=""
RESUME_CHECKPOINT="" 
START_MODE="start_new"
TRACKS=("triangle")
EPOCHS=(300)  

# Run training with specified arguments
python3 -m train.train_nn \
    --n_cars "$N_CARS" \
    --n_steps "$N_STEPS" \
    --lr "$LR" \
    --lr_patience "$LR_PATIENCE" \
    --input_dim "$INPUT_DIM" \
    --output_dim "$OUTPUT_DIM" \
    --steer_noise "$STEER_NOISE" \
    --accel_noise "$ACCEL_NOISE" \
    --track "$TRACK" \
    --device "$DEVICE" \
    --checkpoint "$CHECKPOINT" \
    --start_mode "$START_MODE" \
    --car_start_mode "$CAR_START_MODE" \
    --group_size "$GROUP_SIZE" \
    --update_frequency "$UPDATE_FREQUENCY" \
    --resume_checkpoint "$RESUME_CHECKPOINT" \
    --tracks "${TRACKS[@]}" \
    --epochs "${EPOCHS[@]}" \
    --force_lr