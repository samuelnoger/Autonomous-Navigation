# Autonomous-Navigation

This is a PyTorch-based racing project where a neural network learns to drive a car around 2D tracks using ray-based observations, continuous steering, and acceleration control.

## Overview

The project contains:
- a training pipeline for learning driving behavior on one or more tracks
- a Tkinter simulation viewer for running trained models or random policies
- several model variants, including recurrent and attention-based architectures
- track generation and loading for both built-in layouts and GeoJSON tracks
- live reward plotting during training

## Features

- Continuous control for steering and acceleration
- Ray-cast track perception
- Multiple track layouts, including built-in and GeoJSON-based tracks
- Sequential training across multiple tracks
- Per-track checkpoints and resume support
- Real-time simulation with Tkinter
- Live reward plot in a separate process

## Requirements

The project uses:
- Python 3.11+
- PyTorch
- NumPy
- Matplotlib
- OpenCV
- GeoPandas
- Tkinter

A typical install looks like:

```bash
pip install torch numpy matplotlib opencv-python geopandas
```

If you already have a working virtual environment, activate it before running training or simulation.

## Quick Start

### 1. Train a model

Use the shell script for the default training setup:

```bash
./train_nn.sh
```

Or run the trainer directly:

```bash
python3 -m train.train_nn \
  --track triangle \
  --n_epochs 300 \
  --n_steps 1600 \
  --n_cars 512
```

### 2. Watch a simulation

Run the Tkinter viewer with a random policy:

```bash
python simulate_nn.py
```

Load a trained checkpoint:

```bash
python simulate_nn.py
```

The simulation script currently loads `checkpoints/us_track/last.pth` by default in its example configuration, so update the path in `simulate_nn.py` if you want to visualize another checkpoint.

## Training

### Main trainer

The core training entry point is:

```bash
python -m train.train_nn
```

### Training script

`train_nn.sh` is a convenience wrapper around the Python trainer. It sets a default training configuration and can be edited at the top for your preferred:
- number of cars
- number of epochs
- learning rate
- track sequence
- checkpoint paths

### Checkpoints

Training saves checkpoints per track under:

```text
checkpoints/<track_name>/last.pth
```

If you pass a direct `.pth` file to `--checkpoint`, the trainer uses that exact file.
If you pass a directory, it saves into `<directory>/<track_name>/last.pth`.

## Simulation

The Tkinter viewer is in `simulate_nn.py`.

It:
- loads a trained `CarNet` checkpoint
- builds the selected track
- runs the simulation with a GUI canvas
- uses the same input preprocessing as training

### Example usages

Random policy:

```bash
python3 simulate_nn.py
```

Use a custom track from the command line arguments:

```bash
python simulate_nn.py --config --track triangle
```

## Tracks

Supported named tracks include:
- `simple`
- `square`
- `triangle`
- `square_narrow`
- `us_track`
- `redbull_ring`

You can also load an absolute path to a GeoJSON track file.

## Models

The default network is `CarNet`, which uses:
- a 1D convolutional encoder for ray inputs
- a small MLP for state features
- a final control head that outputs steering and acceleration

Additional architectures are available in `model/model_variants.py`:
- `LSTMCarNet`
- `GRUCarNet`
- `SeparateHeadsCarNet`
- `DeepCarNet`
- `AttentionCarNet`

## Tips

- Start with a simple track like `simple` or `triangle` when debugging training.
- If training is unstable, lower the learning rate or try the recurrent variants.
- If simulation fails to open, verify that Tkinter is available in your Python install.
- For custom GeoJSON tracks, make sure the file is readable from the path you pass in.

## Development Methodology
The core neural network architectures and boilerplate PyTorch code for this project were scaffolded with the assistance of AI coding tools. My primary technical contributions focus on the conceptual design, physics engine integration, hyperparameter tuning, and orchestrating the end-to-end training pipelines on Apple Silicon/MPS.

## Notes

This README is intentionally focused on the current code in this workspace. If you change the training script or checkpoint layout later, update the examples here as well.
