# Autonomous Navigation & Vectorized Physics (Drive_NN)

A PyTorch-based simulation environment and training pipeline where a neural network agent learns to navigate 2D tracks. The project features a custom-built, fully vectorized physics engine and uses 1D convolutional ray-casting for spatial perception.

## Architecture Highlights
- **Vectorized Physics Engine:** Simulates Ackermann kinematics, centripetal drift, grip dynamics, and batched car-to-car collision resolution entirely using PyTorch tensors.
- **Sensor-Based Perception:** Ray-cast distances are treated as 1D spatial signals and processed via a `Conv1d` encoder.
- **Control Strategy:** Continuous action space (acceleration and steering) utilizing state-fusion (ray features + velocity/heading).
- **Model Variants:** Includes LSTM, GRU, and Attention-based network architectures alongside the standard Convolutional model.

## Features
- **Batched Training:** Capable of simulating and training hundreds of agents simultaneously on a single GPU.
- **Real-Time Simulation Viewer:** Tkinter-based GUI for evaluating trained policies or running random agents.
- **Dynamic Track Loading:** Supports built-in procedural tracks and real-world layouts via GeoJSON data.
- **Sequential Curriculum:** Agents can be trained across multiple tracks sequentially with per-track checkpointing.

## Project Structure
* `/model/` - Neural network architectures (`CarNet`, LSTM/GRU variants) and the core RL trainer.
* `/track/` - Procedural generation and GeoJSON parsers for track rendering.
* `/train/` - Training loop execution and hyperparameter configuration.
* `/utils/` - Live reward plotting and utility functions.
* `simulate_nn.py` - Tkinter simulation viewer for trained checkpoints.

## Requirements
```bash
pip install torch numpy matplotlib opencv-python geopandas
