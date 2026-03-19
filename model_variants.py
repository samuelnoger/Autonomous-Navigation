"""
Alternative CarNet architectures for better acceleration/braking learning.

Each architecture addresses different aspects of the problem:
1. LSTMCarNet: Temporal memory for learning action sequences
2. SeparateHeadsCarNet: Specialized processing for steering vs acceleration
3. DeepCarNet: More capacity with residual connections
4. AttentionCarNet: Attention over ray inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LSTMCarNet(nn.Module):
    """
    LSTM-based architecture with temporal memory.

    Advantages:
    - Learns temporal patterns (e.g., "I've been accelerating, now brake")
    - Remembers recent actions and states
    - Can learn smoother control sequences

    Best for: Learning when to transition between accelerating and braking

    Note: Requires maintaining hidden states during training (see usage example)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_rays, lstm_layers=1):
        super().__init__()

        self.n_rays = n_rays
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        state_dim = input_dim - n_rays

        # Ray encoder (same as original)
        self.ray_net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        ray_out_dim = 16 * n_rays

        # State encoder
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # LSTM for temporal processing
        combined_dim = ray_out_dim + hidden_dim // 2
        self.lstm = nn.LSTM(
            input_size=combined_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Output head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, x, hidden=None):
        """
        Args:
            x: Input tensor (batch, input_dim) or (batch, seq_len, input_dim)
            hidden: Optional LSTM hidden state (h, c)

        Returns:
            output: Control output (batch, output_dim)
            hidden: New hidden state for next step
        """
        # Handle both single-step and sequence inputs
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)

        batch_size, seq_len, _ = x.shape

        # Process each timestep
        rays = x[:, :, :self.n_rays]  # (batch, seq_len, n_rays)
        state = x[:, :, self.n_rays:]  # (batch, seq_len, state_dim)

        # Reshape for processing
        rays = rays.reshape(batch_size * seq_len, self.n_rays, 1).transpose(1, 2)
        state = state.reshape(batch_size * seq_len, -1)

        # Encode
        r = self.ray_net(rays)
        s = self.state_net(state)

        # Combine and reshape back to sequence
        combined = torch.cat([r, s], dim=1)
        combined = combined.reshape(batch_size, seq_len, -1)

        # LSTM processing
        lstm_out, hidden = self.lstm(combined, hidden)

        # Take last timestep output
        out = lstm_out[:, -1, :]

        return self.fc(out), hidden

    def init_hidden(self, batch_size, device):
        """Initialize hidden state for LSTM"""
        h = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)


class SeparateHeadsCarNet(nn.Module):
    """
    Architecture with separate processing heads for steering and acceleration.

    Advantages:
    - Steering focuses on ray distances and heading
    - Acceleration focuses on speed, curvature, and distance to gate
    - Each control output has specialized feature processing

    Best for: When steering and acceleration need different information emphasis
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_rays):
        super().__init__()

        self.n_rays = n_rays
        state_dim = input_dim - n_rays

        # Shared ray encoder
        self.ray_net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        ray_out_dim = 16 * n_rays

        # Shared state encoder
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU()
        )

        combined_dim = ray_out_dim + hidden_dim // 2

        # Steering head (focuses on rays and heading)
        self.steering_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

        # Acceleration head (focuses on speed and curvature)
        self.accel_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Split and encode
        rays = x[:, :self.n_rays].unsqueeze(1)
        state = x[:, self.n_rays:]

        r = self.ray_net(rays)
        s = self.state_net(state)

        combined = torch.cat([r, s], dim=1)

        # Separate outputs
        steering = self.steering_head(combined)
        accel = self.accel_head(combined)

        return torch.cat([steering, accel], dim=1)


class DeepCarNet(nn.Module):
    """
    Deeper network with residual connections.

    Advantages:
    - More capacity to learn complex control policies
    - Residual connections improve gradient flow
    - Better representation learning

    Best for: When the problem needs more model capacity
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_rays, n_blocks=3):
        super().__init__()

        self.n_rays = n_rays
        state_dim = input_dim - n_rays

        # Ray encoder with more layers
        self.ray_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        ray_out_dim = 32 * n_rays

        # State encoder
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Initial projection
        combined_dim = ray_out_dim + hidden_dim
        self.input_proj = nn.Linear(combined_dim, hidden_dim)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_blocks)
        ])

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        rays = x[:, :self.n_rays].unsqueeze(1)
        state = x[:, self.n_rays:]

        r = self.ray_net(rays)
        s = self.state_net(state)

        combined = torch.cat([r, s], dim=1)
        x = self.input_proj(combined)

        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)

        return self.output_head(x)


class ResidualBlock(nn.Module):
    """Residual block with layer normalization"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.ln(x + residual)
        return F.relu(x)


class AttentionCarNet(nn.Module):
    """
    Architecture with attention over ray inputs.

    Advantages:
    - Learns which rays are most important for current decision
    - Can focus on closest walls when braking needed
    - Weighted combination of ray information

    Best for: Dynamic focus on relevant parts of ray inputs
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_rays):
        super().__init__()

        self.n_rays = n_rays
        state_dim = input_dim - n_rays

        # Ray embedding
        self.ray_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )

        # State encoder
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Attention mechanism
        self.query_net = nn.Linear(hidden_dim, 32)
        self.key_net = nn.Linear(32, 32)
        self.value_net = nn.Linear(32, 32)

        # Output network
        self.fc = nn.Sequential(
            nn.Linear(32 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        rays = x[:, :self.n_rays]  # (batch, n_rays)
        state = x[:, self.n_rays:]  # (batch, state_dim)

        # Embed each ray
        ray_embeds = self.ray_embed(rays.unsqueeze(-1))  # (batch, n_rays, 32)

        # Encode state
        state_feat = self.state_net(state)  # (batch, hidden_dim)

        # Attention: state queries ray information
        query = self.query_net(state_feat).unsqueeze(1)  # (batch, 1, 32)
        keys = self.key_net(ray_embeds)  # (batch, n_rays, 32)
        values = self.value_net(ray_embeds)  # (batch, n_rays, 32)

        # Scaled dot-product attention
        scores = torch.matmul(query, keys.transpose(1, 2)) / math.sqrt(32)  # (batch, 1, n_rays)
        attention_weights = F.softmax(scores, dim=-1)

        # Weighted sum of values
        attended = torch.matmul(attention_weights, values).squeeze(1)  # (batch, 32)

        # Combine with state and predict
        combined = torch.cat([attended, state_feat], dim=1)
        return self.fc(combined)


# ============================================================================
# Usage Example: How to use these architectures in train_nn.py
# ============================================================================

"""
OPTION 1: Simple drop-in replacement (SeparateHeadsCarNet, DeepCarNet, AttentionCarNet)
----------

In train_nn.py, replace the model initialization:

    # Original:
    from model import CarNet
    model = CarNet(input_dim=input_dim, hidden_dim=hidden_dim,
                   output_dim=output_dim, n_rays=n_rays).to(device)

    # New (choose one):
    from model_variants import SeparateHeadsCarNet
    model = SeparateHeadsCarNet(input_dim=input_dim, hidden_dim=hidden_dim,
                                output_dim=output_dim, n_rays=n_rays).to(device)

    # Or:
    from model_variants import DeepCarNet
    model = DeepCarNet(input_dim=input_dim, hidden_dim=hidden_dim,
                       output_dim=output_dim, n_rays=n_rays, n_blocks=3).to(device)

    # Or:
    from model_variants import AttentionCarNet
    model = AttentionCarNet(input_dim=input_dim, hidden_dim=hidden_dim,
                            output_dim=output_dim, n_rays=n_rays).to(device)


OPTION 2: LSTM (requires additional changes for hidden state management)
----------

1. Import and create model:
    from model_variants import LSTMCarNet
    model = LSTMCarNet(input_dim=input_dim, hidden_dim=hidden_dim,
                       output_dim=output_dim, n_rays=n_rays, lstm_layers=1).to(device)

2. Initialize hidden states at start of epoch (in train_model function):
    # After "for epoch in epoch_bar:" and before the step loop
    hidden = model.init_hidden(n_cars, device)

3. Pass hidden state through forward pass (modify the forward call):
    # Original:
    outputs = model(inputs)

    # New:
    outputs, hidden = model(inputs, hidden)
    # Detach hidden to prevent backprop through time across steps
    hidden = (hidden[0].detach(), hidden[1].detach())

4. Reset hidden when cars become inactive (optional, for better performance):
    # After collision check or when resetting cars
    if not cars.active.any():
        hidden = model.init_hidden(n_cars, device)


RECOMMENDATION:
--------------
For your acceleration/braking problem, I'd try in this order:

1. **SeparateHeadsCarNet** (easiest, no training loop changes)
   - Dedicated processing for steering vs acceleration
   - Works well with the adaptive speed reward

2. **DeepCarNet** (easy, more capacity)
   - If the model needs more learning capacity
   - Try hidden_dim=128 or n_blocks=4

3. **LSTMCarNet** (moderate difficulty, best for temporal patterns)
   - Best for learning "accelerate → brake → accelerate" sequences
   - Requires training loop modifications

4. **AttentionCarNet** (easy, interpretable)
   - Can visualize which rays model focuses on
   - Good for understanding decision-making
"""
