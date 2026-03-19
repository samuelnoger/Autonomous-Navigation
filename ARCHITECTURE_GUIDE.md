# Architecture Variants Guide

I've created 4 alternative architectures in `model_variants.py` to help with acceleration/braking learning:

## Quick Comparison

| Architecture | Difficulty | Best For | Key Advantage |
|-------------|-----------|----------|---------------|
| **SeparateHeadsCarNet** | ⚡ Easy | Different steering/accel complexity | Specialized processing |
| **DeepCarNet** | ⚡ Easy | Need more capacity | Deeper with residual connections |
| **AttentionCarNet** | ⚡ Easy | Understanding decisions | Focuses on relevant rays |
| **LSTMCarNet** | 🔧 Moderate | Temporal patterns | Remembers past actions |

## How to Use (Step by Step)

### Option 1: SeparateHeadsCarNet (RECOMMENDED TO TRY FIRST)

**Easiest drop-in replacement. Gives acceleration its own specialized processing.**

1. Edit `train_nn.py` line ~352, change:
```python
# FROM:
from model import CarNet
model = CarNet(
    input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_rays=n_rays
).to(device)

# TO:
from model_variants import SeparateHeadsCarNet
model = SeparateHeadsCarNet(
    input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_rays=n_rays
).to(device)
```

2. Train:
```bash
python train_nn.py --checkpoint separate_heads_ckpt.pth --start_mode start_new --input_dim 26
```

That's it! No other changes needed.

---

### Option 2: DeepCarNet

**Use when you need more model capacity (more parameters).**

1. Edit `train_nn.py` line ~352:
```python
from model_variants import DeepCarNet
model = DeepCarNet(
    input_dim=input_dim,
    hidden_dim=hidden_dim,  # try 128 instead of 64
    output_dim=output_dim,
    n_rays=n_rays,
    n_blocks=3  # more blocks = deeper network
).to(device)
```

2. Train:
```bash
python train_nn.py --checkpoint deep_ckpt.pth --start_mode start_new --input_dim 26 --hidden_dim 128
```

---

### Option 3: AttentionCarNet

**Use to understand which rays the model focuses on when making decisions.**

1. Edit `train_nn.py` line ~352:
```python
from model_variants import AttentionCarNet
model = AttentionCarNet(
    input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_rays=n_rays
).to(device)
```

2. Train:
```bash
python train_nn.py --checkpoint attention_ckpt.pth --start_mode start_new --input_dim 26
```

---

### Option 4: LSTMCarNet (requires training loop changes)

**Best for learning temporal patterns like "accelerate on straight → brake for corner".**

#### Step 1: Modify `train_nn.py` to handle LSTM hidden states

Add hidden state initialization in `train_model()` function, after line ~108 (after `for epoch in epoch_bar:`):

```python
# Add this right after: for epoch in epoch_bar:
# Initialize LSTM hidden state for this epoch
if hasattr(model, 'init_hidden'):
    hidden = model.init_hidden(n_cars, device)
else:
    hidden = None
```

#### Step 2: Modify the forward pass (around line ~149)

```python
# Change from:
outputs = model(inputs)

# To:
if hidden is not None:
    outputs, hidden = model(inputs, hidden)
    # Detach to prevent backprop through entire episode
    hidden = (hidden[0].detach(), hidden[1].detach())
else:
    outputs = model(inputs)
```

#### Step 3: Change model initialization (line ~352)

```python
from model_variants import LSTMCarNet
model = LSTMCarNet(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    n_rays=n_rays,
    lstm_layers=1  # try 2 for more memory
).to(device)
```

#### Step 4: Train

```bash
python train_nn.py --checkpoint lstm_ckpt.pth --start_mode start_new --input_dim 26 --hidden_dim 128
```

---

## My Recommendation

**Try them in this order:**

1. **Start with SeparateHeadsCarNet**
   - Easiest to implement (no training loop changes)
   - Addresses your specific problem (acceleration needs different processing than steering)
   - Works great with the adaptive speed reward I added
   - Train for 200 epochs and see if acceleration/braking improves

2. **If that's not enough, try DeepCarNet with `--hidden_dim 128`**
   - More capacity might help learn complex acceleration patterns
   - Still easy to implement

3. **Last resort: LSTMCarNet**
   - Most powerful for temporal patterns
   - But requires training loop modifications
   - Only if feedforward models don't work

---

## Checking Results

When training, watch the `AdaptiveSpeed` reward in the logs:
```
AdaptiveSpeed:X.XX
```

- **Negative or near zero** = Model isn't learning appropriate speeds
- **Positive and increasing** = Model is learning to go fast on straights, slow in corners ✅

Also watch car behavior in simulation - you should see:
- Speed varies significantly (50-140 range)
- Accelerates on straights
- Brakes before turns

---

## Troubleshooting

**"Input dimension mismatch"**
- Make sure you use `--input_dim 26` (with new input features I added)
- Use `--start_mode start_new` to reset training

**"Still driving at constant speed"**
- Try increasing `adaptive_speed_reward_rate` in `utils.py` from 2.0 to 5.0
- Reduce acceleration noise further (change 0.15 to 0.1 in `train_nn.py`)

**"Training is unstable"**
- Reduce learning rate: `--lr 5e-4` instead of default 1e-3
- Use gradient clipping (add to `train_nn.py` after loss.backward()):
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  ```
