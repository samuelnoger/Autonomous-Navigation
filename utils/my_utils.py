import torch
import math
import numpy as np

def save_checkpoint(
    epoch,
    model,
    optimizer,
    checkpoint_path="checkpoints/last_ckpt.pth",
    total_epochs=None,
    scheduler=None,
    best_reward=-float("inf"),
):
    """Save model and optimizer state for checkpoint resuming.

    Saves model weights, optimizer state, learning rate scheduler state,
    and training metadata needed to resume training from this epoch.

    Args:
        epoch: Current epoch number.
        model: Model with state to save.
        optimizer: Optimizer with state to save.
        checkpoint_path: Where to save the checkpoint.
        total_epochs: Total number of epochs planned for training.
        scheduler: Optional LR scheduler with state to save.
        best_reward: Best reward achieved so far.
    """
    torch.save(
        {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "current_lr": optimizer.param_groups[0]["lr"],
            "best_reward": best_reward,
        },
        checkpoint_path,
    )
            
def passed_gate(car, gate_line, threshold=15):
    x1, y1, x2, y2 = gate_line
    # distance from point to line segment formula
    px, py = car.x, car.y
    line_mag = math.hypot(x2 - x1, y2 - y1)
    if line_mag < 1e-6:
        return False
    u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag**2)
    u = max(0, min(1, u))
    ix = x1 + u * (x2 - x1)
    iy = y1 + u * (y2 - y1)
    dist = math.hypot(px - ix, py - iy)
    return dist < threshold


def generate_simple_track(screen_width=1000, screen_height=600,
                          width=600, height=400,
                          corner_points=5, corner_radius=140):
    """
    Generates a rectangular track with rounded corners.

    Returns:
        centerline: list of (x, y) points
    """

    cx = screen_width / 2
    cy = screen_height / 2

    hw = width / 2
    hh = height / 2

    centerline = []

    # corner radius
    r = corner_radius

    # corner centers
    corners = [
        (cx + hw - r, cy - hh + r, -np.pi/2, 0),        # top-right
        (cx + hw - r, cy + hh - r, 0, np.pi/2),         # bottom-right
        (cx - hw + r, cy + hh - r, np.pi/2, np.pi),     # bottom-left
        (cx - hw + r, cy - hh + r, np.pi, 3*np.pi/2)    # top-left
    ]

    for cx_c, cy_c, a0, a1 in corners:
        angles = np.linspace(a0, a1, corner_points)
        for a in angles:
            x = cx_c + r * np.cos(a)
            y = cy_c + r * np.sin(a)
            centerline.append((x, y))
            
    centerline.append(centerline[0])

    return centerline

def get_inputs(track, cars, next_gate_centers, max_ray_dist=200.0, gates_tensor=None, gate_indices=None):
    """Compute neural network inputs from track and car state.

    Returns NN inputs per car:
    - Ray distances (log-normalized for features)
    - Speed (normalized)
    - sin(heading_error)
    - cos(heading_error)
    - Distance to next gate (normalized)
    - Min ray distance (wall proximity)
    - Lookahead curvatures (gates 1, 2, 3 ahead)

    Args:
        track: Track object
        cars: CarState object
        next_gate_centers: (N,2) tensor with gate centers
        max_ray_dist: max distance for rays (default 200)
        gates_tensor: (n_gates, 4) tensor of gate coordinates [optional for curvature]
        gate_indices: (N,) tensor of current gate indices [optional for curvature]

    Returns:
        inputs: (N, n_rays+6+curvature) tensor with log-normalized ray features and other inputs
        ray_dists: (N, n_rays) tensor of original ray distances in pixels
    """
    N = cars.pos.shape[0]
    device = cars.pos.device

    # ---- Ray distances ----
    ray_angles = cars.ray_angles  # (N,R)
    ray_dists_original = track.get_lines_along_rays(cars.pos, ray_angles, max_ray_dist)  # (N,R)
    ray_dists = ray_dists_original / max_ray_dist

    # ---- Heading error to next gate ----
    delta = next_gate_centers - cars.pos  # (N,2)
    target_angle = torch.atan2(delta[:, 1], delta[:, 0])
    heading_error = target_angle - cars.angle
    heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
    sin_error = torch.sin(heading_error)
    cos_error = torch.cos(heading_error)

    # ---- Speed input (normalize) ----
    speed_input = cars.speed.unsqueeze(1) / 140.0

    # ---- Distance to next gate (normalized) ----
    dist_to_gate = delta.norm(dim=1, keepdim=True) / 200.0  # normalize by typical gate distance

    # ---- Min ray distance (wall proximity) ----
    min_ray = ray_dists.min(dim=1, keepdim=True).values  # extract values tensor

    # ---- Lookahead curvature (gates 1, 2, 3 ahead) ----
    curvatures = []
    if gates_tensor is not None and gate_indices is not None:
        n_gates = gates_tensor.shape[0]
        for lookahead in [1, 2, 3]:
            curr_idx = gate_indices
            next_idx = (gate_indices + lookahead * cars.direction) % n_gates  # Account for driving direction

            # Extract gate vectors and compute direction vectors
            # direction = R(+90) * gate_vec / ||gate_vec|| = (-(y2-y1), x2-x1) / norm
            curr_gate = gates_tensor[curr_idx]  # (N, 4)
            next_gate = gates_tensor[next_idx]  # (N, 4)

            curr_gx = curr_gate[:, 2] - curr_gate[:, 0]
            curr_gy = curr_gate[:, 3] - curr_gate[:, 1]
            curr_norm = torch.hypot(curr_gx, curr_gy)
            curr_dir_x = -curr_gy / (curr_norm + 1e-6)
            curr_dir_y = curr_gx / (curr_norm + 1e-6)

            next_gx = next_gate[:, 2] - next_gate[:, 0]
            next_gy = next_gate[:, 3] - next_gate[:, 1]
            next_norm = torch.hypot(next_gx, next_gy)
            next_dir_x = -next_gy / (next_norm + 1e-6)
            next_dir_y = next_gx / (next_norm + 1e-6)

            # Curvature: angle between current and next gate directions
            cos_curv = curr_dir_x * next_dir_x + curr_dir_y * next_dir_y  # dot product
            sin_curv = curr_dir_x * next_dir_y - curr_dir_y * next_dir_x  # cross product

            curvatures.append(cos_curv.unsqueeze(1))
            curvatures.append(sin_curv.unsqueeze(1))

    # ---- Concatenate all features ----
    feature_list = [
        ray_dists,           # n_rays features
        speed_input,         # 1 feature
        sin_error.unsqueeze(1),  # 1 feature
        cos_error.unsqueeze(1),  # 1 feature
    ]
    if curvatures:
        feature_list.extend(curvatures)  # 6 features (3 gates * 2 for sin/cos)

    inputs = torch.cat(feature_list, dim=1)
    return inputs, ray_dists_original


def compute_step_reward(
    cars,
    track,
    gates_tensor,
    gate_indices,
    last_gate_indices,
    ray_dists,
    prev_dist,
    step,
    n_steps,
    max_ray_dist,
):
    """
    Compute step-wise rewards and update car states for episode-level cumulative reward.

    Returns:
        step_rewards: dict of reward components (each tensor of shape [n_cars])
        prev_dist: updated distance to next gate (tensor [n_cars])
    """
    pos = cars.pos
    speed = cars.speed

    speed_reward_rate = 0.005  # reward per unit speed (increased from 0.001)
    collision_penalty_rate = 50  # penalty for collision
    gate_pass_reward_rate = 2.0  # (10.0 when starting new later 1.0)reward for passing a gate
    wall_penalty_rate = 5  #(1.0 when starting new later 0.5) penalty for hitting the wall
    direction_reward_rate = 0.01  #(0.05 when starting new later 0.005) reward for heading toward next gate
    alive_reward_rate = 0.025  # (0.1 when starting new later 0.01) default alive reward

    if track.track_name == "simple":
        speed_reward_rate = 0.005
        collision_penalty_rate = 100.0  # penalty for collision
        gate_pass_reward_rate = 10.0  # reward for passing a gate
        wall_penalty_rate = 10.0  # penalty for hitting the wall
        direction_reward_rate = 0.05  # reward for heading toward next gate
        alive_reward_rate = 0.1


    # -----------------------------
    # Distance to next gate
    # -----------------------------
    gate_coords = gates_tensor[gate_indices]
    x1, y1 = gate_coords[:, 0], gate_coords[:, 1]
    x2, y2 = gate_coords[:, 2], gate_coords[:, 3]

    line_vec = torch.stack([x2 - x1, y2 - y1], dim=1)
    p_vec = pos - torch.stack([x1, y1], dim=1)
    line_len2 = (line_vec**2).sum(dim=1)
    u = torch.clamp((p_vec * line_vec).sum(dim=1) / line_len2, 0.0, 1.0)
    closest = torch.stack([x1, y1], dim=1) + u.unsqueeze(1) * line_vec
    dist = (pos - closest).norm(dim=1)

    # -----------------------------
    # Reward components
    # -----------------------------
    
    # Speed reward: encourage faster speeds
    speed_reward = speed_reward_rate * speed * cars.active

    # Gate passing: reward cars that passed through the gate this step
    passed_mask = (
        (dist < 10.0)
        & (gate_indices == (last_gate_indices + cars.direction) % track.gates.shape[0])
        & cars.active
    )
    gate_reward = torch.zeros_like(speed)
    with torch.no_grad():
        cars.gates_passed[passed_mask] += cars.direction[passed_mask]  # Increment or decrement based on direction
        gate_reward[passed_mask] = gate_pass_reward_rate
        last_gate_indices[passed_mask] = gate_indices[passed_mask]
        gate_indices[passed_mask] = (
            gate_indices[passed_mask] + cars.direction[passed_mask]
        ) % gates_tensor.shape[0]

    prev_active = cars.active.clone()
    cars.check_collisions(track)

    # The penalty is scaled by how late in the episode the collision occurs, to encourage longer survival.
    if track.track_name == "simple":  # simpler track with fewer collision opportunities, so use scaled penalty
        collision_penalty = (
            -(~cars.active & prev_active).float() * (n_steps - step) / n_steps * collision_penalty_rate
        )
    else:
        # Constant penalty for collision, regardless of when it happens, to strongly encourage avoiding collisions.
        collision_penalty = (
            -(~cars.active & prev_active).float() *(n_steps - 0.6*step)/n_steps * collision_penalty_rate
        )


    # Direction reward: encourage heading toward next gate

    track_width = track.track_width
    min_dist = ray_dists.min(dim=1).values  # Already in original units (pixels)
    safe_margin = 0.2 * track_width
    scaled_min_dist = min_dist / safe_margin
    wall_penalty = (-wall_penalty_rate * (1.0 - torch.clamp(scaled_min_dist, 0, 1)) ** 2 * cars.active)

    next_gate_centers = (
        gates_tensor[gate_indices, 0:2] + gates_tensor[gate_indices, 2:4]
    ) / 2
    gate_vec = next_gate_centers - cars.pos
    gate_dir = gate_vec / (gate_vec.norm(dim=1, keepdim=True) + 1e-6)

    vel = cars.vel
    vel_norm = vel / (vel.norm(dim=1, keepdim=True) + 1e-6)

    direction_reward = (vel_norm * gate_dir).sum(dim=1) * cars.active * direction_reward_rate

    # print("Before: ",progress_reward[:20])
    # -----------------------------
    # Alive reward: small reward for each timestep the car is active
    alive_reward = alive_reward_rate * cars.active

    # Aggregate step reward
    step_rewards = {
        "speed": speed_reward,
        "gate": gate_reward,
        "collision": collision_penalty,
        "wall": wall_penalty,
        "direction": direction_reward,
        "alive": alive_reward,
    }

    prev_dist = dist.detach()

    return step_rewards, prev_dist


def line_intersection(line1, line2):
    """
    Compute intersection between two lines
    line1: (...,4)
    line2: (...,4)
    Returns (...,2) or NaN if no intersection
    """
    x1, y1, x2, y2 = line1.unbind(-1)
    x3, y3, x4, y4 = line2.unbind(-1)
    rx = x2 - x1
    ry = y2 - y1
    sx = x4 - x3
    sy = y4 - y3
    # 2D cross products for parametric segment intersection.
    den = rx * sy - ry * sx
    qpx = x3 - x1
    qpy = y3 - y1
    eps = 1e-6
    valid_den = den.abs() > eps
    safe_den = torch.where(valid_den, den, torch.ones_like(den))
    t = (qpx * sy - qpy * sx) / safe_den
    u = (qpx * ry - qpy * rx) / safe_den
    mask = valid_den & (t >= -eps) & (t <= 1.0 + eps) & (u >= -eps) & (u <= 1.0 + eps)
    px = x1 + t * rx
    py = y1 + t * ry
    px = torch.where(mask, px, torch.full_like(px, float("nan")))
    py = torch.where(mask, py, torch.full_like(py, float("nan")))
    return torch.stack([px, py], dim=-1)
