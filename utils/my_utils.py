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
    track_name=None,
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
            "track_name": track_name,
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

def generate_triangle_track(screen_width=1000, screen_height=600,width=600, height=400, corner_points=5, corner_radius=60):

    """
    Generates a triangular track with rounded corners.

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

    # triangle vertices
    vertices = [
        (cx, cy - hh + r,-3*np.pi/4, -np.pi/4),          # top vertex
        (cx + hw - r, cy + hh - r,-np.pi/4,np.pi/2),     # bottom-right vertex
        (cx - hw + r, cy + hh - r,np.pi/2,5*np.pi/4)     # bottom-left vertex
    ]

    for cx_c, cy_c, a0, a1 in vertices:
        angles = np.linspace(a0, a1, corner_points)
        for a in angles:
            x = cx_c + r * np.cos(a)
            y = cy_c + r * np.sin(a)
            centerline.append((x, y))

    centerline.append(centerline[0])

    return centerline

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

def get_inputs(cars,track, gates_tensor, gate_indices, max_ray_dist):
        """Get neural network inputs for all cars.

        Computes ray distances, heading error, speed, curvature lookahead, etc.

        Args:
            cars: CarState object.
            gates_tensor: Tensor containing gate coordinates.
            gate_indices: Current gate indices for all cars.

        Returns:
            inputs: Model input tensor (N, input_dim).
            ray_dists: Ray distances for reward computation (N, n_rays).
        """
        # Compute next gate centers
        next_gate_centers = (
            gates_tensor[gate_indices, 0:2] + gates_tensor[gate_indices, 2:4]
        ) / 2

        # ---- Ray distances ----
        ray_angles = cars.ray_angles  # (N, R)
        ray_dists_original = track.get_lines_along_rays(cars.pos, ray_angles, max_ray_dist)  # (N, R)
        ray_dists = ray_dists_original / max_ray_dist

        # ---- Heading error to next gate ----
        #delta = next_gate_centers - cars.pos  # (N, 2)
        #target_angle = torch.atan2(delta[:, 1], delta[:, 0])
        #heading_error = target_angle - cars.angle
        #heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
        #sin_error = torch.sin(heading_error)
        #cos_error = torch.cos(heading_error)

        # ---- Speed input (normalize) ----
        speed_input = cars.speed.unsqueeze(1) / 140.0

        # ---- Lookahead curvature (gates 1, 2, 3 ahead) ----
        # Compute car's heading direction
        car_heading_x = torch.cos(cars.angle)
        car_heading_y = torch.sin(cars.angle)

        curvatures = []
        distances = []
        n_gates = gates_tensor.shape[0]
        for lookahead in [1, 2, 3]:
            next_idx = (gate_indices + lookahead * cars.direction) % n_gates

            # Extract gate direction for upcoming gate
            next_gate = gates_tensor[next_idx]  # (N, 4)

            next_gx = next_gate[:, 2] - next_gate[:, 0]
            next_gy = next_gate[:, 3] - next_gate[:, 1]

            # Distance from car position to the center of the lookahead gate
            next_center = (next_gate[:, 0:2] + next_gate[:, 2:4]) / 2
            next_dist_to_center = (next_center - cars.pos).norm(dim=1)

            # Normalize and clip by max_ray_dist
            next_dist_norm = torch.clamp(next_dist_to_center / max_ray_dist, max=1.0)
            distances.append(next_dist_norm.unsqueeze(1))

            next_norm = torch.hypot(next_gx, next_gy)
            next_dir_x = -next_gy / (next_norm + 1e-6)
            next_dir_y = next_gx / (next_norm + 1e-6)

            next_dir_x = next_dir_x * cars.direction
            next_dir_y = next_dir_y * cars.direction

            # Curvature: angle difference between car heading and next gate direction
            # cos_curv: dot product = cos(angle)
            # sin_curv: cross product = sin(angle)
            cos_curv = car_heading_x * next_dir_x + car_heading_y * next_dir_y
            sin_curv = car_heading_x * next_dir_y - car_heading_y * next_dir_x

            curvatures.append(cos_curv.unsqueeze(1))
            curvatures.append(sin_curv.unsqueeze(1))


        # ---- Concatenate all features ----
        feature_list = [
            ray_dists,  # n_rays features
            speed_input
        ]
        feature_list.extend(curvatures)  # 6 features (3 gates * 2 for sin/cos)
        feature_list.extend(distances)   # 3 features (distances to lookahead gate centers)

        inputs = torch.cat(feature_list, dim=1)
        
        return inputs, ray_dists_original



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
