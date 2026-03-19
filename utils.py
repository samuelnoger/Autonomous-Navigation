import torch
import math
import numpy as np
import geopandas as gpd # type: ignore

            
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

def compute_centerline(inner_border, outer_border):
    """
    Compute the centerline of a track given the inner and outer borders.
    
    inner_border, outer_border: lists of (x, y) points of same length
    Returns: list of (x, y) points representing the centerline
    """
    if len(inner_border) != len(outer_border):
        raise ValueError("Inner and outer borders must have the same number of points")

    centerline = []
    for (ix, iy), (ox, oy) in zip(inner_border, outer_border):
        cx = (ix + ox) / 2
        cy = (iy + oy) / 2
        centerline.append((cx, cy))
    
    return centerline

def generate_gates(centerline, track_width, gates_per_segment=1):
    """
    Generate gates along the track.

    centerline: list of (x, y) points along the track
    track_width: full width of the track (distance from inner to outer border)
    gates_per_segment: number of gates per segment between two centerline points

    Returns: list of gates as (x1, y1, x2, y2)
    """
    gates = []
    n = len(centerline)

    for i in range(n):
        p1 = np.array(centerline[i])
        p2 = np.array(centerline[(i + 1) % n])  # wrap around for closed track

        # direction along track segment
        dir_vec = p2 - p1
        length = np.linalg.norm(dir_vec)
        if length == 0:
            continue
        dir_vec /= length

        # perpendicular direction
        normal = np.array([-dir_vec[1], dir_vec[0]])
        half_width = track_width / 2

        # create gates along this segment
        for j in range(gates_per_segment):
            t = (j + 0.5) / gates_per_segment  # fractional position along segment
            point = p1 + dir_vec * length * t
            gate_start = point + normal * half_width
            gate_end = point - normal * half_width
            gates.append((gate_start[0], gate_start[1], gate_end[0], gate_end[1]))

    return gates

def offset_track_borders(centerline, outer_width=80, inner_width=20):
    """
    Returns a tuple: (inner_border, outer_border)
    - outer_border: centerline pushed outward by outer_width
    - inner_border: centerline pushed slightly inward by inner_width
    """
    outer_border = []
    inner_border = []
    n = len(centerline)

    for i in range(n):
        p0 = centerline[(i - 1) % n]
        p1 = centerline[i]
        p2 = centerline[(i + 1) % n]

        # Direction vectors
        dx1, dy1 = p1[0] - p0[0], p1[1] - p0[1]
        dx2, dy2 = p2[0] - p1[0], p2[1] - p1[1]

        # Normalize
        len1 = math.hypot(dx1, dy1) or 1
        len2 = math.hypot(dx2, dy2) or 1
        dx1 /= len1
        dy1 /= len1
        dx2 /= len2
        dy2 /= len2

        # Average direction
        avg_dx = dx1 + dx2
        avg_dy = dy1 + dy2
        avg_len = math.hypot(avg_dx, avg_dy) or 1
        avg_dx /= avg_len
        avg_dy /= avg_len

        # Perpendicular (normal) vector
        nx = avg_dy
        ny = -avg_dx

        # Outer border (full offset)
        outer_x = p1[0] + nx * outer_width
        outer_y = p1[1] + ny * outer_width
        outer_border.append((outer_x, outer_y))

        # Inner border (smaller offset inward)
        inner_x = p1[0] - nx * inner_width
        inner_y = p1[1] - ny * inner_width
        inner_border.append((inner_x, inner_y))

    return inner_border, outer_border
            
            
def load_track(geojson_path, canvas_width=800, canvas_height=600, padding=50):
    # Load GeoJSON
    gdf = gpd.read_file(geojson_path)
    track_line = gdf.geometry.iloc[0]
    coords = list(track_line.coords)

    # Extract longitudes and latitudes
    lons, lats = zip(*coords)
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    # Compute scale factors
    scale_x = (canvas_width - 2*padding) / (max_lon - min_lon)
    scale_y = (canvas_height - 2*padding) / (max_lat - min_lat)

    # Use the smaller scale to keep aspect ratio
    scale = min(scale_x, scale_y)

    # Apply scaling and add padding
    scaled_coords = [
        (
            padding + (lon - min_lon) * scale,
            canvas_height - (padding + (lat - min_lat) * scale)  # flip Y-axis
        )
        for lon, lat in coords
    ]

    return scaled_coords

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
    # Log-scale normalization: amplifies close-range signals (near walls) while allowing far lookahead
    # Maps: 0 → 0, 30 → 0.55, 500 → 1.0
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
            next_idx = (gate_indices + lookahead) % n_gates

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
    gate_pass_reward_rate = 5.0  # (10.0 when starting new later 1.0)reward for passing a gate
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
        & (gate_indices == (last_gate_indices + 1) % track.gates.shape[0])
        & cars.active
    )
    cars.gates_passed[passed_mask] += 1
    gate_reward = torch.zeros_like(speed)
    gate_reward[passed_mask] = gate_pass_reward_rate

    with torch.no_grad():
        last_gate_indices[passed_mask] = gate_indices[passed_mask]
        gate_indices[passed_mask] = (
            gate_indices[passed_mask] + 1
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

def get_items(track_name, outer_width=50, inner_width=10,
                  screen_width=800, screen_height=600):
        """Load track geometry from track name or geojson file."""
        if track_name == "simple" or track_name == "square":
            gates_per_segment = 1
            inner_width = 10
            outer_width = 45
            corner_radius = 140 if track_name == "simple" else 60
            centerline = generate_simple_track(
                screen_width,
                screen_height,
                width=600,
                height=400,
                corner_points=6,
                corner_radius=corner_radius
            )

            # Remove duplicate closure point before offsetting to avoid degenerate segments.
            if len(centerline) > 1 and np.allclose(centerline[0], centerline[-1]):
                centerline = centerline[:-1]

            track_borders = offset_track_borders(
                centerline,
                outer_width=outer_width,
                inner_width=inner_width
            )
            # Close loops without destroying the first vertex.
            track_borders[0][-1] = track_borders[0][0]  # inner_border
            track_borders[1][-1] = track_borders[1][0]  # outer_border

        elif track_name == "square_narrow":
            gates_per_segment = 1
            inner_width = 7.5
            outer_width = 42.5
            corner_radius = 25
            centerline = generate_simple_track(
                screen_width,
                screen_height,
                width=600,
                height=400,
                corner_points=5,
                corner_radius=corner_radius
            )

             # Remove duplicate closure point before offsetting to avoid degenerate segments.
            if len(centerline) > 1 and np.allclose(centerline[0], centerline[-1]):
                centerline = centerline[:-1]

            track_borders = offset_track_borders(
                centerline,
                outer_width=outer_width,
                inner_width=inner_width
            )
            # Close loops without destroying the first vertex.
            track_borders[0][-1] = track_borders[0][0]  # inner_border
            track_borders[1][-1] = track_borders[1][0]  # outer_border

        else:
            gates_per_segment = 1
            # Load custom track from geojson file
            import os
            track_path = track_name if os.path.isabs(track_name) else os.path.join(os.path.dirname(__file__), f"{track_name}.geojson")
            scaled_coords = load_track(
                track_path,
                canvas_width=screen_width,
                canvas_height=screen_height,
                padding=100
            )

            if len(scaled_coords) > 1 and np.allclose(scaled_coords[0], scaled_coords[-1]):
                scaled_coords = scaled_coords[:-1]

            track_borders = offset_track_borders(
                scaled_coords,
                outer_width=outer_width,
                inner_width=inner_width
            )

        centerline = compute_centerline(
            track_borders[0],
            track_borders[1]
        )

        gates = generate_gates(
            centerline,
            track_width=inner_width + outer_width,
            gates_per_segment=gates_per_segment
        )

        return track_borders[0], track_borders[1], gates

def args_nn():
    import argparse
    parser = argparse.ArgumentParser(description="Train racing neural network model")
    parser.add_argument("--n_cars", type=int, default=512, help="Number of cars per batch")
    parser.add_argument("--n_rays", type=int, default=15, help="Number of rays per car")
    parser.add_argument("--n_epochs", type=int, default=500, help="Number of epochs to train")
    parser.add_argument("--n_steps", type=int, default=1000, help="Steps per epoch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--input_dim", type=int, default=26, help="Input dimension size (n_rays[15] + speed[1] + heading[2] + dist_to_gate[1] + min_ray[1] + curvature[6] = 26)")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size")
    parser.add_argument("--output_dim", type=int, default=2, help="Output dimension size")
    parser.add_argument("--track", type=str, default="simple", help="Track name or path to geojson")
    parser.add_argument("--multi_track", action="store_true", help="Train on multiple tracks (simple and square_narrow)")
    parser.add_argument("--device",type=str,default=None,help="Device (mps/cpu/cuda), auto-detect if not specified",)
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file path")
    parser.add_argument("--lr_factor",type=float,default=0.5,help="ReduceLROnPlateau factor (new_lr = lr * factor)")
    parser.add_argument("--lr_patience",type=int,default=20,help="Epochs without reward improvement before reducing LR",)
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--disable_lr_scheduler",action="store_true",help="Disable automatic learning-rate reduction on plateaus",)
    parser.add_argument("--force_lr", action="store_true", help="Force learning rate to --lr even when resuming from checkpoint")
    parser.add_argument("--ray_method", type=str, default="sphere", choices=["sphere", "line"], help="Ray casting method: 'sphere' (distance field) or 'line' (segment intersection)")
    parser.add_argument("--start_mode", type=str, default="resume", choices=["resume", "start_new"], help="'resume' to continue from last epoch, 'start_new' to start from epoch 0 even if checkpoint exists.")
    parser.add_argument("--max_ray_dist", type=float, default=200.0, help="Maximum distance for ray inputs (for normalization)")
    parser.add_argument("--steer_smooth_alpha", type=float, default=0.0, help="Steering smoothing factor (0 disables smoothing)")
    parser.add_argument("--model", type=str, default="gru", choices=["gru", "lstm", "carnet"], help="Model type to train: 'gru', 'lstm', or 'carnet'")

    return parser