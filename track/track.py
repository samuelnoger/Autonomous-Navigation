import torch
import math
import numpy as np
import cv2 # type: ignore
from utils.my_utils import line_intersection
import geopandas as gpd # type: ignore

class Track:
    def __init__(
        self,
        track_name,
        w,
        h,
        cell_size=100,
        device='mps',
        ray_method='sphere',
        sphere_max_steps=48,
        sphere_hit_epsilon=1.5,
        sphere_min_step=0.5,
    ):
        """Initialize track with geometry and collision detection.

        Args:
            track_name: Name of the track ('simple' or path to geojson)
            w: Track width in pixels
            h: Track height in pixels
            cell_size: Grid cell size for spatial indexing
            device: Torch device for tensors
            ray_method: 'sphere' (distance-field tracing) or 'line' (segment intersection)
            sphere_max_steps: max sphere-tracing steps per ray
            sphere_hit_epsilon: hit threshold in pixels
            sphere_min_step: minimum march step in pixels
        """
        self.w = w
        self.h = h
        self.car_radius = 2
        self.outer_width = 32.5
        self.inner_width = 7.5
        self.track_width = self.outer_width + self.inner_width
        self.track_name = track_name
        self.ray_method = ray_method
        self.sphere_max_steps = sphere_max_steps
        self.sphere_hit_epsilon = sphere_hit_epsilon
        self.sphere_min_step = sphere_min_step

        self.left_border, self.right_border, self.gates = self.get_items(
            track_name,
            outer_width=self.outer_width,
            inner_width=self.inner_width,
            screen_width=w,
            screen_height=h
        )

        # Convert borders and gates to tensors
        self.left_border = torch.tensor(self.left_border, dtype=torch.float32)  # (L,2)
        self.right_border = torch.tensor(self.right_border, dtype=torch.float32)  # (R,2)
        self.gates = torch.tensor(self.gates, dtype=torch.float32)  # (G,4) -> x1,y1,x2,y2

        # Precompute gate distances for normalization
        gate_centers = (self.gates[:, 0:2] + self.gates[:, 2:4]) / 2  # (G,2)
        gate_diffs = gate_centers[1:] - gate_centers[:-1]
        gate_distances = torch.norm(gate_diffs, dim=1)
        self.min_gate_dist = float(gate_distances.min().item()) if len(gate_distances) > 0 else 1.0
        self.max_gate_dist = float(gate_distances.max().item()) if len(gate_distances) > 0 else 1.0

        # Prepare all lines
        self.all_lines = self.get_all_lines()  # (N,4) tensor
        self._all_lines_by_device = {}

        self.build_distance_field(device)

        # Spatial grid
        self.cell_size = cell_size
        self.grid_width = (w // cell_size) + 1
        self.grid_height = (h // cell_size) + 1
        self.grid = [[[] for _ in range(self.grid_height)] for _ in range(self.grid_width)]

        # Fill grid
        for idx, line in enumerate(self.all_lines):
            x1, y1, x2, y2 = line
            min_x = int(min(x1, x2) // cell_size)
            max_x = int(max(x1, x2) // cell_size)
            min_y = int(min(y1, y2) // cell_size)
            max_y = int(max(y1, y2) // cell_size)
            for gx in range(min_x, max_x + 1):
                for gy in range(min_y, max_y + 1):
                    if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                        self.grid[gx][gy].append(idx)

        max_dist = 0.0
        for gates_idx,_ in enumerate(self.gates):
            gate_center_1 = (self.gates[gates_idx, 0:2] + self.gates[gates_idx, 2:4]) / 2
            gate_center_2 = (self.gates[(gates_idx+1)%self.gates.shape[0], 0:2] + self.gates[(gates_idx+1)%self.gates.shape[0], 2:4]) / 2
            dist_between_gates = torch.norm(gate_center_2 - gate_center_1)
            if dist_between_gates > max_dist:
                max_dist = dist_between_gates

        self.max_gate_dist = max_dist


    def get_all_lines(self):
        """Return all track lines as a tensor (N,4).

        Extracts line segments from left and right borders.
        """
        lines = []
        for border in [self.left_border, self.right_border]:
            for i in range(len(border)):
                p1 = border[i]
                p2 = border[(i + 1) % len(border)]
                if torch.allclose(p1, p2):
                    continue
                lines.append(torch.cat([p1, p2]))
        return torch.stack(lines, dim=0)  # (N,4)
    
    def build_distance_field(self, device="mps"):
        """Build distance field for collision detection using distance transform.

        Creates a 2D distance field where each pixel contains the distance to the nearest wall.
        """
        # Create white image (track space)
        img = np.ones((self.h, self.w), dtype=np.uint8) * 255

        # Draw track walls
        for line in self.all_lines:
            x1, y1, x2, y2 = map(int, line.tolist())
            cv2.line(img, (x1, y1), (x2, y2), 0, 2)

        # Compute distance transform
        dist = cv2.distanceTransform(img, cv2.DIST_L2, 5)

        # Convert to torch tensor
        self.distance_field = torch.tensor(dist, dtype=torch.float32, device=device)

    def _sample_distance_field_bilinear(self, points):
        """Bilinear sample distance field at points of shape (K,2)."""
        x = points[:, 0]
        y = points[:, 1]

        # Ensure all index tensors are on the same device as distance_field
        df_device = self.distance_field.device
        x0 = torch.floor(x).long().clamp(0, self.w - 1).to(df_device)
        y0 = torch.floor(y).long().clamp(0, self.h - 1).to(df_device)
        x1 = (x0 + 1).clamp(0, self.w - 1)
        y1 = (y0 + 1).clamp(0, self.h - 1)

        wx = (x - x0.float().to(x.device)).to(df_device)
        wy = (y - y0.float().to(y.device)).to(df_device)

        x1 = x1.to(df_device)
        y1 = y1.to(df_device)

        d00 = self.distance_field[y0, x0]
        d10 = self.distance_field[y0, x1]
        d01 = self.distance_field[y1, x0]
        d11 = self.distance_field[y1, x1]

        d0 = d00 * (1.0 - wx) + d10 * wx
        d1 = d01 * (1.0 - wx) + d11 * wx
        return d0 * (1.0 - wy) + d1 * wy

    def _get_lines_along_rays_sphere(self, positions, ray_angles, max_dist=200.0):
        """Distance-field sphere tracing ray cast."""
        n_cars, n_rays = ray_angles.shape
        device = positions.device

        origins = positions.unsqueeze(1).expand(-1, n_rays, -1).reshape(-1, 2)
        dirs = torch.stack([torch.cos(ray_angles), torch.sin(ray_angles)], dim=-1).reshape(-1, 2)

        n_total = origins.shape[0]
        t = torch.zeros(n_total, device=device)
        active = torch.ones(n_total, dtype=torch.bool, device=device)
        hit = torch.zeros(n_total, dtype=torch.bool, device=device)

        for _ in range(self.sphere_max_steps):
            if not active.any():
                break

            points = origins + dirs * t.unsqueeze(1)
            x = points[:, 0]
            y = points[:, 1]

            in_bounds = (x >= 0.0) & (x <= self.w - 1) & (y >= 0.0) & (y <= self.h - 1)
            still_valid = active & in_bounds & (t < max_dist)
            if not still_valid.any():
                break

            dist = self._sample_distance_field_bilinear(points, device)
            # Ensure dist is on the same device as positions
            dist = dist.to(positions.device)

            just_hit = still_valid & (dist <= self.sphere_hit_epsilon)
            hit = hit | just_hit
            active[just_hit] = False

            to_march = still_valid & ~just_hit
            step = dist.clamp(min=self.sphere_min_step)
            t[to_march] = t[to_march] + step[to_march]

            active = active & (t < max_dist)

        t = torch.clamp(t, max=max_dist)
        t[~hit & (t >= max_dist)] = max_dist
        return t.reshape(n_cars, n_rays)

    def _get_lines_along_rays_line(self, positions, ray_angles, max_dist=200.0):
        """Original segment-intersection ray cast."""
        N, R = ray_angles.shape
        device = positions.device

        # Compute ray endpoints
        dx = torch.cos(ray_angles) * max_dist  # (N,R)
        dy = torch.sin(ray_angles) * max_dist
        ray_ends = positions.unsqueeze(1) + torch.stack([dx, dy], dim=-1)  # (N,R,2)

        # Flatten rays for vectorized line intersection
        rays_start = positions.unsqueeze(1).expand(-1, R, 2).reshape(-1, 2)  # (N*R,2)
        rays_end = ray_ends.reshape(-1, 2)  # (N*R,2)
        ray_lines = torch.cat([rays_start, rays_end], dim=1)  # (N*R,4)

        # Cache line tensor on the active device to avoid repeated transfers.
        device_key = str(device)
        if device_key not in self._all_lines_by_device:
            self._all_lines_by_device[device_key] = self.all_lines.to(device)
        track_lines = self._all_lines_by_device[device_key]  # (M,4)
        M = track_lines.shape[0]
        ray_lines_exp = ray_lines.unsqueeze(1).expand(-1, M, 4)  # (N*R,M,4)
        track_lines_exp = track_lines.unsqueeze(0).expand(N * R, -1, -1)  # (N*R,M,4)

        # Compute intersections
        inter = line_intersection(ray_lines_exp, track_lines_exp)  # (N*R,M,2)
        inter_valid = ~torch.isnan(inter[..., 0])

        # Distances
        ray_pos = rays_start.unsqueeze(1).expand(-1, M, 2)
        dists2 = ((inter - ray_pos) ** 2).sum(dim=-1)  # (N*R,M)
        dists2[~inter_valid] = max_dist * max_dist

        # Minimum distance per ray
        min_dists2, _ = dists2.min(dim=1)  # (N*R,)
        return torch.sqrt(min_dists2).reshape(N, R)
    
    def get_nearby_lines(self, x, y):
        gx = int((x / self.cell_size).item())
        gy = int((y / self.cell_size).item())
        
        x0 = max(gx - 1, 0)
        x1 = min(gx + 1, self.grid_width - 1)
        y0 = max(gy - 1, 0)
        y1 = min(gy + 1, self.grid_height - 1)
    
        nearby = []
    
        for nx in range(x0, x1 + 1):
            for ny in range(y0, y1 + 1):
                nearby.extend(self.grid[nx][ny])

        return nearby

    def get_lines_along_rays(self, positions, ray_angles, max_dist=200.0):
        """
        Vectorized ray-line intersection for a batch of cars and multiple rays.

        positions: (N,2) tensor of car positions
        ray_angles: (N,R) tensor of ray angles relative to global frame
        max_dist: maximum ray length

        Returns: (N,R) tensor of distances to nearest line along each ray
        """
        if self.ray_method == 'line':
            return self._get_lines_along_rays_line(positions, ray_angles, max_dist=max_dist)
        return self._get_lines_along_rays_sphere(positions, ray_angles, max_dist=max_dist)
    
    def offset_track_borders(self,centerline, outer_width=80, inner_width=20):
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


    def get_items(self,track_name, outer_width=50, inner_width=10,
                  screen_width=800, screen_height=600):
        """Load track geometry from track name or geojson file."""
        if track_name == "simple" or track_name == "square":
            gates_per_segment = 1
            inner_width = 10
            outer_width = 45
            corner_radius = 140 if track_name == "simple" else 60
            centerline = self.generate_simple_track(
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

            track_borders = self.offset_track_borders(
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
            centerline = self.generate_simple_track(
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

            track_borders = self.offset_track_borders(
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
            scaled_coords = self.load_track(
                track_path,
                canvas_width=screen_width,
                canvas_height=screen_height,
                padding=100
            )

            if len(scaled_coords) > 1 and np.allclose(scaled_coords[0], scaled_coords[-1]):
                scaled_coords = scaled_coords[:-1]

            track_borders = self.offset_track_borders(
                scaled_coords,
                outer_width=outer_width,
                inner_width=inner_width
            )

        centerline = self.compute_centerline(
            track_borders[0],
            track_borders[1]
        )

        gates = self.generate_gates(
            centerline,
            track_width=inner_width + outer_width,
            gates_per_segment=gates_per_segment
        )

        return track_borders[0], track_borders[1], gates
    
    def compute_centerline(self, inner_border, outer_border):
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

    def generate_gates(self, centerline, track_width, gates_per_segment=1):
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


    def load_track(self, geojson_path, canvas_width=800, canvas_height=600, padding=50):
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