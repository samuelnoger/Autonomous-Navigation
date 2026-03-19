import numpy as np
import math
from utils import get_items



class Car:
    def __init__(self, x, y, angle=0):
        self.x = x
        self.y = y
        self.angle = angle  # in radiants
        self.length = 18  # half-width for drawing the car as a rectangle
        self.width = 8
        self.ray_angles = [
           -np.pi/2,
           -3*np.pi/8,
           -np.pi/4,
           -np.pi/8,
           0,
           np.pi/8,
           np.pi/4,
           3*np.pi/8,
           np.pi/2
        ]
        self.ray_length = 1000
        self.rays = []
        self.velocity_vector = np.array([0.0, 0.0])  # x and y velocity components
        self.acceleration = 0
        self.max_speed = 15
        self.max_acceleration = 0.25
        self.drift_factor = 0.2
        self.color = "blue"
        self.steering = 0.2

    def update(self,track,net,sin_input,cos_input):
        # Update angle first (angle already updated in NEAT or controls)
        
        ray_dists = self.sense(track)
        norm_dists = [d / self.ray_length for d in ray_dists]
        speed = self.get_speed()
        inputs = [sin_input, cos_input] + norm_dists + [speed / self.max_speed]
        steer, accel = net.activate(inputs)
        self.angle += steer*net.steering
        self.acceleration = accel * self.max_acceleration
    
        # Compute facing direction
        heading = np.array([np.cos(self.angle), np.sin(self.angle)])
    
        # Accelerate in the direction the car is facing
        self.velocity_vector *= 0.98
        self.velocity_vector += heading * self.acceleration
    
        # Clamp speed
        speed = np.linalg.norm(self.velocity_vector)
        if speed > self.max_speed:
            self.velocity_vector = self.velocity_vector / speed * self.max_speed
            
    
        # Drift: blend velocity toward heading direction
        projected_speed = np.dot(self.velocity_vector, heading)
        ideal_velocity = heading * projected_speed
        self.velocity_vector = (
            ideal_velocity * self.drift_factor + self.velocity_vector * (1 - self.drift_factor)
        )
    
        # Update position
        self.x += self.velocity_vector[0]
        self.y += self.velocity_vector[1]
        
    def get_speed(self):
        return np.linalg.norm(self.velocity_vector)
    

    def draw(self,track_lines,canvas,color="blue"):
        # Draw the car as a rectangle
        half_w = self.width / 2
        half_l = self.length / 2

        # Define the four corners relative to center before rotation
        corners = [
            (-half_l, -half_w),  # front-left
            (-half_l, half_w),   # front-right
            (half_l, half_w),    # rear-right
            (half_l, -half_w)    # rear-left
        ]

        rotated_corners = []
        for dx, dy in corners:
            # Rotate and translate corner
            rotated_x = self.x + dx * np.cos(self.angle) - dy * np.sin(self.angle)
            rotated_y = self.y + dx * np.sin(self.angle) + dy * np.cos(self.angle)
            rotated_corners.append((rotated_x, rotated_y))

        # Flatten list of points for tkinter polygon
        points = [coord for point in rotated_corners for coord in point]
        canvas.create_polygon(points, fill=self.color, outline=color,tag = "car")

        # Draw rays
        self.rays = []
        for rel_angle in self.ray_angles:
            ray_angle = self.angle + rel_angle
            end_x = self.x + np.cos(ray_angle) * self.ray_length
            end_y = self.y + np.sin(ray_angle) * self.ray_length

            ray_line = (self.x, self.y, end_x, end_y)

            
            closest_dist = self.ray_length
            closest_point = None
            for border_line in track_lines:
                ix, iy = self.line_intersection(ray_line, border_line)
                if ix is not None:
                    dist = math.hypot(ix - self.x, iy - self.y)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_point = (ix, iy)
            
            
            if closest_point:
                ray_line = (self.x, self.y, closest_point[0], closest_point[1])
                #r = 3
                #self.canvas.create_oval(
                #   closest_point[0] - r, closest_point[1] - r,
                #   closest_point[0] + r, closest_point[1] + r,
                #   fill="red", outline="black"
                #)
            #self.canvas.create_line(*ray_line, fill="red")  # Draw the ray
            self.rays.append(ray_line)
            
    def sense(self, track):
        distances = []

        for rel_angle in self.ray_angles:

            ray_angle = self.angle + rel_angle

            end_x = self.x + math.cos(ray_angle) * self.ray_length
            end_y = self.y + math.sin(ray_angle) * self.ray_length

            ray_line = (self.x, self.y, end_x, end_y)

            closest_dist = self.ray_length

            nearby_lines = track.get_lines_along_ray(
                self.x,
                self.y,
                ray_angle,
                self.ray_length
            )

            for border_line in nearby_lines:
                ix, iy = self.line_intersection(ray_line, border_line)
                if ix is not None:
                    dist = math.hypot(ix - self.x, iy - self.y)
                    if dist < closest_dist:
                        closest_dist = dist

            distances.append(closest_dist)

        return distances

    def get_ray_distances(self, track_lines):
        # Return distances from car to track borders for each ray
        distances = []
        for x1, y1, x2, y2 in self.rays:
            min_dist = self.ray_length
            for line in track_lines:
                ix, iy = self.line_intersection((x1, y1, x2, y2), line)
                if ix is not None:
                    dist = math.hypot(ix - x1, iy - y1)
                    min_dist = min(min_dist, dist)
            distances.append(min_dist)
        return distances
    
    def check_collision(self, track_lines):
        # Check if any of the car's corners intersect with a track border segment
        half_w = self.width / 2
        half_l = self.length / 2

        corners = [
            (-half_l, -half_w),
            (-half_l, half_w),
            (half_l, half_w),
            (half_l, -half_w)
        ]

        rotated_corners = []
        for dx, dy in corners:
            rotated_x = self.x + dx * math.cos(self.angle) - dy * math.sin(self.angle)
            rotated_y = self.y + dx * math.sin(self.angle) + dy * math.cos(self.angle)
            rotated_corners.append((rotated_x, rotated_y))

        # Check for intersection between car edges and track lines
        for i in range(len(rotated_corners)):
            x1, y1 = rotated_corners[i]
            x2, y2 = rotated_corners[(i + 1) % 4]
            for line in track_lines:
                if self.line_intersection((x1, y1, x2, y2), line)[0] is not None:
                    return True
        return False
    
    def get_front_bumper_line(self):
        # Make line extend forward by, e.g., car.length * 1.5
        length_mult = 3
        front_dx = np.cos(self.angle) * self.length * length_mult
        front_dy = np.sin(self.angle) * self.length * length_mult
        x1 = self.x + front_dx
        y1 = self.y + front_dy
        x2 = self.x
        y2 = self.y
        return (x1, y1, x2, y2)

    @staticmethod
    def line_intersection(line1, line2):
        # Compute intersection point of two lines
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return None, None  # parallel

        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / den
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / den

        eps = 1e-6

        if (min(x1,x2)-eps <= px <= max(x1,x2)+eps and
            min(y1,y2)-eps <= py <= max(y1,y2)+eps and
            min(x3,x4)-eps <= px <= max(x3,x4)+eps and
            min(y3,y4)-eps <= py <= max(y3,y4)+eps):
            return px, py
        
        return None, None


class Track:
    def __init__(self, w, h, cell_size=100):
        self.w = w
        self.h = h
        self.outer_width = 50
        self.inner_width = 10
        self.track_width = 60

        self.left_border, self.right_border, self.gates = get_items(
            "redbull_ring",
            outer_width=self.outer_width,
            inner_width=self.inner_width,
            screen_width=w,
            screen_height=h
        )

        # ---- Spatial grid ----
        self.cell_size = cell_size
        self.grid_width = (w // cell_size) + 1
        self.grid_height = (h // cell_size) + 1

        # Each cell stores a list of line indices
        self.grid = [[[] for _ in range(self.grid_height)] for _ in range(self.grid_width)]
        self.all_lines = self.get_all_lines()

        # Fill the grid
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

    
    def get_nearby_lines(self, x, y):
        gx = int(x // self.cell_size)
        gy = int(y // self.cell_size)
        nearby = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx = gx + dx
                ny = gy + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    for idx in self.grid[nx][ny]:
                        nearby.append(self.all_lines[idx])
        return nearby
    
    def get_lines_along_ray(self, x, y, angle, max_dist):
        """Return border lines along a ray using grid traversal."""
        lines = []

        dx = math.cos(angle)
        dy = math.sin(angle)

        step = self.cell_size * 0.5
        steps = int(max_dist / step)

        visited = set()

        for i in range(steps):
            px = x + dx * step * i
            py = y + dy * step * i

            gx = int(px // self.cell_size)
            gy = int(py // self.cell_size)

            if (gx, gy) in visited:
                continue
            visited.add((gx, gy))

            if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                for idx in self.grid[gx][gy]:
                    lines.append(self.all_lines[idx])

        return lines
    

    def draw(self,canvas):
        # Draw borders as lines
        for i in range(len(self.left_border) - 1):
            canvas.create_line(*self.left_border[i], *self.left_border[i + 1], fill="black", width=3,tag =  "track")
        for i in range(len(self.right_border) - 1):
            canvas.create_line(*self.right_border[i], *self.right_border[i + 1], fill="black", width=3,tag =  "track")
            
        for i, [x1, y1, x2, y2] in enumerate(self.gates):
            canvas.create_line(x1, y1, x2, y2, fill="green", dash=(4, 2), tag="track")

    def get_all_lines(self):
        # Returns all border segments for ray intersection
        lines = []
        for i in range(len(self.left_border)):
            p1 = self.left_border[i]
            p2 = self.left_border[(i+1) % len(self.left_border)]
            lines.append((*p1, *p2))
        for i in range(len(self.right_border)):
            p1 = self.right_border[i]
            p2 = self.right_border[(i+1) % len(self.right_border)]
            lines.append((*p1, *p2))
        return lines