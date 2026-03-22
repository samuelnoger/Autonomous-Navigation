import tkinter as tk
import torch
from model import CarState, CarNet # your trained model classes
from model import LSTMCarNet,GRUCarNet, SeparateHeadsCarNet, DeepCarNet, AttentionCarNet  # Import all variants
from track import Track  # your Track class
from utils import get_inputs  # function to compute NN inputs from track and car state
from utils import args_nn  # argument parser for simulation parameters

# -----------------------------
# Simulation parameters
# -----------------------------
PHYSICS_DT = 0.1  # seconds per physics update
CAR_WIDTH = 5
CAR_LENGTH = 10
MAX_RENDER_FPS = 60  # optional cap on rendering speed


# -----------------------------
# Tkinter visualization
# -----------------------------
class Simulation:

    def __init__(self, track, model, n_cars=1, n_rays=13, start_idx=0, device="cpu", max_ray_dist=500, steer_smooth_alpha=0.6):
        self.track = track
        self.model = model.to(device)
        self.n_cars = n_cars
        self.n_rays = n_rays
        self.device = device
        self.max_ray_dist = max_ray_dist
        self.steer_smooth_alpha = steer_smooth_alpha
        self.start_idx = start_idx
        self.timestep = 0
        self.epoch = 0

        # Initialize LSTM hidden state if using LSTM model
        self.hidden = None
        if hasattr(self.model, 'init_hidden'):
            self.hidden = self.model.init_hidden(n_cars, device)

        self.cars = CarState(n_cars, n_rays=n_rays, device=device)
        self.gates_tensor = self.track.gates.to(self.device)
        self.gate_indices = torch.full((n_cars,), start_idx, dtype=torch.long, device=device)
        self._reset_cars()

        # Tkinter setup
        self.root = tk.Tk()
        self.root.title("Simulation")
        self.canvas = tk.Canvas(self.root, width=track.w, height=track.h, bg="white")
        self.canvas.pack()

        # Draw track once
        self.track_lines = []
        for line in track.all_lines:
            x1, y1, x2, y2 = line.tolist()
            self.track_lines.append(
                self.canvas.create_line(x1, y1, x2, y2, fill="black", width=2)
            )

        # Draw gates and store references
        self.gate_lines = []
        for gate in track.gates:
            x1, y1, x2, y2 = gate.tolist()
            self.gate_lines.append(
                self.canvas.create_line(x1, y1, x2, y2, fill="green", dash=(4, 2), width=1)
            )

        # Car rectangles
        self.car_rects = []
        for i in range(n_cars):
            rect = self.canvas.create_polygon(0, 0, 0, 0, 0, 0, 0, 0, fill="blue")
            self.car_rects.append(rect)

        self.accum_time = 0.0
        self.last_time = None
        t = 0
        self.root.bind("r", self.on_restart)
        self.root.after(0, self.update, t)
        self.root.mainloop()

    def _reset_cars(self):
        self.cars.reset(self.track, self.start_idx, self.epoch) 
        self.gate_indices.fill_(self.start_idx)
        # For reverse cars, move one step in their direction so they aim at the correct next gate
        self.gate_indices = (self.gate_indices + self.cars.direction) % self.gates_tensor.shape[0]
        if hasattr(self.cars, "prev_steer"):
            self.cars.prev_steer.zero_()
        # Reset LSTM hidden state when resetting cars
        if hasattr(self.model, 'init_hidden'):
            self.hidden = self.model.init_hidden(self.n_cars, self.device)

    def on_restart(self, _event=None):
        self.timestep = 0
        self.epoch += 1
        self._reset_cars()

    # Helper: rectangle corners based on position and angle
    def get_car_corners(self, pos, angle):
        cx, cy = pos
        w, l = CAR_WIDTH / 2, CAR_LENGTH / 2

        # rectangle points relative to center
        corners = torch.tensor(
            [[-l, -w], [-l, w], [l, w], [l, -w]], dtype=torch.float32
        )

        # rotation matrix
        c, s = torch.cos(angle), torch.sin(angle)
        rot = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
        rotated = corners @ rot.T
        rotated += pos
        return rotated.flatten().tolist()

    def update(self, t):
        now = (
            self.root.winfo_pointerx()
        )  # approximate time, we just use fixed step here
        # advance physics fixed dt
        self.cars.prev_pos = self.cars.pos.clone()

        # --- Compute NN inputs ---
        next_gate_centers = (
            self.gates_tensor[self.gate_indices, 0:2]
            + self.gates_tensor[self.gate_indices, 2:4]
        ) / 2
        inputs, ray_dists = get_inputs(
            self.track,
            self.cars,
            next_gate_centers=next_gate_centers,
            max_ray_dist=self.max_ray_dist,
            gates_tensor=self.gates_tensor,
            gate_indices=self.gate_indices
        )

        with torch.no_grad():
            # Handle LSTM models with hidden state
            if self.hidden is not None:
                outputs, self.hidden = self.model(inputs, self.hidden)
            else:
                outputs = self.model(inputs)
            steer = outputs[:, 0]
            accel = outputs[:, 1]
            self.cars.physics_update(steer, accel, dt=PHYSICS_DT, steer_smooth_alpha=self.steer_smooth_alpha)
            self.cars.check_collisions(self.track)

            # Advance gate indices when car is close to the current gate line
            gate_coords = self.gates_tensor[self.gate_indices]
            gx1, gy1 = gate_coords[:, 0], gate_coords[:, 1]
            gx2, gy2 = gate_coords[:, 2], gate_coords[:, 3]
            line_vec = torch.stack([gx2 - gx1, gy2 - gy1], dim=1)
            p_vec = self.cars.pos - torch.stack([gx1, gy1], dim=1)
            line_len2 = (line_vec ** 2).sum(dim=1)
            u = torch.clamp((p_vec * line_vec).sum(dim=1) / line_len2, 0.0, 1.0)
            closest = torch.stack([gx1, gy1], dim=1) + u.unsqueeze(1) * line_vec
            dist_to_gate = (self.cars.pos - closest).norm(dim=1)
            passed = dist_to_gate < 10.0
            self.gate_indices[passed] = (self.gate_indices[passed] + self.cars.direction[passed]) % self.gates_tensor.shape[0]

        # Recompute rays for the updated car state so drawn rays match current car positions.
        _, ray_dists_viz = get_inputs(
            self.track,
            self.cars,
            next_gate_centers=next_gate_centers,
            max_ray_dist=self.max_ray_dist,
            gates_tensor=self.gates_tensor,
            gate_indices=self.gate_indices
        )
        # ray_dists_viz are already in original units (pixels), no need to invert log normalization
        ray_lengths = ray_dists_viz
        ray_angles = self.cars.ray_angles  # [n_cars, n_rays]
        dx = torch.cos(ray_angles) * ray_lengths  # [n_cars, n_rays]
        dy = torch.sin(ray_angles) * ray_lengths
        ray_ends = self.cars.pos.unsqueeze(1) + torch.stack([dx, dy], dim=2)

        # --- Draw cars ---
        for i in range(self.n_cars):
            pos = self.cars.pos[i].cpu()
            angle = self.cars.angle[i].cpu()
            corners = self.get_car_corners(pos, angle)
            self.canvas.coords(self.car_rects[i], *corners)
            color = "red" if self.cars.active[i] else "gray"
            self.canvas.itemconfig(self.car_rects[i], fill=color)

        self.canvas.delete("ray")

        for i in range(self.n_cars):
            if not self.cars.active[i]:
                continue  # skip inactive cars
            start = self.cars.pos[i].cpu()
            for j in range(ray_ends.shape[1]):
                end = ray_ends[i, j].cpu()
                x0, y0 = start.tolist()
                x1, y1 = end.tolist()
                self.canvas.create_line(
                    x0, y0, x1, y1, fill="orange", width=1, tags="ray"
                )

        self.canvas.delete("text")
        self.canvas.create_text(
            50, 50, text="Timestep:" + str(self.timestep), fill="black", tags="text"
        )

        # Update gate colors: current gate is red, others are green
        for gate_idx, line_obj in enumerate(self.gate_lines):
            color = "red" if gate_idx == self.gate_indices[0].item() else "green"
            self.canvas.itemconfig(line_obj, fill=color)

        if self.n_cars == 1:
            steer_val = steer[0].item()
            accel_val = accel[0].item()
            speed_val = self.cars.speed[0].item()
            self.canvas.create_text(
                10, 80, anchor="w",
                text=f"Steer: {steer_val:+.2f}",
                fill="blue", font=("Courier", 12), tags="text"
            )
            self.canvas.create_text(
                10, 100, anchor="w",
                text=f"Accel: {accel_val:+.2f}",
                fill="blue", font=("Courier", 12), tags="text"
            )
            self.canvas.create_text(
                10, 120, anchor="w",
                text=f"Speed: {speed_val:+.1f} px/s",
                fill="blue", font=("Courier", 12), tags="text"
            )

        self.timestep += 1
        # schedule next update
        self.root.after(int(1000 / MAX_RENDER_FPS), self.update, self.timestep)


# -----------------------------
# Model to simulate (change this variable to switch models)
MODEL_NAME = "carnet"  # Options: "gru", "lstm", "carnet", "separate_heads", "deep", "attention"

# Model class mapping
MODEL_MAP = {
    "gru": (GRUCarNet, "last_ckpt_GRUCarNet.pth"),
    "lstm": (LSTMCarNet, "last_ckpt_LSTMCarNet.pth"),
    "carnet": (CarNet, "last_ckpt_CarNet.pth"),
    "separate_heads": (SeparateHeadsCarNet, "last_ckpt_SeparateHeadsCarNet.pth"),
    "deep": (DeepCarNet, "last_ckpt_DeepCarNet.pth"),
    "attention": (AttentionCarNet, "last_ckpt_AttentionCarNet.pth"),
}

if __name__ == "__main__":
    n_cars = 1

    # Load config.json if it exists
    import json, os
    config = {}
    if os.path.exists("config.json"):
        with open("config.json") as f:
            config = json.load(f)
        print(f"Loaded config from config.json: {config}")

    parser = args_nn()
    # Set defaults from config
    if config:
        parser.set_defaults(**config)

    args = parser.parse_args()

    device = torch.device("cpu")

    # Handle multi-track config - default to simple track for simulation
    track_name = args.track
    if getattr(args, 'multi_track', False):
        track_name = "simple"
        print(f"Multi-track training detected. Using '{track_name}' track for simulation.")
    else:
        print(f"Loading track: {track_name}")

    track = Track(track_name, 1000, 600, device=device, ray_method=args.ray_method)

    # Select model based on MODEL_NAME variable
    if MODEL_NAME not in MODEL_MAP:
        print(f"Error: MODEL_NAME='{MODEL_NAME}' not in {list(MODEL_MAP.keys())}")
        exit(1)

    model_class, default_checkpoint = MODEL_MAP[MODEL_NAME]

    # Determine checkpoint path - prioritize MODEL_NAME over args.checkpoint
    checkpoint_candidates = [
        f"checkpoints/{default_checkpoint}",
        default_checkpoint,
    ]

    checkpoint_path = None
    for candidate in checkpoint_candidates:
        if os.path.exists(candidate):
            checkpoint_path = candidate
            break

    if not checkpoint_path:
        print(f"Error: Could not find checkpoint for {MODEL_NAME}")
        print(f"Tried: {checkpoint_candidates}")
        exit(1)

    print(f"Using model: {MODEL_NAME}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    if MODEL_NAME == "gru":
        model = model_class(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            n_rays=args.n_rays,
            gru_layers=1
        )
    elif MODEL_NAME == "lstm":
        model = model_class(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            n_rays=args.n_rays,
            lstm_layers=1
        )
    else:
        # carnet, separate_heads, deep, attention
        model = model_class(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            n_rays=args.n_rays
        )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    sim = Simulation(
        track, model, n_cars=n_cars, n_rays=args.n_rays, start_idx=track.gates.shape[0] - 1, device=device, max_ray_dist=args.max_ray_dist, steer_smooth_alpha=args.steer_smooth_alpha
    )
