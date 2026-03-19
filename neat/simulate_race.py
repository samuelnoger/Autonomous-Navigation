import neat
import numpy as np
import pickle
import tkinter as tk
from Race.neat.neat_model import Car
from Race.neat.neat_model import Track
from utils import passed_gate

def step_visualization(cars, nets,track,root,canvas=None,timestep=0):
    global speed_text, accel_text
    
    colors = ["red", "blue", "green", "orange", "purple"]
    
    canvas.delete("car","track")  # Clear previous car and track drawings
    track.draw(canvas)
    
    gate_index = []
    for car in cars:
        gate_index.append(63)

    for i, (car, net) in enumerate(zip(cars, nets)):
        
        gate_line = track.gates[gate_index[i]]
            
        x1, y1, x2, y2 = gate_line
        gate_x = (x1 + x2) / 2
        gate_y = (y1 + y2) / 2
        
        dx = gate_x - car.x
        dy = gate_y - car.y
        
        target_angle = np.arctan2(-dy, dx)
        heading_error = target_angle - car.angle
        heading_error = np.arctan2(-np.sin(heading_error), np.cos(heading_error))
        
        sin_input = np.sin(heading_error)
        cos_input = np.cos(heading_error)
        
        speed = car.get_speed()
        
        if passed_gate(car, gate_line): 
            gate_index[i] = (gate_index[i] + 1) % len(track.gates)
        
        car.update(track,nets[i],sin_input,cos_input)
        car.draw(track.get_all_lines(), canvas,colors[i])
    
    canvas.itemconfig(timestep_text, text=f"Timestep: {timestep}")
    canvas.itemconfig(speed_text, text=f"Speed: {speed:.2f}")
    #canvas.itemconfig(accel_text, text=f"Acceleration: {accel:.2f}")
    timestep += 1
    
    canvas.update()

    # schedule next step (~60 FPS)
    root.after(16, step_visualization, cars, nets, track, root, canvas, timestep)
    
def visualize_genome(genomes, config, canvas,track, root=None):
    global speed_text, accel_text, timestep_text
    if root is None:
        return
    
    timestep = 0

    speed_text = canvas.create_text(
    50, 50,
    anchor="nw",
    text="Speed: 0",
    font=("Arial", 16),
    fill="black"
)

    accel_text = canvas.create_text(
    50, 80,
    anchor="nw",
    text="Acceleration: 0",
    font=("Arial", 16),
    fill="black"
)
    timestep_text = canvas.create_text(
    50, 120,
    anchor="nw",
    text="timestep: 0",
    font=("Arial", 16),
    fill="black"
)
    x1, y1, x2, y2 = track.gates[63]
    start_x = (x1 + x2) / 2
    start_y = (y1 + y2) / 2
    cars = []
    nets = []
    for genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        car = Car(x=start_x, y=start_y, angle=0)
        cars.append(car)
        nets.append(net)

    step_visualization(cars, nets, track,root,canvas,timestep)
    
def main():
    
    n_cars = 1
    
    config_path = "/Users/samuelnoger/Programms/Race/neat_config.txt"
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    root = tk.Tk()

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    track = Track( screen_width, screen_height)
    
    best_genomes = [None] * n_cars  # Assuming you have n_cars best genomes to load
    print("Loading best genome...")
    for i in range(n_cars):
        with open(f"best_genome_{i}.pkl", "rb") as f:
            best_genomes[i] = pickle.load(f)
    print("Best genome loaded")
    
    root.attributes("-fullscreen", True)
    root.bind("<f>", lambda e: root.attributes("-fullscreen", True))
    root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))

    canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg="white")
    canvas.pack()

    visualize_genome(best_genomes, config,canvas,track, root)
    root.mainloop()
    
if __name__ == "__main__":
    main()