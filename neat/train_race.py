import neat
import tkinter as tk
import numpy as np
import pickle
from Race.neat.neat_model import Car
from Race.neat.neat_model import Track
from utils import passed_gate
from utils import SaveBestReporter



def eval_population(genomes, config, track, track_lines, gates,starting_gate):
    nets = []
    cars = []
    fitness_total = []
    fitness_speed = []
    fitness_gates = []
    fitness_time_penalty = []
    fitness_progress = []
    gate_index = []
    last_gate_index = []
    gate_time = []
    active = []
    prev_dist = [] # Track distance to current gate for progress reward
    collided = []

    # Starting position
    x1, y1, x2, y2 = gates[starting_gate]
    start_x = (x1 + x2)/2
    start_y = (y1 + y2)/2
    dx = x2 - x1
    dy = y2 - y1
    

    angle = np.arctan2(-dy, dx)-np.pi/2

    # Initialize genomes
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        car = Car(start_x, start_y, angle)
        gate_passed = False
        nets.append(net)
        cars.append(car)
        fitness_total.append(0)
        fitness_speed.append(0)
        fitness_gates.append(0)
        fitness_time_penalty.append(0)
        fitness_progress.append(0)
        gate_index.append(starting_gate)
        last_gate_index.append(None)
        gate_time.append(0)
        active.append(True)
        prev_dist.append(0)
        collided.append(False)

    max_steps = 1000

    for t in range(max_steps):
        if not any(active):
            break

        for i, car in enumerate(cars):
            if not active[i]:
                continue
            
            
            gate_line = gates[gate_index[i]]
            
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
            
            #if i == 0 and t % 5 == 0:
            #    print(sin_input, cos_input)

            car.update(track,nets[i],sin_input,cos_input)

            # Speed contribution
            speed_contrib = 0.05 * abs(speed)
            fitness_speed[i] += speed_contrib
            fitness_total[i] += speed_contrib

            
            dt = 0.2 * t - gate_time[i]

            dist = np.hypot(car.x - gate_x, car.y - gate_y)
            progress = prev_dist[i] - dist

            if not gate_passed:
                fitness_progress[i] += 0.1*progress
                fitness_total[i] += 0.1*progress
                
            gate_passed = False
            
            #if i == 0:
            #    print("acceleration:", accel, "progress:", progress, "dist to gate:", dist, "last dist:", prev_dist[i])
            
    
            prev_dist[i] = dist
            
            if passed_gate(car, gate_line): 
                gate_passed = True
                gates_contrib = 20
                dt = 0.2 * t - gate_time[i]
                time_penalty = max(0, 0.5 * dt)  # avoid negative penalty

                fitness_gates[i] += gates_contrib
                fitness_time_penalty[i] -= time_penalty
                fitness_total[i] += gates_contrib - time_penalty

                gate_time[i] = 0.2 * t
                last_gate_index[i] = gate_index[i]        # store current gate as last
                gate_index[i] = (gate_index[i] + 1) % len(gates)  # then advance to next gate

            # Collisions or stuck
            if car.check_collision(track_lines):
                collided[i] = True
                fitness_total[i] -= 600
                active[i] = False
                
            elif speed < 0.5 and t > 60:
                fitness_total[i] -= 400
                active[i] = False
                
            if t - gate_time[i] > 200:
                fitness_total[i] -= 400
                #print(f"Car {i} timed out at time {t*0.2:.1f}s")
                active[i] = False

    # Assign total fitness to genome
    for (genome_id, genome), fit in zip(genomes, fitness_total):
        genome.fitness = fit
    
    winner_idx = np.argmax(fitness_total)
    print("\nWinner genome fitness breakdown:")
    if collided[winner_idx] == True:
        print(f"Car {i} collided at gate {gate_index[winner_idx]}")
    print(f"  Total fitness: {fitness_total[winner_idx]:.2f}")
    print(f"    Speed contrib: {fitness_speed[winner_idx]:.2f}")
    print(f"    Gates contrib: {fitness_gates[winner_idx]:.2f}")
    print(f"    Time penalty : {fitness_time_penalty[winner_idx]:.2f}")
    print(f"    Progress contrib: {fitness_progress[winner_idx]:.2f}")
        
            
def run_neat(config, track, n=10, i=0):
    import neat
    import multiprocessing
    from functools import partial
    import pickle
    from tqdm import tqdm
    
    track_lines = track.get_all_lines()  # Get all track border lines for collision detection
    gates = track.gates  # Get gates for fitness evaluation
    
    # Partial function to pass to Pool

    p = neat.Population(config)
    #p.add_reporter(SaveBestReporter())
    # Print progress to the console
    #p.add_reporter(neat.StdOutReporter(True))
    # Keep track of statistics for plotting later
    p.add_reporter(neat.StatisticsReporter())

    winner = None
    starting_gate = 63  

    # ---- Custom generation-level loop ----
    for gen in tqdm(range(n), desc="NEAT Generations"):

        genomes = list(p.population.items())

        eval_population(genomes, config, track, track_lines, gates,starting_gate)

        winner = max(p.population.values(), key=lambda g: g.fitness)

        p.reporters.post_evaluate(config, p.population, p.species, winner)
        p.population = p.reproduction.reproduce(config, p.species, config.pop_size, p.generation)
        p.species.speciate(config, p.population, p.generation)
        p.generation += 1

        # Optionally save every 10 generations
        if gen % 10 == 0:
            with open(f"best_genome_{i}.pkl", "wb") as f:
                pickle.dump(winner, f)

    return winner

def main():
    
    n_gens = 50
    n_cars = 1
    # ---- load NEAT config ----
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
    
    for i in range(n_cars):
        
        print(f"Starting training car {i}")
        winner = run_neat(config, track, n_gens,i)

        with open(f"best_genome_{i}.pkl", "wb") as f:
                pickle.dump(winner, f)

    print("Training finished")
    
if __name__ == "__main__":
    main()