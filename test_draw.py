import tkinter as tk
import cv2  # type: ignore
import numpy as np
from track import Track 


def build_distance_field_image(track):
    dist = track.distance_field.detach().cpu().numpy()
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(dist_norm, cv2.COLORMAP_TURBO)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    height, width, _ = heatmap_rgb.shape
    ppm_header = f"P6 {width} {height} 255\n".encode("ascii")
    ppm_data = ppm_header + heatmap_rgb.tobytes()
    return tk.PhotoImage(data=ppm_data, format="PPM")


def draw_distance_field(track, canvas):
    image = build_distance_field_image(track)
    canvas.distance_field_image = image
    canvas.create_image(0, 0, anchor="nw", image=image)

def draw_track(track, canvas):
    canvas.delete("all")

    draw_distance_field(track, canvas)
    
    # Draw left border
    for i in range(len(track.left_border)-1):
        x1, y1 = track.left_border[i].tolist()
        x2, y2 = track.left_border[i+1].tolist()
        canvas.create_line(x1, y1, x2, y2, fill="black", width=1)
        
    # Draw right border
    for i in range(len(track.right_border)-1):
        x1, y1 = track.right_border[i].tolist()
        x2, y2 = track.right_border[i+1].tolist()
        canvas.create_line(x1, y1, x2, y2, fill="black", width=1)
        
    # Draw gates
    for i,gate in enumerate(track.gates):
        x1, y1, x2, y2 = gate.tolist()
        if i == len(track.gates)-1:
            canvas.create_line(x1, y1, x2, y2, fill="red", dash=(4,2), width=1)
        else:
            canvas.create_line(x1, y1, x2, y2, fill="green", dash=(4,2), width=1)

def main():
    screen_width = 1000
    screen_height = 600
    
    root = tk.Tk()
    root.title("Track Visualization")
    
    canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg="white")
    canvas.pack()
    
    track = Track("redbull_ring",screen_width, screen_height)
    
    draw_track(track, canvas)
    
    root.mainloop()

if __name__ == "__main__":
    main()