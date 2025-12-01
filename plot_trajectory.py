import json
import matplotlib.pyplot as plt
import os
import sys

# Define path to json
JSON_PATH = "trajectory.json"
OUTPUT_IMAGE = "trajectory_plot2.png"

def plot_traj():
    if not os.path.exists(JSON_PATH):
        print(f"File not found: {JSON_PATH}")
        print("Please run the Blender script or generate_shot.py first.")
        return

    try:
        with open(JSON_PATH, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("Error decoding JSON.")
        return

    if "error" in data:
        print(f"Error in trajectory generation: {data['error']}")
        return

    trajectory = data.get("trajectory", [])
    if not trajectory:
        print("No trajectory points found.")
        return

    # Extract Coordinates
    # position is [x, y, z]
    xs = [pt['position'][0] for pt in trajectory]
    ys = [pt['position'][1] for pt in trajectory]
    zs = [pt['position'][2] for pt in trajectory]
    
    # Parameters for title
    params = data.get("parameters", {})
    title_str = (f"V: {params.get('v_mag',0):.1f} m/s | "
                 f"Spin: {params.get('w_mag',0):.0f} rad/s @ {params.get('w_angle',0):.0f}°")

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Cricket Ball Trajectory\n{title_str}", fontsize=14)

    # 1. Top View (X vs Y)
    ax1.plot(xs, ys, 'b-', label='Path')
    ax1.set_ylabel('Pitch Width (Y) [m]')
    ax1.set_title('Top View (Swing/Drift)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Draw Stumps and Crease Lines
    ax1.axhline(0, color='k', linewidth=0.8, alpha=0.5) # Center line
    ax1.axvline(0, color='k', linewidth=1) # Bowling Crease
    ax1.axvline(20.12, color='r', linewidth=2, label='Stumps') # Batting Crease
    ax1.fill_between([0, 20.12], -1.52, 1.52, color='green', alpha=0.1, label='Pitch Area') # Approx pitch width
    ax1.legend()
    # Set limits to make it look like a pitch
    ax1.set_ylim(-2.0, 2.0)

    # 2. Side View (X vs Z)
    ax2.plot(xs, zs, 'g-', label='Height')
    ax2.set_xlabel('Distance down Pitch (X) [m]')
    ax2.set_ylabel('Height (Z) [m]')
    ax2.set_title('Side View (Bounce)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Draw Ground and Stumps
    ax2.axhline(0, color='k', linewidth=2) # Ground
    ax2.axvline(20.12, color='r', linewidth=2, linestyle='--', label='Stumps (0.71m)')
    
    # Standard stump height is roughly 0.71m (28 inches)
    ax2.plot([20.12, 20.12], [0, 0.711], 'r-', linewidth=3)
    
    ax2.legend()
    ax2.set_ylim(0, max(zs) + 0.5)

    plt.tight_layout()
    
    # SAVE instead of show
    print(f"Saving plot to {OUTPUT_IMAGE}...")
    plt.savefig(OUTPUT_IMAGE)
    print("Done.")

if __name__ == "__main__":
    plot_traj()