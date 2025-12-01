import sys
import json
import numpy as np
import pandas as pd
import os
import argparse
from keras.models import load_model
from joblib import load

# --- PATH SETUP (CRITICAL FIX) ---
# Get the absolute path of the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Add this directory to sys.path so we can import forward_sim regardless of CWD
sys.path.append(BASE_DIR)

# Import Physics
from forward_sim import air_sim, GRAVITY_VEC, BALL_MASS

# --- Configuration using Absolute Paths ---
MODEL_PATH = os.path.join(BASE_DIR, "cricket_trajectory_model.keras")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "scaler_X.joblib")
SCALER_Y_PATH = os.path.join(BASE_DIR, "scaler_y.joblib")

# --- Bins (Must match Training exactly) ---
V_BINS = [0, 22, 26, np.inf]
V_LABELS = ['Low', 'Medium', 'High']

W_MAG_BINS = [0, 150, 220, np.inf]
W_MAG_LABELS = ['Low', 'Medium', 'High']

W_ANGLE_BINS = [-np.inf, -10, 10, np.inf]
W_ANGLE_LABELS = ['Negative_Spin', 'Neutral_Spin', 'Positive_Spin']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=float, nargs=3, required=True, help="Start X Y Z")
    parser.add_argument("--target", type=float, nargs=2, required=True, help="Target X Y")
    parser.add_argument("--v_cat", type=str, required=True)
    parser.add_argument("--w_mag_cat", type=str, required=True)
    parser.add_argument("--w_angle_cat", type=str, required=True)
    parser.add_argument("--output", type=str, default="trajectory.json")
    return parser.parse_args()

def write_error(output_path, message):
    """Writes an error message to the output JSON so Blender can see it."""
    try:
        with open(output_path, 'w') as f:
            json.dump({"error": message}, f, indent=4)
    except Exception as e:
        print(f"Critical Error: Could not write error to file: {e}", file=sys.stderr)

def main():
    args = get_args()

    # 1. Load AI Artifacts
    try:
        # Check if files exist explicitly to give better errors
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
            
        model = load_model(MODEL_PATH)
        preprocessor = load(PREPROCESSOR_PATH)
        scaler_y = load(SCALER_Y_PATH)
    except Exception as e:
        write_error(args.output, f"Failed to load AI models: {str(e)}")
        return

    # 2. Construct Input DataFrame
    input_data = pd.DataFrame([{
        'land_x': args.target[0],
        'land_y': args.target[1],
        'p_x': args.start[0],
        'p_y': args.start[1],
        'p_z': args.start[2],
        'v_cat': args.v_cat,
        'w_mag_cat': args.w_mag_cat,
        'w_angle_cat': args.w_angle_cat
    }])

    # 3. Predict Initial Conditions
    try:
        X_scaled = preprocessor.transform(input_data)
        y_pred_scaled = model.predict(X_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
        # Unpack predictions
        v_mag, phi, theta, w_mag, w_angle = y_pred[0]
        
    except Exception as e:
        write_error(args.output, f"Prediction logic failed: {str(e)}")
        return

    # 4. Run Physics Simulation
    try:
        # Convert parameters to vectors
        phi_rad = np.radians(phi)
        theta_rad = np.radians(theta)
        w_angle_rad = np.radians(w_angle)
        
        pos_in = np.array(args.start)
        
        v_z = v_mag * np.sin(phi_rad)
        v_xy = v_mag * np.cos(phi_rad)
        v_in = np.array([v_xy * np.cos(theta_rad), v_xy * np.sin(theta_rad), v_z])
        
        w_x = w_mag * np.sin(w_angle_rad)
        w_y = w_mag * np.cos(w_angle_rad)
        w_in = np.array([w_x, w_y, 0])
        
        # Run full trajectory sim with bounce
        trajectory = air_sim(pos_in, v_in, 60, GRAVITY_VEC, BALL_MASS, w_in, 
                             return_trajectory=True, simulate_bounce=True)
    except Exception as e:
        write_error(args.output, f"Physics engine failed: {str(e)}")
        return

    # 5. Export
    output_data = {
        "status": "success",
        "parameters": {
            "v_mag": float(v_mag),
            "phi": float(phi),
            "theta": float(theta),
            "w_mag": float(w_mag),
            "w_angle": float(w_angle)
        },
        "trajectory": trajectory
    }
    
    try:
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=4)
    except Exception as e:
        print(f"Failed to write output file: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()