import argparse
import pandas as pd
import numpy as np
import sys

def analyze_movement(csv_file: str, move_threshold: float):
    """
    Analyzes the movement data from a CSV file.

    Args:
        csv_file: Path to the input CSV file.
        move_threshold: The threshold for detecting movement (ratio to person_scale).
    """
    try:
        df = pd.read_csv(csv_file)
        
        threshold = 1e-6
        df_filtered = df[(df["hip_center_x"].abs() > threshold) & (df["hip_center_y"].abs() > threshold)].copy()

        if len(df_filtered) <= 1:
            print("Not enough data to analyze movement.")
            return

        # Calculate pixel distance
        df_filtered.loc[:, "prev_x"] = df_filtered["hip_center_x"].shift(1)
        df_filtered.loc[:, "prev_y"] = df_filtered["hip_center_y"].shift(1)
        df_filtered.dropna(subset=["prev_x", "prev_y"], inplace=True)
        
        if df_filtered.empty:
            print("No valid data after shift operation.")
            return

        df_filtered.loc[:, "distance_px"] = np.sqrt(
            (df_filtered["hip_center_x"] - df_filtered["prev_x"])**2 +
            (df_filtered["hip_center_y"] - df_filtered["prev_y"])**2
        )

        # Calculate normalized distance
        valid_scale = df_filtered["person_scale"] > threshold
        df_filtered.loc[valid_scale, "distance_normalized"] = \
            df_filtered.loc[valid_scale, "distance_px"] / df_filtered.loc[valid_scale, "person_scale"]
        df_filtered.dropna(subset=["distance_normalized"], inplace=True)

        if df_filtered.empty:
            print("No valid normalized distance data to analyze.")
            return
            
        # --- Output Analysis ---
        total_comparisons = len(df_filtered)
        movement_frames = len(df_filtered[df_filtered["distance_normalized"] >= move_threshold])
        movement_ratio = (movement_frames / total_comparisons) * 100 if total_comparisons > 0 else 0

        print(f"--- Analysis for {csv_file} ---")
        print(f"Movement Threshold: {move_threshold} (ratio to torso length)")
        print("\n--- Movement Detection Ratio ---")
        print(f"Total frames compared: {total_comparisons}")
        print(f"Frames detected as 'Movement': {movement_frames}")
        print(f"Movement detection ratio: {movement_ratio:.2f}%")
        print("--------------------------------")

        print("\n--- Statistics of Normalized Movement ---")
        print(f"Average movement: {df_filtered['distance_normalized'].mean():.4f}")
        print(f"Median movement:  {df_filtered['distance_normalized'].median():.4f}")
        print(f"Max movement:     {df_filtered['distance_normalized'].max():.4f}")
        print("-----------------------------------------")


    except FileNotFoundError:
        print(f"Error: The file {csv_file} was not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze hip movement from pose estimation CSV.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file to analyze.")
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.2, 
        help="Movement threshold as a ratio of torso length (default: 0.2)."
    )
    args = parser.parse_args()
    
    analyze_movement(args.csv_file, args.threshold)
