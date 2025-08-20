import argparse
import pandas as pd
import numpy as np
import sys

def analyze_ideas(csv_file: str):
    """
    Analyzes a CSV file using 5 different movement detection ideas.
    Returns a dictionary with the number of movement frames for each idea.
    """
    try:
        df = pd.read_csv(csv_file)
        
        # --- Pre-computation ---
        threshold = 1e-6
        df_filtered = df[(df["person_scale"].abs() > threshold)].copy()
        
        df_filtered.loc[:, "prev_x"] = df_filtered["hip_center_x"].shift(1)
        df_filtered.loc[:, "prev_y"] = df_filtered["hip_center_y"].shift(1)
        df_filtered.dropna(subset=["prev_x", "prev_y"], inplace=True)
        
        if df_filtered.empty: return None

        df_filtered.loc[:, "distance_px"] = np.sqrt(
            (df_filtered["hip_center_x"] - df_filtered["prev_x"])**2 +
            (df_filtered["hip_center_y"] - df_filtered["prev_y"])**2
        )
        df_filtered.loc[:, "distance_normalized"] = df_filtered["distance_px"] / df_filtered["person_scale"]
        
        total_frames = len(df_filtered)
        if total_frames == 0: return None

        # --- Apply 5 Ideas ---

        # Idea 1: Simple Threshold (0.1)
        idea1_frames = (df_filtered['distance_normalized'] >= 0.1).sum()

        # Idea 2: Spike Detection (1.5 over 1 sec/30 frames)
        is_spike = df_filtered['distance_normalized'].rolling(window=30, min_periods=1).max() >= 1.5
        idea2_frames = is_spike.sum()
        
        # Idea 3: Cumulative Movement (sum > 5.0 over 3 sec/90 frames)
        is_cumulative_move = df_filtered['distance_normalized'].rolling(window=90, min_periods=1).sum() > 5.0
        idea3_frames = is_cumulative_move.sum()

        # Idea 4: Stability (std dev of scale > 50px over 2 sec/60 frames)
        is_unstable = df_filtered['person_scale'].rolling(window=60, min_periods=1).std().fillna(0) > 50.0
        idea4_frames = is_unstable.sum()
        
        # Idea 5: Composite (Idea 2 OR Idea 4)
        is_composite_move = is_spike | is_unstable
        idea5_frames = is_composite_move.sum()
        
        return {
            "total": total_frames,
            "idea1": idea1_frames,
            "idea2": idea2_frames,
            "idea3": idea3_frames,
            "idea4": idea4_frames,
            "idea5": idea5_frames,
        }

    except Exception as e:
        print(f"Error processing {csv_file}: {e}", file=sys.stderr)
        return None

def main(file1: str, file2: str):
    """Compares the results of the 5 ideas on two CSV files."""
    results1 = analyze_ideas(file1)
    results2 = analyze_ideas(file2)

    if not results1 or not results2:
        sys.exit(1)

    # --- Print Comparison Table ---
    header = f"| {'Idea & Condition':<50} | {'Video 1':<20} | {'Video 2':<20} |"
    separator = "-" * len(header)
    
    print(separator)
    print(header)
    print(separator)

    def print_row(name, desc, res1, res2):
        total1, total2 = results1['total'], results2['total']
        val1 = f"{res1} ({res1/total1:.1%})"
        val2 = f"{res2} ({res2/total2:.1%})"
        print(f"| {name:<50} | {val1:<20} | {val2:<20} |")
        print(f"|  - {desc:<47} | {'':<20} | {'':<20} |")

    print_row("Idea 1: Simple Threshold", "Normalized distance >= 0.1", results1['idea1'], results2['idea1'])
    print(separator)
    print_row("Idea 2: Spike Detection", "Max norm. dist in 1s >= 1.5", results1['idea2'], results2['idea2'])
    print(separator)
    print_row("Idea 3: Cumulative Movement", "Sum of norm. dist in 3s > 5.0", results1['idea3'], results2['idea3'])
    print(separator)
    print_row("Idea 4: Stability Check", "Std Dev of scale in 2s > 50px", results1['idea4'], results2['idea4'])
    print(separator)
    print_row("Idea 5: Composite (2 or 4)", "Spike detection OR Stability check is true", results1['idea5'], results2['idea5'])
    print(separator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate and compare 5 movement detection ideas.")
    parser.add_argument("file1", type=str, help="Path to the first CSV file.")
    parser.add_argument("file2", type=str, help="Path to the second CSV file.")
    args = parser.parse_args()
    
    main(args.file1, args.file2)
