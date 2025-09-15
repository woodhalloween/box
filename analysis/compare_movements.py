import argparse
import pandas as pd
import numpy as np
import sys

def get_movement_stats(csv_file: str):
    """Calculates and returns movement statistics from a CSV file."""
    try:
        df = pd.read_csv(csv_file)
        
        threshold = 1e-6
        df_filtered = df[(df["hip_center_x"].abs() > threshold) & (df["hip_center_y"].abs() > threshold)].copy()

        if len(df_filtered) <= 1:
            return None

        # Pixel distance
        df_filtered.loc[:, "prev_x"] = df_filtered["hip_center_x"].shift(1)
        df_filtered.loc[:, "prev_y"] = df_filtered["hip_center_y"].shift(1)
        df_filtered.dropna(subset=["prev_x", "prev_y"], inplace=True)
        
        if df_filtered.empty:
            return None

        df_filtered.loc[:, "distance_px"] = np.sqrt(
            (df_filtered["hip_center_x"] - df_filtered["prev_x"])**2 +
            (df_filtered["hip_center_y"] - df_filtered["prev_y"])**2
        )

        # Normalized distance
        valid_scale = df_filtered["person_scale"] > threshold
        df_filtered.loc[valid_scale, "distance_normalized"] = \
            df_filtered.loc[valid_scale, "distance_px"] / df_filtered.loc[valid_scale, "person_scale"]
        df_filtered.dropna(subset=["distance_normalized"], inplace=True)
        
        if df_filtered.empty:
            return None

        stats = {
            "px_mean": df_filtered['distance_px'].mean(),
            "px_median": df_filtered['distance_px'].median(),
            "px_max": df_filtered['distance_px'].max(),
            "px_std": df_filtered['distance_px'].std(),
            "norm_mean": df_filtered['distance_normalized'].mean(),
            "norm_median": df_filtered['distance_normalized'].median(),
            "norm_max": df_filtered['distance_normalized'].max(),
            "norm_std": df_filtered['distance_normalized'].std(),
            "scale_mean": df_filtered['person_scale'].mean(),
            "scale_median": df_filtered['person_scale'].median(),
            "scale_std": df_filtered['person_scale'].std(),
        }
        return stats

    except FileNotFoundError:
        print(f"Error: File not found - {csv_file}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An error occurred while processing {csv_file}: {e}", file=sys.stderr)
        return None

def main(file1: str, file2: str):
    """Compares movement analysis of two CSV files."""
    stats1 = get_movement_stats(file1)
    stats2 = get_movement_stats(file2)

    if not stats1 or not stats2:
        sys.exit(1)

    # --- Print Comparison Table ---
    header = f"| {'Metric':<28} | {'Video 1 (.._1.csv)':<25} | {'Video 2 (.._2.csv)':<25} |"
    separator = "-" * len(header)
    
    print(separator)
    print(header)
    print(separator)
    
    # --- Pixel Movement ---
    print(f"| {'Pixel Movement (px)':<55} |")
    print(f"| {'  Mean':<26} | {stats1['px_mean']:<25.2f} | {stats2['px_mean']:<25.2f} |")
    print(f"| {'  Median':<26} | {stats1['px_median']:<25.2f} | {stats2['px_median']:<25.2f} |")
    print(f"| {'  Max':<26} | {stats1['px_max']:<25.2f} | {stats2['px_max']:<25.2f} |")
    print(f"| {'  Std Dev':<26} | {stats1['px_std']:<25.2f} | {stats2['px_std']:<25.2f} |")
    print(separator)

    # --- Normalized Movement ---
    print(f"| {'Normalized Movement (ratio)':<55} |")
    print(f"| {'  Mean':<26} | {stats1['norm_mean']:<25.4f} | {stats2['norm_mean']:<25.4f} |")
    print(f"| {'  Median':<26} | {stats1['norm_median']:<25.4f} | {stats2['norm_median']:<25.4f} |")
    print(f"| {'  Max':<26} | {stats1['norm_max']:<25.4f} | {stats2['norm_max']:<25.4f} |")
    print(f"| {'  Std Dev':<26} | {stats1['norm_std']:<25.4f} | {stats2['norm_std']:<25.4f} |")
    print(separator)

    # --- Person Scale ---
    print(f"| {'Person Scale (torso, px)':<55} |")
    print(f"| {'  Mean':<26} | {stats1['scale_mean']:<25.2f} | {stats2['scale_mean']:<25.2f} |")
    print(f"| {'  Median':<26} | {stats1['scale_median']:<25.2f} | {stats2['scale_median']:<25.2f} |")
    print(f"| {'  Std Dev':<26} | {stats1['scale_std']:<25.2f} | {stats2['scale_std']:<25.2f} |")
    print(separator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare movement analysis from two CSV files.")
    parser.add_argument("file1", type=str, help="Path to the first CSV file.")
    parser.add_argument("file2", type=str, help="Path to the second CSV file.")
    args = parser.parse_args()
    
    main(args.file1, args.file2)
