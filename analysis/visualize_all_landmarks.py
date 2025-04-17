"""visualize_all_landmarks.py
MediaPipe Poseのすべてのランドマーク(x,y)を時系列でプロットし、visibilityフィルタを適用した点を強調表示します。
Usage:
    python analysis/visualize_all_landmarks.py <skeleton_csv> --vis_th 0.5 --output_dir output/all_landmarks
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp

# Mediapipe PoseLandmarkの名前マッピング
_landmark_enum = mp.solutions.pose.PoseLandmark
_landmark_names = {l.value: l.name for l in _landmark_enum}


def save_plot(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_all_landmarks(df: pd.DataFrame, vis_th: float, output_dir: Path):
    # ランドマークIDを抽出
    ids = sorted({int(col.split('_')[1]) for col in df.columns if col.startswith('landmark_') and col.endswith('_x')})
    for idx in ids:
        x = df[f"landmark_{idx}_x"]
        y = df[f"landmark_{idx}_y"]
        vis = df[f"landmark_{idx}_visibility"] >= vis_th

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        # X座標プロット
        axs[0].plot(x, label='x (all)', color='blue')
        axs[0].plot(df.index[vis], x[vis], '.', label=f'x (vis>={vis_th})', color='orange')
        axs[0].set_ylabel('Normalized x')
        axs[0].legend(loc='upper right')
        # Y座標プロット
        axs[1].plot(y, label='y (all)', color='blue')
        axs[1].plot(df.index[vis], y[vis], '.', label=f'y (vis>={vis_th})', color='orange')
        axs[1].set_ylabel('Normalized y')
        axs[1].set_xlabel('Frame')
        axs[1].legend(loc='upper right')

        lm_name = _landmark_names.get(idx, f'ID{idx}')
        fig.suptitle(f'Landmark {idx} - {lm_name}')

        out_file = output_dir / f"landmark_{idx}_{lm_name}.png"
        save_plot(fig, out_file)


def main():
    parser = argparse.ArgumentParser(description='Visualize all Mediapipe Pose landmarks')
    parser.add_argument('csv_path', help='Path to skeleton CSV log')
    parser.add_argument('--vis_th', type=float, default=0.5, help='Visibility threshold')
    parser.add_argument('--output_dir', default='output/all_landmarks', help='Directory to save plots')
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    output_dir = Path(args.output_dir)

    plot_all_landmarks(df, args.vis_th, output_dir)


if __name__ == '__main__':
    main() 