"""visualize_landmarks.py
骨格ログ CSV から以下の可視化 PNG を自動生成します。
1. 各ランドマークの速度（例：鼻 dx/dt, dy/dt）
2. ランドマーク座標の時系列（visibility フィルタ付き）
3. キーポイント差分（耳 x 差、鼻と肩中心 y 差）

Usage:
    python analysis/visualize_landmarks.py logs/skeleton_data_20250330_215959.csv \
        --vis_th 0.5 --output_prefix output/vis_result
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd

# MediaPipe Pose のランドマーク ID (主要部位のみ)
NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def velocity(series: pd.Series, timestamps: pd.Series) -> pd.Series:
    """差分から速度を計算 (単位: 座標/秒)"""
    dt = timestamps.diff()
    return series.diff() / dt


def save_plot(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")


def plot_velocity(df: pd.DataFrame, output_prefix: Path):
    ts = df["timestamp"]
    dx = velocity(df[f"landmark_{NOSE}_x"], ts)
    dy = velocity(df[f"landmark_{NOSE}_y"], ts)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dx, label="dx/dt")
    ax.plot(dy, label="dy/dt")
    ax.set_title("Nose velocity")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Velocity (norm/秒)")
    ax.legend()

    save_plot(fig, output_prefix.parent / f"{output_prefix.name}_nose_velocity.png")
    plt.close(fig)


def plot_coordinates(df: pd.DataFrame, vis_th: float, output_prefix: Path):
    nose_x = df[f"landmark_{NOSE}_x"]
    nose_y = df[f"landmark_{NOSE}_y"]
    vis = df[f"landmark_{NOSE}_visibility"] >= vis_th

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(nose_x, label="nose_x (all)")
    ax.plot(nose_y, label="nose_y (all)")
    ax.plot(df.index[vis], nose_x[vis], "o", markersize=2, label=f"nose_x (vis>={vis_th})")
    ax.set_title("Nose coordinates with visibility filter")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Normalized coordinate")
    ax.legend()

    save_plot(fig, output_prefix.parent / f"{output_prefix.name}_nose_coords.png")
    plt.close(fig)


def plot_keypoint_diffs(df: pd.DataFrame, vis_th: float, output_prefix: Path):
    # 耳 X 差分
    left_x = df[f"landmark_{LEFT_EAR}_x"]
    right_x = df[f"landmark_{RIGHT_EAR}_x"]
    ear_diff = right_x - left_x
    mask_ear = (df[f"landmark_{LEFT_EAR}_visibility"] >= vis_th) & (
        df[f"landmark_{RIGHT_EAR}_visibility"] >= vis_th
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ear_diff, label="Ear X diff (all)")
    ax.plot(df.index[mask_ear], ear_diff[mask_ear], "o", markersize=2, label="filtered")
    ax.set_title("Difference in X between ears")
    ax.set_xlabel("Frame")
    ax.set_ylabel("ΔX (norm)")
    ax.legend()

    save_plot(fig, output_prefix.parent / f"{output_prefix.name}_ear_x_diff.png")
    plt.close(fig)

    # 鼻と肩中心 Y 差分
    nose_y = df[f"landmark_{NOSE}_y"]
    lsy = df[f"landmark_{LEFT_SHOULDER}_y"]
    rsy = df[f"landmark_{RIGHT_SHOULDER}_y"]
    shoulder_center_y = (lsy + rsy) / 2.0
    diff_y = nose_y - shoulder_center_y
    mask_shoulder = (
        (df[f"landmark_{LEFT_SHOULDER}_visibility"] >= vis_th)
        & (df[f"landmark_{RIGHT_SHOULDER}_visibility"] >= vis_th)
        & (df[f"landmark_{NOSE}_visibility"] >= vis_th)
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(diff_y, label="Nose - ShoulderCenter Y (all)")
    ax.plot(df.index[mask_shoulder], diff_y[mask_shoulder], "o", markersize=2, label="filtered")
    ax.set_title("Vertical diff: Nose vs. Shoulder Center")
    ax.set_xlabel("Frame")
    ax.set_ylabel("ΔY (norm)")
    ax.legend()

    save_plot(fig, output_prefix.parent / f"{output_prefix.name}_nose_shoulder_y_diff.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Skeleton landmark visualization")
    parser.add_argument("csv_path", help="Path to skeleton CSV log")
    parser.add_argument("--vis_th", type=float, default=0.5, help="Visibility threshold (0-1)")
    parser.add_argument(
        "--output_prefix", default="visualization", help="Prefix path for output PNGs"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    output_prefix = Path(args.output_prefix)

    plot_velocity(df, output_prefix)
    plot_coordinates(df, args.vis_th, output_prefix)
    plot_keypoint_diffs(df, args.vis_th, output_prefix)


if __name__ == "__main__":
    main()
