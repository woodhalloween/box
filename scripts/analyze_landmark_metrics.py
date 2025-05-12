#!/usr/bin/env python3
# coding: utf-8
"""
CSVを読み込み、各ランドマークのフレーム間移動距離を計算し、統計値および距離推移グラフを出力するスクリプト。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    input_path = "output/landmark_metrics.csv"
    df = pd.read_csv(input_path, header=None)
    n_cols = df.shape[1]
    n_landmarks = (n_cols - 2) // 3

    col_names = ["frame_index", "timestamp"]
    for i in range(n_landmarks):
        col_names.extend([f"lmk{i}_x", f"lmk{i}_y", f"lmk{i}_z"])
    df.columns = col_names

    # 座標列を数値型に変換（文字列をfloatに）
    coord_cols = col_names[2:]
    df[coord_cols] = df[coord_cols].apply(pd.to_numeric, errors="coerce")

    # 各ランドマークのフレーム間移動距離を計算
    for i in range(n_landmarks):
        coords = df[[f"lmk{i}_x", f"lmk{i}_y", f"lmk{i}_z"]]
        df[f"lmk{i}_dist"] = np.linalg.norm(coords.diff(), axis=1)

    # 統計値を集計
    stats = {
        "mean": df[[f"lmk{i}_dist" for i in range(n_landmarks)]].mean().values,
        "max": df[[f"lmk{i}_dist" for i in range(n_landmarks)]].max().values,
        "sum": df[[f"lmk{i}_dist" for i in range(n_landmarks)]].sum().values,
    }
    summary = pd.DataFrame(stats, index=[f"lm{i}" for i in range(n_landmarks)])
    os.makedirs("analysis", exist_ok=True)
    summary.to_csv("analysis/summary_stats.csv", encoding="utf-8-sig")

    # 距離推移をプロット
    plt.figure(figsize=(12, 6))
    for i in range(n_landmarks):
        plt.plot(df["timestamp"], df[f"lmk{i}_dist"], label=f"lm{i}")
    plt.xlabel("Time (s)")
    plt.ylabel("Movement Distance")
    plt.title("Landmark Movement Over Time")
    plt.legend(ncol=4, fontsize="small")
    plt.tight_layout()
    plt.savefig("analysis/landmark_distances.png")
    plt.close()

    print(
        "分析完了: analysis/summary_stats.csv と analysis/landmark_distances.png を出力しました。"
    )


if __name__ == "__main__":
    main()
