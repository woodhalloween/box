import pandas as pd
import numpy as np

# CSVファイルを読み込む
df = pd.read_csv('log_WIN_20250319_10_03_53_Pro_bytetrack_yolov8n_20250326_084243.csv', 
                 comment='#')  # '#'で始まる行はヘッダーとしてスキップ

# 基本的な統計情報を計算
stats = {
    '検出時間 (ms)': {
        '平均': df['Detection_Time_ms'].mean(),
        '標準偏差': df['Detection_Time_ms'].std(),
        '最小': df['Detection_Time_ms'].min(),
        '最大': df['Detection_Time_ms'].max(),
    },
    'トラッキング時間 (ms)': {
        '平均': df['Tracking_Time_ms'].mean(),
        '標準偏差': df['Tracking_Time_ms'].std(),
        '最小': df['Tracking_Time_ms'].min(),
        '最大': df['Tracking_Time_ms'].max(),
    },
    '合計処理時間 (ms)': {
        '平均': df['Total_Time_ms'].mean(),
        '標準偏差': df['Total_Time_ms'].std(),
        '最小': df['Total_Time_ms'].min(),
        '最大': df['Total_Time_ms'].max(),
    },
    'FPS': {
        '平均': df['FPS'].mean(),
        '標準偏差': df['FPS'].std(),
        '最小': df['FPS'].min(),
        '最大': df['FPS'].max(),
    }
}

# 検出されたオブジェクトの統計
total_frames = len(df)
frames_with_objects = len(df[df['Objects_Detected'] > 0])
detection_rate = frames_with_objects / total_frames * 100

print("=== 処理性能の統計情報 ===")
for metric, values in stats.items():
    print(f"\n{metric}:")
    for stat, value in values.items():
        print(f"{stat}: {value:.2f}")

print(f"\n=== 検出率 ===")
print(f"総フレーム数: {total_frames}")
print(f"オブジェクト検出フレーム数: {frames_with_objects}")
print(f"検出率: {detection_rate:.2f}%") 