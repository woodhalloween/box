import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ログファイルのパス
skeleton_file = 'logs/skeleton_data_20250413_161655.csv'
events_file = 'logs/stick_figure_events_20250413_161655.csv'

# データの読み込み
skeleton_data = pd.read_csv(skeleton_file)
events_data = pd.read_csv(events_file)

print(f"骨格データ: {len(skeleton_data)}行")
print(f"首振りイベント: {len(events_data)}行")

# 頭部の動きを分析するためのランドマーク
# MediaPipeのランドマーク番号
# 0: 鼻
# 7: 左耳
# 8: 右耳
head_landmarks = [0, 7, 8]

# 各イベントの前後の骨格データを抽出して分析
for i, event in events_data.iloc[:5].iterrows():
    start_frame = int(event['start_frame'])
    end_frame = int(event['end_frame'])
    duration = event['duration']
    
    print(f"\nイベント {i+1} の分析:")
    print(f"フレーム範囲: {start_frame} - {end_frame}")
    print(f"持続時間: {duration:.2f}秒")
    
    # フレーム番号に基づいてデータをフィルタリング
    # イベントの前後5フレームを含む
    window_size = 5
    min_frame = max(0, start_frame - window_size)
    max_frame = end_frame + window_size
    
    event_data = skeleton_data[
        (skeleton_data['frame_idx'] >= min_frame) & 
        (skeleton_data['frame_idx'] <= max_frame)
    ].copy()
    
    if len(event_data) == 0:
        print(f"イベント {i+1} のデータが見つかりません")
        continue
    
    # 鼻のX座標の変化（微分）を計算
    for lm in head_landmarks:
        x_col = f'landmark_{lm}_x'
        diff_col = f'landmark_{lm}_x_diff'
        event_data[diff_col] = event_data[x_col].diff()
    
    # 変化量の閾値
    threshold = 0.01
    
    # 方向の変化をカウント（符号の変化）
    direction_changes = 0
    prev_sign = None
    
    # イベント期間内のみの変化を分析
    event_period = event_data[
        (event_data['frame_idx'] >= start_frame) & 
        (event_data['frame_idx'] <= end_frame)
    ]
    
    for diff in event_period['landmark_0_x_diff'].dropna():
        if abs(diff) > threshold:
            current_sign = np.sign(diff)
            if prev_sign is not None and current_sign != prev_sign:
                direction_changes += 1
            prev_sign = current_sign
    
    print(f"方向変化回数: {direction_changes}")
    
    # 保存先ディレクトリがなければ作成
    os.makedirs('analysis', exist_ok=True)
    
    # ランドマークのX座標をプロット
    plt.figure(figsize=(12, 6))
    
    for lm in head_landmarks:
        x_col = f'landmark_{lm}_x'
        plt.plot(event_data['frame_idx'], event_data[x_col], 
                label=f'Landmark {lm} X')
    
    # イベント範囲を表示
    plt.axvline(x=start_frame, color='r', linestyle='--', label='イベント開始')
    plt.axvline(x=end_frame, color='g', linestyle='--', label='イベント終了')
    
    plt.title(f'イベント {i+1} の頭部X座標の変化')
    plt.xlabel('フレームインデックス')
    plt.ylabel('X座標（正規化）')
    plt.legend()
    
    plt.savefig(f'analysis/event_{i+1}_head_movement.png')
    plt.close()
    
    # X座標の変化量をプロット
    plt.figure(figsize=(12, 6))
    
    for lm in head_landmarks:
        diff_col = f'landmark_{lm}_x_diff'
        plt.plot(event_data['frame_idx'].iloc[1:], event_data[diff_col].iloc[1:], 
                label=f'Landmark {lm} X diff')
    
    plt.axvline(x=start_frame, color='r', linestyle='--', label='イベント開始')
    plt.axvline(x=end_frame, color='g', linestyle='--', label='イベント終了')
    plt.axhline(y=threshold, color='k', linestyle=':', label='閾値+')
    plt.axhline(y=-threshold, color='k', linestyle=':', label='閾値-')
    
    plt.title(f'イベント {i+1} の頭部X座標の変化量')
    plt.xlabel('フレームインデックス')
    plt.ylabel('X座標の変化量（フレーム間）')
    plt.legend()
    
    plt.savefig(f'analysis/event_{i+1}_head_movement_diff.png')
    plt.close()

# 首振り検出のための改良アルゴリズム提案
print("\n首振り検出のためのアルゴリズム案:")
print("1. 頭部のランドマーク（特に鼻、左耳、右耳）のX座標の変化を監視")
print("2. X座標の変化量（微分）を計算し、閾値（例: 0.01）以上の変化を有意な動きとみなす")
print("3. 有意な動きの中で、方向の変化（符号の変化）をカウント")
print("4. 短時間（0.2秒〜0.3秒程度）の間に2回以上の方向変化がある場合を首振りと判定")
print("5. 連続した首振りを検出するために、一定時間（例: 1秒）経過後に再度検出を開始") 