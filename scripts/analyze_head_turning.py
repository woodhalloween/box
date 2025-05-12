import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ログファイルのパス
skeleton_file = "logs/skeleton_data_20250413_161655.csv"
events_file = "logs/stick_figure_events_20250413_161655.csv"

# データの読み込み
skeleton_data = pd.read_csv(skeleton_file)
events_data = pd.read_csv(events_file)

print(f"骨格データ: {len(skeleton_data)}行")
print(f"首振りイベント: {len(events_data)}行")


# タイムスタンプをフレームインデックスに変換
def find_closest_frames(events, skeleton):
    """イベントのタイムスタンプに最も近いフレームを特定"""
    event_frames = []
    for i, row in events.iterrows():
        start_time = row["start_time"]
        end_time = row["end_time"]

        # 開始時間に最も近いフレームを特定
        start_diff = abs(skeleton["timestamp"] - start_time)
        start_idx = int(start_diff.idxmin())  # 整数型に変換

        # 終了時間に最も近いフレームを特定
        end_diff = abs(skeleton["timestamp"] - end_time)
        end_idx = int(end_diff.idxmin())  # 整数型に変換

        event_frames.append(
            {
                "event_id": i,
                "start_frame": int(row["start_frame"]),  # 整数型に変換
                "end_frame": int(row["end_frame"]),  # 整数型に変換
                "start_idx": start_idx,
                "end_idx": end_idx,
                "duration": row["duration"],
            }
        )

    return pd.DataFrame(event_frames)


# 首振りイベントとフレームの対応を取得
event_frames = find_closest_frames(events_data, skeleton_data)
print("イベントとフレームの対応:")
print(event_frames.head())

# 頭部の動きを分析するためのランドマーク
# MediaPipeのランドマーク番号
# 0: 鼻
# 7: 左耳
# 8: 右耳
head_landmarks = [0, 7, 8]


# 頭部の左右の動き（X軸の変化）を分析
def analyze_head_movement(event_id, window_size=5):
    """特定のイベント周辺の頭部の動きを分析"""
    event = event_frames.loc[event_id]
    start_idx = max(0, event["start_idx"] - window_size)
    end_idx = min(len(skeleton_data) - 1, event["end_idx"] + window_size)

    # イベント範囲の骨格データを取得
    event_skeleton = skeleton_data.iloc[start_idx : end_idx + 1].copy()

    # 頭部のX座標の変化を計算
    nose_x = [f"landmark_{lm}_x" for lm in head_landmarks]

    # ランドマークのX座標を時系列で表示
    plt.figure(figsize=(12, 6))

    for lm in head_landmarks:
        x_col = f"landmark_{lm}_x"
        plt.plot(event_skeleton["frame_idx"], event_skeleton[x_col], label=f"Landmark {lm} X")

    # イベント範囲を表示
    plt.axvline(x=event["start_frame"], color="r", linestyle="--", label="イベント開始")
    plt.axvline(x=event["end_frame"], color="g", linestyle="--", label="イベント終了")

    plt.title(f"イベント {event_id + 1} の頭部X座標の変化")
    plt.xlabel("フレームインデックス")
    plt.ylabel("X座標（正規化）")
    plt.legend()

    # 保存先ディレクトリがなければ作成
    os.makedirs("analysis", exist_ok=True)
    plt.savefig(f"analysis/event_{event_id + 1}_head_movement.png")
    plt.close()

    # X座標の微分（フレーム間の変化量）を計算
    for lm in head_landmarks:
        x_col = f"landmark_{lm}_x"
        diff_col = f"landmark_{lm}_x_diff"
        event_skeleton[diff_col] = event_skeleton[x_col].diff()

    # 変化量の絶対値が大きい部分を特定
    threshold = 0.01  # 変化量の閾値
    for lm in head_landmarks:
        diff_col = f"landmark_{lm}_x_diff"
        event_skeleton[f"{diff_col}_significant"] = abs(event_skeleton[diff_col]) > threshold

    # 頭部の左右の動きの特徴を抽出
    # 変化量の符号の変化回数をカウント
    direction_changes = 0
    prev_sign = None

    # 鼻のX座標の変化を使用
    nose_diff = event_skeleton["landmark_0_x_diff"].dropna()

    for diff in nose_diff:
        if abs(diff) > threshold:
            current_sign = np.sign(diff)
            if prev_sign is not None and current_sign != prev_sign:
                direction_changes += 1
            prev_sign = current_sign

    # 結果の表示
    print(f"\nイベント {event_id + 1} の分析結果:")
    print(f"フレーム範囲: {event['start_frame']} - {event['end_frame']}")
    print(f"持続時間: {event['duration']:.2f}秒")
    print(f"方向変化回数: {direction_changes}")

    # X座標の変化量をプロット
    plt.figure(figsize=(12, 6))

    for lm in head_landmarks:
        diff_col = f"landmark_{lm}_x_diff"
        plt.plot(
            event_skeleton["frame_idx"].iloc[1:],
            event_skeleton[diff_col].iloc[1:],
            label=f"Landmark {lm} X diff",
        )

    plt.axvline(x=event["start_frame"], color="r", linestyle="--", label="イベント開始")
    plt.axvline(x=event["end_frame"], color="g", linestyle="--", label="イベント終了")
    plt.axhline(y=threshold, color="k", linestyle=":", label="閾値+")
    plt.axhline(y=-threshold, color="k", linestyle=":", label="閾値-")

    plt.title(f"イベント {event_id + 1} の頭部X座標の変化量")
    plt.xlabel("フレームインデックス")
    plt.ylabel("X座標の変化量（フレーム間）")
    plt.legend()

    plt.savefig(f"analysis/event_{event_id + 1}_head_movement_diff.png")
    plt.close()

    return {
        "event_id": event_id,
        "start_frame": event["start_frame"],
        "end_frame": event["end_frame"],
        "duration": event["duration"],
        "direction_changes": direction_changes,
    }


# 最初の5つのイベントを分析
results = []
for i in range(min(5, len(event_frames))):
    result = analyze_head_movement(i)
    results.append(result)

# 分析結果のサマリー
results_df = pd.DataFrame(results)
print("\n首振りイベントの分析サマリー:")
print(results_df)

# 首振り検出のためのアルゴリズム案
print("\n首振り検出のためのアルゴリズム案:")
print("1. 頭部のランドマーク（特に鼻、左耳、右耳）のX座標の変化を監視")
print("2. X座標の変化量（微分）を計算し、閾値以上の変化を有意な動きとみなす")
print("3. 有意な動きの中で、方向の変化（符号の変化）をカウント")
print("4. 短時間（0.2秒〜0.3秒程度）の間に複数回の方向変化がある場合を首振りと判定")
print("5. 方向変化の数と動きの大きさに基づいて、首振りの強度を評価")
