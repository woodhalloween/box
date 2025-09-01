import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd


def analyze_and_find_segments(csv_path):
    """
    CSVを分析し、「アイデア5」に基づいて移動区間と滞在区間の
    フレーム範囲を特定する。
    """
    try:
        df = pd.read_csv(csv_path)
        if "person_scale" not in df.columns:
            print(f"エラー: 必要な列 'person_scale' が見つかりません in {csv_path}", file=sys.stderr)
            return None, None

        # --- アイデア5のロジックを再計算 ---
        # 準備
        threshold = 1e-6
        df.loc[df["person_scale"].abs() <= threshold, "person_scale"] = np.nan
        df["person_scale"] = df["person_scale"].ffill().bfill()

        df["prev_x"] = df["hip_center_x"].shift(1)
        df["prev_y"] = df["hip_center_y"].shift(1)
        df["distance_px"] = np.sqrt((df["hip_center_x"] - df["prev_x"]) ** 2 + (df["hip_center_y"] - df["prev_y"]) ** 2)
        df["distance_normalized"] = df["distance_px"] / df["person_scale"]
        df.fillna(0, inplace=True)

        # アイデア2：スパイク検知
        is_spike = df["distance_normalized"].rolling(window=30, min_periods=1).max() >= 1.5
        # アイデア4：安定性
        is_unstable = df["person_scale"].rolling(window=60, min_periods=1).std().fillna(0) > 50.0
        # アイデア5：複合条件
        df["is_movement"] = is_spike | is_unstable

        # --- 連続区間のフレーム範囲を抽出 ---
        move_segments = []
        stay_segments = []

        # 状態が変わるインデックスを見つける
        df["change"] = df["is_movement"].diff()
        change_indices = df[df["change"] != 0].index

        start_idx = 0
        for end_idx in change_indices:
            # end_idxが0の場合は最初のセグメントがないためスキップ
            if end_idx == 0:
                start_idx = end_idx
                continue

            segment_state = df["is_movement"][start_idx]
            frame_start = int(df["frame_number"][start_idx])
            frame_end = int(df["frame_number"][end_idx - 1])

            if frame_start < frame_end:
                if segment_state:
                    move_segments.append((frame_start, frame_end))
                else:
                    stay_segments.append((frame_start, frame_end))
            start_idx = end_idx

        # 最後のセグメントを追加
        last_state = df["is_movement"][start_idx]
        frame_start = int(df["frame_number"][start_idx])
        frame_end = int(df["frame_number"].iloc[-1])
        if frame_start < frame_end:
            if last_state:
                move_segments.append((frame_start, frame_end))
            else:
                stay_segments.append((frame_start, frame_end))

        return move_segments, stay_segments

    except Exception as e:
        print(f"セグメント分析中にエラー: {e}", file=sys.stderr)
        return None, None


def create_video_from_segments(video_path, output_path, segments):
    """
    フレーム範囲のリストから1本の動画を作成する。
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"エラー: 動画ファイルを開けません: {video_path}", file=sys.stderr)
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"'{os.path.basename(output_path)}' を作成中 ({len(segments)}区間)...")

        for start_frame, end_frame in segments:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame = start_frame
            while current_frame <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                current_frame += 1

        cap.release()
        out.release()
        print(f" -> 保存完了: {output_path}")

    except Exception as e:
        print(f"動画作成中にエラー: {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV分析に基づき、動画を移動区間と滞在区間に分割します。")
    parser.add_argument("csv_path", help="分析用のCSVファイルパス")
    parser.add_argument("video_path", help="元の動画ファイルパス")
    parser.add_argument("move_output", help="移動区間を結合した動画の出力パス")
    parser.add_argument("stay_output", help="滞在区間を結合した動画の出力パス")
    args = parser.parse_args()

    move_segments, stay_segments = analyze_and_find_segments(args.csv_path)

    if move_segments is not None and stay_segments is not None:
        if move_segments:
            create_video_from_segments(args.video_path, args.move_output, move_segments)
        else:
            print("移動区間は見つかりませんでした。")

        if stay_segments:
            create_video_from_segments(args.video_path, args.stay_output, stay_segments)
        else:
            print("滞在区間は見つかりませんでした。")
