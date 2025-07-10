"""src/detect_joint_movement.py

This script is the main entry point for the joint movement detection application.
It captures video from a file, processes each frame to detect pose landmarks,
analyzes the landmarks to calculate joint angles, and displays the results in
real-time.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import IO, Any

import cv2

from src.definitions import Angle
from src.drawing_utils import draw_analysis_results, draw_landmarks
from src.movement_analyzer import MovementAnalyzer
from src.pose_estimator import PoseEstimator


def open_output_file(path: str) -> IO[Any]:
    """出力用のCSVファイルを開き、ヘッダーを書き込む"""
    try:
        output_file = open(path, "w", newline="")
        # ヘッダーを準備
        header = ["timestamp"]
        all_angles = list(Angle)  # Enumの全メンバーを取得
        for angle in all_angles:
            header.extend([f"{angle.value}_ANGLE", f"{angle.value}_STATE"])

        csv_writer = csv.writer(output_file)
        csv_writer.writerow(header)
        return output_file
    except OSError as e:
        sys.exit(f"Error: Cannot open output file {path} - {e}")


def write_results_to_csv(writer: Any, timestamp: float, frame_number: int, results: dict) -> None:
    """分析結果をCSVファイルに書き込む"""
    row = [timestamp, frame_number]
    # ANGLE_DEFINITIONSの順序に基づいて結果を並び替える
    all_angles = [angle for angle in Angle]
    for joint_name in MovementAnalyzer.ANGLE_DEFINITIONS:
        angle_data = results.get(joint_name)
        if angle_data:
            row.append(f"{angle_data['angle']:.2f}")
            row.append(angle_data["state"].value)
        else:
            # データがない場合は空欄を追加
            row.extend(["", ""])
    writer.writerow(row)


def setup_csv_writer(csv_file: IO[Any]) -> csv.writer:
    """CSVファイルのヘッダーを書き込む"""
    header = ["timestamp"]
    all_angles = list(Angle)  # Enumの全メンバーを取得
    for angle in all_angles:
        header.extend([f"{angle.value}_ANGLE", f"{angle.value}_STATE"])
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)
    return csv_writer


def setup_video_writer(cap: cv2.VideoCapture, output_video_path: str) -> cv2.VideoWriter:
    """ビデオライターを初期化し、ビデオを書き込む"""
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    return video_writer


def process_video(video_path: str, output_csv_path: str | None, output_video_path: str | None):
    """
    ビデオを処理し、関節の動きを分析して結果を出力する。
    """
    # 入力ビデオを開く
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # 出力ファイルパスが指定されていない場合、デフォルトパスを生成
    p = Path(video_path)
    if output_csv_path is None:
        output_csv_path = f"output/{p.stem}_analysis.csv"
    if output_video_path is None:
        output_video_path = f"output/{p.stem}_output.mp4"

    # 出力ディレクトリを作成
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)

    # CSVライターとビデオライターをセットアップ
    csv_file = open(output_csv_path, "w", newline="")
    csv_writer = setup_csv_writer(csv_file)
    video_writer = setup_video_writer(cap, output_video_path)

    pose_estimator = PoseEstimator()
    analyzer = MovementAnalyzer()

    # --- パフォーマンス計測用の変数を初期化 ---
    frame_count = 0
    total_time_spent = 0.0
    time_reading = 0.0
    time_posing = 0.0
    time_analyzing = 0.0
    time_drawing = 0.0
    time_writing = 0.0
    time_showing = 0.0
    # ------------------------------------

    while cap.isOpened():
        loop_start_time = time.perf_counter()

        start_time = time.perf_counter()
        success, frame = cap.read()
        if not success:
            break
        time_reading += time.perf_counter() - start_time

        start_time = time.perf_counter()
        landmarks = pose_estimator.estimate(frame)
        time_posing += time.perf_counter() - start_time

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        analysis_results = {}
        if landmarks is not None:
            # 2. Movement Analysis
            start_time = time.perf_counter()
            analysis_results = analyzer.analyze(landmarks)
            time_analyzing += time.perf_counter() - start_time

            # 3. Write to CSV
            start_time = time.perf_counter()
            write_results_to_csv(csv_writer, timestamp, frame_count, analysis_results)
            time_writing += time.perf_counter() - start_time

        # 4. Drawing
        start_time = time.perf_counter()
        if landmarks is not None:
            # draw_analysis_resultsが描画済みの画像を返すように変更されたため、
            # 戻り値で変数を更新する
            frame = draw_analysis_results(frame, analysis_results, landmarks)
            draw_landmarks(frame, landmarks)
        time_drawing += time.perf_counter() - start_time

        # 描画されたフレームをビデオに書き込む
        start_time = time.perf_counter()
        if video_writer:
            video_writer.write(frame)
        time_writing += time.perf_counter() - start_time

        # 5. Display
        cv2.imshow("Movement Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

        total_time_spent += time.perf_counter() - loop_start_time

    # クリーンアップ
    cap.release()
    video_writer.release()
    csv_file.close()
    cv2.destroyAllWindows()

    # --- パフォーマンス分析レポートを出力 ---
    if frame_count > 0 and total_time_spent > 0:
        print("\n--- Performance Analysis Report ---")
        print(f"Total frames processed: {frame_count}")
        print(f"Total processing time: {total_time_spent:.2f} seconds")
        print(f"Average FPS (including waitKey): {frame_count / total_time_spent:.2f}")
        print("-" * 33)

        # waitKeyを除いた、純粋な処理時間を計算
        total_tracked_time = time_reading + time_posing + time_analyzing + time_drawing + time_writing + time_showing
        if total_tracked_time == 0:
            total_tracked_time = 1  # ゼロ除算を避ける

        print(f"Bottleneck Analysis (based on {total_tracked_time:.2f}s of tracked processing time):")
        print(f"  - AI Pose Estimation:   {time_posing:7.2f}s ({time_posing / total_tracked_time * 100:5.1f}%)")
        print(
            f"  - OpenCV Operations:    {time_reading + time_drawing + time_writing + time_showing:7.2f}s ({(time_reading + time_drawing + time_writing + time_showing) / total_tracked_time * 100:5.1f}%)"
        )
        print(f"  - Other (Analyzing):    {time_analyzing:7.2f}s ({time_analyzing / total_tracked_time * 100:5.1f}%)")
        print("-" * 33)

        print("Detailed Breakdown:")
        print(f"  - AI Pose Estimation:       {time_posing:7.2f}s ({time_posing / total_tracked_time * 100:5.1f}%)")
        print(f"  - OpenCV: Video Reading     {time_reading:7.2f}s ({time_reading / total_tracked_time * 100:5.1f}%)")
        print(f"  - OpenCV: Drawing Results   {time_drawing:7.2f}s ({time_drawing / total_tracked_time * 100:5.1f}%)")
        print(f"  - OpenCV: Video Writing     {time_writing:7.2f}s ({time_writing / total_tracked_time * 100:5.1f}%)")
        print(f"  - OpenCV: Displaying        {time_showing:7.2f}s ({time_showing / total_tracked_time * 100:5.1f}%)")
        print(
            f"  - Python: Joint Analyzing   {time_analyzing:7.2f}s ({time_analyzing / total_tracked_time * 100:5.1f}%)"
        )

        untracked_time = total_time_spent - total_tracked_time
        print(f"\nUntracked time (mostly cv2.waitKey): {untracked_time:.2f}s")
        print("--- End of Report ---")


def main() -> None:
    """
    Main function to parse arguments and start the video processing.
    """
    parser = argparse.ArgumentParser(description="Analyze joint movements from a video.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output-csv", type=str, help="Path to the output CSV file to save results.")
    parser.add_argument("--output-video", type=str, help="Path to the output video file to save the processed video.")
    args = parser.parse_args()

    process_video(args.video, args.output_csv, args.output_video)


if __name__ == "__main__":
    main()
