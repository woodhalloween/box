import argparse
import csv
from pathlib import Path

import cv2
from tqdm import tqdm

from src.config import AppConfig
from src.detectors.hand_raise_detector import HandRaiseDetector
from src.drawing_utils import draw_hand_raise_status, draw_landmarks
from src.pose_estimator import PoseEstimator


def main():
    """手挙げ検出のメイン処理を実行する。"""
    parser = argparse.ArgumentParser(description="手挙げ検出アルゴリズムを実行するスクリプト")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="処理対象の動画ファイルパス",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default=None,
        help="結果を描画した動画の出力先パス。省略時は入力動画名に `_processed` を付与。",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="検出結果のCSVファイルの出力先パス。省略時は入力動画名に `_results.csv` を付与。",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="処理中の映像をウィンドウ表示しない",
    )
    parser.add_argument(
        "--no_video_output",
        action="store_true",
        help="処理済み動画の書き出しを行わない",
    )
    parser.add_argument(
        "--no_csv_output",
        action="store_true",
        help="検出結果CSVの書き出しを行わない",
    )
    parser.add_argument(
        "--draw_skeleton",
        action="store_true",
        help="骨格（ランドマーク）描画を有効化する",
    )
    args = parser.parse_args()

    display_window = not args.no_display
    write_video = not args.no_video_output
    write_csv = not args.no_csv_output
    draw_skeleton = args.draw_skeleton

    if not any([display_window, write_video, write_csv]):
        print("警告: 表示も出力も全て無効化されています。最低1つは有効にします。")
        display_window = True

    # --- 1. 初期化フェーズ ---
    # 設定ファイルの読み込み
    try:
        config = AppConfig("config.yaml")
        visibility_threshold = config.getfloat("hand_raise.visibility_threshold", 0.5)
        min_consecutive_frames = config.getint("hand_raise.min_consecutive_frames", 5)
    except FileNotFoundError:
        print("警告: config.yamlが見つかりません。デフォルト値を使用します。")
        visibility_threshold = 0.5
        min_consecutive_frames = 5

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"エラー: 動画ファイルが見つかりません: {input_path}")
        return

    # 出力パスの決定
    output_video_path = None
    output_csv_path = None
    if write_video:
        output_video_path = args.output_video or str(input_path.parent / f"{input_path.stem}_processed.mp4")
    if write_csv:
        output_csv_path = args.output_csv or str(input_path.parent / f"{input_path.stem}_results.csv")

    # モジュールのインスタンス化
    pose_estimator = PoseEstimator()
    hand_raise_detector = HandRaiseDetector(
        visibility_threshold=visibility_threshold,
        min_consecutive_frames=min_consecutive_frames,
    )

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"エラー: 動画ファイルが開けません: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_writer = None
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not video_writer.isOpened():
            print(f"警告: 動画を書き出せませんでした: {output_video_path}")
            video_writer.release()
            video_writer = None
            write_video = False

    csv_file = None
    csv_writer = None
    if write_csv:
        try:
            csv_file = open(output_csv_path, "w", newline="", encoding="utf-8")
        except OSError as exc:
            print(f"警告: CSVを書き出せませんでした ({exc}). CSV出力を無効化します。")
            write_csv = False
        else:
            csv_fieldnames = [
                "frame_number",
                "timestamp",
                "left_hand_raised",
                "right_hand_raised",
            ]
            csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
            csv_writer.writeheader()

    frame_number = 0
    user_aborted = False
    progress_total = total_frames if total_frames > 0 else None

    try:
        with tqdm(total=progress_total, desc="Processing video", unit="frame") as progress_bar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_number / fps if fps else frame_number

                # 姿勢推定
                landmarks = pose_estimator.estimate(frame)

                # 手挙げ状態判定
                hand_statuses = hand_raise_detector.detect(landmarks)

                if write_csv and csv_writer is not None:
                    csv_writer.writerow(
                        {
                            "frame_number": frame_number,
                            "timestamp": f"{timestamp:.3f}",
                            "left_hand_raised": hand_statuses["left_hand_raised"],
                            "right_hand_raised": hand_statuses["right_hand_raised"],
                        }
                    )

                processed_frame = frame.copy()
                if draw_skeleton and landmarks is not None:
                    draw_landmarks(processed_frame, landmarks)
                processed_frame = draw_hand_raise_status(processed_frame, hand_statuses)

                if write_video and video_writer is not None:
                    video_writer.write(processed_frame)

                frame_number += 1
                progress_bar.update(1)

                if display_window:
                    cv2.imshow("Hand Raise Detection", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        user_aborted = True
                        break
    except KeyboardInterrupt:
        user_aborted = True
        print("ユーザー操作により処理を中断しました。")
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if display_window:
            cv2.destroyAllWindows()
        if csv_file is not None:
            csv_file.close()
        pose_estimator.close()

    if user_aborted:
        print("処理を途中で終了しました。")
    else:
        print("処理が完了しました。")

    if write_video and output_video_path:
        print(f"  - 処理済み動画: {output_video_path}")
    if write_csv and output_csv_path:
        print(f"  - 検出結果CSV: {output_csv_path}")


if __name__ == "__main__":
    main()
