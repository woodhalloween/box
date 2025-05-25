"""
長時間滞在検出システム + 顔検出機能（シンプル版）
基本の人物追跡にMediaPipeの顔検出機能のみを追加

このバージョンでは:
- YOLO + ByteTrackによる人物追跡（既存）
- MediaPipeによる顔検出（新機能）
- 顔認識は行わない（シンプル化）
"""

import os
import sys

# スクリプトの親ディレクトリの親ディレクトリ (プロジェクトルート) をsys.pathに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import psutil

# 共通ユーティリティをインポート
from src.tracking.bytetrack_utils import (
    draw_tracking_info,
    initialize_bytetrack,
    initialize_perf_log,
    load_yolo_model,
    process_frame_for_tracking,
    update_stay_times,
)


class SimpleFaceDetector:
    """MediaPipeを使用したシンプルな顔検出クラス"""

    def __init__(self, confidence: float = 0.7):
        """
        Args:
            confidence: 顔検出の信頼度閾値
        """
        self.confidence = confidence
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0: 2m以内の顔検出用, 1: 5m以内の顔検出用
            min_detection_confidence=confidence,
        )

    def detect_faces(self, frame_rgb: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """
        フレームから顔を検出

        Args:
            frame_rgb: RGB形式のフレーム

        Returns:
            List of (x1, y1, x2, y2, confidence) 顔のバウンディングボックス
        """
        height, width = frame_rgb.shape[:2]
        results = self.face_detection.process(frame_rgb)

        faces = []
        if results.detections:
            for detection in results.detections:
                # 相対座標から絶対座標に変換
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * width)
                y1 = int(bbox.ymin * height)
                x2 = int((bbox.xmin + bbox.width) * width)
                y2 = int((bbox.ymin + bbox.height) * height)
                confidence = detection.score[0]

                faces.append((x1, y1, x2, y2, confidence))

        return faces

    def close(self):
        """リソースをクリーンアップ"""
        self.face_detection.close()


def draw_faces_on_frame(
    frame: np.ndarray, faces: list[tuple[int, int, int, int, float]]
) -> np.ndarray:
    """
    フレームに顔検出結果を描画

    Args:
        frame: 描画対象のフレーム
        faces: 顔のバウンディングボックスリスト

    Returns:
        顔検出結果が描画されたフレーム
    """
    for x1, y1, x2, y2, confidence in faces:
        # 顔の枠を描画（緑色）
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 信頼度を表示
        label = f"Face: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(
            frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1
        )
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return frame


def main(
    input_file: str,
    output_file: str,
    model_path: str = "yolo11n.pt",
    enable_perf_log: bool = False,
    enable_video_display: bool = False,
    device: str = "",
    stay_threshold_sec: float = 5.0,
    move_threshold_px: float = 30.0,
    conf: float = 0.3,
    face_confidence: float = 0.7,
):
    """
    長時間滞在検出 + 顔検出を実行するメイン関数

    Args:
        input_file: 入力動画ファイル
        output_file: 出力動画ファイル
        model_path: YOLOモデルのパス
        enable_perf_log: パフォーマンスログを有効にするかどうか
        enable_video_display: 処理中のビデオをリアルタイム表示するかどうか
        device: 推論デバイス
        stay_threshold_sec: 長時間滞在と判定する閾値（秒）
        move_threshold_px: 移動と判定する距離の閾値（ピクセル）
        conf: YOLO検出信頼度の閾値
        face_confidence: 顔検出信頼度の閾値
    """
    # 入力ファイルチェック
    if not os.path.exists(input_file):
        print(f"エラー: 入力ファイルが見つかりません: {input_file}")
        return

    # キャプチャ設定
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"エラー: 動画ファイルを開けません: {input_file}")
        return

    # 入力動画の情報を取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"入力動画: {width}x{height}, {fps}fps, {frame_count}フレーム")

    # モデルとトラッカーの初期化
    print("YOLOモデルを読み込み中...")
    model = load_yolo_model(model_path, device)
    print("ByteTrackトラッカーを初期化中...")
    tracker = initialize_bytetrack()
    print("顔検出器を初期化中...")
    face_detector = SimpleFaceDetector(confidence=face_confidence)

    # 出力設定
    if output_file:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Mac互換コーデック
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    else:
        out = None

    # ログファイル設定
    perf_log_file, perf_log_f, perf_log_writer = initialize_perf_log(
        enable_perf_log, input_file, model_path, log_type="face_detection"
    )

    if perf_log_writer:
        # ヘッダーに顔検出情報を追加
        perf_log_writer.writerow(["# Video Properties", f"{width}x{height}", f"{fps}fps"])
        perf_log_writer.writerow(["# Face Detection", f"confidence: {face_confidence}"])
        perf_log_writer.writerow([])

    # 滞在時間情報を保持する辞書
    stay_info = {}

    # 統計情報
    total_faces_detected = 0
    frames_with_faces = 0

    try:
        frame_idx = 0
        start_time = time.time()
        fps_buffer = []

        print("処理開始...")

        # 処理開始
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_idx += 1
            current_time = time.time()

            # フレーム処理
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # 1. 人物検出と追跡
            tracking_start = time.time()
            tracks, detection_time_ms, tracking_time_ms, num_detections, num_tracks, _ = (
                process_frame_for_tracking(frame_rgb, model, tracker)
            )

            # 2. 顔検出
            face_detection_start = time.time()
            faces = face_detector.detect_faces(frame_rgb)
            face_detection_time_ms = (time.time() - face_detection_start) * 1000

            # 統計更新
            if faces:
                frames_with_faces += 1
                total_faces_detected += len(faces)

            # 3. 長時間滞在チェック
            stay_info, notifications, stay_check_time_ms = update_stay_times(
                tracks, stay_info, current_time, move_threshold_px, stay_threshold_sec
            )

            # 通知があれば表示
            for notification in notifications:
                print(f"Frame {frame_idx}: {notification}")

            # 描画処理
            if out or enable_video_display:
                # トラッキング情報を描画
                frame_bgr = draw_tracking_info(
                    frame_bgr, tracks, show_duration=True, stay_info=stay_info
                )

                # 顔検出結果を描画
                frame_bgr = draw_faces_on_frame(frame_bgr, faces)

                # FPS情報などを追加
                current_fps = (
                    1.0 / (time.time() - current_time) if (time.time() - current_time) > 0 else 0
                )
                fps_buffer.append(current_fps)
                if len(fps_buffer) > 10:
                    fps_buffer.pop(0)
                avg_fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0

                # フレーム情報テキスト
                frame_info = f"Frame: {frame_idx}/{frame_count} FPS: {avg_fps:.1f}"
                stats_info = f"Person: {num_tracks} Faces: {len(faces)}"
                timing_info = f"Det: {detection_time_ms:.1f}ms Face: {face_detection_time_ms:.1f}ms"

                # テキスト背景用の黒枠描画
                cv2.rectangle(frame_bgr, (10, 10), (450, 115), (0, 0, 0), -1)

                # テキスト描画
                cv2.putText(
                    frame_bgr,
                    frame_info,
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    stats_info,
                    (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    timing_info,
                    (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                # 長期滞在者のリスト表示
                long_stayers = [
                    f"ID {id}: {info['stay_duration']:.1f}s"
                    for id, info in stay_info.items()
                    if info["stay_duration"] >= stay_threshold_sec
                ]

                if long_stayers:
                    cv2.rectangle(
                        frame_bgr,
                        (width - 210, 10),
                        (width - 10, 30 + 25 * len(long_stayers)),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.putText(
                        frame_bgr,
                        "長時間滞在者:",
                        (width - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                    for i, stayer in enumerate(long_stayers):
                        cv2.putText(
                            frame_bgr,
                            stayer,
                            (width - 200, 30 + 25 * (i + 1)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )

                # 出力処理
                if out:
                    out.write(frame_bgr)

                # 表示処理
                if enable_video_display:
                    cv2.imshow("Long Stay Detection with Face Detection", frame_bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESCキーで終了
                        break

            # パフォーマンスログ
            if perf_log_writer and frame_idx % max(1, int(fps)) == 0:
                current_mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                total_time_ms = (
                    detection_time_ms
                    + tracking_time_ms
                    + face_detection_time_ms
                    + stay_check_time_ms
                )
                perf_log_writer.writerow(
                    [
                        frame_idx,
                        datetime.now().strftime("%H:%M:%S.%f")[:-3],
                        f"{detection_time_ms:.2f}",
                        f"{tracking_time_ms:.2f}",
                        f"{face_detection_time_ms:.2f}",
                        f"{stay_check_time_ms:.2f}",
                        f"{total_time_ms:.2f}",
                        num_detections,
                        num_tracks,
                        len(faces),
                        f"{avg_fps:.2f}",
                        f"{current_mem_usage:.2f}",
                        Path(model_path).name,
                        "bytetrack+face_detection",
                        f"Stay:{stay_threshold_sec}s Face:{face_confidence}",
                    ]
                )

            # 進捗表示 (30フレームごと)
            if frame_idx % 30 == 0:
                elapsed_time = time.time() - start_time
                processing_speed = frame_idx / elapsed_time if elapsed_time > 0 else 0
                face_detection_rate = frames_with_faces / frame_idx * 100 if frame_idx > 0 else 0
                print(
                    f"進捗: {frame_idx}/{frame_count} ({frame_idx / frame_count * 100:.1f}%) | "
                    f"処理速度: {processing_speed:.2f} FPS | "
                    f"顔検出率: {face_detection_rate:.1f}% | "
                    f"総顔数: {total_faces_detected}"
                )

    except KeyboardInterrupt:
        print("\n処理がユーザーによって中断されました。")
    finally:
        # 終了処理
        cap.release()
        if out:
            out.release()
        if enable_video_display:
            cv2.destroyAllWindows()
        face_detector.close()
        if perf_log_f:
            perf_log_f.close()

        # 最終統計表示
        print("\n=== 処理完了 ===")
        print(f"総フレーム数: {frame_idx}")
        print(
            f"顔を検出したフレーム: {frames_with_faces} ({frames_with_faces / frame_idx * 100:.1f}%)"
        )
        print(f"総検出顔数: {total_faces_detected}")
        print(f"出力ファイル: {output_file}")
        print(f"ログファイル: {perf_log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="動画から人物追跡と顔検出を行い、長時間滞在を検出します（シンプル版）。"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="入力動画ファイルのパス。 (例: data/sample.mp4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/face_detection_tracking.mp4",
        help="出力動画ファイルのパス。 (例: output/tracked_sample.mp4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="YOLOモデルファイルのパス。 (例: models/yolo11n.pt)",
    )
    parser.add_argument(
        "--enable_perf_log",
        action="store_true",
        help="パフォーマンスログをCSVファイルに保存します。",
    )
    parser.add_argument(
        "--enable_video_display",
        action="store_true",
        help="処理中の動画をリアルタイムで表示します。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="推論に使用するデバイスを指定します。(例: cpu, mps, 0 for cuda:0)",
    )
    parser.add_argument(
        "--stay_threshold_sec",
        type=float,
        default=5.0,
        help="長時間滞在と判定する閾値（秒）。デフォルト: 5.0秒",
    )
    parser.add_argument(
        "--move_threshold_px",
        type=float,
        default=30.0,
        help="移動と判定するためのピクセル単位の閾値。デフォルト: 30.0ピクセル",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="YOLOの検出信頼度の閾値。デフォルト: 0.3",
    )
    parser.add_argument(
        "--face_confidence",
        type=float,
        default=0.7,
        help="顔検出の信頼度閾値。デフォルト: 0.7",
    )

    args = parser.parse_args()

    # 出力ディレクトリの作成
    if args.output:
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)

    main(
        input_file=args.input,
        output_file=args.output,
        model_path=args.model,
        enable_perf_log=args.enable_perf_log,
        enable_video_display=args.enable_video_display,
        device=args.device,
        stay_threshold_sec=args.stay_threshold_sec,
        move_threshold_px=args.move_threshold_px,
        conf=args.conf,
        face_confidence=args.face_confidence,
    )
