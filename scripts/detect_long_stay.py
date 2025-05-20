import os
import sys

# スクリプトの親ディレクトリの親ディレクトリ (プロジェクトルート) をsys.pathに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
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

# Skeleton structure for YOLO keypoints visualization
# (kept for potential future use, but not primary focus)
SKELETON = [
    (0, 1),
    (1, 3),
    (0, 2),
    (1, 2),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def main(
    input_file: str,
    output_file: str,
    model_path: str = "yolov11n-pose.pt",
    enable_perf_log: bool = False,
    enable_video_display: bool = False,
    device: str = "",
    stay_threshold_sec: float = 4.0,
    move_threshold_px: float = 20.0,
    conf: float = 0.3,
):
    """
    長時間滞在検出を実行するメイン関数

    Args:
        input_file: 入力動画ファイル
        output_file: 出力動画ファイル
        model_path: YOLOモデルのパス
        enable_perf_log: パフォーマンスログを有効にするかどうか
        enable_video_display: 処理中のビデオをリアルタイム表示するかどうか
        device: 推論デバイス ("cpu", "cuda", "mps" など)
        stay_threshold_sec: 長時間滞在と判定する閾値（秒）
        move_threshold_px: 移動と判定する距離の閾値（ピクセル）
        conf: 検出信頼度の閾値
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

    # モデルの読み込み
    model = load_yolo_model(model_path, device)

    # トラッカーの初期化
    tracker = initialize_bytetrack()

    # 出力設定
    if output_file:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Mac互換コーデック
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    else:
        out = None

    # ログファイル設定
    perf_log_file, perf_log_f, perf_log_writer = initialize_perf_log(
        enable_perf_log, input_file, model_path, log_type="long_stay"
    )

    if perf_log_writer:
        # 動画のプロパティ情報を追加
        perf_log_writer.writerow(["# Video Properties", f"{width}x{height}", f"{fps}fps"])
        perf_log_writer.writerow([])

    # 滞在時間情報を保持する辞書
    stay_info = {}

    try:
        frame_idx = 0
        start_time = time.time()
        last_output_time = start_time
        last_fps_update = start_time
        fps_buffer = []

        # 処理開始
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_idx += 1
            current_time = time.time()

            # フレーム処理
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # 検出と追跡
            tracks, detection_time_ms, tracking_time_ms, num_detections, num_tracks, _ = (
                process_frame_for_tracking(frame_rgb, model, tracker)
            )

            # 長時間滞在チェック
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

                # FPS情報などを追加
                process_time = (time.time() - current_time) * 1000
                current_fps = (
                    1.0 / (time.time() - last_fps_update)
                    if (time.time() - last_fps_update) > 0
                    else 0
                )
                fps_buffer.append(current_fps)
                if len(fps_buffer) > 10:
                    fps_buffer.pop(0)
                avg_fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0
                last_fps_update = time.time()

                # フレーム情報テキスト
                frame_info = f"Frame: {frame_idx}/{frame_count} FPS: {avg_fps:.1f}"
                stats_info = f"Det: {detection_time_ms:.1f}ms Track: {tracking_time_ms:.1f}ms Stay: {stay_check_time_ms:.1f}ms"
                objects_info = f"Detected: {num_detections} Tracked: {num_tracks}"

                # テキスト背景用の黒枠描画
                cv2.rectangle(frame_bgr, (10, 10), (400, 90), (0, 0, 0), -1)

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
                    objects_info,
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
                    cv2.imshow("Long Stay Detection", frame_bgr)
                    key = cv2.waitKey(1)
                    if key == 27:  # ESCキーで終了
                        break

            # パフォーマンスログ
            if perf_log_writer and frame_idx % max(1, int(fps)) == 0:  # 1秒に1回程度ログを取る
                current_mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                total_time_ms = detection_time_ms + tracking_time_ms + stay_check_time_ms
                perf_log_writer.writerow(
                    [
                        frame_idx,
                        datetime.now().strftime("%H:%M:%S.%f")[:-3],
                        f"{detection_time_ms:.2f}",
                        f"{tracking_time_ms:.2f}",
                        f"{stay_check_time_ms:.2f}",
                        f"{total_time_ms:.2f}",
                        num_detections,
                        num_tracks,
                        f"{avg_fps:.2f}",
                        f"{current_mem_usage:.2f}",
                        Path(model_path).name,
                        "bytetrack",
                        f"Stay:{stay_threshold_sec}s Move:{move_threshold_px}px",
                    ]
                )

            # 進捗表示 (30フレームごと)
            if frame_idx % 30 == 0:
                elapsed_time = time.time() - start_time
                processing_speed = frame_idx / elapsed_time if elapsed_time > 0 else 0
                print(
                    f"進捗: {frame_idx}/{frame_count} ({frame_idx / frame_count * 100:.1f}%) | "
                    f"処理速度: {processing_speed:.2f} FPS | "
                    f"現在時刻: {time.strftime('%H:%M:%S', time.localtime(current_time))}"
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
        if perf_log_f:
            perf_log_f.close()
        print(f"処理完了。出力ファイル: {output_file}, ログ: {perf_log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="動画からByteTrackを用いて長時間滞在を検出します。"
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
        default="output/tracked_video.mp4",
        help="出力動画ファイルのパス。 (例: output/tracked_sample.mp4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",  # 修正: yolo11n-pose.pt から yolov8n.pt へ変更の提案（一般的であるため）
        help="YOLOモデルファイルのパス。 (例: models/yolov8n.pt)",
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
        default="",  # 自動選択を促す
        help="推論に使用するデバイスを指定します。(例: cpu, mps, 0 for cuda:0)",
    )
    parser.add_argument(
        "--stay_threshold_sec",
        type=float,
        default=5.0,  # 修正: 4.0から5.0へ変更の提案（より実用的か）
        help="長時間滞在と判定する閾値（秒）。デフォルト: 5.0秒",
    )
    parser.add_argument(
        "--move_threshold_px",
        type=float,
        default=30.0,  # 修正: 20.0から30.0へ変更の提案（より実用的か）
        help="移動と判定するためのピクセル単位の閾値。デフォルト: 30.0ピクセル",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,  # yolov8のデフォルトに近い値
        help="YOLOの検出信頼度の閾値。デフォルト: 0.3",
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
    )
