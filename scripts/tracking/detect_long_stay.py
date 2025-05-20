import argparse
import csv
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import cv2
import psutil
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

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


def get_system_info():
    """システム情報を取得する関数"""
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu": platform.processor(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_total": round(psutil.virtual_memory().total / (1024**3), 2),  # GB単位
    }


# Simplified CSV initialization for performance logs only
def initialize_perf_log(enable_perf_log: bool, input_file: str, model_path: str):
    """Initialize performance log CSV if enabled."""
    if not enable_perf_log:
        return None, None, None

    system_info = get_system_info()
    timestamp_log = datetime.now().strftime("%Y%m%d_%H%M%S")
    perf_log_file = (
        f"log_{Path(input_file).stem}_long_stay_{Path(model_path).stem}_{timestamp_log}.csv"
    )
    perf_columns = [
        "Frame",
        "Time",
        "Detection_Time_ms",
        "Tracking_Time_ms",
        "Stay_Check_Time_ms",
        "Total_Time_ms",
        "Objects_Detected",
        "Objects_Tracked",
        "FPS",
        "Model",
        "Tracker",
        "Notes",
    ]

    try:
        with open(perf_log_file, "w", newline="") as perf_log_f:
            perf_log_writer = csv.writer(perf_log_f)
            perf_log_writer.writerow(["# System Information"])
            perf_log_writer.writerow(["# OS", system_info["os"], system_info["os_version"]])
            perf_log_writer.writerow(["# Python", system_info["python_version"]])
            perf_log_writer.writerow(
                [
                    "# CPU",
                    system_info["cpu"],
                    f"{system_info['cpu_cores']} cores, {system_info['cpu_threads']} threads",
                ]
            )
            perf_log_writer.writerow(["# RAM", f"{system_info['ram_total']} GB"])
            perf_log_writer.writerow([])
            perf_log_writer.writerow(["# YOLO Model", model_path])  # Simplified model info
            perf_log_writer.writerow(["# Tracker", "deepsort"])
            perf_log_writer.writerow([])
            perf_log_writer.writerow(["# Video", input_file])
            # Video properties (width, height, fps) will be added later in main
            perf_log_writer.writerow([])
            perf_log_writer.writerow(perf_columns)
            print(f"パフォーマンスログ: 有効 ({perf_log_file})")
            return perf_log_file, perf_log_f, perf_log_writer
    except OSError as e:
        print(f"エラー: パフォーマンスログファイル '{perf_log_file}' を開けません: {e}")
        return None, None, None


def process_frame_for_tracking(frame_rgb, model, tracker):
    """Process a single frame for detection and tracking, returning tracks."""
    detection_start_time = time.time()
    results = model.predict(frame_rgb, verbose=False, classes=[0])  # Predict only class 0 (person)
    detection_time_ms = (time.time() - detection_start_time) * 1000
    result = results[0]

    detections = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()

    formatted_detections = [
        ([x1, y1, x2 - x1, y2 - y1], float(conf), int(cls))
        for (x1, y1, x2, y2), conf, cls in zip(detections, confidences, class_ids, strict=False)
    ]

    tracking_start_time = time.time()
    tracks = tracker.update_tracks(formatted_detections, frame=frame_rgb)
    tracking_time_ms = (time.time() - tracking_start_time) * 1000

    confirmed_tracks = [t for t in tracks if t.is_confirmed()]

    return (
        confirmed_tracks,
        detection_time_ms,
        tracking_time_ms,
        len(detections),
        len(confirmed_tracks),
    )


def update_stay_times(tracks, stay_info, current_time, move_threshold_px, stay_threshold_sec):
    """Update stay duration for each tracked object and check for long stays."""
    stay_check_start = time.time()
    current_track_ids = set()
    notifications = []

    for track in tracks:
        track_id = track.track_id
        current_track_ids.add(track_id)
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        current_pos = (center_x, center_y)

        if track_id not in stay_info:
            # New track detected
            stay_info[track_id] = {
                "last_pos": current_pos,
                "last_time": current_time,
                "stay_duration": 0.0,
                "notified": False,
            }
        else:
            # Existing track update
            last_pos = stay_info[track_id]["last_pos"]
            last_time = stay_info[track_id]["last_time"]
            distance = (
                (current_pos[0] - last_pos[0]) ** 2 + (current_pos[1] - last_pos[1]) ** 2
            ) ** 0.5
            time_diff = current_time - last_time

            if distance < move_threshold_px:
                # Stayed in the same area
                stay_info[track_id]["stay_duration"] += time_diff
            else:
                # Moved significantly
                stay_info[track_id]["stay_duration"] = 0.0

            # Update position and time regardless of movement
            stay_info[track_id]["last_pos"] = current_pos
            stay_info[track_id]["last_time"] = current_time

            # Check for notification
            if (
                stay_info[track_id]["stay_duration"] >= stay_threshold_sec
                and not stay_info[track_id]["notified"]
            ):
                # 長い通知メッセージを複数行に分割
                notification = (
                    f"通知: ID {track_id} が座標 ({int(center_x)}, {int(center_y)}) "
                    f"付近に {stay_threshold_sec}秒 以上滞在中。"
                )
                notifications.append(notification)
                stay_info[track_id]["notified"] = True  # Mark as notified

    # Clean up lost tracks
    lost_track_ids = set(stay_info.keys()) - current_track_ids
    for track_id in lost_track_ids:
        del stay_info[track_id]

    stay_check_time_ms = (time.time() - stay_check_start) * 1000
    return stay_info, notifications, stay_check_time_ms


def draw_tracking_info(frame_bgr, tracks, stay_info):
    """Draw bounding boxes, IDs, and stay duration on the frame."""
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # Draw bounding box
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display ID and stay duration
        duration = stay_info.get(track_id, {}).get("stay_duration", 0.0)
        label = f"ID {track_id} ({duration:.1f}s)"
        cv2.putText(frame_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main(
    input_file: str,
    output_file: str,
    model_path: str = "yolov11n-pose.pt",
    enable_perf_log: bool = False,
    enable_video_display: bool = False,
    device: str = "",
    stay_threshold_sec: float = 4.0,
    move_threshold_px: float = 20.0,
):
    """
    指定座標での滞在時間を検知するメイン関数
    Args:
        input_file: 入力動画のパス
        output_file: 出力動画のパス
        model_path: 使用するYOLOモデルのパス (人物検出用, poseモデルでなくても良い)
        enable_perf_log: パフォーマンスログをCSVに出力するかどうか
        enable_video_display: プレビューを表示するかどうか
        device: 使用するデバイス (例: cpu, 0)
        stay_threshold_sec: 滞在通知を行う時間の閾値 (秒)
        move_threshold_px: 同一場所とみなす移動距離の閾値 (ピクセル)
    """
    # モデルファイルの存在確認
    if not os.path.exists(model_path):
        print(f"エラー: モデルファイル '{model_path}' が見つかりません。")
        exit()

    # Read input file
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"Error: Unable to open video: {input_file}")
        exit()

    # Prepare output file
    fps, width, height = (
        cap.get(cv2.CAP_PROP_FPS),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Mac互換コーデック
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    print(f"ビデオ情報: {width}x{height} @ {fps:.2f}fps")
    print("出力コーデック: H.264 (avc1)")
    print(f"滞在検知閾値: {stay_threshold_sec} 秒")
    print(f"移動検知閾値: {move_threshold_px} ピクセル")

    # Initialize YOLO model (use a standard detection model if pose isn't needed)
    print(f"YOLOモデル: {model_path}")
    try:
        if device.isdigit():
            yolo_device = int(device)
        elif device.lower() == "cpu":
            yolo_device = "cpu"
        else:
            yolo_device = None  # Auto-select
            if device:
                print(f"警告: 不正なデバイス指定 '{device}' です。自動選択を使用します。")
    except ValueError:
        print(f"警告: 不正なデバイス指定 '{device}' です。自動選択を使用します。")
        yolo_device = None

    model = YOLO(model_path).to(yolo_device)
    if model.task == "pose":
        print("情報: ポーズ推定モデルが指定されましたが、このスクリプトでは人物検出のみ行います。")

    # Initialize DeepSORT tracker
    print("DeepSORTトラッキング: 有効")
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)  # Standard parameters

    # Initialize Performance Log
    perf_log_file, perf_log_f, perf_log_writer = initialize_perf_log(
        enable_perf_log, input_file, model_path
    )
    if enable_perf_log and perf_log_writer:
        # Add video info to perf log header now that we have it
        perf_log_writer.writerow(["# Resolution", f"{width}x{height}", "FPS", f"{fps:.2f}"])
        perf_log_writer.writerow(["# Stay Threshold (s)", stay_threshold_sec])
        perf_log_writer.writerow(["# Move Threshold (px)", move_threshold_px])
        perf_log_writer.writerow([])  # Add separator before data rows

    frame_idx = 0
    loop_start_time = time.time()
    detection_times = []
    tracking_times = []
    stay_check_times = []
    stay_info = {}  # Dictionary to store stay duration per track_id

    try:
        print("処理を開始します...")
        while True:
            frame_start_time = time.time()
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_idx += 1
            current_timestamp = time.time()

            # Convert to RGB for YOLO
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # 1. Process frame for tracking
            tracks, detection_time_ms, tracking_time_ms, objects_detected, objects_tracked = (
                process_frame_for_tracking(frame_rgb, model, tracker)
            )
            detection_times.append(detection_time_ms)
            tracking_times.append(tracking_time_ms)

            # 2. Update stay times and check for notifications
            stay_info, notifications, stay_check_time_ms = update_stay_times(
                tracks, stay_info, current_timestamp, move_threshold_px, stay_threshold_sec
            )
            stay_check_times.append(stay_check_time_ms)

            # Print notifications
            for msg in notifications:
                print(f"フレーム {frame_idx}: {msg}")

            # 3. Draw visuals
            draw_tracking_info(frame_bgr, tracks, stay_info)

            # Write performance log entry
            if enable_perf_log and perf_log_writer:
                frame_end_time = time.time()
                total_frame_time_ms = (frame_end_time - frame_start_time) * 1000
                elapsed_loop_time = time.time() - loop_start_time
                current_overall_fps = frame_idx / elapsed_loop_time if elapsed_loop_time > 0 else 0

                perf_log_writer.writerow(
                    [
                        frame_idx,
                        f"{elapsed_loop_time:.3f}",
                        f"{detection_time_ms:.1f}",
                        f"{tracking_time_ms:.1f}",
                        f"{stay_check_time_ms:.1f}",
                        f"{total_frame_time_ms:.1f}",
                        objects_detected,
                        objects_tracked,
                        f"{current_overall_fps:.1f}",
                        Path(model_path).stem,
                        "deepsort",
                        f"{len(notifications)} notifications" if notifications else "",
                    ]
                )

            # Display progress
            if frame_idx % 30 == 0 or frame_idx == 1:
                elapsed = time.time() - loop_start_time
                current_fps = frame_idx / elapsed if elapsed > 0 else 0
                print(f"処理中: {frame_idx}フレーム完了 (現在の処理速度: {current_fps:.1f}fps)")

            # Save output frame
            out.write(frame_bgr)

            # Display live window if enabled
            if enable_video_display:
                cv2.imshow("Long Stay Detection", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("処理が中断されました")
    finally:
        print("リソース解放中...")
        cap.release()
        out.release()
        if enable_video_display:
            cv2.destroyAllWindows()

        # Print summary
        total_time = time.time() - loop_start_time
        avg_fps = frame_idx / total_time if total_time > 0 else 0
        avg_detection_time = sum(detection_times) / len(detection_times) if detection_times else 0
        avg_tracking_time = sum(tracking_times) / len(tracking_times) if tracking_times else 0
        avg_stay_check_time = (
            sum(stay_check_times) / len(stay_check_times) if stay_check_times else 0
        )

        print("--- 処理結果サマリ ---")
        print(f"合計処理時間: {total_time:.2f} 秒")
        print(f"処理フレーム数: {frame_idx}")
        print(f"平均処理速度 (FPS): {avg_fps:.2f}")
        print(f"平均検出時間: {avg_detection_time:.1f} ms")
        print(f"平均トラッキング時間: {avg_tracking_time:.1f} ms")
        print(f"平均滞在チェック時間: {avg_stay_check_time:.1f} ms")

        # 長い行を分割
        total_avg_time = avg_detection_time + avg_tracking_time + avg_stay_check_time
        print(f"平均合計処理時間 (フレームあたり): {total_avg_time:.1f} ms")
        print(f"出力動画: {output_file}")
        if enable_perf_log and perf_log_file:
            print(f"パフォーマンスログ: {perf_log_file}")
            if perf_log_f:
                print(f"パフォーマンスログファイル '{perf_log_file}' を閉じています。")
                perf_log_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect long stays of tracked objects (people) in a video."
    )
    parser.add_argument("--input-mp4", required=True, help="入力動画ファイルのパス")
    parser.add_argument("--output-mp4", required=True, help="出力動画ファイルのパス")
    parser.add_argument(
        "--model",
        default="models/yolov8n.pt",
        help="YOLOモデルのパス (人物検出用, デフォルト: models/yolov8n.pt)",
    )
    parser.add_argument(
        "--enable-perf-log", action="store_true", help="パフォーマンスログをCSVに出力"
    )
    parser.add_argument("--device", type=str, default="", help="使用するデバイス (例: cpu, 0, '')")
    parser.add_argument(
        "--enable-video-display", action="store_true", help="処理中のプレビューを表示"
    )
    parser.add_argument(
        "--stay-threshold",
        type=float,
        default=6.0,
        help="滞在通知を行う時間の閾値 (秒, デフォルト: 4.0)",
    )
    parser.add_argument(
        "--move-threshold",
        type=float,
        default=20.0,
        help="同一場所とみなす移動距離の閾値 (ピクセル, デフォルト: 20.0)",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_mp4), exist_ok=True)

    main(
        args.input_mp4,
        args.output_mp4,
        args.model,
        args.enable_perf_log,
        args.enable_video_display,
        args.device,
        args.stay_threshold,
        args.move_threshold,
    )
