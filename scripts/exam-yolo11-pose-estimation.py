import argparse
import csv
from datetime import datetime
import time
import cv2
import os
import platform
import psutil
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Skeleton structure for YOLO keypoints visualization
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
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu": platform.processor(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_total": round(psutil.virtual_memory().total / (1024**3), 2),  # GB単位
    }
    return info


def initialize_csv(enable_csv_output: bool):
    """Initialize CSV file if enabled and return the file path and writer."""
    if not enable_csv_output:
        return None, None

    timestamp_output = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"pose_output_{timestamp_output}.csv"
    columns = ["frame_idx", "timestamp", "track_id", "detection_idx"] + [
        f"kpt_{i}_{axis}" for i in range(17) for axis in ("x", "y", "conf")
    ]

    # Initialize CSV and write the header
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

    return csv_file, columns


def process_frame(frame_idx, frame_rgb, model, tracker=None, enable_tracking=True):
    """Process a single frame: Run YOLO detection and DeepSORT tracking if enabled."""
    # Run YOLO Pose inference
    results = model.predict(frame_rgb, verbose=False)
    result = results[0]

    # Extract keypoints
    keypoints = result.keypoints.cpu().numpy() if result.keypoints is not None else []

    # Skip tracking if disabled
    if not enable_tracking or tracker is None:
        return keypoints, [], 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0

    # Extract bounding boxes for tracking
    detections = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()

    # Prepare detection data for DeepSORT
    formatted_detections = [
        ([x1, y1, x2 - x1, y2 - y1], float(conf), int(cls))
        for (x1, y1, x2, y2), conf, cls in zip(detections, confidences, class_ids)
    ]

    # Update tracker
    tracks = tracker.update_tracks(formatted_detections, frame=frame_rgb)
    return (
        keypoints,
        tracks,
        0.0,
        0.0,
        len(detections),
        len([t for t in tracks if t.is_confirmed()]),
        0.0,
        0.0,
        0.0,
    )


def write_to_csv(csv_file, columns, frame_idx, timestamp, tracks, keypoints):
    """Write pose and tracking data to CSV if enabled."""
    if not csv_file:
        return

    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)

        # If no tracking, just write keypoints with dummy track_id
        if not tracks:
            for detection_idx, kp_obj in enumerate(keypoints):
                if kp_obj.shape[0] == 0:
                    continue
                row = {
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "track_id": -1,
                    "detection_idx": detection_idx,
                }

                # Add keypoints data
                kp_array = kp_obj.data[0]
                for kpt_idx, (x, y, conf) in enumerate(kp_array):
                    row[f"kpt_{kpt_idx}_x"] = float(x)
                    row[f"kpt_{kpt_idx}_y"] = float(y)
                    row[f"kpt_{kpt_idx}_conf"] = float(conf)

                writer.writerow(row)
            return

        # With tracking, associate tracks with keypoints
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            for detection_idx, kp_obj in enumerate(keypoints):
                if kp_obj.shape[0] == 0:
                    continue
                row = {
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "track_id": track_id,
                    "detection_idx": detection_idx,
                }

                # Add keypoints data
                kp_array = kp_obj.data[0]
                for kpt_idx, (x, y, conf) in enumerate(kp_array):
                    row[f"kpt_{kpt_idx}_x"] = float(x)
                    row[f"kpt_{kpt_idx}_y"] = float(y)
                    row[f"kpt_{kpt_idx}_conf"] = float(conf)

                writer.writerow(row)


def draw_visuals(frame_bgr, tracks, keypoints, enable_tracking=True):
    """Draw bounding boxes, IDs, and skeletons on the frame."""

    # Draw bounding boxes & IDs on the BGR frame if tracking is enabled
    if enable_tracking and tracks:
        for track in tracks:
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = map(int, track.to_ltrb())  # left, top, right, bottom

            # Deep SORT Plot
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame_bgr,
                f"ID {track.track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    # Draw the skeleton lines for each person
    for kp_obj in keypoints:
        if kp_obj.shape[0] == 0:
            continue

        kp_array = kp_obj.data[0]
        for p1, p2 in SKELETON:
            if p1 >= len(kp_array) or p2 >= len(kp_array):
                continue

            x1, y1, c1 = kp_array[p1]
            x2, y2, c2 = kp_array[p2]
            if c1 > 0.5 and c2 > 0.5:
                cv2.line(frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        # Draw circles for each visible keypoint
        for x, y, c in kp_array:
            if c > 0.5:
                cv2.circle(frame_bgr, (int(x), int(y)), 3, (255, 0, 0), -1)


def main(
    input_file: str,
    output_file: str,
    enable_csv_output: bool = False,
    enable_video_display: bool = False,
    enable_tracking: bool = True,
    model_path: str = "yolov11n-pose.pt",
    enable_perf_log: bool = False,
    device: str = "",
):
    """
    ポーズ推定処理のメイン関数
    Args:
        input_file: 入力動画のパス
        output_file: 出力動画のパス
        enable_csv_output: CSVログを出力するかどうか
        enable_video_display: プレビューを表示するかどうか
        enable_tracking: DeepSORTによる追跡を行うかどうか
        model_path: 使用するYOLOモデルのパス
        enable_perf_log: パフォーマンスログをCSVに出力するかどうか
        device: 使用するデバイス (例: cpu, 0)
    """
    # モデルファイルの存在確認
    if not os.path.exists(model_path):
        print(f"エラー: モデルファイル '{model_path}' が見つかりません。")
        print(f"現在の作業ディレクトリ: {os.getcwd()}")
        print(f"絶対パスでの指定が必要かもしれません。")
        print(
            f"例: python exam-yolo11-pose-estimation.py --input-mp4 [入力ファイル] --output-mp4 [出力ファイル] --model [モデルへの絶対パス]"
        )
        exit()

    # Read input file
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"Error: Unable to open video: {input_file}")
        exit()

    # Prepare output file with OpenCV
    fps, width, height = (
        cap.get(cv2.CAP_PROP_FPS),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    # Mac互換コーデックを使用（avc1 = H.264）
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Video properties を出力
    print(f"ビデオ情報: {width}x{height} @ {fps}fps")
    print(f"出力コーデック: H.264 (avc1) - Mac互換")

    # Initialize YOLO Pose model
    print(f"YOLOモデル: {model_path}")
    try:
        if device.isdigit():
            # '0', '1' など数字の場合は整数に変換
            yolo_device = int(device)
        elif device.lower() == "cpu":
            yolo_device = "cpu"
        else:
            # 空文字や他の文字列の場合は自動選択 (None)
            print(f"警告: 不正なデバイス指定 '{device}' です。自動選択を使用します。")
            yolo_device = None
    except ValueError:
        # 整数に変換できない場合は自動選択
        print(f"警告: 不正なデバイス指定 '{device}' です。自動選択を使用します。")
        yolo_device = None

    model = YOLO(model_path).to(yolo_device)  # デバイスを指定

    # Initialize DeepSORT if tracking is enabled
    tracker = None
    if enable_tracking:
        print("DeepSORTトラッキング: 有効")
        # トラッキングパラメータ最適化: max_ageを短くして速度向上
        tracker = DeepSort(max_age=15, n_init=2, nn_budget=50)
    else:
        print("DeepSORTトラッキング: 無効 (処理速度優先)")

    pose_csv_file, pose_columns = initialize_csv(enable_csv_output)
    perf_log_file = None
    perf_log_f = None
    perf_log_writer = None

    if enable_perf_log:
        system_info = get_system_info()
        timestamp_log = datetime.now().strftime("%Y%m%d_%H%M%S")
        perf_log_file = (
            f"log_{Path(input_file).stem}_deepsort_{Path(model_path).stem}_{timestamp_log}.csv"
        )
        perf_columns = [
            "Frame",
            "Time",
            "Detection_Time_ms",
            "Tracking_Time_ms",
            "Total_Time_ms",
            "Objects_Detected",
            "Objects_Tracked",
            "FPS",
            "YOLO_Preprocess_ms",
            "YOLO_Inference_ms",
            "YOLO_Postprocess_ms",
            "CUDA_Used",
            "Model",
            "Tracker",
            "Notes",
        ]

        try:
            perf_log_f = open(perf_log_file, "w", newline="")
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
            perf_log_writer.writerow(
                [
                    "# YOLO Model",
                    model_path,
                    "Task",
                    model.task if hasattr(model, "task") else "pose",
                ]
            )
            perf_log_writer.writerow(["# Tracker", "deepsort" if enable_tracking else "None"])
            perf_log_writer.writerow(["# Device", device if device else "auto"])
            perf_log_writer.writerow([])
            perf_log_writer.writerow(["# Video", input_file])
            perf_log_writer.writerow(["# Resolution", f"{width}x{height}", "FPS", fps])
            perf_log_writer.writerow([])
            perf_log_writer.writerow(perf_columns)
            print(f"パフォーマンスログ: 有効 ({perf_log_file})")
        except IOError as e:
            print(f"エラー: パフォーマンスログファイル '{perf_log_file}' を開けません: {e}")
            enable_perf_log = False
            perf_log_file = None
            perf_log_f = None
            perf_log_writer = None
    else:
        print("パフォーマンスログ: 無効")

    frame_idx = 0
    loop_start_time = time.time()
    detection_times = []
    tracking_times = []
    skip_frame = 0  # 処理負荷を下げるためのフレームスキップカウンタ

    try:
        print("処理を開始します...")
        while True:
            # Read a raw BGR frame from the video
            frame_start_time = time.time()
            ret, frame_bgr = cap.read()
            if not ret:
                # End of video or read error
                break

            frame_idx += 1

            # フレームスキップ処理（トラッキング有効時のみ）
            if enable_tracking and skip_frame > 0:
                skip_frame -= 1
                # スキップフレームは単にコピーする
                out.write(frame_bgr)
                if enable_perf_log and perf_log_writer:
                    elapsed_loop_time = time.time() - loop_start_time
                    current_overall_fps = (
                        frame_idx / elapsed_loop_time if elapsed_loop_time > 0 else 0
                    )
                    cuda_used_str = "Yes" if device and device != "cpu" else "No"
                    perf_log_writer.writerow(
                        [
                            frame_idx,
                            f"{elapsed_loop_time:.3f}",
                            "0.0",
                            "0.0",
                            "0.0",
                            0,
                            0,
                            f"{current_overall_fps:.1f}",
                            "0.0",
                            "0.0",
                            "0.0",
                            cuda_used_str,
                            Path(model_path).stem,
                            "deepsort" if enable_tracking else "None",
                            "Skipped Frame",
                        ]
                    )
                continue

            timestamp = time.time()

            # Convert to RGB once for YOLO
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            (
                keypoints,
                tracks,
                detection_time_ms,
                tracking_time_ms,
                objects_detected,
                objects_tracked,
                yolo_preprocess,
                yolo_inference,
                yolo_postprocess,
            ) = process_frame(frame_idx, frame_rgb, model, tracker, enable_tracking)
            detection_times.append(detection_time_ms)
            if enable_tracking:
                tracking_times.append(tracking_time_ms)

            if enable_csv_output:
                write_to_csv(pose_csv_file, pose_columns, frame_idx, timestamp, tracks, keypoints)

            draw_visuals(frame_bgr, tracks, keypoints, enable_tracking)

            # トラッキング有効時は一部フレームをスキップして処理負荷を軽減
            if enable_tracking and frame_idx % 60 == 0:
                skip_frame = 2  # 2フレームスキップ

            if enable_perf_log and perf_log_writer:
                frame_end_time = time.time()
                total_frame_time_ms = (frame_end_time - frame_start_time) * 1000
                elapsed_loop_time = time.time() - loop_start_time
                current_overall_fps = frame_idx / elapsed_loop_time if elapsed_loop_time > 0 else 0
                cuda_used_str = "Yes" if device and device != "cpu" else "No"

                perf_log_writer.writerow(
                    [
                        frame_idx,
                        f"{elapsed_loop_time:.3f}",
                        f"{detection_time_ms:.1f}",
                        f"{tracking_time_ms:.1f}" if enable_tracking else "0.0",
                        f"{detection_time_ms + tracking_time_ms:.1f}"
                        if enable_tracking
                        else f"{detection_time_ms:.1f}",
                        objects_detected,
                        objects_tracked if enable_tracking else 0,
                        f"{current_overall_fps:.1f}",
                        f"{yolo_preprocess:.1f}",
                        f"{yolo_inference:.1f}",
                        f"{yolo_postprocess:.1f}",
                        cuda_used_str,
                        Path(model_path).stem,
                        "deepsort" if enable_tracking else "None",
                        "",
                    ]
                )

            # 処理の進捗状況を表示
            if frame_idx % 30 == 0 or frame_idx == 1:
                elapsed = time.time() - loop_start_time
                current_fps = frame_idx / elapsed if elapsed > 0 else 0
                print(f"処理中: {frame_idx}フレーム完了 (現在の処理速度: {current_fps:.1f}fps)")

                detect_text = f"Detection: {detection_time_ms:.1f}ms"
                track_text = (
                    f"Tracking: {tracking_time_ms:.1f}ms" if enable_tracking else "Tracking: N/A"
                )
                fps_text = f"FPS: {current_fps:.1f}"
                cv2.putText(
                    frame_bgr, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
                cv2.putText(
                    frame_bgr, detect_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
                cv2.putText(
                    frame_bgr, track_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )

            # Save the processed frame to our output file
            out.write(frame_bgr)

            # Optionally, also display if you want a live window (can be omitted if running headless)
            if enable_video_display:
                cv2.imshow("YOLO Pose", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n処理が中断されました")
    finally:
        # 処理の完了
        print("\nリソース解放中...")
        cap.release()
        out.release()
        if enable_video_display:
            cv2.destroyAllWindows()

        # 性能情報の表示
        total_time = time.time() - loop_start_time
        avg_fps = frame_idx / total_time if total_time > 0 else 0
        avg_detection_time = sum(detection_times) / len(detection_times) if detection_times else 0
        avg_tracking_time = (
            sum(tracking_times) / len(tracking_times) if tracking_times and enable_tracking else 0
        )

        print(f"\n--- 処理結果サマリ ---")
        print(f"合計処理時間: {total_time:.2f} 秒")
        print(f"処理フレーム数: {frame_idx}")
        print(f"平均処理速度 (FPS): {avg_fps:.2f}")
        print(f"平均検出時間: {avg_detection_time:.1f} ms")
        if enable_tracking:
            print(f"平均トラッキング時間: {avg_tracking_time:.1f} ms")
            print(f"平均合計処理時間 (検出+追跡): {avg_detection_time + avg_tracking_time:.1f} ms")
        print(f"出力動画: {output_file}")
        if enable_csv_output:
            print(f"ポーズCSV: {pose_csv_file}")
        if enable_perf_log:
            print(f"パフォーマンスログ: {perf_log_file}")
            if perf_log_f:
                print(f"パフォーマンスログファイル '{perf_log_file}' を閉じています。")
                perf_log_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO Pose Estimation Tool with DeepSORT Tracking and Logging"
    )
    parser.add_argument("--input-mp4", required=True, help="入力動画ファイルのパス")
    parser.add_argument("--output-mp4", required=True, help="出力動画ファイルのパス")
    parser.add_argument(
        "--enable-csv-output", action="store_true", help="ポーズデータをCSVに出力 (キーポイント用)"
    )
    parser.add_argument(
        "--enable-perf-log", action="store_true", help="パフォーマンスログをCSVに出力"
    )
    parser.add_argument("--device", type=str, default="", help="使用するデバイス (例: cpu, 0)")
    parser.add_argument(
        "--enable-video-display", action="store_true", help="処理中のプレビューを表示"
    )
    parser.add_argument(
        "--model",
        default="yolov11n-pose.pt",
        help="YOLOモデルのパス (デフォルト: yolov11n-pose.pt)",
    )
    parser.add_argument(
        "--disable-tracking", action="store_true", help="DeepSORTトラッキングを無効化(処理速度向上)"
    )
    args = parser.parse_args()

    # 出力ディレクトリの確保
    os.makedirs(os.path.dirname(args.output_mp4), exist_ok=True)
    if args.enable_perf_log:
        log_dir = Path(
            f"log_{Path(args.input_mp4).stem}_deepsort_{Path(args.model).stem}_YYYYMMDD_HHMMSS.csv"
        ).parent
        pass

    main(
        args.input_mp4,
        args.output_mp4,
        args.enable_csv_output,
        args.enable_video_display,
        not args.disable_tracking,
        args.model,
        args.enable_perf_log,
        args.device,
    )
