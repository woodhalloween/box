# detect_long_stay_core.py
#
# Daisy's core logic module for long-stay detection

import csv
import os
import time
from datetime import datetime

import cv2
import psutil

from behavior.analyzer import BehaviorAnalyzer


def run_long_stay_detection(
    input_path,
    output_path,
    model_path,
    device,
    stay_threshold,
    move_threshold,
    conf,
    enable_perf_log,
    draw_fn,
    load_model_fn,
    tracker_init_fn,
    perf_log_fn,
    process_frame_fn,
    update_stay_fn,
    enable_video_display=True,  # critical
    enable_pose=False,  # Add enable_pose argument
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise OSError(f"Cannot open video file: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model = load_model_fn(model_path, device)
    tracker = tracker_init_fn()

    # AOI座標は設定ファイルなどから読み込めるようにするのが望ましい
    aoi_coordinates = [100, 150, width - 100, height - 150]
    behavior_analyzer = BehaviorAnalyzer(aoi_coords=aoi_coordinates)

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            # fallback for environments without H.264
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    perf_log_file = perf_log_fn(enable_perf_log, input_path, model_path, log_type="long_stay")

    if enable_perf_log:
        with open(perf_log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["# Video Properties", f"{width}x{height}", f"{fps}fps"])
            writer.writerow([])

    # --- パフォーマンス計測用の変数を初期化 ---
    time_reading_s = 0.0
    time_cvtColor_s = 0.0
    time_processing_s = 0.0  # det_ms + track_ms
    time_behavior_s = 0.0
    time_stay_update_s = 0.0
    time_drawing_s = 0.0
    time_writing_s = 0.0
    time_showing_s = 0.0
    # ------------------------------------

    stay_info = {}
    frame_idx = 0
    start_time = time.time()
    last_fps_update = start_time
    fps_buffer = []

    try:
        while True:
            t_start = time.perf_counter()
            ret, frame_bgr = cap.read()
            if not ret:
                break
            time_reading_s += time.perf_counter() - t_start

            frame_idx += 1
            current_time = time.time()

            t_start = time.perf_counter()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            time_cvtColor_s += time.perf_counter() - t_start

            # Pass conf and enable_pose to process_frame_fn
            tracks, det_ms, track_ms, num_det, num_track, keypoints = process_frame_fn(
                frame_rgb, model, tracker, conf, enable_pose
            )
            time_processing_s += (det_ms + track_ms) / 1000.0

            t_start = time.perf_counter()
            behavior_notifications = behavior_analyzer.analyze_frame(tracks, current_time)
            time_behavior_s += time.perf_counter() - t_start
            for note in behavior_notifications:
                print(f"Frame {frame_idx}: {note}")

            stay_info, notifications, stay_ms = update_stay_fn(
                tracks, stay_info, current_time, move_threshold, stay_threshold
            )
            time_stay_update_s += stay_ms / 1000.0

            for note in notifications:
                print(f"Frame {frame_idx}: {note}")

            if out or enable_video_display:
                t_start = time.perf_counter()
                # AOIの矩形を描画
                x_min, y_min, x_max, y_max = aoi_coordinates
                cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
                cv2.putText(
                    frame_bgr,
                    "Target Area",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )
                # Pass keypoints and enable_pose to draw_fn
                frame_bgr = draw_fn(
                    frame_bgr,
                    tracks,
                    keypoints=keypoints,
                    enable_pose=enable_pose,
                    show_duration=True,
                    stay_info=stay_info,
                )
                current_fps = 1.0 / (time.time() - last_fps_update) if (time.time() - last_fps_update) > 0 else 0
                fps_buffer.append(current_fps)
                if len(fps_buffer) > 10:
                    fps_buffer.pop(0)
                avg_fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0
                last_fps_update = time.time()

                info_texts = [
                    f"Frame: {frame_idx}/{frame_count} FPS: {avg_fps:.1f}",
                    f"Det: {det_ms:.1f}ms Track: {track_ms:.1f}ms Stay: {stay_ms:.1f}ms",
                    f"Detected: {num_det} Tracked: {num_track}",
                ]
                for i, text in enumerate(info_texts):
                    cv2.putText(frame_bgr, text, (15, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                long_stayers = [
                    f"ID {id}: {info['stay_duration']:.1f}s"
                    for id, info in stay_info.items()
                    if info["stay_duration"] >= stay_threshold
                ]

                if long_stayers:
                    cv2.rectangle(
                        frame_bgr, (width - 210, 10), (width - 10, 30 + 25 * len(long_stayers)), (0, 0, 0), -1
                    )
                    cv2.putText(
                        frame_bgr, "長時間滞在者:", (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
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
                time_drawing_s += time.perf_counter() - t_start

                t_start = time.perf_counter()
                if out:
                    out.write(frame_bgr)
                time_writing_s += time.perf_counter() - t_start

                t_start = time.perf_counter()
                if enable_video_display:
                    cv2.imshow("Long Stay Detection", frame_bgr)
                    if cv2.waitKey(1) == 27:
                        break
                time_showing_s += time.perf_counter() - t_start

            if enable_perf_log and frame_idx % max(1, int(fps)) == 0:
                with open(perf_log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    current_mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                    total_time_ms = det_ms + track_ms + stay_ms
                    writer.writerow(
                        [
                            frame_idx,
                            datetime.now().strftime("%H:%M:%S.%f")[:-3],
                            f"{det_ms:.2f}",
                            f"{track_ms:.2f}",
                            f"{stay_ms:.2f}",
                            f"{total_time_ms:.2f}",
                            num_det,
                            num_track,
                            f"{avg_fps:.2f}",
                            f"{current_mem_usage:.2f}",
                            os.path.basename(model_path),
                            "bytetrack",
                            f"Stay:{stay_threshold}s Move:{move_threshold}px",
                        ]
                    )

            if frame_idx % 30 == 0:
                elapsed_time = time.time() - start_time
                proc_fps = frame_idx / elapsed_time if elapsed_time > 0 else 0
                print(
                    f"進捗: {frame_idx}/{frame_count} ({frame_idx / frame_count * 100:.1f}%) | "
                    f"処理速度: {proc_fps:.2f} FPS | "
                    f"現在時刻: {time.strftime('%H:%M:%S', time.localtime(current_time))}"
                )

    except KeyboardInterrupt:
        print("\n処理がユーザーによって中断されました。")

    finally:
        cap.release()
        if out:
            out.release()
        if enable_video_display:
            cv2.destroyAllWindows()
        print(f"処理完了。出力ファイル: {output_path}, ログ: {perf_log_file}")

        # --- パフォーマンス分析レポートを出力 ---
        total_time_spent = time.time() - start_time
        if frame_idx > 0 and total_time_spent > 0:
            print("\n--- Performance Analysis Report ---")
            print(f"Total frames processed: {frame_idx}")
            print(f"Total processing time: {total_time_spent:.2f} seconds")
            print(f"Average FPS: {frame_idx / total_time_spent:.2f}")
            print("-" * 33)

            total_tracked_time = (
                time_reading_s
                + time_cvtColor_s
                + time_processing_s
                + time_behavior_s
                + time_stay_update_s
                + time_drawing_s
                + time_writing_s
                + time_showing_s
            )
            if total_tracked_time == 0:
                total_tracked_time = 1  # ゼロ除算を避ける

            print(f"Bottleneck Analysis (based on {total_tracked_time:.2f}s of tracked processing time):")
            opencv_total_s = time_reading_s + time_cvtColor_s + time_drawing_s + time_writing_s + time_showing_s
            ai_total_s = time_processing_s
            other_total_s = time_behavior_s + time_stay_update_s

            print(f"  - AI Processing:    {ai_total_s:7.2f}s ({ai_total_s / total_tracked_time * 100:5.1f}%)")
            print(f"  - OpenCV Operations:  {opencv_total_s:7.2f}s ({opencv_total_s / total_tracked_time * 100:5.1f}%)")
            print(f"  - Other (Python Logic):{other_total_s:7.2f}s ({other_total_s / total_tracked_time * 100:5.1f}%)")
            print("-" * 33)

            print("Detailed Breakdown:")
            print(
                f"  - AI (Detection+Track):     {time_processing_s:7.2f}s ({time_processing_s / total_tracked_time * 100:5.1f}%)"
            )
            print(
                f"  - OpenCV: Video Reading     {time_reading_s:7.2f}s ({time_reading_s / total_tracked_time * 100:5.1f}%)"
            )
            print(
                f"  - OpenCV: Color Convert     {time_cvtColor_s:7.2f}s ({time_cvtColor_s / total_tracked_time * 100:5.1f}%)"
            )
            print(
                f"  - OpenCV: Drawing           {time_drawing_s:7.2f}s ({time_drawing_s / total_tracked_time * 100:5.1f}%)"
            )
            print(
                f"  - OpenCV: Video Writing     {time_writing_s:7.2f}s ({time_writing_s / total_tracked_time * 100:5.1f}%)"
            )
            print(
                f"  - OpenCV: Displaying        {time_showing_s:7.2f}s ({time_showing_s / total_tracked_time * 100:5.1f}%)"
            )
            print(
                f"  - Python: Behavior Logic    {time_behavior_s:7.2f}s ({time_behavior_s / total_tracked_time * 100:5.1f}%)"
            )
            print(
                f"  - Python: Stay-Time Logic   {time_stay_update_s:7.2f}s ({time_stay_update_s / total_tracked_time * 100:5.1f}%)"
            )

            untracked_time = total_time_spent - total_tracked_time
            print(f"\nUntracked time (e.g., cv2.waitKey, perf_log I/O): {untracked_time:.2f}s")
            print("--- End of Report ---")
