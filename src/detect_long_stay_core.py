# detect_long_stay_core.py
# 
# Daisy's core logic module for long-stay detection

import csv
import os
import time
from datetime import datetime

import cv2
import psutil


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
    enable_pose=False, # Add enable_pose argument
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

    stay_info = {}
    frame_idx = 0
    start_time = time.time()
    last_fps_update = start_time
    fps_buffer = []

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_idx += 1
            current_time = time.time()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Pass conf and enable_pose to process_frame_fn
            tracks, det_ms, track_ms, num_det, num_track, keypoints = process_frame_fn(
                frame_rgb, model, tracker, conf, enable_pose
            )

            stay_info, notifications, stay_ms = update_stay_fn(
                tracks, stay_info, current_time, move_threshold, stay_threshold
            )

            for note in notifications:
                print(f"Frame {frame_idx}: {note}")

            if out or enable_video_display:
                # Pass keypoints and enable_pose to draw_fn
                frame_bgr = draw_fn(frame_bgr, tracks, keypoints=keypoints, enable_pose=enable_pose, show_duration=True, stay_info=stay_info)
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

                if out:
                    out.write(frame_bgr)
                if enable_video_display:
                    cv2.imshow("Long Stay Detection", frame_bgr)
                    if cv2.waitKey(1) == 27:
                        break

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