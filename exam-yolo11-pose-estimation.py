import argparse
import csv
from datetime import datetime
import time
import cv2
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Skeleton structure for YOLO keypoints visualization
SKELETON = [
    (0, 1), (1, 3), (0, 2), (1, 2), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8),
    (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]


def initialize_csv(enable_csv_output: bool):
    """Initialize CSV file if enabled and return the file path and writer."""
    if not enable_csv_output:
        return None, None

    timestamp_output = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"pose_output_{timestamp_output}.csv"
    columns = ["frame_idx", "timestamp", "track_id", "detection_idx"] + \
              [f"kpt_{i}_{axis}" for i in range(17) for axis in ("x", "y", "conf")]

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
        return keypoints, []
    
    # Extract bounding boxes for tracking
    detections = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()

    # Prepare detection data for DeepSORT
    formatted_detections = [([x1, y1, x2 - x1, y2 - y1], float(conf), int(cls))
                            for (x1, y1, x2, y2), conf, cls in zip(detections, confidences, class_ids)]

    # Update tracker
    tracks = tracker.update_tracks(formatted_detections, frame=frame_rgb)
    return keypoints, tracks


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
                row = {"frame_idx": frame_idx, "timestamp": timestamp, "track_id": -1,
                       "detection_idx": detection_idx}

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
                row = {"frame_idx": frame_idx, "timestamp": timestamp, "track_id": track_id,
                       "detection_idx": detection_idx}

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
            cv2.putText(frame_bgr, f"ID {track.track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw the skeleton lines for each person
    for kp_obj in keypoints:
        if kp_obj.shape[0] == 0:
            continue

        kp_array = kp_obj.data[0]
        for (p1, p2) in SKELETON:
            if p1 >= len(kp_array) or p2 >= len(kp_array):
                continue

            x1, y1, c1 = kp_array[p1]
            x2, y2, c2 = kp_array[p2]
            if c1 > 0.5 and c2 > 0.5:
                cv2.line(frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        # Draw circles for each visible keypoint
        for (x, y, c) in kp_array:
            if c > 0.5:
                cv2.circle(frame_bgr, (int(x), int(y)), 3, (255, 0, 0), -1)


def main(input_file: str, output_file: str, 
         enable_csv_output: bool = False, 
         enable_video_display: bool = False,
         enable_tracking: bool = True, 
         model_path: str = "yolov11n-pose.pt"):
    """
    ポーズ推定処理のメイン関数
    Args:
        input_file: 入力動画のパス
        output_file: 出力動画のパス
        enable_csv_output: CSVログを出力するかどうか
        enable_video_display: プレビューを表示するかどうか
        enable_tracking: DeepSORTによる追跡を行うかどうか
        model_path: 使用するYOLOモデルのパス
    """
    # Read input file
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"Error: Unable to open video: {input_file}")
        exit()

    # Prepare output file with OpenCV
    fps, width, height = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Mac互換コーデックを使用（avc1 = H.264）
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Video properties を出力
    print(f"ビデオ情報: {width}x{height} @ {fps}fps")
    print(f"出力コーデック: H.264 (avc1) - Mac互換")

    # Initialize YOLO Pose model
    print(f"YOLOモデル: {model_path}")
    model = YOLO(model_path)
    
    # Initialize DeepSORT if tracking is enabled
    tracker = None
    if enable_tracking:
        print("DeepSORTトラッキング: 有効")
        # トラッキングパラメータ最適化: max_ageを短くして速度向上
        tracker = DeepSort(max_age=15, n_init=2, nn_budget=50)
    else:
        print("DeepSORTトラッキング: 無効 (処理速度優先)")
        
    csv_file, columns = initialize_csv(enable_csv_output)

    frame_idx = 0
    start_time = time.time()
    skip_frame = 0  # 処理負荷を下げるためのフレームスキップカウンタ
    
    try:
        print("処理を開始します...")
        while True:
            # Read a raw BGR frame from the video
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
                continue
                
            timestamp = time.time()

            # Convert to RGB once for YOLO
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            keypoints, tracks = process_frame(frame_idx, frame_rgb, model, tracker, enable_tracking)
            
            if enable_csv_output:
                write_to_csv(csv_file, columns, frame_idx, timestamp, tracks, keypoints)
                
            draw_visuals(frame_bgr, tracks, keypoints, enable_tracking)
            
            # トラッキング有効時は一部フレームをスキップして処理負荷を軽減
            if enable_tracking and frame_idx % 60 == 0:
                skip_frame = 2  # 2フレームスキップ
            
            # 処理の進捗状況を表示
            if frame_idx % 30 == 0 or frame_idx == 1:
                elapsed = time.time() - start_time
                current_fps = frame_idx / elapsed if elapsed > 0 else 0
                print(f"処理中: {frame_idx}フレーム完了 (現在の処理速度: {current_fps:.1f}fps)")
                
                # 進捗情報をフレームに表示
                cv2.putText(frame_bgr, f"FPS: {current_fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Save the processed frame to our output file
            out.write(frame_bgr)

            # Optionally, also display if you want a live window (can be omitted if running headless)
            if enable_video_display:
                cv2.imshow("YOLO Pose", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        print("\n処理が中断されました")
    finally:
        # 処理の完了
        cap.release()
        out.release()
        if enable_video_display:
            cv2.destroyAllWindows()
        
        # 性能情報の表示
        total_time = time.time() - start_time
        print(f"\n処理完了: {frame_idx}フレーム ({total_time:.2f}秒)")
        print(f"平均処理速度: {frame_idx / total_time:.2f}fps")
        print(f"出力動画: {output_file}")
        if enable_csv_output:
            print(f"ポーズCSV: {csv_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO Pose Estimation Tool")
    parser.add_argument("--input-mp4", required=True, help="入力動画ファイルのパス")
    parser.add_argument("--output-mp4", required=True, help="出力動画ファイルのパス")
    parser.add_argument("--enable-csv-output", action='store_true', help="ポーズデータをCSVに出力")
    parser.add_argument("--enable-video-display", action='store_true', help="処理中のプレビューを表示")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="YOLOモデルのパス (デフォルト: yolov8n-pose.pt)")
    parser.add_argument("--disable-tracking", action='store_true', help="DeepSORTトラッキングを無効化(処理速度向上)")
    args = parser.parse_args()
    
    # 出力ディレクトリの確保
    os.makedirs(os.path.dirname(args.output_mp4), exist_ok=True)
    
    main(args.input_mp4, args.output_mp4, 
         args.enable_csv_output, 
         args.enable_video_display,
         not args.disable_tracking,
         args.model)
