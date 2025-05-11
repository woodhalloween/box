import argparse
import csv
from datetime import datetime
import time
import cv2
import os
import platform
import psutil
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from boxmot.trackers.bytetrack.bytetrack import ByteTrack

# Skeleton structure for YOLO keypoints visualization (kept for potential future use, but not primary focus)
SKELETON = [
    (0, 1), (1, 3), (0, 2), (1, 2), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8),
    (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

def get_system_info():
    """システム情報を取得する関数"""
    info = {
        'os': platform.system(),
        'os_version': platform.version(),
        'python_version': platform.python_version(),
        'cpu': platform.processor(),
        'cpu_cores': psutil.cpu_count(logical=False),
        'cpu_threads': psutil.cpu_count(logical=True),
        'ram_total': round(psutil.virtual_memory().total / (1024**3), 2),  # GB単位
    }
    return info

# Simplified CSV initialization for performance logs only
def initialize_perf_log(enable_perf_log: bool, input_file: str, model_path: str):
    """Initialize performance log CSV if enabled."""
    if not enable_perf_log:
        return None, None, None

    system_info = get_system_info()
    timestamp_log = datetime.now().strftime("%Y%m%d_%H%M%S")
    perf_log_file = f'log_{Path(input_file).stem}_long_stay_{Path(model_path).stem}_{timestamp_log}.csv'
    perf_columns = ['Frame', 'Time', 'Detection_Time_ms', 'Tracking_Time_ms', 'Stay_Check_Time_ms', 'Total_Time_ms',
                    'Objects_Detected', 'Objects_Tracked', 'FPS', 'Memory_MB', 'Model', 'Tracker', 'Notes']

    try:
        perf_log_f = open(perf_log_file, 'w', newline='')
        perf_log_writer = csv.writer(perf_log_f)
        perf_log_writer.writerow(['# System Information'])
        perf_log_writer.writerow(['# OS', system_info['os'], system_info['os_version']])
        perf_log_writer.writerow(['# Python', system_info['python_version']])
        perf_log_writer.writerow(['# CPU', system_info['cpu'], f'{system_info["cpu_cores"]} cores, {system_info["cpu_threads"]} threads'])
        perf_log_writer.writerow(['# RAM', f'{system_info["ram_total"]} GB'])
        perf_log_writer.writerow([])
        perf_log_writer.writerow(['# YOLO Model', model_path]) # Simplified model info
        perf_log_writer.writerow(['# Tracker', 'bytetrack'])
        perf_log_writer.writerow([])
        perf_log_writer.writerow(['# Video', input_file])
        # Video properties (width, height, fps) will be added later in main
        perf_log_writer.writerow([])
        perf_log_writer.writerow(perf_columns)
        print(f"パフォーマンスログ: 有効 ({perf_log_file})")
        return perf_log_file, perf_log_f, perf_log_writer
    except IOError as e:
        print(f"エラー: パフォーマンスログファイル '{perf_log_file}' を開けません: {e}")
        return None, None, None


def process_frame_for_tracking(frame_rgb, model, tracker):
    """Process a single frame for detection and tracking, returning tracks."""
    detection_start_time = time.time()
    results = model.predict(frame_rgb, verbose=False, classes=[0]) # Predict only class 0 (person)
    detection_time_ms = (time.time() - detection_start_time) * 1000
    result = results[0]

    boxes = result.boxes
    # 検出結果をboxmot用の形式に変換
    dets_for_tracker = []
    
    if len(boxes) > 0:
        for i in range(len(boxes)):
            box = boxes[i].xyxy.cpu().numpy()[0]  # [x1, y1, x2, y2]
            conf = float(boxes[i].conf.cpu().numpy()[0])
            cls = int(boxes[i].cls.cpu().numpy()[0])
            
            # [x1, y1, x2, y2, conf, class]の形式
            dets_for_tracker.append([box[0], box[1], box[2], box[3], conf, cls])
        
        dets_for_tracker = np.array(dets_for_tracker)
    else:
        dets_for_tracker = np.empty((0, 6))
    
    tracking_start_time = time.time()
    # ByteTrackのupdateメソッドにnumpy配列を渡す - 結果もnumpy配列
    tracks = tracker.update(dets_for_tracker, frame_rgb)
    tracking_time_ms = (time.time() - tracking_start_time) * 1000

    return tracks, detection_time_ms, tracking_time_ms, len(boxes), len(tracks)


def update_stay_times(tracks, stay_info, current_time, move_threshold_px, stay_threshold_sec):
    """Update stay duration for each tracked object and check for long stays."""
    stay_check_start = time.time()
    current_track_ids = set()
    notifications = []

    for track in tracks:
        # トラックの形式: [x1, y1, x2, y2, track_id, conf, cls_id, ...]
        x1, y1, x2, y2, track_id = track[:5]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        track_id = int(track_id)
        
        current_track_ids.add(track_id)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        current_pos = (center_x, center_y)
        
        # 人物の高さを計算（遠近感の基準として使用）
        person_height = y2 - y1
        
        # 基準高さの設定（画面の半分の高さを基準と仮定）
        reference_height = 200  # この値は環境によって調整可能
        
        # 正規化係数の計算（小さい人物ほど係数が大きくなる）
        normalization_factor = reference_height / max(person_height, 1)  # ゼロ除算防止

        if track_id not in stay_info:
            # New track detected
            stay_info[track_id] = {
                'last_pos': current_pos,
                'last_time': current_time,
                'stay_duration': 0.0,
                'notified': False,
                'person_height': person_height,  # 人物の高さを記録
                'last_normalization_factor': normalization_factor  # 正規化係数を記録
            }
        else:
            # Existing track update
            last_pos = stay_info[track_id]['last_pos']
            last_time = stay_info[track_id]['last_time']
            
            # 記録されている正規化係数と現在の正規化係数の平均を使用
            last_normalization_factor = stay_info[track_id]['last_normalization_factor']
            avg_normalization_factor = (last_normalization_factor + normalization_factor) / 2
            
            # ピクセル単位の距離を計算
            pixel_distance = ((current_pos[0] - last_pos[0])**2 + (current_pos[1] - last_pos[1])**2)**0.5
            
            # 人物の大きさで正規化した距離を計算
            normalized_distance = pixel_distance * avg_normalization_factor
            
            time_diff = current_time - last_time

            # ログ出力（デバッグ用、必要に応じて有効化）
            # print(f"ID {track_id}: pixel_dist={pixel_distance:.1f}, norm_dist={normalized_distance:.1f}, height={person_height}, factor={avg_normalization_factor:.2f}")

            if normalized_distance < move_threshold_px:
                # Stayed in the same area (using normalized distance)
                stay_info[track_id]['stay_duration'] += time_diff
            else:
                # Moved significantly based on normalized distance
                stay_info[track_id]['stay_duration'] = 0.0

            # Update position, time, and person size regardless of movement
            stay_info[track_id]['last_pos'] = current_pos
            stay_info[track_id]['last_time'] = current_time
            stay_info[track_id]['person_height'] = person_height
            stay_info[track_id]['last_normalization_factor'] = normalization_factor

            # Check for notification
            if stay_info[track_id]['stay_duration'] >= stay_threshold_sec and not stay_info[track_id]['notified']:
                notifications.append(f"通知: ID {track_id} が座標 ({int(center_x)}, {int(center_y)}) 付近に {stay_threshold_sec}秒 以上滞在中。人物高さ: {person_height}px")
                stay_info[track_id]['notified'] = True # Mark as notified

    # Clean up lost tracks
    lost_track_ids = set(stay_info.keys()) - current_track_ids
    for track_id in lost_track_ids:
        del stay_info[track_id]

    stay_check_time_ms = (time.time() - stay_check_start) * 1000
    return stay_info, notifications, stay_check_time_ms


def draw_tracking_info(frame_bgr, tracks, stay_info):
    """Draw bounding boxes, IDs, and stay duration on the frame."""
    for track in tracks:
        # トラックの形式: [x1, y1, x2, y2, track_id, conf, cls_id, ...]
        x1, y1, x2, y2, track_id = track[:5]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        track_id = int(track_id)

        # 人物の高さを計算
        person_height = y2 - y1
        
        # Draw bounding box
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display ID, stay duration, and person height
        duration = stay_info.get(track_id, {}).get('stay_duration', 0.0)
        label = f"ID {track_id} ({duration:.1f}s, h:{person_height}px)"
        cv2.putText(frame_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main(input_file: str, output_file: str,
         model_path: str = "yolov11n-pose.pt",
         enable_perf_log: bool = False,
         enable_video_display: bool = False,
         device: str = '',
         stay_threshold_sec: float = 4.0,
         move_threshold_px: float = 20.0,
         conf: float = 0.3):
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
        move_threshold_px: 同一場所とみなす移動距離の閾値 (ピクセル、正規化後の値)
        conf: ByteTrackのtrack_thresh値 (検出信頼度閾値)
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
    fps, width, height = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1') # Mac互換コーデック
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    print(f"ビデオ情報: {width}x{height} @ {fps:.2f}fps")
    print(f"出力コーデック: H.264 (avc1)")
    print(f"滞在検知閾値: {stay_threshold_sec} 秒")
    print(f"移動検知閾値: {move_threshold_px} ピクセル (正規化後)")
    print(f"※遠近を考慮し、人物の高さで正規化した座標変化検知を使用")

    # Initialize YOLO model (use a standard detection model if pose isn't needed)
    print(f"YOLOモデル: {model_path}")
    try:
        if device.isdigit():
            yolo_device = int(device)
        elif device.lower() == 'cpu':
            yolo_device = 'cpu'
        else:
            yolo_device = None # Auto-select
            if device: print(f"警告: 不正なデバイス指定 '{device}' です。自動選択を使用します。")
    except ValueError:
         print(f"警告: 不正なデバイス指定 '{device}' です。自動選択を使用します。")
         yolo_device = None

    model = YOLO(model_path).to(yolo_device)
    if model.task == 'pose':
        print("情報: ポーズ推定モデルが指定されましたが、このスクリプトでは人物検出のみ行います。")

    # Initialize ByteTrack tracker
    print("ByteTrackトラッキング: 有効")
    tracker = ByteTrack(
        track_thresh=conf,
        track_buffer=30,
        match_thresh=0.8
    )

    # Initialize Performance Log
    perf_log_file, perf_log_f, perf_log_writer = initialize_perf_log(enable_perf_log, input_file, model_path)
    if enable_perf_log and perf_log_writer:
        # Add video info to perf log header now that we have it
        perf_log_writer.writerow(['# Resolution', f'{width}x{height}', 'FPS', f'{fps:.2f}'])
        perf_log_writer.writerow(['# Stay Threshold (s)', stay_threshold_sec])
        perf_log_writer.writerow(['# Move Threshold (px)', move_threshold_px])
        perf_log_writer.writerow([]) # Add separator before data rows

    frame_idx = 0
    loop_start_time = time.time()
    detection_times = []
    tracking_times = []
    stay_check_times = []
    memory_usages = []  # メモリ使用量を記録するリスト
    max_memory_usage = 0  # 最大メモリ使用量
    stay_info = {} # Dictionary to store stay duration per track_id

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
            tracks, detection_time_ms, tracking_time_ms, objects_detected, objects_tracked = process_frame_for_tracking(frame_rgb, model, tracker)
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

            # 現在のメモリ使用量を測定（MB単位）
            current_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            memory_usages.append(current_memory_mb)
            max_memory_usage = max(max_memory_usage, current_memory_mb)

            # Write performance log entry
            if enable_perf_log and perf_log_writer:
                frame_end_time = time.time()
                total_frame_time_ms = (frame_end_time - frame_start_time) * 1000
                elapsed_loop_time = time.time() - loop_start_time
                current_overall_fps = frame_idx / elapsed_loop_time if elapsed_loop_time > 0 else 0

                perf_log_writer.writerow([
                    frame_idx,                                  # フレーム番号
                    f'{elapsed_loop_time:.3f}',                 # 経過時間（秒）
                    f'{detection_time_ms:.1f}',                 # 検出時間（ms）
                    f'{tracking_time_ms:.1f}',                  # 追跡時間（ms）
                    f'{stay_check_time_ms:.1f}',                # 滞在チェック時間（ms）
                    f'{total_frame_time_ms:.1f}',               # 合計処理時間（ms）
                    objects_detected,                           # 検出されたオブジェクト数
                    objects_tracked,                            # 追跡されたオブジェクト数
                    f'{current_overall_fps:.1f}',               # 現在のFPS
                    f'{current_memory_mb:.1f}',                 # 現在のメモリ使用量（MB）
                    Path(model_path).stem,                      # モデル名
                    'bytetrack',                                # トラッカー名
                    f'{len(notifications)} notifications' if notifications else ''  # メモ欄
                ])

            # Display progress
            if frame_idx % 30 == 0 or frame_idx == 1:
                elapsed = time.time() - loop_start_time
                current_fps = frame_idx / elapsed if elapsed > 0 else 0
                print(f"処理中: {frame_idx}フレーム完了 (現在の処理速度: {current_fps:.1f}fps, メモリ: {current_memory_mb:.1f}MB)")

            # Save output frame
            out.write(frame_bgr)

            # Display live window if enabled
            if enable_video_display:
                cv2.imshow("Long Stay Detection", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
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
        avg_stay_check_time = sum(stay_check_times) / len(stay_check_times) if stay_check_times else 0
        avg_memory_usage = sum(memory_usages) / len(memory_usages) if memory_usages else 0

        print(f"--- 処理結果サマリ ---")
        print(f"合計処理時間: {total_time:.2f} 秒")
        print(f"処理フレーム数: {frame_idx}")
        print(f"平均処理速度 (FPS): {avg_fps:.2f}")
        print(f"平均検出時間: {avg_detection_time:.1f} ms")
        print(f"平均トラッキング時間: {avg_tracking_time:.1f} ms")
        print(f"平均滞在チェック時間: {avg_stay_check_time:.1f} ms")
        print(f"平均合計処理時間 (フレームあたり): {avg_detection_time + avg_tracking_time + avg_stay_check_time:.1f} ms")
        print(f"平均メモリ使用量: {avg_memory_usage:.1f} MB")
        print(f"最大メモリ使用量: {max_memory_usage:.1f} MB")
        print(f"出力動画: {output_file}")
        if enable_perf_log and perf_log_file:
            print(f"パフォーマンスログ: {perf_log_file}")
            if perf_log_f:
                print(f"パフォーマンスログファイル '{perf_log_file}' を閉じています。")
                perf_log_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect long stays of tracked objects (people) in a video.")
    parser.add_argument("--input-mp4", required=True, help="入力動画ファイルのパス")
    parser.add_argument("--output-mp4", required=True, help="出力動画ファイルのパス")
    parser.add_argument("--model", default="models/yolov8n.pt", help="YOLOモデルのパス (人物検出用, デフォルト: models/yolov8n.pt)")
    parser.add_argument("--enable-perf-log", action='store_true', help="パフォーマンスログをCSVに出力")
    parser.add_argument("--device", type=str, default='', help="使用するデバイス (例: cpu, 0, '')")
    parser.add_argument("--enable-video-display", action='store_true', help="処理中のプレビューを表示")
    parser.add_argument("--stay-threshold", type=float, default=6.0, help="滞在通知を行う時間の閾値 (秒, デフォルト: 6.0)")
    parser.add_argument("--move-threshold", type=float, default=20.0, help="同一場所とみなす移動距離の閾値 (ピクセル, デフォルト: 20.0)")
    parser.add_argument("--conf", type=float, default=0.3, help="ByteTrackのtrack_thresh (検出信頼度閾値, デフォルト: 0.3)")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_mp4), exist_ok=True)

    main(args.input_mp4, args.output_mp4,
         args.model,
         args.enable_perf_log,
         args.enable_video_display,
         args.device,
         args.stay_threshold,
         args.move_threshold,
         args.conf) 