import csv
import platform
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import psutil
from boxmot.trackers.bytetrack.bytetrack import ByteTrack
from ultralytics import YOLO


# データ保存用の構造体
class DetectionResults:
    def __init__(self):
        self.frame_count = 0
        self.detection_times = []  # 検出時間（ms）
        self.tracking_times = []  # 追跡時間（ms）
        self.fps_values = []  # FPS値
        self.objects_detected = []  # 検出されたオブジェクト数
        self.objects_tracked = []  # 追跡されたオブジェクト数
        self.memory_usages = []  # メモリ使用量（MB）
        self.detection_confidence = []  # 検出信頼度の平均

    def add_frame_result(
        self,
        detection_time,
        tracking_time,
        fps,
        objects_detected,
        objects_tracked,
        memory_usage,
        detection_conf=None,
    ):
        self.frame_count += 1
        self.detection_times.append(detection_time)
        self.tracking_times.append(tracking_time)
        self.fps_values.append(fps)
        self.objects_detected.append(objects_detected)
        self.objects_tracked.append(objects_tracked)
        self.memory_usages.append(memory_usage)
        if detection_conf is not None:
            self.detection_confidence.append(detection_conf)

    def get_summary(self):
        """結果のサマリを辞書で返す"""
        return {
            "frame_count": self.frame_count,
            "avg_detection_time": np.mean(self.detection_times) if self.detection_times else 0,
            "avg_tracking_time": np.mean(self.tracking_times) if self.tracking_times else 0,
            "avg_fps": np.mean(self.fps_values) if self.fps_values else 0,
            "avg_objects_detected": np.mean(self.objects_detected) if self.objects_detected else 0,
            "avg_objects_tracked": np.mean(self.objects_tracked) if self.objects_tracked else 0,
            "avg_memory_usage": np.mean(self.memory_usages) if self.memory_usages else 0,
            "avg_detection_conf": np.mean(self.detection_confidence)
            if self.detection_confidence
            else 0,
            "max_objects_detected": max(self.objects_detected) if self.objects_detected else 0,
            "max_objects_tracked": max(self.objects_tracked) if self.objects_tracked else 0,
        }


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


def process_frame_for_tracking(frame_rgb, model, tracker):
    """1フレームを処理して検出と追跡を行う"""
    detection_start_time = time.time()
    # 人物クラス(0)のみを検出
    results = model.predict(frame_rgb, verbose=False, classes=[0])
    detection_time_ms = (time.time() - detection_start_time) * 1000
    result = results[0]

    boxes = result.boxes
    # 検出結果をboxmot用の形式に変換
    dets_for_tracker = []
    avg_conf = 0.0

    if len(boxes) > 0:
        total_conf = 0.0
        for i in range(len(boxes)):
            box = boxes[i].xyxy.cpu().numpy()[0]  # [x1, y1, x2, y2]
            conf = float(boxes[i].conf.cpu().numpy()[0])
            cls = int(boxes[i].cls.cpu().numpy()[0])
            total_conf += conf

            # [x1, y1, x2, y2, conf, class]の形式
            dets_for_tracker.append([box[0], box[1], box[2], box[3], conf, cls])

        dets_for_tracker = np.array(dets_for_tracker)
        avg_conf = total_conf / len(boxes) if len(boxes) > 0 else 0
    else:
        dets_for_tracker = np.empty((0, 6))

    tracking_start_time = time.time()
    # ByteTrackのupdateメソッドにnumpy配列を渡す
    tracks = tracker.update(dets_for_tracker, frame_rgb)
    tracking_time_ms = (time.time() - tracking_start_time) * 1000

    return tracks, detection_time_ms, tracking_time_ms, len(boxes), len(tracks), avg_conf


def resize_frame(frame, target_width, target_height):
    """フレームを指定サイズにリサイズ"""
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def draw_tracking_info(frame, tracks, show_duration=False, stay_info=None):
    """トラッキング情報を描画

    Args:
        frame: フレーム画像（BGRフォーマット）
        tracks: 追跡情報 [x1, y1, x2, y2, track_id, ...]
        show_duration: 滞在時間を表示するかどうか (default: False)
        stay_info: 滞在情報の辞書 (show_durationがTrueの場合必要)
    """
    for track in tracks:
        # トラックの形式: [x1, y1, x2, y2, track_id, conf, cls_id, ...]
        x1, y1, x2, y2, track_id = track[:5]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        track_id = int(track_id)

        # バウンディングボックスの描画
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # IDの表示
        if show_duration and stay_info and track_id in stay_info:
            stay_duration = stay_info[track_id]["stay_duration"]
            person_height = stay_info[track_id]["person_height"]
            label = f"ID:{track_id} 滞在:{stay_duration:.1f}s 高さ:{person_height}px"

            # 長時間滞在の場合は色を変える
            if stay_duration >= 4.0:  # デフォルトのしきい値
                color = (0, 0, 255)  # 赤色
            else:
                color = (0, 255, 0)  # 緑色

            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


def initialize_perf_log(enable_perf_log, input_file, model_path, log_type="generic"):
    """パフォーマンスログの初期化

    Args:
        enable_perf_log: ログを有効にするかどうか
        input_file: 入力動画ファイル名
        model_path: モデルファイルパス
        log_type: ログタイプ ("generic", "long_stay", "fps", "resolution")
    """
    if not enable_perf_log:
        return None, None, None

    system_info = get_system_info()
    timestamp_log = datetime.now().strftime("%Y%m%d_%H%M%S")

    input_stem = Path(input_file).stem
    model_stem = Path(model_path).stem

    perf_log_file = f"log_{input_stem}_{log_type}_{model_stem}_{timestamp_log}.csv"

    perf_columns = [
        "Frame",
        "Time",
        "Detection_Time_ms",
        "Tracking_Time_ms",
        "Total_Time_ms",
        "Objects_Detected",
        "Objects_Tracked",
        "FPS",
        "Memory_MB",
        "Model",
        "Tracker",
        "Notes",
    ]

    # long_stay用に追加カラム
    if log_type == "long_stay":
        perf_columns.insert(4, "Stay_Check_Time_ms")

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
            perf_log_writer.writerow(["# YOLO Model", model_path])
            perf_log_writer.writerow(["# Tracker", "bytetrack"])
            perf_log_writer.writerow([])
            perf_log_writer.writerow(["# Video", input_file])
            perf_log_writer.writerow([])
            perf_log_writer.writerow(perf_columns)
            print(f"パフォーマンスログ: 有効 ({perf_log_file})")
            return perf_log_file, perf_log_f, perf_log_writer
    except OSError as e:
        print(f"エラー: パフォーマンスログファイル '{perf_log_file}' を開けません: {e}")
        return None, None, None


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
                "last_pos": current_pos,
                "last_time": current_time,
                "stay_duration": 0.0,
                "notified": False,
                "person_height": person_height,  # 人物の高さを記録
                "last_normalization_factor": normalization_factor,  # 正規化係数を記録
            }
        else:
            # Existing track update
            last_pos = stay_info[track_id]["last_pos"]
            last_time = stay_info[track_id]["last_time"]

            # 記録されている正規化係数と現在の正規化係数の平均を使用
            last_normalization_factor = stay_info[track_id]["last_normalization_factor"]
            avg_normalization_factor = (last_normalization_factor + normalization_factor) / 2

            # ピクセル単位の距離を計算
            pixel_distance = (
                (current_pos[0] - last_pos[0]) ** 2 + (current_pos[1] - last_pos[1]) ** 2
            ) ** 0.5

            # 人物の大きさで正規化した距離を計算
            normalized_distance = pixel_distance * avg_normalization_factor

            time_diff = current_time - last_time

            if normalized_distance < move_threshold_px:
                # Stayed in the same area (using normalized distance)
                stay_info[track_id]["stay_duration"] += time_diff
            else:
                # Moved significantly based on normalized distance
                stay_info[track_id]["stay_duration"] = 0.0

            # Update position, time, and person size regardless of movement
            stay_info[track_id]["last_pos"] = current_pos
            stay_info[track_id]["last_time"] = current_time
            stay_info[track_id]["person_height"] = person_height
            stay_info[track_id]["last_normalization_factor"] = normalization_factor

            # Check for notification
            if (
                stay_info[track_id]["stay_duration"] >= stay_threshold_sec
                and not stay_info[track_id]["notified"]
            ):
                # 通知メッセージを複数行に分割
                notification = (
                    f"通知: ID {track_id} が座標 ({int(center_x)}, {int(center_y)}) "
                    f"付近に {stay_threshold_sec}秒 以上滞在中。人物高さ: {person_height}px"
                )
                notifications.append(notification)
                stay_info[track_id]["notified"] = True  # Mark as notified

    # Clean up lost tracks
    lost_track_ids = set(stay_info.keys()) - current_track_ids
    for track_id in lost_track_ids:
        del stay_info[track_id]

    stay_check_time_ms = (time.time() - stay_check_start) * 1000
    return stay_info, notifications, stay_check_time_ms


def initialize_bytetrack():
    """ByteTrackトラッカーの初期化"""
    return ByteTrack(
        fps=30, delta_t=3, detection_threshold=0.35, track_threshold=0.2, match_threshold=0.8
    )


def load_yolo_model(model_path, device=""):
    """YOLOモデルの読み込み"""
    if device == "":
        model = YOLO(model_path)
    else:
        model = YOLO(model_path, device=device)
    return model
