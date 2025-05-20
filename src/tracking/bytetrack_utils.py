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
            stay_info[track_id] = {
                "start_time": current_time,
                "last_pos": current_pos,
                "stay_duration": 0,
                "notified": False,
                "person_height": person_height,
                "history": [(current_time, current_pos, person_height)],
            }
        else:
            last_time, last_pos, _ = stay_info[track_id]["history"][-1]
            time_diff = current_time - last_time

            # 移動距離の計算 (正規化された距離)
            # dist_moved = np.sqrt(
            #     (current_pos[0] - last_pos[0]) ** 2 + (current_pos[1] - last_pos[1]) ** 2
            # )
            # dist_moved_normalized = dist_moved * normalization_factor

            # 過去1秒間の平均位置と比較して移動判定（より安定した判定のため）
            history_seconds = 1.0  # 過去何秒間の履歴を参照するか
            relevant_history = [
                p
                for t, p, _ in stay_info[track_id]["history"]
                if current_time - t <= history_seconds
            ]
            if not relevant_history:
                relevant_history = [last_pos]  # 履歴が足りない場合は最後の位置を使用

            avg_prev_x = np.mean([p[0] for p in relevant_history])
            avg_prev_y = np.mean([p[1] for p in relevant_history])
            avg_prev_pos = (avg_prev_x, avg_prev_y)

            dist_moved_from_avg = np.sqrt(
                (current_pos[0] - avg_prev_pos[0]) ** 2 + (current_pos[1] - avg_prev_pos[1]) ** 2
            )
            dist_moved_normalized = dist_moved_from_avg * normalization_factor

            # 閾値も正規化
            # dynamic_move_threshold = move_threshold_px / normalization_factor
            # print(f"ID:{track_id}, dist_norm:{dist_moved_normalized:.2f}, thresh_norm:{dynamic_move_threshold:.2f}, factor:{normalization_factor:.2f}, height:{person_height}")

            if dist_moved_normalized < move_threshold_px:  # 正規化後の閾値と比較
                stay_info[track_id]["stay_duration"] += time_diff
            else:
                # 移動があった場合は滞在時間をリセットし、開始時刻と位置を更新
                stay_info[track_id]["start_time"] = current_time
                stay_info[track_id]["stay_duration"] = 0
                stay_info[track_id]["notified"] = False  # 通知状態もリセット

            stay_info[track_id]["last_pos"] = current_pos
            stay_info[track_id]["person_height"] = person_height  # 高さを更新
            stay_info[track_id]["history"].append((current_time, current_pos, person_height))
            # 古い履歴を削除（例: 過去5秒分のみ保持）
            max_history_duration = 5.0
            stay_info[track_id]["history"] = [
                h
                for h in stay_info[track_id]["history"]
                if current_time - h[0] <= max_history_duration
            ]

            # 長時間滞在の通知チェック
            if (
                stay_info[track_id]["stay_duration"] >= stay_threshold_sec
                and not stay_info[track_id]["notified"]
            ):
                notifications.append(
                    {
                        "id": track_id,
                        "duration": stay_info[track_id]["stay_duration"],
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                stay_info[track_id]["notified"] = True

    # 追跡が途切れたIDをstay_infoから削除
    # （必要に応じて、すぐ消さずに一定時間後に消すなどの工夫も可能）
    lost_ids = set(stay_info.keys()) - current_track_ids
    for lost_id in lost_ids:
        del stay_info[lost_id]

    stay_check_time_ms = (time.time() - stay_check_start) * 1000
    return stay_info, notifications, stay_check_time_ms


def initialize_bytetrack():
    """ByteTrackトラッカーを初期化"""
    # track_thresh: 追跡を開始するための信頼度の閾値。デフォルトは0.5。
    # track_buffer: 追跡が途切れた後、IDを保持するフレーム数。デフォルトは30。
    # match_thresh: 追跡と検出を関連付けるためのIoUの閾値。デフォルトは0.8。
    # frame_rate: ビデオのフレームレート。デフォルトは30。
    return ByteTrack(track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30)


def load_yolo_model(model_path, device=""):
    """YOLOモデルをロード"""
    try:
        model = YOLO(model_path)
        if device:  # 'cpu' or 'mps' or '0' (for CUDA GPU 0)
            model.to(device)
        print(f"YOLOモデル '{model_path}' を正常にロードしました。")
        return model
    except Exception as e:
        print(f"エラー: YOLOモデル '{model_path}' のロードに失敗しました: {e}")
        return None
