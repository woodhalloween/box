import csv
import platform
from datetime import datetime
from pathlib import Path
import numpy as np
import psutil

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

    system_info = get_system_info() # This call should work as get_system_info is in the same file.
    timestamp_log = datetime.now().strftime("%Y%m%d_%H%M%S")

    input_stem = Path(input_file).stem
    model_stem = Path(model_path).stem

    # outputディレクトリの作成
    output_dir = Path("output/logs")
    output_dir.mkdir(parents=True, exist_ok=True)
    perf_log_file = output_dir / f"log_{input_stem}_{log_type}_{model_stem}_{timestamp_log}.csv"

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

    perf_log_f = None  # Initialize perf_log_f
    try:
        # with open(perf_log_file, "w", newline="") as perf_log_f:
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
        perf_log_writer.writerow(["# YOLO Model", model_path])
        perf_log_writer.writerow(["# Tracker", "bytetrack"]) # Note: This is hardcoded, might need review if tracker changes
        perf_log_writer.writerow([])
        perf_log_writer.writerow(["# Video", input_file])
        perf_log_writer.writerow([])
        perf_log_writer.writerow(perf_columns)
        print(f"パフォーマンスログ: 有効 ({perf_log_file})")
        return str(perf_log_file), perf_log_f, perf_log_writer  # Return path as string
    except OSError as e:
        print(f"エラー: パフォーマンスログファイル '{perf_log_file}' を開けません: {e}")
        if perf_log_f:
            perf_log_f.close()  # Close if an error occurs after opening
        return None, None, None
