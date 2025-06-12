import glob
from pathlib import Path

import pandas as pd

# パフォーマンスログの列名 (ヘッダーがないため、ここで定義)
PERF_LOG_COLUMNS = [
    "frame",
    "time",
    "detect_ms",
    "track_ms",
    "stay_ms",
    "total_ms",
    "detected",
    "tracked",
    "fps",
    "mem_mb",
    "avg_conf",
    "model_name",
    "tracker",
    "params",
]

# イベントログの列名
EVENT_LOG_COLUMNS = [
    "frame_id",
    "timestamp",
    "track_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "confidence",
    "class_id",
    "stay_duration_s",
]


def analyze_performance_log(log_path: Path) -> dict:
    """単一のパフォーマンスログファイルを分析し、集計結果を返す"""
    try:
        # ヘッダーなし、コメント行をスキップして読み込む
        df = pd.read_csv(log_path, header=None, comment="#", names=PERF_LOG_COLUMNS, skipinitialspace=True)
        if df.empty:
            return {}

        # 数値に変換できない可能性のある列を処理
        for col in ["detect_ms", "track_ms", "stay_ms", "fps", "mem_mb"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return {
            "avg_detect_ms": df["detect_ms"].mean(),
            "avg_track_ms": df["track_ms"].mean(),
            "avg_fps": df["fps"].mean(),
            "peak_mem_mb": df["mem_mb"].max(),
        }
    except Exception as e:
        print(f"  - 個別ログの分析エラー ({log_path.name}): {e}")
        return {}


def analyze_event_log(log_path: Path) -> dict:
    """単一のイベントログファイルを分析し、集計結果を返す"""
    try:
        df = pd.read_csv(log_path, header=0, names=EVENT_LOG_COLUMNS, skipinitialspace=True)
        if df.empty:
            return {}

        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

        return {
            "avg_confidence": df["confidence"].mean(),
        }
    except Exception as e:
        print(f"  - イベントログの分析エラー ({log_path.name}): {e}")
        return {}


def find_log(video_name: str, model_name: str, log_type: str) -> Path | None:
    """動画名とモデル名に一致するログファイルを見つける"""
    base_name = Path(video_name).stem

    # モデル名に応じて検索タグを決定
    if model_name == "YOLOv8n":
        model_tag = "yolov8n"
    elif model_name == "YOLOv11n":
        model_tag = "yolo11n"
    else:
        model_tag = model_name.lower()  # フォールバック

    # 検索パターンを作成 (タイムスタンプ部分はワイルドカード)
    pattern = f"output/logs/{log_type}_{base_name}_long_stay_*{model_tag}_*.csv"

    if log_type == "events":
        pattern = f"output/logs/events_{base_name}_{model_tag}_*.csv"

    files = glob.glob(pattern)
    if files:
        return Path(files[0])
    return None


def generate_report():
    """本日の全ログを分析し、Markdownレポートを生成する"""
    batch_logs = {
        "YOLOv8n": "output/detected_comparison_v8_final/batch_log_*.csv",
        "YOLOv11n": "output/detected_comparison_v11_final/batch_log_*.csv",
    }

    results = []
    for model, batch_log_pattern in batch_logs.items():
        log_file = glob.glob(batch_log_pattern)
        if not log_file:
            print(f"バッチログが見つかりません: {batch_log_pattern}")
            continue

        batch_df = pd.read_csv(log_file[0])
        for _, row in batch_df.iterrows():
            video_name = row["relative_path"]
            print(f"分析中: {model} - {video_name}")

            perf_data, event_data = {}, {}
            perf_log_path = find_log(video_name, model, "log")
            event_log_path = find_log(video_name, model, "events")

            if perf_log_path:
                print(f"  - パフォーマンスログ発見: {perf_log_path.name}")
                perf_data = analyze_performance_log(perf_log_path)

            if event_log_path:
                print(f"  - イベントログ発見: {event_log_path.name}")
                event_data = analyze_event_log(event_log_path)

            results.append(
                {
                    "condition": extract_condition(video_name),
                    "model": model,
                    "event_count": row["long_stay_events"],
                    **perf_data,
                    **event_data,
                }
            )

    if not results:
        print("分析データがありません。")
        return

    # データフレームに変換してピボット
    final_df = pd.DataFrame(results).pivot(index="condition", columns="model")

    # 表示順と列名を調整
    ordered_index = ["Original", "Grayscale", "FPS15", "Scaled"]
    final_df = final_df.reindex(ordered_index)

    final_df = final_df[
        [
            ("event_count", "YOLOv8n"),
            ("event_count", "YOLOv11n"),
            ("avg_confidence", "YOLOv8n"),
            ("avg_confidence", "YOLOv11n"),
            ("avg_fps", "YOLOv8n"),
            ("avg_fps", "YOLOv11n"),
            ("avg_detect_ms", "YOLOv8n"),
            ("avg_detect_ms", "YOLOv11n"),
            ("peak_mem_mb", "YOLOv8n"),
            ("peak_mem_mb", "YOLOv11n"),
        ]
    ]
    final_df.columns = [
        "イベント数 (v8n)",
        "イベント数 (v11n)",
        "平均信頼度 (v8n)",
        "平均信頼度 (v11n)",
        "平均FPS (v8n)",
        "平均FPS (v11n)",
        "平均推論時間ms (v8n)",
        "平均推論時間ms (v11n)",
        "ピークメモリMB (v8n)",
        "ピークメモリMB (v11n)",
    ]

    print("\n\n--- 最終分析レポート ---")
    print("## 長時間滞在検出 総合パフォーマンス比較\n")
    print(final_df.to_markdown(floatfmt=".3f"))


def extract_condition(filename: str) -> str:
    """ファイル名から動画の加工条件を判定します。"""
    if "gray" in filename:
        return "Grayscale"
    if "fps15" in filename:
        return "FPS15"
    if "scale" in filename:
        return "Scaled"
    return "Original"


if __name__ == "__main__":
    generate_report()
