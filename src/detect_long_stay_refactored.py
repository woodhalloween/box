# detect_long_stay_refactored.py
# 🖥️ Refactored by Daisy for maximal pytest testability and modularity

import argparse
from pathlib import Path

from src.detect_long_stay_core import run_long_stay_detection
from src.tracking.bytetrack_utils import (
    draw_tracking_info,
    initialize_bytetrack,
    initialize_perf_log,
    load_yolo_model,
    process_frame_for_tracking,
    update_stay_times,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="動画からByteTrackを用いて長時間滞在を検出します。")
    parser.add_argument("--input", type=str, required=True, help="入力動画ファイルのパス。")
    parser.add_argument("--output", type=str, default="output/tracked_video.mp4", help="出力動画ファイルのパス。")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOモデルファイルのパス。")
    parser.add_argument("--enable_perf_log", action="store_true", help="パフォーマンスログをCSVファイルに保存します。")
    parser.add_argument("--enable_video_display", action="store_true", help="処理中の動画をリアルタイムで表示します。")
    parser.add_argument("--device", type=str, default="", help="推論に使用するデバイスを指定します。")
    parser.add_argument("--stay_threshold_sec", type=float, default=5.0, help="長時間滞在と判定する閾値（秒）。")
    parser.add_argument("--move_threshold_px", type=float, default=30.0, help="移動と判定するピクセルの閾値。")
    parser.add_argument("--conf", type=float, default=0.3, help="YOLOの検出信頼度の閾値。")
    return parser.parse_args()


def main():
    args = parse_arguments()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    run_long_stay_detection(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        device=args.device,
        stay_threshold=args.stay_threshold_sec,
        move_threshold=args.move_threshold_px,
        conf=args.conf,
        enable_perf_log=args.enable_perf_log,
        enable_video_display=args.enable_video_display,
        draw_fn=draw_tracking_info,
        load_model_fn=load_yolo_model,
        tracker_init_fn=initialize_bytetrack,
        perf_log_fn=initialize_perf_log,
        process_frame_fn=process_frame_for_tracking,
        update_stay_fn=update_stay_times,
    )


if __name__ == "__main__":
    main()
