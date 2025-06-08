import os
import sys

# スクリプトの親ディレクトリの親ディレクトリ (プロジェクトルート) をsys.pathに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import psutil

# 共通ユーティリティをインポート
from src.tracking.bytetrack_utils import (
    draw_tracking_info,
    initialize_bytetrack,
    initialize_perf_log,
    load_yolo_model,
    process_frame_for_tracking,
    update_stay_times,
)

# Skeleton structure for YOLO keypoints visualization
# (kept for potential future use, but not primary focus)
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


class LongStayDetector:
    """
    YOLOとByteTrackを使用して、動画ファイル内の長時間滞在オブジェクトを検出します。
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "",
        stay_threshold_sec: float = 5.0,
        move_threshold_px: float = 30.0,
        conf: float = 0.3,
        enable_video_display: bool = False,
    ):
        """
        検出器を初期化します。

        Args:
            model_path (str): YOLOモデルファイルへのパス。
            device (str): 推論デバイス（"cpu", "cuda", "mps"など）。
            stay_threshold_sec (float): 長時間滞在と見なすための秒単位の閾値。
            move_threshold_px (float): 滞在タイマーをリセットするためのピクセル単位の移動閾値。
            conf (float): 検出の信頼度の閾値。
            enable_video_display (bool): 処理中にビデオを表示するかどうか。
        """
        self.model = load_yolo_model(model_path, device)
        self.tracker = initialize_bytetrack()
        self.stay_threshold_sec = stay_threshold_sec
        self.move_threshold_px = move_threshold_px
        self.conf = conf
        self.enable_video_display = enable_video_display
        self.stay_info = {}

    def _draw_overlay(self, frame, frame_idx, frame_count, avg_fps, perf_stats, object_counts):
        """ビデオフレームに情報オーバーレイを描画します。"""
        height, width, _ = frame.shape

        frame_info = f"Frame: {frame_idx}/{frame_count} FPS: {avg_fps:.1f}"
        stats_info = (
            f"Det: {perf_stats['detection']:.1f}ms "
            f"Track: {perf_stats['tracking']:.1f}ms "
            f"Stay: {perf_stats['stay_check']:.1f}ms"
        )
        objects_info = f"Detected: {object_counts['detected']} Tracked: {object_counts['tracked']}"

        cv2.rectangle(frame, (10, 10), (400, 90), (0, 0, 0), -1)
        cv2.putText(frame, frame_info, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, stats_info, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, objects_info, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        long_stayers = [
            f"ID {id}: {info['stay_duration']:.1f}s"
            for id, info in self.stay_info.items()
            if info["stay_duration"] >= self.stay_threshold_sec
        ]

        if long_stayers:
            cv2.rectangle(frame, (width - 210, 10), (width - 10, 30 + 25 * len(long_stayers)), (0, 0, 0), -1)
            cv2.putText(frame, "長時間滞在者:", (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            for i, stayer in enumerate(long_stayers):
                cv2.putText(
                    frame, stayer, (width - 200, 30 + 25 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                )

        return frame

    def process_video(self, input_file: str, output_file: str, enable_perf_log: bool = False):
        """単一の動画ファイルを処理して長時間滞在を検出します。"""
        if not os.path.exists(input_file):
            print(f"エラー: 入力ファイルが見つかりません: {input_file}")
            return

        cap = cv2.VideoCapture(input_file)
        if not cap.isOpened():
            print(f"エラー: 動画ファイルを開けません: {input_file}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"入力動画: {width}x{height}, {fps}fps, {frame_count}フレーム")

        out = None
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        perf_log_file = None
        if enable_perf_log:
            model_name = Path(self.model.ckpt_path).name if hasattr(self.model, "ckpt_path") else "N/A"
            perf_log_file = initialize_perf_log(True, input_file, model_name, log_type="long_stay")
            with open(perf_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["# Video Properties", f"{width}x{height}", f"{fps}fps"])
                writer.writerow([])

        self.stay_info = {}
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

                tracks, det_time, track_time, num_det, num_track, _ = process_frame_for_tracking(
                    frame_rgb, self.model, self.tracker, conf=self.conf
                )

                self.stay_info, notifications, stay_time = update_stay_times(
                    tracks, self.stay_info, current_time, self.move_threshold_px, self.stay_threshold_sec
                )

                for notification in notifications:
                    print(f"Frame {frame_idx}: {notification}")

                if out or self.enable_video_display:
                    frame_bgr = draw_tracking_info(frame_bgr, tracks, show_duration=True, stay_info=self.stay_info)

                    current_fps = 1.0 / (time.time() - last_fps_update) if (time.time() - last_fps_update) > 0 else 0
                    if len(fps_buffer) > 10:
                        fps_buffer.pop(0)
                    fps_buffer.append(current_fps)
                    avg_fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0
                    last_fps_update = time.time()

                    perf_stats = {"detection": det_time, "tracking": track_time, "stay_check": stay_time}
                    object_counts = {"detected": num_det, "tracked": num_track}

                    frame_bgr = self._draw_overlay(
                        frame_bgr, frame_idx, frame_count, avg_fps, perf_stats, object_counts
                    )

                    if out:
                        out.write(frame_bgr)

                    if self.enable_video_display:
                        cv2.imshow("Long Stay Detection", frame_bgr)
                        if cv2.waitKey(1) & 0xFF == 27:  # ESC
                            break

                if perf_log_file and frame_idx % max(1, int(fps or 1)) == 0:
                    with open(perf_log_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                        total_time = det_time + track_time + stay_time
                        model_name = Path(self.model.ckpt_path).name if hasattr(self.model, "ckpt_path") else "N/A"
                        writer.writerow(
                            [
                                frame_idx,
                                datetime.now().strftime("%H:%M:%S.%f")[:-3],
                                f"{det_time:.2f}",
                                f"{track_time:.2f}",
                                f"{stay_time:.2f}",
                                f"{total_time:.2f}",
                                num_det,
                                num_track,
                                f"{avg_fps:.2f}",
                                f"{mem_usage:.2f}",
                                model_name,
                                "bytetrack",
                                f"Stay:{self.stay_threshold_sec}s Move:{self.move_threshold_px}px",
                            ]
                        )

                if frame_idx % 30 == 0:
                    elapsed = time.time() - start_time
                    speed = frame_idx / elapsed if elapsed > 0 else 0
                    print(
                        f"進捗: {frame_idx}/{frame_count} ({frame_idx / frame_count * 100:.1f}%) | 処理速度: {speed:.2f} FPS"
                    )

        except KeyboardInterrupt:
            print("\n処理がユーザーによって中断されました。")
        finally:
            cap.release()
            if out:
                out.release()
            if self.enable_video_display:
                cv2.destroyAllWindows()
            print(f"処理完了。出力ファイル: {output_file}, ログ: {perf_log_file}")


def main():
    """コマンドラインインターフェースのエントリーポイント。"""
    parser = argparse.ArgumentParser(description="動画からByteTrackを用いて長時間滞在を検出します。")
    parser.add_argument("--input", type=str, required=True, help="入力動画ファイルのパス。")
    parser.add_argument("--output", type=str, default="output/tracked_video.mp4", help="出力動画ファイルのパス。")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOモデルファイルのパス。")
    parser.add_argument("--enable_perf_log", action="store_true", help="パフォーマンスログをCSVファイルに保存します。")
    parser.add_argument("--enable_video_display", action="store_true", help="処理中の動画をリアルタイムで表示します。")
    parser.add_argument(
        "--device", type=str, default="", help="推論に使用するデバイスを指定します。(例: cpu, mps, 0 for cuda:0)"
    )
    parser.add_argument("--stay_threshold_sec", type=float, default=5.0, help="長時間滞在と判定する閾値（秒）。")
    parser.add_argument(
        "--move_threshold_px", type=float, default=30.0, help="移動と判定するためのピクセル単位の閾値。"
    )
    parser.add_argument("--conf", type=float, default=0.3, help="YOLOの検出信頼度の閾値。")
    args = parser.parse_args()

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    try:
        detector = LongStayDetector(
            model_path=args.model,
            device=args.device,
            stay_threshold_sec=args.stay_threshold_sec,
            move_threshold_px=args.move_threshold_px,
            conf=args.conf,
            enable_video_display=args.enable_video_display,
        )
        detector.process_video(
            input_file=args.input,
            output_file=args.output,
            enable_perf_log=args.enable_perf_log,
        )
    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
