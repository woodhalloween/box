import os
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

# スクリプトの親ディレクトリの親ディレクトリ (プロジェクトルート) をsys.pathに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import time

import cv2
import psutil
from ultralytics import YOLO

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
# SKELETON = [
#     (0, 1),
#     (1, 3),
#     (0, 2),
#     (1, 2),
#     (2, 4),
#     (0, 5),
#     (0, 6),
#     (5, 7),
#     (7, 9),
#     (6, 8),
#     (8, 10),
#     (5, 6),
#     (5, 11),
#     (6, 12),
#     (11, 12),
#     (11, 13),
#     (13, 15),
#     (12, 14),
#     (14, 16),
# ]


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
        enable_pose: bool = False,
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
            enable_pose (bool): YOLO-Poseによる姿勢推定を有効にするかどうか。
        """
        model = load_yolo_model(model_path, device)
        if model is None:
            raise ValueError(f"YOLOモデルのロードに失敗しました: {model_path}")
        self.model: YOLO = model
        self.tracker = initialize_bytetrack()
        self.stay_threshold_sec = stay_threshold_sec
        self.move_threshold_px = move_threshold_px
        self.conf = conf
        self.enable_video_display = enable_video_display
        self.enable_pose = enable_pose
        if self.enable_pose:
            print("姿勢推定モードが有効です。")
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

    def process_video(
        self,
        input_file: str,
        output_file: str | None,
        enable_perf_log: bool = False,
    ) -> None:
        """指定された動画ファイルを処理し、滞留を検出する"""
        if not os.path.exists(input_file):
            print(f"エラー: 入力ファイルが見つかりません: {input_file}")
            return

        cap = cv2.VideoCapture(input_file)
        if not cap.isOpened():
            print(f"エラー: 動画ファイルを開けません: {input_file}")
            return

        # 動画プロパティの取得
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"入力動画: {width}x{height}, {fps:.2f}fps, {frame_count}フレーム")

        # 出力動画ライターの初期化
        out = None
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        # パフォーマンスログの初期化
        perf_log_file = None
        if enable_perf_log:
            model_name = (
                Path(self.model.ckpt_path).name
                if hasattr(self.model, "ckpt_path") and self.model.ckpt_path
                else "yolo_model"
            )
            perf_log_file = initialize_perf_log(True, input_file, model_name, log_type="long_stay")
            if perf_log_file:
                try:
                    with open(perf_log_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["# Video Properties", f"{width}x{height}", f"{fps:.2f}fps"])
                        writer.writerow([])
                except OSError as e:
                    print(f"警告: パフォーマンスログファイルに書き込めません: {e}")
                    perf_log_file = None  # 書き込めない場合は無効化

        # 処理ループの変数を初期化
        self.stay_info.clear()
        frame_idx = 0
        fps_buffer = deque(maxlen=30)  # FPS計算用のバッファ
        long_stay_event_count = 0
        avg_fps = 0.0

        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame_idx += 1
                loop_start_time = time.time()

                # メインの処理
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                tracks, det_time, track_time, num_det, num_track, keypoints = process_frame_for_tracking(
                    frame_rgb, self.model, self.tracker, self.conf, self.enable_pose
                )
                self.stay_info, notifications, stay_time = update_stay_times(
                    tracks, self.stay_info, loop_start_time, self.move_threshold_px, self.stay_threshold_sec
                )

                # 滞留イベントの記録と通知
                if notifications:
                    long_stay_event_count += len(notifications)
                    for notification in notifications:
                        print(f"Frame {frame_idx}: {notification}")

                # 描画処理
                if out or self.enable_video_display:
                    frame_bgr = draw_tracking_info(
                        frame_bgr,
                        tracks,
                        keypoints=keypoints,
                        enable_pose=self.enable_pose,
                        show_duration=True,
                        stay_info=self.stay_info,
                    )
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
                            print("処理がユーザーによって中断されました。")
                            break

                # FPS計算
                loop_time = time.time() - loop_start_time
                if loop_time > 0:
                    fps_buffer.append(1.0 / loop_time)
                    avg_fps = sum(fps_buffer) / len(fps_buffer)

                # パフォーマンスログ書き込み
                if perf_log_file and frame_idx % max(1, int(fps or 1)) == 0:
                    try:
                        with open(perf_log_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                            total_time = det_time + track_time + stay_time
                            model_name = (
                                Path(self.model.ckpt_path).name
                                if hasattr(self.model, "ckpt_path") and self.model.ckpt_path
                                else "yolo_model"
                            )
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
                                ]
                            )
                    except OSError as e:
                        print(f"警告: パフォーマンスログファイルへの書き込み中にエラーが発生しました: {e}")
                        perf_log_file = None  # エラー発生後は無効化

                # 進捗表示
                if frame_idx % 30 == 0:
                    print(
                        f"進捗: {frame_idx}/{frame_count} フレーム "
                        f"({100 * frame_idx / frame_count:.1f}%) - "
                        f"FPS: {avg_fps:.1f}"
                    )
        finally:
            # リソースの解放
            cap.release()
            if out:
                out.release()
            if self.enable_video_display:
                cv2.destroyAllWindows()

            # 最終サマリーの表示
            print(f"\n処理完了: {Path(input_file).name}")
            print(f"検出された長時間滞在イベントの総数: {long_stay_event_count}")
            if output_file:
                print(f"出力ファイル: {output_file}")
            if perf_log_file:
                print(f"パフォーマンスログ: {perf_log_file}")
            print("-" * 50)


class LongStayBatchProcessor:
    """
    ディレクトリ内の複数の動画ファイルに対して、長時間滞在検出をバッチ処理します。
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        detector_options: dict,
        video_extensions: list[str] | None = None,
        enable_perf_log: bool = False,
    ):
        """
        バッチプロセッサを初期化します。

        Args:
            input_dir (Path): 処理対象の動画ファイルが含まれる入力ディレクトリ。
            output_dir (Path): 処理結果を保存する出力ディレクトリ。
            detector_options (dict): LongStayDetectorに渡す設定オプションの辞書。
            video_extensions (Optional[list[str]]): 処理対象とする動画の拡張子リスト。
            enable_perf_log (bool): 個別のパフォーマンスログを有効にするかどうか。
        """
        if not input_dir.is_dir():
            raise ValueError(f"入力パスはディレクトリである必要があります: {input_dir}")

        self.input_dir = input_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.detector_options = detector_options
        self.video_extensions = video_extensions or [".mp4", ".mov", ".avi", ".mkv"]
        self.enable_perf_log = enable_perf_log
        self.detector = LongStayDetector(**self.detector_options)

        # ログファイルの設定
        self.output_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_file = self.output_dir / f"batch_log_{date_str}.csv"
        self.log_fp = open(self.log_file, "w", newline="", encoding="utf-8-sig")  # noqa: SIM115
        self.logger = csv.writer(self.log_fp)
        self.logger.writerow(["relative_path", "output_path", "status", "long_stay_events", "error_message"])

    def run(self):
        """バッチ処理を実行します。"""
        print(f"===== バッチ処理開始: {self.input_dir} =====")
        print(f"ログファイル: {self.log_file}")

        video_files = [p for p in self.input_dir.rglob("*") if p.suffix.lower() in self.video_extensions]

        if not video_files:
            print("処理対象の動画ファイルが見つかりませんでした。")
            return

        print(f"{len(video_files)} 件の動画ファイルを処理します。")

        for i, video_file in enumerate(video_files, 1):
            status = "Success"
            error_message = ""
            event_count = 0
            rel_path = video_file.relative_to(self.input_dir)
            output_path = self.output_dir / rel_path

            should_stop = False
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)

                print(f"\n--- [{i}/{len(video_files)}] 処理中: {rel_path} ---")

                event_count = self.detector.process_video(
                    input_file=str(video_file),
                    output_file=str(output_path),
                    enable_perf_log=self.enable_perf_log,
                )
                print(f"--- ✓ 完了: {rel_path} ---")

            except KeyboardInterrupt:
                status = "Interrupted"
                error_message = "User interrupted the process"
                print("\n[中断] ユーザーによってバッチ処理が中断されました。")
                should_stop = True

            except Exception as e:
                status = "Failed"
                error_message = str(e)
                print(f"[エラー] {video_file.name} の処理中にエラーが発生しました: {e}")
                should_stop = True

            self.logger.writerow([str(rel_path), str(output_path), status, event_count, error_message])

            if should_stop:
                print("バッチ処理を停止します。")
                break

        print("\n===== バッチ処理が終了しました =====")
        self.log_fp.close()


def main():
    """コマンドラインインターフェースのエントリーポイント。"""
    parser = argparse.ArgumentParser(description="動画ファイルまたはディレクトリから長時間滞在を検出します。")
    parser.add_argument("--input", type=str, required=True, help="入力動画ファイルまたはディレクトリのパス。")
    parser.add_argument("--output", type=str, required=True, help="出力ファイルまたはディレクトリのパス。")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOモデルファイルのパス。")
    parser.add_argument(
        "--enable_perf_log",
        action="store_true",
        help="パフォーマンスログをCSVファイルに保存します。（単一ファイル処理時のみ有効）",
    )
    parser.add_argument(
        "--enable_batch_perf_log", action="store_true", help="バッチ処理時に個別のパフォーマンスログを有効にします。"
    )
    parser.add_argument("--enable_video_display", action="store_true", help="処理中の動画をリアルタイムで表示します。")
    parser.add_argument("--device", type=str, default="", help="推論に使用するデバイス。(例: cpu, mps, 0)")
    parser.add_argument("--stay_threshold_sec", type=float, default=5.0, help="長時間滞在と判定する閾値（秒）。")
    parser.add_argument("--move_threshold_px", type=float, default=30.0, help="移動と判定するピクセル単位の閾値。")
    parser.add_argument("--conf", type=float, default=0.3, help="YOLOの検出信頼度の閾値。")
    parser.add_argument(
        "--enable_pose", action="store_true", help="YOLO-Poseによる姿勢推定を有効にし、骨格を描画します。"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    detector_options = {
        "model_path": args.model,
        "device": args.device,
        "stay_threshold_sec": args.stay_threshold_sec,
        "move_threshold_px": args.move_threshold_px,
        "conf": args.conf,
        "enable_video_display": args.enable_video_display,
        "enable_pose": args.enable_pose,
    }

    try:
        if input_path.is_dir():
            batch_processor = LongStayBatchProcessor(
                input_dir=input_path,
                output_dir=output_path,
                detector_options=detector_options,
                enable_perf_log=args.enable_batch_perf_log,
            )
            batch_processor.run()
        elif input_path.is_file():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            event_count = LongStayDetector(**detector_options).process_video(
                input_file=str(input_path),
                output_file=str(output_path),
                enable_perf_log=args.enable_perf_log,
            )
            print(f"検出された長時間滞在イベントの総数: {event_count}")
        else:
            print(f"エラー: 指定されたパスが見つかりません: {input_path}")

    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
