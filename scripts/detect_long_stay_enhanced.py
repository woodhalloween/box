"""
顔認識統合長時間滞在検出スクリプト
ハイブリッド追跡システムを使用した高精度なID付与と長時間滞在検知
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

# パスの追加
sys.path.append(str(Path(__file__).parent.parent))

# 必要なモジュールのインポート
import numpy as np

# ByteTrack関連のインポート
from ultralytics import YOLO

from src.tracking.hybrid_tracker import HybridTracker
from src.utils.enhanced_visualization import EnhancedVisualizer


def setup_logging(log_level: str = "INFO"):
    """ログ設定"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/enhanced_tracking.log")],
    )


def parse_arguments():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description="Enhanced Face Recognition Tracking System")

    # 基本設定
    parser.add_argument("--input", type=str, required=True, help="Input video file path")
    parser.add_argument("--output", type=str, help="Output video file path")
    parser.add_argument(
        "--config",
        type=str,
        default="config/face_recognition_config.yaml",
        help="Configuration file path",
    )

    # モデル設定
    parser.add_argument("--yolo-model", type=str, default="yolo11n.pt", help="YOLO model path")
    parser.add_argument(
        "--confidence", type=float, default=0.3, help="YOLO detection confidence threshold"
    )

    # 追跡設定
    parser.add_argument(
        "--stay-threshold", type=float, default=5.0, help="Long stay threshold in seconds"
    )
    parser.add_argument(
        "--move-threshold", type=int, default=20, help="Movement threshold in pixels"
    )

    # 表示設定
    parser.add_argument("--show", action="store_true", help="Show real-time video display")
    parser.add_argument("--save-debug", action="store_true", help="Save debug images")
    parser.add_argument("--show-trails", action="store_true", help="Show tracking trails")

    # その他
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


class LongStayDetector:
    """長時間滞在検出クラス"""

    def __init__(self, stay_threshold: float, move_threshold: int):
        self.stay_threshold = stay_threshold
        self.move_threshold = move_threshold
        self.stay_info = {}  # stable_id -> stay_info

    def update_stay_times(self, enhanced_tracks: list, current_time: float) -> list:
        """滞在時間の更新と長時間滞在の検出"""
        current_track_ids = set()
        notifications = []

        for track in enhanced_tracks:
            stable_id = track["stable_id"]
            x1, y1, x2, y2 = track["bbox"]

            current_track_ids.add(stable_id)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            current_pos = (center_x, center_y)

            if stable_id not in self.stay_info:
                # 新しい追跡の初期化
                self.stay_info[stable_id] = {
                    "first_seen": current_time,
                    "last_update": current_time,
                    "last_position": current_pos,
                    "stay_duration": 0.0,
                    "has_moved": False,
                    "notified": False,
                    "person_name": track.get("person_name", "Unknown"),
                    "is_known": track.get("is_known_person", False),
                }
            else:
                # 既存の追跡の更新
                stay_data = self.stay_info[stable_id]
                last_pos = stay_data["last_position"]

                # 移動距離の計算
                distance = np.sqrt(
                    (current_pos[0] - last_pos[0]) ** 2 + (current_pos[1] - last_pos[1]) ** 2
                )

                if distance > self.move_threshold:
                    # 大きな移動があった場合、リセット
                    stay_data["first_seen"] = current_time
                    stay_data["has_moved"] = True
                    stay_data["notified"] = False

                # 位置と時間の更新
                stay_data["last_position"] = current_pos
                stay_data["last_update"] = current_time
                stay_data["stay_duration"] = current_time - stay_data["first_seen"]
                stay_data["person_name"] = track.get("person_name", stay_data["person_name"])
                stay_data["is_known"] = track.get("is_known_person", stay_data["is_known"])

                # 長時間滞在の通知
                if stay_data["stay_duration"] >= self.stay_threshold and not stay_data["notified"]:
                    notification = {
                        "stable_id": stable_id,
                        "person_name": stay_data["person_name"],
                        "stay_duration": stay_data["stay_duration"],
                        "position": current_pos,
                        "is_known": stay_data["is_known"],
                    }
                    notifications.append(notification)
                    stay_data["notified"] = True

        # 消失した追跡のクリーンアップ
        disappeared_ids = set(self.stay_info.keys()) - current_track_ids
        for stable_id in list(disappeared_ids):
            del self.stay_info[stable_id]

        return notifications


def main():
    """メイン処理"""
    args = parse_arguments()

    # ログ設定
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Enhanced Face Recognition Tracking System started")
    logger.info(f"Input: {args.input}")
    logger.info(f"Config: {args.config}")

    try:
        # ハイブリッド追跡システムの初期化
        hybrid_tracker = HybridTracker(args.config)

        # 可視化システムの初期化
        with open(args.config, encoding="utf-8") as f:
            import yaml

            config = yaml.safe_load(f)

        visualizer = EnhancedVisualizer(config)

        # YOLO + ByteTrackの初期化
        yolo_model = YOLO(args.yolo_model)
        byte_tracker = ByteTrack(
            track_thresh=args.confidence, track_buffer=30, match_thresh=0.8, frame_rate=25
        )

        # 長時間滞在検出器の初期化
        stay_detector = LongStayDetector(args.stay_threshold, args.move_threshold)

        # 動画の読み込み
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {args.input}")

        # 動画情報取得
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video info: {width}x{height}, {fps}FPS, {total_frames} frames")

        # 出力動画ライターの設定
        video_writer = None
        if args.output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

        # 処理ループ
        frame_count = 0
        start_time = time.time()
        processing_times = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start_time = time.time()
            current_time = time.time()

            try:
                # 1. YOLO検出とByteTrackによる基本追跡
                results = yolo_model(frame, verbose=False)
                detections = []
                if results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()

                    for box, conf, cls in zip(boxes, confs, classes, strict=False):
                        if conf >= args.confidence and int(cls) == 0:  # person class
                            detections.append([box[0], box[1], box[2], box[3], conf, cls])

                detections_array = np.array(detections) if detections else np.empty((0, 6))
                basic_tracks = byte_tracker.update(detections_array, frame)

                # 2. ハイブリッド追跡による強化
                enhanced_tracks, performance_metrics = hybrid_tracker.process_frame(
                    frame, basic_tracks
                )

                # 3. 長時間滞在検出
                notifications = stay_detector.update_stay_times(enhanced_tracks, current_time)

                # 4. 通知の処理
                for notification in notifications:
                    logger.warning(f"Long stay detected: {notification}")

                # 5. 可視化
                # 顔検出結果を取得（デバッグ用）
                detected_faces = []
                if hasattr(hybrid_tracker, "face_detector"):
                    person_boxes = [(track["bbox"]) for track in enhanced_tracks]
                    detected_faces = hybrid_tracker.face_detector.detect_faces(frame, person_boxes)

                # フレームに描画
                result_frame = visualizer.draw_enhanced_tracks(
                    frame, enhanced_tracks, detected_faces, performance_metrics
                )

                # 追跡軌跡の描画（オプション）
                if args.show_trails and hasattr(hybrid_tracker, "track_history"):
                    result_frame = visualizer.draw_tracking_trails(
                        result_frame, hybrid_tracker.track_history
                    )

                # 滞在時間情報の追加
                for track in enhanced_tracks:
                    stable_id = track["stable_id"]
                    if stable_id in stay_detector.stay_info:
                        stay_duration = stay_detector.stay_info[stable_id]["stay_duration"]
                        x1, y1, x2, y2 = track["bbox"]

                        # 長時間滞在の場合は赤でハイライト
                        if stay_duration >= args.stay_threshold:
                            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

                        # 滞在時間表示
                        stay_text = f"Stay: {stay_duration:.1f}s"
                        cv2.putText(
                            result_frame,
                            stay_text,
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )

                # パフォーマンス情報の表示
                frame_time = (time.time() - frame_start_time) * 1000
                processing_times.append(frame_time)

                if frame_count > 0:
                    avg_fps = frame_count / (time.time() - start_time)
                    cv2.putText(
                        result_frame,
                        f"FPS: {avg_fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                # デバッグ画像の保存
                if args.save_debug and frame_count % 100 == 0:
                    debug_filename = f"debug_frame_{frame_count:06d}.jpg"
                    visualizer.save_debug_image(
                        result_frame, debug_filename, detected_faces, enhanced_tracks
                    )

                # 結果の保存
                if video_writer:
                    video_writer.write(result_frame)

                # リアルタイム表示
                if args.show:
                    cv2.imshow("Enhanced Face Recognition Tracking", result_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1

                # プログレス表示
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    avg_time = np.mean(processing_times[-100:])
                    logger.info(
                        f"Progress: {progress:.1f}% ({frame_count}/{total_frames}), "
                        f"Avg time: {avg_time:.1f}ms"
                    )

            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
                continue

        # 統計の表示
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        stats = hybrid_tracker.get_performance_stats()
        stats.update(
            {"total_frames": frame_count, "total_time_seconds": total_time, "average_fps": avg_fps}
        )

        logger.info("Processing completed!")
        logger.info(f"Statistics: {stats}")

        # サマリー可視化の作成と保存
        summary_frame = visualizer.create_summary_visualization(stats)
        if args.output:
            summary_path = args.output.replace(".mp4", "_summary.jpg")
            cv2.imwrite(summary_path, summary_frame)
            logger.info(f"Summary saved to: {summary_path}")

    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        raise

    finally:
        # クリーンアップ
        if "cap" in locals():
            cap.release()
        if "video_writer" in locals() and video_writer:
            video_writer.release()
        if args.show:
            cv2.destroyAllWindows()

        logger.info("Enhanced Face Recognition Tracking System finished")


if __name__ == "__main__":
    main()
