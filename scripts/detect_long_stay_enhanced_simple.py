"""
シンプルな顔認識統合長時間滞在検出スクリプト
既存のByteTrackシステムに顔認識機能を追加
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

# パスの追加
sys.path.append(str(Path(__file__).parent.parent))

# 既存システムのインポート
from src.face_recognition.face_database import FaceDatabase

# 顔認識関連のインポート
from src.face_recognition.face_detector import FaceDetector
from src.face_recognition.face_recognizer import FaceRecognizer
from src.tracking.bytetrack_utils import (
    initialize_bytetrack,
    initialize_perf_log,
    load_yolo_model,
    process_frame_for_tracking,
    update_stay_times,
)


def setup_logging(log_level: str = "INFO"):
    """ログ設定"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/enhanced_tracking.log")],
    )


def parse_arguments():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description="Simple Face Recognition Enhanced Tracking")

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
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")

    # 追跡設定
    parser.add_argument(
        "--stay_threshold_sec", type=float, default=5.0, help="Long stay threshold in seconds"
    )
    parser.add_argument(
        "--move_threshold_px", type=int, default=20, help="Movement threshold in pixels"
    )

    # 表示設定
    parser.add_argument("--enable_video_display", action="store_true", help="Show real-time video")
    parser.add_argument("--enable_perf_log", action="store_true", help="Enable performance logging")

    return parser.parse_args()


class SimpleFaceTracker:
    """シンプルな顔認識統合追跡システム"""

    def __init__(self, config_path: str):
        """初期化"""
        # 設定読み込み
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # 顔認識コンポーネント初期化
        self.face_detector = FaceDetector(self.config)
        self.face_recognizer = FaceRecognizer(self.config)
        self.face_database = FaceDatabase(self.config)

        # 既知の顔を読み込み
        self._load_known_faces()

        # ID管理
        self.track_to_person_id = {}  # track_id -> person_id
        self.person_id_to_name = {}  # person_id -> name

    def _load_known_faces(self):
        """データベースから既知の顔を読み込み"""
        try:
            face_encodings = self.face_database.get_all_face_encodings()
            for person_id, name, encoding in face_encodings:
                self.face_recognizer.add_known_person(person_id, name, encoding)
                self.person_id_to_name[person_id] = name
            logging.info(f"Loaded {len(face_encodings)} known faces")
        except Exception as e:
            logging.error(f"Error loading known faces: {e}")

    def enhance_tracks(self, frame: np.ndarray, tracks: list) -> list:
        """追跡結果に顔認識情報を追加"""
        enhanced_tracks = []

        if not tracks:
            return enhanced_tracks

        # 人物ボックスの抽出
        person_boxes = []
        for track in tracks:
            if hasattr(track, "xyxy"):
                x1, y1, x2, y2 = map(int, track.xyxy)
            elif len(track) >= 4:
                x1, y1, x2, y2 = map(int, track[:4])
            else:
                continue
            person_boxes.append((x1, y1, x2, y2))

        # 顔検出
        detected_faces = self.face_detector.detect_faces(frame, person_boxes)

        # 顔認識
        face_crops = []
        for face in detected_faces:
            face_crop = self.face_detector.crop_face(frame, face["bbox"])
            if face_crop is not None:
                face_crops.append(face_crop)

        recognition_results = self.face_recognizer.recognize_faces(face_crops)

        # 追跡と顔認識の対応付け
        for i, track in enumerate(tracks):
            if hasattr(track, "track_id"):
                track_id = int(track.track_id)
                x1, y1, x2, y2 = map(int, track.xyxy)
            elif len(track) >= 5:
                x1, y1, x2, y2, track_id = map(int, track[:5])
            else:
                continue

            # 顔認識結果の取得
            person_id = -1
            person_name = "Unknown"
            face_confidence = 0.0
            is_known = False

            # 最も近い顔を探す
            if i < len(recognition_results):
                result = recognition_results[i]
                if result["is_known"]:
                    person_id = result["person_id"]
                    person_name = result["name"]
                    face_confidence = result["confidence"]
                    is_known = True

                    # ID関連付けの更新
                    self.track_to_person_id[track_id] = person_id
                    self.person_id_to_name[person_id] = person_name

                    # データベース更新
                    self.face_database.update_person_appearance(person_id)

            # 既存の関連付けをチェック
            elif track_id in self.track_to_person_id:
                person_id = self.track_to_person_id[track_id]
                person_name = self.person_id_to_name.get(person_id, "Unknown")
                is_known = True

            # 拡張追跡情報を作成
            enhanced_track = {
                "track_id": track_id,
                "bbox": (x1, y1, x2, y2),
                "person_id": person_id,
                "person_name": person_name,
                "face_confidence": face_confidence,
                "is_known": is_known,
                "xyxy": [x1, y1, x2, y2],  # 既存互換性のため
            }

            enhanced_tracks.append(enhanced_track)

        return enhanced_tracks


def draw_enhanced_tracking_info(frame, enhanced_tracks, stay_info=None):
    """拡張追跡情報の描画"""
    for track in enhanced_tracks:
        x1, y1, x2, y2 = track["bbox"]
        track_id = track["track_id"]
        person_name = track["person_name"]
        is_known = track["is_known"]
        face_confidence = track["face_confidence"]

        # ボックスの色決定
        if is_known:
            color = (0, 255, 255)  # 黄色：既知の人物
            thickness = 3
        else:
            color = (0, 0, 255)  # 赤色：未知の人物
            thickness = 2

        # ボックス描画
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # ラベル作成
        if is_known:
            label = f"ID {track_id}: {person_name} ({face_confidence:.2f})"
        else:
            label = f"ID {track_id}: Unknown"

        # 滞在時間情報の追加
        if stay_info and track_id in stay_info:
            duration = stay_info[track_id].get("duration", 0)
            label += f" [{duration:.1f}s]"

        # ラベル背景
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0], y1), color, -1)

        # ラベルテキスト
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame


def main():
    """メイン処理"""
    args = parse_arguments()
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    logger.info("Simple Face Recognition Enhanced Tracking started")

    try:
        # 顔認識システム初期化
        face_tracker = SimpleFaceTracker(args.config)

        # 基本システム初期化
        model = load_yolo_model(args.model, "")
        tracker = initialize_bytetrack()

        # 動画読み込み
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {args.input}")

        # 動画情報取得
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video info: {width}x{height}, {fps}FPS, {total_frames} frames")

        # 出力動画設定
        video_writer = None
        if args.output:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            video_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

        # ログファイル設定
        perf_log_file, perf_log_f, perf_log_writer = initialize_perf_log(
            args.enable_perf_log, args.input, args.model, log_type="enhanced"
        )

        # 滞在時間管理
        stay_info = {}

        # 処理ループ
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start_time = time.time()
            current_time = time.time()

            try:
                # 1. 基本追跡（YOLO + ByteTrack）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tracks, detection_time_ms, tracking_time_ms, num_detections, num_tracks, _ = (
                    process_frame_for_tracking(frame_rgb, model, tracker)
                )

                # 2. 顔認識による強化
                face_start_time = time.time()
                enhanced_tracks = face_tracker.enhance_tracks(frame, tracks)
                face_time_ms = (time.time() - face_start_time) * 1000

                # 3. 長時間滞在検出
                stay_info, notifications, stay_check_time_ms = update_stay_times(
                    enhanced_tracks,
                    stay_info,
                    current_time,
                    args.move_threshold_px,
                    args.stay_threshold_sec,
                )

                # 通知表示
                for notification in notifications:
                    logger.warning(f"Frame {frame_count}: Long stay detected - {notification}")

                # 4. 可視化
                result_frame = draw_enhanced_tracking_info(frame, enhanced_tracks, stay_info)

                # パフォーマンス情報表示
                frame_time = (time.time() - frame_start_time) * 1000
                if frame_count > 0:
                    avg_fps = frame_count / (time.time() - start_time)
                    cv2.putText(
                        result_frame,
                        f"FPS: {avg_fps:.1f} | Face: {face_time_ms:.1f}ms",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                # 出力保存
                if video_writer:
                    video_writer.write(result_frame)

                # リアルタイム表示
                if args.enable_video_display:
                    cv2.imshow("Enhanced Face Recognition Tracking", result_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1

                # プログレス表示
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")

            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
                continue

        # 統計表示
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        logger.info("Processing completed!")
        logger.info(f"Total frames: {frame_count}, Time: {total_time:.1f}s, Avg FPS: {avg_fps:.1f}")

    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        raise

    finally:
        # クリーンアップ
        if "cap" in locals():
            cap.release()
        if "video_writer" in locals() and video_writer:
            video_writer.release()
        if args.enable_video_display:
            cv2.destroyAllWindows()

        logger.info("Enhanced Face Recognition Tracking finished")


if __name__ == "__main__":
    main()
