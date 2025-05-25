"""
長時間滞在検出システム + 顔検出・認識機能（段階的実装版）
基本の人物追跡にMediaPipeの顔検出とface_recognitionライブラリを統合

このバージョンでは:
- YOLO + ByteTrackによる人物追跡（既存）
- MediaPipeによる顔検出（新機能）
- face_recognitionによる顔認識（新機能）
- 日本語フォント対応（文字化け解決）
"""

import os
import sys

# スクリプトの親ディレクトリの親ディレクトリ (プロジェクトルート) をsys.pathに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import psutil
from PIL import Image, ImageDraw, ImageFont

import face_recognition

# 共通ユーティリティをインポート
from src.tracking.bytetrack_utils import (
    draw_tracking_info,
    initialize_bytetrack,
    initialize_perf_log,
    load_yolo_model,
    process_frame_for_tracking,
    update_stay_times,
)


class FaceRecognitionSystem:
    """顔検出・認識システム"""

    def __init__(self, face_confidence: float = 0.7, recognition_tolerance: float = 0.6):
        """
        Args:
            face_confidence: 顔検出の信頼度閾値
            recognition_tolerance: 顔認識の許容値（小さいほど厳密）
        """
        self.face_confidence = face_confidence
        self.recognition_tolerance = recognition_tolerance

        # MediaPipe 顔検出
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=face_confidence
        )

        # 既知の顔データベース
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_database_path = "data/face_database"
        Path(self.face_database_path).mkdir(parents=True, exist_ok=True)

        # 統計情報
        self.unknown_face_count = 0

        # 日本語フォント設定
        self.setup_japanese_font()

        print(f"顔認識システム初期化完了 (tolerance: {recognition_tolerance})")

    def setup_japanese_font(self):
        """日本語フォントを設定"""
        try:
            # macOSの日本語フォントを試行
            font_paths = [
                "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
                "/System/Library/Fonts/Arial Unicode.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
            ]

            self.font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        self.font = ImageFont.truetype(font_path, 16)
                        print(f"日本語フォント使用: {font_path}")
                        break
                    except:
                        continue

            if self.font is None:
                self.font = ImageFont.load_default()
                print("デフォルトフォントを使用")

        except Exception as e:
            print(f"フォント設定エラー: {e}")
            self.font = ImageFont.load_default()

    def detect_faces(self, frame_rgb: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """MediaPipeで顔を検出"""
        height, width = frame_rgb.shape[:2]
        results = self.face_detection.process(frame_rgb)

        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * width)
                y1 = int(bbox.ymin * height)
                x2 = int((bbox.xmin + bbox.width) * width)
                y2 = int((bbox.ymin + bbox.height) * height)
                confidence = detection.score[0]

                faces.append((x1, y1, x2, y2, confidence))

        return faces

    def recognize_faces(
        self, frame_rgb: np.ndarray, face_locations: list[tuple[int, int, int, int, float]]
    ) -> list[dict[str, Any]]:
        """顔認識を実行"""
        if not face_locations:
            return []

        # face_recognition用の座標形式に変換 (top, right, bottom, left)
        recognition_locations = []
        for x1, y1, x2, y2, confidence in face_locations:
            recognition_locations.append((y1, x2, y2, x1))

        # 顔エンコーディングを生成
        try:
            face_encodings = face_recognition.face_encodings(frame_rgb, recognition_locations)
        except Exception as e:
            print(f"顔エンコーディング生成エラー: {e}")
            return []

        face_results = []
        for i, (encoding, (x1, y1, x2, y2, confidence)) in enumerate(
            zip(face_encodings, face_locations, strict=False)
        ):
            if len(encoding) == 0:
                continue

            # 既知の顔と比較
            if self.known_face_encodings:
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, encoding, tolerance=self.recognition_tolerance
                )
                distances = face_recognition.face_distance(self.known_face_encodings, encoding)

                best_match_index = np.argmin(distances) if len(distances) > 0 else None

                if best_match_index is not None and matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    distance = distances[best_match_index]
                else:
                    name = f"未知の人物_{self.unknown_face_count}"
                    distance = 1.0
                    self.unknown_face_count += 1
            else:
                name = f"未知の人物_{self.unknown_face_count}"
                distance = 1.0
                self.unknown_face_count += 1

            face_results.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "name": name,
                    "distance": distance,
                    "encoding": encoding,
                }
            )

        return face_results

    def add_known_face(self, encoding: np.ndarray, name: str):
        """既知の顔を追加"""
        self.known_face_encodings.append(encoding)
        self.known_face_names.append(name)
        print(f"新しい顔を登録: {name}")

    def save_face_database(self):
        """顔データベースを保存"""
        try:
            database = {
                "encodings": [enc.tolist() for enc in self.known_face_encodings],
                "names": self.known_face_names,
            }
            with open(f"{self.face_database_path}/face_database.json", "w", encoding="utf-8") as f:
                json.dump(database, f, ensure_ascii=False, indent=2)
            print("顔データベースを保存しました")
        except Exception as e:
            print(f"顔データベース保存エラー: {e}")

    def load_face_database(self):
        """顔データベースを読み込み"""
        try:
            database_file = f"{self.face_database_path}/face_database.json"
            if os.path.exists(database_file):
                with open(database_file, encoding="utf-8") as f:
                    database = json.load(f)

                self.known_face_encodings = [np.array(enc) for enc in database["encodings"]]
                self.known_face_names = database["names"]
                print(f"顔データベースを読み込み: {len(self.known_face_names)}人")
            else:
                print("顔データベースファイルが見つかりません")
        except Exception as e:
            print(f"顔データベース読み込みエラー: {e}")

    def close(self):
        """リソースをクリーンアップ"""
        self.face_detection.close()


def draw_faces_with_japanese(
    frame: np.ndarray, face_results: list[dict[str, Any]], font
) -> np.ndarray:
    """
    PILを使って日本語対応で顔認識結果を描画
    """
    # OpenCV (BGR) から PIL (RGB) に変換
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_image)

    for result in face_results:
        x1, y1, x2, y2 = result["bbox"]
        confidence = result["confidence"]
        name = result["name"]
        distance = result["distance"]

        # 顔枠を描画（緑色）
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

        # ラベルテキスト
        if name.startswith("未知"):
            label = f"{name} ({confidence:.2f})"
            text_color = (255, 100, 100)  # 赤系
        else:
            label = f"{name} ({distance:.2f})"
            text_color = (100, 255, 100)  # 緑系

        # テキスト背景
        bbox = draw.textbbox((x1, y1 - 25), label, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0, 128))

        # テキスト描画
        draw.text((x1, y1 - 25), label, font=font, fill=text_color)

    # PIL (RGB) から OpenCV (BGR) に変換
    frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return frame_bgr


def main(
    input_file: str,
    output_file: str,
    model_path: str = "yolo11n.pt",
    enable_perf_log: bool = False,
    enable_video_display: bool = False,
    device: str = "",
    stay_threshold_sec: float = 5.0,
    move_threshold_px: float = 30.0,
    conf: float = 0.3,
    face_confidence: float = 0.7,
    recognition_tolerance: float = 0.6,
):
    """
    長時間滞在検出 + 顔検出・認識を実行するメイン関数
    """
    # 入力ファイルチェック
    if not os.path.exists(input_file):
        print(f"エラー: 入力ファイルが見つかりません: {input_file}")
        return

    # キャプチャ設定
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"エラー: 動画ファイルを開けません: {input_file}")
        return

    # 入力動画の情報を取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"入力動画: {width}x{height}, {fps}fps, {frame_count}フレーム")

    # システム初期化
    print("YOLOモデルを読み込み中...")
    model = load_yolo_model(model_path, device)
    print("ByteTrackトラッカーを初期化中...")
    tracker = initialize_bytetrack()
    print("顔認識システムを初期化中...")
    face_system = FaceRecognitionSystem(
        face_confidence=face_confidence, recognition_tolerance=recognition_tolerance
    )
    # 既存の顔データベースを読み込み
    face_system.load_face_database()

    # 出力設定
    if output_file:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    else:
        out = None

    # ログファイル設定
    perf_log_file, perf_log_f, perf_log_writer = initialize_perf_log(
        enable_perf_log, input_file, model_path, log_type="face_recognition"
    )

    if perf_log_writer:
        perf_log_writer.writerow(["# Video Properties", f"{width}x{height}", f"{fps}fps"])
        perf_log_writer.writerow(
            [
                "# Face Recognition",
                f"confidence: {face_confidence}, tolerance: {recognition_tolerance}",
            ]
        )
        perf_log_writer.writerow([])

    # 統計情報
    stay_info = {}
    total_faces_detected = 0
    total_faces_recognized = 0
    frames_with_faces = 0
    unique_people = set()

    try:
        frame_idx = 0
        start_time = time.time()
        fps_buffer = []
        print("処理開始...")

        # 処理開始
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_idx += 1
            current_time = time.time()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # 1. 人物検出と追跡
            tracks, detection_time_ms, tracking_time_ms, num_detections, num_tracks, _ = (
                process_frame_for_tracking(frame_rgb, model, tracker)
            )

            # 2. 顔検出（5フレームに1回実行でパフォーマンス向上）
            face_detection_time_ms = 0
            face_recognition_time_ms = 0
            face_results = []

            if frame_idx % 5 == 0:  # 5フレームごとに顔検出・認識
                # 顔検出
                face_start = time.time()
                face_locations = face_system.detect_faces(frame_rgb)
                face_detection_time_ms = (time.time() - face_start) * 1000

                if face_locations:
                    frames_with_faces += 1
                    total_faces_detected += len(face_locations)

                    # 顔認識
                    recognition_start = time.time()
                    face_results = face_system.recognize_faces(frame_rgb, face_locations)
                    face_recognition_time_ms = (time.time() - recognition_start) * 1000

                    total_faces_recognized += len(face_results)
                    for result in face_results:
                        unique_people.add(result["name"])

            # 3. 長時間滞在チェック
            stay_info, notifications, stay_check_time_ms = update_stay_times(
                tracks, stay_info, current_time, move_threshold_px, stay_threshold_sec
            )

            # 通知があれば表示
            for notification in notifications:
                print(f"Frame {frame_idx}: {notification}")

            # 描画処理
            if out or enable_video_display:
                # トラッキング情報を描画
                frame_bgr = draw_tracking_info(
                    frame_bgr, tracks, show_duration=True, stay_info=stay_info
                )

                # 顔認識結果を描画（日本語対応）
                if face_results:
                    frame_bgr = draw_faces_with_japanese(frame_bgr, face_results, face_system.font)

                # FPS情報
                current_fps = (
                    1.0 / (time.time() - current_time) if (time.time() - current_time) > 0 else 0
                )
                fps_buffer.append(current_fps)
                if len(fps_buffer) > 10:
                    fps_buffer.pop(0)
                avg_fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0

                # 統計情報テキスト
                frame_info = f"Frame: {frame_idx}/{frame_count} FPS: {avg_fps:.1f}"
                stats_info = (
                    f"Person: {num_tracks} Faces: {len(face_results)} People: {len(unique_people)}"
                )
                timing_info = f"Det: {detection_time_ms:.1f}ms Face: {face_detection_time_ms:.1f}ms Recog: {face_recognition_time_ms:.1f}ms"

                # テキスト背景
                cv2.rectangle(frame_bgr, (10, 10), (520, 115), (0, 0, 0), -1)

                # テキスト描画
                cv2.putText(
                    frame_bgr,
                    frame_info,
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    stats_info,
                    (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    timing_info,
                    (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                # 長期滞在者リスト（英語で表示）
                long_stayers = [
                    f"ID {id}: {info['stay_duration']:.1f}s"
                    for id, info in stay_info.items()
                    if info["stay_duration"] >= stay_threshold_sec
                ]

                if long_stayers:
                    cv2.rectangle(
                        frame_bgr,
                        (width - 210, 10),
                        (width - 10, 30 + 25 * len(long_stayers)),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.putText(
                        frame_bgr,
                        "Long Stay:",
                        (width - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                    for i, stayer in enumerate(long_stayers):
                        cv2.putText(
                            frame_bgr,
                            stayer,
                            (width - 200, 30 + 25 * (i + 1)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )

                # 出力・表示
                if out:
                    out.write(frame_bgr)
                if enable_video_display:
                    cv2.imshow("Face Recognition + Long Stay Detection", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        break

            # ログ記録
            if perf_log_writer and frame_idx % max(1, int(fps)) == 0:
                current_mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                total_time_ms = (
                    detection_time_ms
                    + tracking_time_ms
                    + face_detection_time_ms
                    + face_recognition_time_ms
                    + stay_check_time_ms
                )
                perf_log_writer.writerow(
                    [
                        frame_idx,
                        datetime.now().strftime("%H:%M:%S.%f")[:-3],
                        f"{detection_time_ms:.2f}",
                        f"{tracking_time_ms:.2f}",
                        f"{face_detection_time_ms:.2f}",
                        f"{face_recognition_time_ms:.2f}",
                        f"{stay_check_time_ms:.2f}",
                        f"{total_time_ms:.2f}",
                        num_detections,
                        num_tracks,
                        len(face_results),
                        len(unique_people),
                        f"{avg_fps:.2f}",
                        f"{current_mem_usage:.2f}",
                        Path(model_path).name,
                        "bytetrack+face_recognition",
                        f"Stay:{stay_threshold_sec}s Face:{face_confidence} Tol:{recognition_tolerance}",
                    ]
                )

            # 進捗表示
            if frame_idx % 30 == 0:
                elapsed_time = time.time() - start_time
                processing_speed = frame_idx / elapsed_time if elapsed_time > 0 else 0
                face_detection_rate = (
                    frames_with_faces / (frame_idx // 5) * 100 if frame_idx > 0 else 0
                )
                print(
                    f"進捗: {frame_idx}/{frame_count} ({frame_idx / frame_count * 100:.1f}%) | "
                    f"処理速度: {processing_speed:.2f} FPS | "
                    f"顔検出率: {face_detection_rate:.1f}% | "
                    f"認識済み人数: {len(unique_people)}"
                )

    except KeyboardInterrupt:
        print("\n処理がユーザーによって中断されました。")
    finally:
        # 終了処理
        cap.release()
        if out:
            out.release()
        if enable_video_display:
            cv2.destroyAllWindows()

        # 顔データベース保存
        face_system.save_face_database()
        face_system.close()

        if perf_log_f:
            perf_log_f.close()

        # 最終統計
        print("\n=== 処理完了 ===")
        print(f"総フレーム数: {frame_idx}")
        print(f"顔検出フレーム: {frames_with_faces}")
        print(f"総検出顔数: {total_faces_detected}")
        print(f"総認識顔数: {total_faces_recognized}")
        print(f"認識済み人数: {len(unique_people)}")
        print(f"出力ファイル: {output_file}")
        print(f"ログファイル: {perf_log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="顔認識統合長時間滞在検出システム（段階的実装版）")
    parser.add_argument("--input", type=str, required=True, help="入力動画ファイル")
    parser.add_argument(
        "--output",
        type=str,
        default="output/face_recognition_tracking.mp4",
        help="出力動画ファイル",
    )
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLOモデル")
    parser.add_argument("--enable_perf_log", action="store_true", help="パフォーマンスログ有効化")
    parser.add_argument("--enable_video_display", action="store_true", help="リアルタイム表示")
    parser.add_argument("--device", type=str, default="", help="推論デバイス")
    parser.add_argument(
        "--stay_threshold_sec", type=float, default=5.0, help="長時間滞在閾値（秒）"
    )
    parser.add_argument(
        "--move_threshold_px", type=float, default=30.0, help="移動判定閾値（ピクセル）"
    )
    parser.add_argument("--conf", type=float, default=0.3, help="YOLO信頼度閾値")
    parser.add_argument("--face_confidence", type=float, default=0.7, help="顔検出信頼度閾値")
    parser.add_argument("--recognition_tolerance", type=float, default=0.6, help="顔認識許容値")

    args = parser.parse_args()

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    main(
        input_file=args.input,
        output_file=args.output,
        model_path=args.model,
        enable_perf_log=args.enable_perf_log,
        enable_video_display=args.enable_video_display,
        device=args.device,
        stay_threshold_sec=args.stay_threshold_sec,
        move_threshold_px=args.move_threshold_px,
        conf=args.conf,
        face_confidence=args.face_confidence,
        recognition_tolerance=args.recognition_tolerance,
    )
