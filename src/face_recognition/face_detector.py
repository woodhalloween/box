"""
顔検出モジュール
MediaPipeを使用した高精度な顔検出機能
"""

import logging
import time
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    MediaPipeを使用した顔検出クラス
    """

    def __init__(self, config: dict[str, Any]):
        """
        顔検出器の初期化

        Args:
            config: 設定辞書
        """
        self.config = config
        face_config = config.get("face_detection", {})

        # MediaPipe初期化
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0: 2mの範囲内、1: 5mの範囲内
            min_detection_confidence=face_config.get("min_detection_confidence", 0.7),
        )

        # 設定パラメータ
        self.max_num_faces = face_config.get("max_num_faces", 10)
        self.min_face_size = face_config.get("min_face_size", 30)

        # パフォーマンス追跡
        self.detection_times = []
        self.frame_count = 0

        logger.info(f"FaceDetector initialized with config: {face_config}")

    def detect_faces(
        self, frame: np.ndarray, person_boxes: list[tuple[int, int, int, int]] | None = None
    ) -> list[dict[str, Any]]:
        """
        フレーム内の顔を検出

        Args:
            frame: 入力フレーム (BGR)
            person_boxes: 人物検出のバウンディングボックス [(x1, y1, x2, y2), ...]

        Returns:
            検出された顔のリスト [{'bbox': (x1, y1, x2, y2), 'confidence': float, 'landmarks': dict}, ...]
        """
        start_time = time.time()

        # BGRからRGBに変換
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]

        detected_faces = []

        try:
            # 人物検出ボックスが提供されている場合、その領域内で顔検出
            if person_boxes:
                for person_box in person_boxes:
                    x1, y1, x2, y2 = person_box
                    # 少し領域を拡張
                    margin = 20
                    x1_exp = max(0, x1 - margin)
                    y1_exp = max(0, y1 - margin)
                    x2_exp = min(width, x2 + margin)
                    y2_exp = min(height, y2 + margin)

                    # 人物領域をクロップ
                    person_crop = rgb_frame[y1_exp:y2_exp, x1_exp:x2_exp]

                    if person_crop.size > 0:
                        faces_in_person = self._detect_faces_in_crop(person_crop, x1_exp, y1_exp)
                        detected_faces.extend(faces_in_person)
            else:
                # フレーム全体で顔検出
                faces_in_frame = self._detect_faces_in_crop(rgb_frame, 0, 0)
                detected_faces.extend(faces_in_frame)

        except Exception as e:
            logger.error(f"Face detection error: {e}")

        # パフォーマンス追跡
        detection_time = (time.time() - start_time) * 1000
        self.detection_times.append(detection_time)
        self.frame_count += 1

        # 顔サイズフィルタリング
        filtered_faces = self._filter_faces_by_size(detected_faces)

        logger.debug(f"Detected {len(filtered_faces)} faces in {detection_time:.2f}ms")

        return filtered_faces

    def _detect_faces_in_crop(
        self, rgb_crop: np.ndarray, offset_x: int, offset_y: int
    ) -> list[dict[str, Any]]:
        """
        クロップされた画像内での顔検出

        Args:
            rgb_crop: RGBクロップ画像
            offset_x: X方向オフセット
            offset_y: Y方向オフセット

        Returns:
            検出された顔のリスト
        """
        faces = []

        results = self.face_detection.process(rgb_crop)

        if results.detections:
            crop_height, crop_width = rgb_crop.shape[:2]

            for detection in results.detections[: self.max_num_faces]:
                # バウンディングボックスの取得
                bbox = detection.location_data.relative_bounding_box

                # 相対座標を絶対座標に変換
                x1 = int(bbox.xmin * crop_width) + offset_x
                y1 = int(bbox.ymin * crop_height) + offset_y
                x2 = int((bbox.xmin + bbox.width) * crop_width) + offset_x
                y2 = int((bbox.ymin + bbox.height) * crop_height) + offset_y

                # 信頼度取得
                confidence = detection.score[0] if detection.score else 0.0

                # ランドマーク取得
                landmarks = self._extract_landmarks(
                    detection, crop_width, crop_height, offset_x, offset_y
                )

                face_info = {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "landmarks": landmarks,
                    "detection_obj": detection,
                }

                faces.append(face_info)

        return faces

    def _extract_landmarks(
        self, detection, crop_width: int, crop_height: int, offset_x: int, offset_y: int
    ) -> dict[str, tuple[int, int]]:
        """
        顔のランドマークを抽出

        Args:
            detection: MediaPipe検出結果
            crop_width: クロップ幅
            crop_height: クロップ高さ
            offset_x: X方向オフセット
            offset_y: Y方向オフセット

        Returns:
            ランドマーク辞書
        """
        landmarks = {}

        if hasattr(detection.location_data, "relative_keypoints"):
            keypoints = detection.location_data.relative_keypoints

            landmark_names = [
                "right_eye",
                "left_eye",
                "nose_tip",
                "mouth_center",
                "right_ear_tragion",
                "left_ear_tragion",
            ]

            for i, kp in enumerate(keypoints):
                if i < len(landmark_names):
                    x = int(kp.x * crop_width) + offset_x
                    y = int(kp.y * crop_height) + offset_y
                    landmarks[landmark_names[i]] = (x, y)

        return landmarks

    def _filter_faces_by_size(self, faces: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        顔サイズによるフィルタリング

        Args:
            faces: 検出された顔のリスト

        Returns:
            フィルタリングされた顔のリスト
        """
        filtered_faces = []

        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            face_width = x2 - x1
            face_height = y2 - y1

            # 最小サイズチェック
            if face_width >= self.min_face_size and face_height >= self.min_face_size:
                # 顔の品質スコア計算
                face["quality_score"] = self._calculate_face_quality(face)
                filtered_faces.append(face)

        return filtered_faces

    def _calculate_face_quality(self, face: dict[str, Any]) -> float:
        """
        顔の品質スコアを計算

        Args:
            face: 顔情報辞書

        Returns:
            品質スコア (0.0-1.0)
        """
        x1, y1, x2, y2 = face["bbox"]
        confidence = face["confidence"]

        # 基本的な品質スコア（信頼度ベース）
        quality_score = confidence

        # サイズボーナス（大きい顔ほど高品質）
        face_size = (x2 - x1) * (y2 - y1)
        size_bonus = min(face_size / (100 * 100), 1.0) * 0.2
        quality_score += size_bonus

        # ランドマーク完全性ボーナス
        landmarks = face.get("landmarks", {})
        landmark_bonus = len(landmarks) / 6.0 * 0.1
        quality_score += landmark_bonus

        return min(quality_score, 1.0)

    def crop_face(
        self, frame: np.ndarray, face_bbox: tuple[int, int, int, int], padding: float = 0.2
    ) -> np.ndarray | None:
        """
        顔領域をクロップ

        Args:
            frame: 入力フレーム
            face_bbox: 顔のバウンディングボックス (x1, y1, x2, y2)
            padding: パディング率

        Returns:
            クロップされた顔画像
        """
        x1, y1, x2, y2 = face_bbox
        height, width = frame.shape[:2]

        # パディングを追加
        face_width = x2 - x1
        face_height = y2 - y1

        pad_x = int(face_width * padding)
        pad_y = int(face_height * padding)

        # 拡張された領域
        x1_exp = max(0, x1 - pad_x)
        y1_exp = max(0, y1 - pad_y)
        x2_exp = min(width, x2 + pad_x)
        y2_exp = min(height, y2 + pad_y)

        # クロップ
        face_crop = frame[y1_exp:y2_exp, x1_exp:x2_exp]

        if face_crop.size > 0:
            return face_crop
        return None

    def get_performance_stats(self) -> dict[str, float]:
        """
        パフォーマンス統計を取得

        Returns:
            統計辞書
        """
        if not self.detection_times:
            return {}

        avg_time = np.mean(self.detection_times)
        max_time = np.max(self.detection_times)
        min_time = np.min(self.detection_times)
        avg_fps = 1000.0 / avg_time if avg_time > 0 else 0

        return {
            "avg_detection_time_ms": avg_time,
            "max_detection_time_ms": max_time,
            "min_detection_time_ms": min_time,
            "avg_fps": avg_fps,
            "total_frames": self.frame_count,
        }

    def reset_stats(self):
        """統計をリセット"""
        self.detection_times = []
        self.frame_count = 0

    def __del__(self):
        """デストラクタ"""
        if hasattr(self, "face_detection"):
            self.face_detection.close()
