"""
顔認識モジュール
face_recognitionライブラリを使用した高精度な顔認識機能
"""

import logging
import os
import pickle
import time
from typing import Any

import cv2
import numpy as np

import face_recognition

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    face_recognitionライブラリを使用した顔認識クラス
    """

    def __init__(self, config: dict[str, Any]):
        """
        顔認識器の初期化

        Args:
            config: 設定辞書
        """
        self.config = config
        face_config = config.get("face_recognition", {})

        # 設定パラメータ
        self.model = face_config.get("model", "large")
        self.num_jitters = face_config.get("num_jitters", 1)
        self.tolerance = face_config.get("tolerance", 0.6)
        self.unknown_threshold = face_config.get("unknown_threshold", 0.8)
        self.encoding_quality_threshold = face_config.get("encoding_quality_threshold", 0.5)

        # 顔エンコーディング管理
        self.known_face_encodings: list[np.ndarray] = []
        self.known_face_names: list[str] = []
        self.known_face_ids: list[int] = []

        # パフォーマンス追跡
        self.recognition_times = []
        self.frame_count = 0

        # キャッシュ設定
        cache_config = config.get("performance", {})
        self.cache_size = cache_config.get("cache_size", 100)
        self.face_cache: dict[
            str, tuple[int, float, float]
        ] = {}  # face_hash -> (person_id, confidence, timestamp)

        logger.info(
            f"FaceRecognizer initialized with model: {self.model}, tolerance: {self.tolerance}"
        )

    def encode_face(self, face_image: np.ndarray) -> np.ndarray | None:
        """
        顔画像から特徴量エンコーディングを生成

        Args:
            face_image: 顔画像 (BGR)

        Returns:
            顔エンコーディング配列 or None
        """
        try:
            # BGRからRGBに変換
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # 顔位置検出
            face_locations = face_recognition.face_locations(rgb_image, model="hog")

            if not face_locations:
                logger.debug("No face locations found in the image")
                return None

            # 最初の顔のエンコーディングを生成
            face_encodings = face_recognition.face_encodings(
                rgb_image, face_locations, num_jitters=self.num_jitters, model=self.model
            )

            if face_encodings:
                return face_encodings[0]
            return None

        except Exception as e:
            logger.error(f"Face encoding error: {e}")
            return None

    def recognize_faces(self, face_crops: list[np.ndarray]) -> list[dict[str, Any]]:
        """
        複数の顔画像を認識

        Args:
            face_crops: 顔画像のリスト (BGR)

        Returns:
            認識結果のリスト [{'person_id': int, 'name': str, 'confidence': float, 'is_known': bool}, ...]
        """
        start_time = time.time()

        recognition_results = []

        for face_crop in face_crops:
            result = self.recognize_single_face(face_crop)
            recognition_results.append(result)

        # パフォーマンス追跡
        recognition_time = (time.time() - start_time) * 1000
        self.recognition_times.append(recognition_time)
        self.frame_count += 1

        logger.debug(f"Recognized {len(face_crops)} faces in {recognition_time:.2f}ms")

        return recognition_results

    def recognize_single_face(self, face_image: np.ndarray) -> dict[str, Any]:
        """
        単一の顔画像を認識

        Args:
            face_image: 顔画像 (BGR)

        Returns:
            認識結果辞書
        """
        # デフォルト結果
        default_result = {
            "person_id": -1,
            "name": "Unknown",
            "confidence": 0.0,
            "is_known": False,
            "encoding": None,
        }

        # 顔エンコーディング生成
        face_encoding = self.encode_face(face_image)
        if face_encoding is None:
            return default_result

        # キャッシュチェック
        face_hash = self._compute_face_hash(face_encoding)
        if face_hash in self.face_cache:
            cached_result = self.face_cache[face_hash]
            person_id, confidence, timestamp = cached_result

            # キャッシュの有効性チェック（例：1秒以内）
            if time.time() - timestamp < 1.0:
                if person_id >= 0:
                    name = self.known_face_names[self.known_face_ids.index(person_id)]
                    return {
                        "person_id": person_id,
                        "name": name,
                        "confidence": confidence,
                        "is_known": True,
                        "encoding": face_encoding,
                    }
                return default_result

        # 既知の顔と比較
        if self.known_face_encodings:
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]

            # 閾値チェック
            if best_distance <= self.tolerance:
                person_id = self.known_face_ids[best_match_index]
                name = self.known_face_names[best_match_index]
                confidence = 1.0 - best_distance  # 距離を信頼度に変換

                result = {
                    "person_id": person_id,
                    "name": name,
                    "confidence": confidence,
                    "is_known": True,
                    "encoding": face_encoding,
                }

                # キャッシュに保存
                self._update_cache(face_hash, person_id, confidence)

                return result

        # 未知の人物
        self._update_cache(face_hash, -1, 0.0)
        default_result["encoding"] = face_encoding
        return default_result

    def add_known_person(self, person_id: int, name: str, face_encoding: np.ndarray) -> bool:
        """
        既知の人物を追加

        Args:
            person_id: 人物ID
            name: 人物名
            face_encoding: 顔エンコーディング

        Returns:
            追加成功フラグ
        """
        try:
            # 重複チェック
            if person_id in self.known_face_ids:
                # 既存の人物の顔エンコーディングを更新
                index = self.known_face_ids.index(person_id)
                self.known_face_encodings[index] = face_encoding
                self.known_face_names[index] = name
                logger.info(f"Updated face encoding for person {person_id}: {name}")
            else:
                # 新しい人物を追加
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(name)
                self.known_face_ids.append(person_id)
                logger.info(f"Added new person {person_id}: {name}")

            return True

        except Exception as e:
            logger.error(f"Error adding known person: {e}")
            return False

    def remove_known_person(self, person_id: int) -> bool:
        """
        既知の人物を削除

        Args:
            person_id: 人物ID

        Returns:
            削除成功フラグ
        """
        try:
            if person_id in self.known_face_ids:
                index = self.known_face_ids.index(person_id)

                del self.known_face_encodings[index]
                del self.known_face_names[index]
                del self.known_face_ids[index]

                logger.info(f"Removed person {person_id}")
                return True
            logger.warning(f"Person {person_id} not found")
            return False

        except Exception as e:
            logger.error(f"Error removing person: {e}")
            return False

    def load_known_faces(self, encodings_path: str) -> bool:
        """
        保存された顔エンコーディングを読み込み

        Args:
            encodings_path: エンコーディングファイルのパス

        Returns:
            読み込み成功フラグ
        """
        try:
            if os.path.exists(encodings_path):
                with open(encodings_path, "rb") as f:
                    data = pickle.load(f)

                self.known_face_encodings = data.get("encodings", [])
                self.known_face_names = data.get("names", [])
                self.known_face_ids = data.get("ids", [])

                logger.info(
                    f"Loaded {len(self.known_face_encodings)} known faces from {encodings_path}"
                )
                return True
            logger.warning(f"Encodings file not found: {encodings_path}")
            return False

        except Exception as e:
            logger.error(f"Error loading known faces: {e}")
            return False

    def save_known_faces(self, encodings_path: str) -> bool:
        """
        顔エンコーディングを保存

        Args:
            encodings_path: 保存先パス

        Returns:
            保存成功フラグ
        """
        try:
            # ディレクトリ作成
            os.makedirs(os.path.dirname(encodings_path), exist_ok=True)

            data = {
                "encodings": self.known_face_encodings,
                "names": self.known_face_names,
                "ids": self.known_face_ids,
            }

            with open(encodings_path, "wb") as f:
                pickle.dump(data, f)

            logger.info(f"Saved {len(self.known_face_encodings)} known faces to {encodings_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving known faces: {e}")
            return False

    def _compute_face_hash(self, face_encoding: np.ndarray) -> str:
        """
        顔エンコーディングのハッシュを計算

        Args:
            face_encoding: 顔エンコーディング

        Returns:
            ハッシュ文字列
        """
        # エンコーディングを丸めてハッシュ化
        rounded_encoding = np.round(face_encoding, 3)
        return hash(rounded_encoding.tobytes()).__str__()

    def _update_cache(self, face_hash: str, person_id: int, confidence: float):
        """
        顔認識キャッシュを更新

        Args:
            face_hash: 顔ハッシュ
            person_id: 人物ID
            confidence: 信頼度
        """
        current_time = time.time()

        # キャッシュサイズ制限
        if len(self.face_cache) >= self.cache_size:
            # 最も古いエントリを削除
            oldest_key = min(self.face_cache.keys(), key=lambda k: self.face_cache[k][2])
            del self.face_cache[oldest_key]

        self.face_cache[face_hash] = (person_id, confidence, current_time)

    def get_performance_stats(self) -> dict[str, float]:
        """
        パフォーマンス統計を取得

        Returns:
            統計辞書
        """
        if not self.recognition_times:
            return {}

        avg_time = np.mean(self.recognition_times)
        max_time = np.max(self.recognition_times)
        min_time = np.min(self.recognition_times)
        avg_fps = 1000.0 / avg_time if avg_time > 0 else 0

        return {
            "avg_recognition_time_ms": avg_time,
            "max_recognition_time_ms": max_time,
            "min_recognition_time_ms": min_time,
            "avg_fps": avg_fps,
            "total_frames": self.frame_count,
            "known_faces_count": len(self.known_face_encodings),
            "cache_size": len(self.face_cache),
        }

    def reset_stats(self):
        """統計をリセット"""
        self.recognition_times = []
        self.frame_count = 0

    def clear_cache(self):
        """キャッシュをクリア"""
        self.face_cache.clear()
        logger.info("Face recognition cache cleared")
