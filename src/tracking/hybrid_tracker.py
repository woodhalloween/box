"""
ハイブリッド追跡システム
顔認識とByteTrackを統合したID付与精度向上システム
"""

import logging
import time
from collections import defaultdict, deque
from typing import Any

import numpy as np
import yaml

from ..face_recognition.face_database import FaceDatabase

# 顔認識関連モジュール
from ..face_recognition.face_detector import FaceDetector
from ..face_recognition.face_recognizer import FaceRecognizer

logger = logging.getLogger(__name__)


class HybridTracker:
    """
    顔認識とByteTrackを統合したハイブリッド追跡システム
    """

    def __init__(self, config_path: str):
        """
        ハイブリッド追跡器の初期化

        Args:
            config_path: 設定ファイルのパス
        """
        # 設定読み込み
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # コンポーネント初期化
        self.face_detector = FaceDetector(self.config)
        self.face_recognizer = FaceRecognizer(self.config)
        self.face_database = FaceDatabase(self.config)

        # 統合設定
        integration_config = self.config.get("tracking_integration", {})
        self.face_weight = integration_config.get("face_weight", 0.7)
        self.position_weight = integration_config.get("position_weight", 0.3)
        self.id_consistency_buffer = integration_config.get("id_consistency_buffer", 10)
        self.occlusion_timeout = integration_config.get("occlusion_timeout", 30)
        self.re_identification_frames = integration_config.get("re_identification_frames", 5)

        # 追跡状態管理
        self.active_tracks: dict[int, dict[str, Any]] = {}  # track_id -> track_info
        self.face_to_track_mapping: dict[int, int] = {}  # person_id -> track_id
        self.track_to_face_mapping: dict[int, int] = {}  # track_id -> person_id
        self.track_history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.id_consistency_buffer)
        )
        self.occlusion_counters: dict[int, int] = defaultdict(int)

        # ID生成
        self.next_stable_id = 1
        self.stable_id_mapping: dict[int, int] = {}  # track_id -> stable_id

        # パフォーマンス追跡
        self.frame_count = 0
        self.processing_times = []

        # データベースから既知の顔を読み込み
        self._load_known_faces()

        logger.info("HybridTracker initialized")

    def _load_known_faces(self):
        """データベースから既知の顔を読み込み"""
        try:
            face_encodings = self.face_database.get_all_face_encodings()

            for person_id, name, encoding in face_encodings:
                self.face_recognizer.add_known_person(person_id, name, encoding)

            logger.info(f"Loaded {len(face_encodings)} known faces from database")

        except Exception as e:
            logger.error(f"Error loading known faces: {e}")

    def process_frame(
        self, frame: np.ndarray, basic_tracks: list[Any]
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """
        フレームを処理してハイブリッド追跡を実行

        Args:
            frame: 入力フレーム (BGR)
            basic_tracks: ByteTrackからの基本追跡結果

        Returns:
            (enhanced_tracks, performance_metrics)
        """
        start_time = time.time()

        # 1. 人物検出ボックスの抽出
        person_boxes = self._extract_person_boxes(basic_tracks)

        # 2. 顔検出
        face_detection_start = time.time()
        detected_faces = self.face_detector.detect_faces(frame, person_boxes)
        face_detection_time = (time.time() - face_detection_start) * 1000

        # 3. 顔認識
        face_recognition_start = time.time()
        face_crops = []
        for face in detected_faces:
            face_crop = self.face_detector.crop_face(frame, face["bbox"])
            if face_crop is not None:
                face_crops.append(face_crop)

        recognition_results = self.face_recognizer.recognize_faces(face_crops)
        face_recognition_time = (time.time() - face_recognition_start) * 1000

        # 4. ID統合・融合
        enhanced_tracks = self._integrate_ids(basic_tracks, detected_faces, recognition_results)

        # 5. 追跡状態の更新
        self._update_tracking_state(enhanced_tracks)

        # パフォーマンス記録
        total_time = (time.time() - start_time) * 1000
        self.processing_times.append(total_time)
        self.frame_count += 1

        # メトリクス計算
        performance_metrics = {
            "total_processing_time": total_time,
            "face_detection_time": face_detection_time,
            "face_recognition_time": face_recognition_time,
            "active_tracks": len(enhanced_tracks),
            "detected_faces": len(detected_faces),
            "recognized_faces": sum(1 for r in recognition_results if r["is_known"]),
        }

        # データベースにパフォーマンスログを記録
        if self.frame_count % 100 == 0:  # 100フレームごと
            fps = 1000.0 / np.mean(self.processing_times[-100:]) if self.processing_times else 0
            self.face_database.log_performance(
                fps, face_detection_time, face_recognition_time, len(enhanced_tracks)
            )

        return enhanced_tracks, performance_metrics

    def _extract_person_boxes(self, basic_tracks: list[Any]) -> list[tuple[int, int, int, int]]:
        """基本追跡結果から人物ボックスを抽出"""
        person_boxes = []

        for track in basic_tracks:
            # ByteTrackの形式に対応
            if hasattr(track, "to_ltrb"):
                # DeepSORTスタイル
                x1, y1, x2, y2 = map(int, track.to_ltrb())
            elif len(track) >= 4:
                # Numpy配列スタイル
                x1, y1, x2, y2 = map(int, track[:4])
            else:
                continue

            person_boxes.append((x1, y1, x2, y2))

        return person_boxes

    def _integrate_ids(
        self,
        basic_tracks: list[Any],
        detected_faces: list[dict[str, Any]],
        recognition_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """ID統合・融合処理"""
        enhanced_tracks = []

        # 顔と人物の対応付け
        face_track_associations = self._associate_faces_with_tracks(
            basic_tracks, detected_faces, recognition_results
        )

        for i, track in enumerate(basic_tracks):
            # 基本追跡情報の取得
            if hasattr(track, "track_id"):
                basic_track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
            elif len(track) >= 5:
                x1, y1, x2, y2, basic_track_id = map(int, track[:5])
            else:
                continue

            # 対応する顔認識結果を取得
            face_result = face_track_associations.get(i)

            # 統合ID決定
            stable_id = self._determine_stable_id(basic_track_id, face_result)

            # 強化された追跡情報を作成
            enhanced_track = {
                "stable_id": stable_id,
                "basic_track_id": basic_track_id,
                "bbox": (x1, y1, x2, y2),
                "person_id": face_result.get("person_id", -1) if face_result else -1,
                "person_name": face_result.get("name", "Unknown") if face_result else "Unknown",
                "face_confidence": face_result.get("confidence", 0.0) if face_result else 0.0,
                "is_known_person": face_result.get("is_known", False) if face_result else False,
                "tracking_source": self._determine_tracking_source(face_result),
                "frame_id": self.frame_count,
            }

            enhanced_tracks.append(enhanced_track)

        return enhanced_tracks

    def _associate_faces_with_tracks(
        self,
        basic_tracks: list[Any],
        detected_faces: list[dict[str, Any]],
        recognition_results: list[dict[str, Any]],
    ) -> dict[int, dict[str, Any]]:
        """顔と追跡結果の対応付け"""
        associations = {}

        if not detected_faces or not recognition_results:
            return associations

        # 各追跡に対して最も近い顔を探す
        for track_idx, track in enumerate(basic_tracks):
            if hasattr(track, "to_ltrb"):
                tx1, ty1, tx2, ty2 = map(int, track.to_ltrb())
            elif len(track) >= 4:
                tx1, ty1, tx2, ty2 = map(int, track[:4])
            else:
                continue

            track_center = ((tx1 + tx2) / 2, (ty1 + ty2) / 2)

            best_face_idx = -1
            best_distance = float("inf")

            for face_idx, face in enumerate(detected_faces):
                fx1, fy1, fx2, fy2 = face["bbox"]
                face_center = ((fx1 + fx2) / 2, (fy1 + fy2) / 2)

                # 距離計算
                distance = np.sqrt(
                    (track_center[0] - face_center[0]) ** 2
                    + (track_center[1] - face_center[1]) ** 2
                )

                # IoU計算（より正確な対応付けのため）
                iou = self._calculate_iou((tx1, ty1, tx2, ty2), (fx1, fy1, fx2, fy2))

                # 組み合わせスコア（距離とIoUを考慮）
                if iou > 0.1:  # 最小重複要件
                    combined_score = distance / (iou + 0.1)  # IoUが高いほどスコアが低い

                    if combined_score < best_distance:
                        best_distance = combined_score
                        best_face_idx = face_idx

            # 最適な対応が見つかった場合
            if best_face_idx >= 0 and best_face_idx < len(recognition_results):
                associations[track_idx] = recognition_results[best_face_idx]

        return associations

    def _calculate_iou(
        self, box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]
    ) -> float:
        """IoU（Intersection over Union）計算"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # 交差領域
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # 統合領域
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _determine_stable_id(self, basic_track_id: int, face_result: dict[str, Any] | None) -> int:
        """安定ID決定アルゴリズム"""
        # 既存の安定IDがある場合
        if basic_track_id in self.stable_id_mapping:
            stable_id = self.stable_id_mapping[basic_track_id]

            # 顔認識結果による確認・更新
            if face_result and face_result["is_known"]:
                person_id = face_result["person_id"]

                # 顔ベースのマッピングチェック
                if person_id in self.face_to_track_mapping:
                    existing_track_id = self.face_to_track_mapping[person_id]
                    if existing_track_id != basic_track_id:
                        # ID競合解決
                        stable_id = self._resolve_id_conflict(
                            basic_track_id, person_id, face_result
                        )
                else:
                    # 新しい顔-追跡マッピング
                    self.face_to_track_mapping[person_id] = basic_track_id
                    self.track_to_face_mapping[basic_track_id] = person_id

            return stable_id

        # 新規安定ID生成
        if face_result and face_result["is_known"]:
            person_id = face_result["person_id"]

            # 既知の人物の場合、person_idベースのIDを使用
            if person_id in self.face_to_track_mapping:
                # 既存の追跡がある場合
                existing_track_id = self.face_to_track_mapping[person_id]
                existing_stable_id = self.stable_id_mapping.get(existing_track_id, person_id)

                # 新しい追跡に同じ安定IDを割り当て
                self.stable_id_mapping[basic_track_id] = existing_stable_id
                self.face_to_track_mapping[person_id] = basic_track_id
                self.track_to_face_mapping[basic_track_id] = person_id

                return existing_stable_id
            # 新しい人物の再出現
            stable_id = person_id
            self.stable_id_mapping[basic_track_id] = stable_id
            self.face_to_track_mapping[person_id] = basic_track_id
            self.track_to_face_mapping[basic_track_id] = person_id

            return stable_id

        # 未知の人物または顔認識失敗の場合
        stable_id = self.next_stable_id
        self.next_stable_id += 1
        self.stable_id_mapping[basic_track_id] = stable_id

        return stable_id

    def _resolve_id_conflict(
        self, basic_track_id: int, person_id: int, face_result: dict[str, Any]
    ) -> int:
        """ID競合解決"""
        # 信頼度ベースの解決
        current_confidence = face_result["confidence"]

        # より高い信頼度の追跡にIDを割り当て
        existing_track_id = self.face_to_track_mapping[person_id]

        # 既存追跡の履歴をチェック
        if basic_track_id in self.track_history:
            recent_confidences = [
                entry.get("face_confidence", 0.0)
                for entry in list(self.track_history[basic_track_id])[-5:]
            ]
            avg_confidence = np.mean(recent_confidences) if recent_confidences else 0.0

            if current_confidence > avg_confidence + 0.1:  # 閾値
                # 新しい追跡により高い信頼度：IDを移譲
                old_stable_id = self.stable_id_mapping.get(existing_track_id, person_id)
                self.stable_id_mapping[basic_track_id] = old_stable_id
                self.face_to_track_mapping[person_id] = basic_track_id
                self.track_to_face_mapping[basic_track_id] = person_id

                # 古い追跡には新しいIDを割り当て
                if existing_track_id in self.stable_id_mapping:
                    self.stable_id_mapping[existing_track_id] = self.next_stable_id
                    self.next_stable_id += 1

                return old_stable_id

        # デフォルト：既存のIDを維持
        return self.stable_id_mapping.get(basic_track_id, self.next_stable_id)

    def _determine_tracking_source(self, face_result: dict[str, Any] | None) -> str:
        """追跡ソースの決定"""
        if face_result:
            if face_result["is_known"]:
                return "face_recognition"
            return "face_detection"
        return "position_only"

    def _update_tracking_state(self, enhanced_tracks: list[dict[str, Any]]):
        """追跡状態の更新"""
        current_track_ids = set()

        for track in enhanced_tracks:
            stable_id = track["stable_id"]
            basic_track_id = track["basic_track_id"]

            current_track_ids.add(stable_id)

            # 履歴更新
            self.track_history[stable_id].append(
                {
                    "frame_id": self.frame_count,
                    "bbox": track["bbox"],
                    "face_confidence": track["face_confidence"],
                    "person_id": track["person_id"],
                    "tracking_source": track["tracking_source"],
                }
            )

            # 遮蔽カウンターリセット
            self.occlusion_counters[stable_id] = 0

            # データベース更新
            if track["is_known_person"]:
                self.face_database.update_person_appearance(track["person_id"])

        # 消失した追跡の処理
        all_stable_ids = set(self.track_history.keys())
        disappeared_ids = all_stable_ids - current_track_ids

        for stable_id in disappeared_ids:
            self.occlusion_counters[stable_id] += 1

            # タイムアウト処理
            if self.occlusion_counters[stable_id] > self.occlusion_timeout:
                self._cleanup_disappeared_track(stable_id)

    def _cleanup_disappeared_track(self, stable_id: int):
        """消失した追跡のクリーンアップ"""
        if stable_id in self.track_history:
            del self.track_history[stable_id]

        if stable_id in self.occlusion_counters:
            del self.occlusion_counters[stable_id]

        # マッピングのクリーンアップ
        track_ids_to_remove = [
            tid for tid, sid in self.stable_id_mapping.items() if sid == stable_id
        ]
        for track_id in track_ids_to_remove:
            if track_id in self.stable_id_mapping:
                del self.stable_id_mapping[track_id]
            if track_id in self.track_to_face_mapping:
                person_id = self.track_to_face_mapping[track_id]
                del self.track_to_face_mapping[track_id]
                if person_id in self.face_to_track_mapping:
                    del self.face_to_track_mapping[person_id]

    def register_new_person(self, name: str, face_image: np.ndarray) -> int | None:
        """新しい人物の登録"""
        # 顔エンコーディング生成
        face_encoding = self.face_recognizer.encode_face(face_image)
        if face_encoding is None:
            return None

        # データベースに追加
        person_id = self.face_database.add_person(name, face_encoding)
        if person_id:
            # 顔認識器に追加
            self.face_recognizer.add_known_person(person_id, name, face_encoding)
            logger.info(f"Registered new person: {name} (ID: {person_id})")

        return person_id

    def get_performance_stats(self) -> dict[str, Any]:
        """パフォーマンス統計取得"""
        stats = {
            "frame_count": self.frame_count,
            "active_tracks": len(self.track_history),
            "known_persons": len(self.face_to_track_mapping),
        }

        if self.processing_times:
            stats.update(
                {
                    "avg_processing_time_ms": np.mean(self.processing_times),
                    "avg_fps": 1000.0 / np.mean(self.processing_times),
                }
            )

        # コンポーネント統計
        stats.update(self.face_detector.get_performance_stats())
        stats.update(self.face_recognizer.get_performance_stats())
        stats.update(self.face_database.get_statistics())

        return stats
