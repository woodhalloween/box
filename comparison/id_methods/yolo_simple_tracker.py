"""
YOLO簡易トラッカー
位置ベースの簡単なID付与を行う
"""

import os
import sys
from typing import Any

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comparison.id_methods.base_tracker import BaseTracker, Detection, Track


class YoloSimpleTracker(BaseTracker):
    """
    YOLO単体での簡易ID付与
    位置ベースの距離計算による最近傍マッチング
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.3,
        device: str = "",
        distance_threshold: float = 50.0,
        max_disappeared_frames: int = 10,
    ):
        """
        YOLO簡易トラッカーの初期化

        Args:
            model_path: YOLOモデルのパス
            confidence: 検出信頼度閾値
            device: 実行デバイス
            distance_threshold: マッチング距離閾値（ピクセル）
            max_disappeared_frames: 追跡を諦めるまでのフレーム数
        """
        super().__init__(model_path, confidence, device)

        self.distance_threshold = distance_threshold
        self.max_disappeared_frames = max_disappeared_frames

        # 追跡状態管理
        self.active_tracks: list[Track] = []
        self.disappeared_tracks: dict[int, int] = {}  # track_id -> disappeared_frames

    def update_tracks(self, detections: list[Detection]) -> list[Track]:
        """
        検出結果を基に追跡を更新

        Args:
            detections: 現在フレームの検出結果

        Returns:
            更新された追跡結果のリスト
        """
        # 既存トラックとの距離行列を計算
        if len(self.active_tracks) == 0:
            # 初回またはすべてのトラックが消失した場合
            new_tracks = []
            for detection in detections:
                track = Track(
                    track_id=self.get_new_id(),
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    frame_id=detection.frame_id,
                    age=1,
                    time_since_update=0,
                )
                new_tracks.append(track)

            self.active_tracks = new_tracks
            return new_tracks

        # 距離行列の計算
        distance_matrix = self._calculate_distance_matrix(detections, self.active_tracks)

        # マッチング実行
        matched_pairs, unmatched_detections, unmatched_tracks = self._assign_matches(
            distance_matrix, self.distance_threshold
        )

        # マッチしたトラックの更新
        updated_tracks = []
        for det_idx, track_idx in matched_pairs:
            detection = detections[det_idx]
            track = self.active_tracks[track_idx]

            # トラック情報の更新
            updated_track = Track(
                track_id=track.track_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                class_id=detection.class_id,
                frame_id=detection.frame_id,
                age=track.age + 1,
                time_since_update=0,
            )
            updated_tracks.append(updated_track)

            # disappeared_tracksからも削除（再出現の場合）
            if track.track_id in self.disappeared_tracks:
                del self.disappeared_tracks[track.track_id]

        # マッチしなかったトラックの処理
        for track_idx in unmatched_tracks:
            track = self.active_tracks[track_idx]

            # 消失カウンターを増加
            if track.track_id not in self.disappeared_tracks:
                self.disappeared_tracks[track.track_id] = 1
            else:
                self.disappeared_tracks[track.track_id] += 1

            # まだ許容範囲内なら継続
            if self.disappeared_tracks[track.track_id] <= self.max_disappeared_frames:
                disappeared_track = Track(
                    track_id=track.track_id,
                    bbox=track.bbox,
                    confidence=track.confidence,
                    class_id=track.class_id,
                    frame_id=self.frame_count,
                    age=track.age + 1,
                    time_since_update=track.time_since_update + 1,
                )
                updated_tracks.append(disappeared_track)

        # 新規検出の処理
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            new_track = Track(
                track_id=self.get_new_id(),
                bbox=detection.bbox,
                confidence=detection.confidence,
                class_id=detection.class_id,
                frame_id=detection.frame_id,
                age=1,
                time_since_update=0,
            )
            updated_tracks.append(new_track)

        # 古い消失トラックの削除
        tracks_to_remove = []
        for track_id, disappeared_frames in self.disappeared_tracks.items():
            if disappeared_frames > self.max_disappeared_frames:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.disappeared_tracks[track_id]

        # アクティブトラックのリストを更新
        self.active_tracks = [track for track in updated_tracks if track.time_since_update == 0]

        return updated_tracks

    def _calculate_distance_matrix(
        self, detections: list[Detection], tracks: list[Track]
    ) -> np.ndarray:
        """
        検出とトラック間の距離行列を計算

        Args:
            detections: 検出リスト
            tracks: トラックリスト

        Returns:
            距離行列 (len(detections) x len(tracks))
        """
        if len(detections) == 0 or len(tracks) == 0:
            return np.array([])

        distance_matrix = np.zeros((len(detections), len(tracks)))

        for i, detection in enumerate(detections):
            for j, track in enumerate(tracks):
                # 中心点間距離を計算
                distance = self.calculate_center_distance(detection.bbox, track.bbox)
                distance_matrix[i, j] = distance

        return distance_matrix

    def _assign_matches(
        self, distance_matrix: np.ndarray, threshold: float
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        距離行列を基にマッチングを実行（貪欲法）

        Args:
            distance_matrix: 距離行列
            threshold: マッチング閾値

        Returns:
            (マッチしたペア, マッチしなかった検出, マッチしなかったトラック)
        """
        if distance_matrix.size == 0:
            num_detections = 0
            num_tracks = len(self.active_tracks)
            return [], list(range(num_detections)), list(range(num_tracks))

        num_detections, num_tracks = distance_matrix.shape

        matched_pairs = []
        used_detection_indices = set()
        used_track_indices = set()

        # 距離の昇順でマッチングを試行
        flat_indices = np.argsort(distance_matrix.flatten())

        for flat_idx in flat_indices:
            det_idx = flat_idx // num_tracks
            track_idx = flat_idx % num_tracks

            # 閾値を超える場合は終了
            if distance_matrix[det_idx, track_idx] > threshold:
                break

            # 既に使用済みの場合はスキップ
            if det_idx in used_detection_indices or track_idx in used_track_indices:
                continue

            # マッチング成功
            matched_pairs.append((det_idx, track_idx))
            used_detection_indices.add(det_idx)
            used_track_indices.add(track_idx)

        # マッチしなかった検出とトラックを特定
        unmatched_detections = [i for i in range(num_detections) if i not in used_detection_indices]
        unmatched_tracks = [i for i in range(num_tracks) if i not in used_track_indices]

        return matched_pairs, unmatched_detections, unmatched_tracks

    def reset(self):
        """
        トラッカーの状態をリセット
        """
        super().reset()
        self.active_tracks = []
        self.disappeared_tracks = {}

    def get_algorithm_info(self) -> dict[str, Any]:
        """
        アルゴリズム情報を取得

        Returns:
            アルゴリズム情報辞書
        """
        return {
            "method": "YOLO Simple Tracker",
            "algorithm": "Position-based nearest neighbor matching",
            "distance_threshold": self.distance_threshold,
            "max_disappeared_frames": self.max_disappeared_frames,
            "features": [
                "Center distance calculation",
                "Greedy matching algorithm",
                "Simple disappearance handling",
            ],
            "pros": ["Very fast processing", "Low memory usage", "Simple implementation"],
            "cons": [
                "Poor occlusion handling",
                "No appearance modeling",
                "Frequent ID switches in crowded scenes",
            ],
        }
