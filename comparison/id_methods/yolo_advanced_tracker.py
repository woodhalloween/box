"""
YOLO高度トラッカー（簡易版）
外観特徴量を使った高度なID付与を行う
"""

import os
import sys
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comparison.id_methods.base_tracker import BaseTracker, Detection, Track


class YoloAdvancedTracker(BaseTracker):
    """
    YOLO単体での高度ID付与
    外観特徴量と位置情報を組み合わせたマッチング
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.3,
        device: str = "",
        appearance_weight: float = 0.4,
        position_weight: float = 0.6,
        iou_threshold: float = 0.3,
        max_disappeared_frames: int = 20,
        similarity_threshold: float = 0.5,
    ):
        """
        YOLO高度トラッカーの初期化

        Args:
            model_path: YOLOモデルのパス
            confidence: 検出信頼度閾値
            device: 実行デバイス
            appearance_weight: 外観類似度の重み
            position_weight: 位置類似度の重み
            iou_threshold: IoU閾値
            max_disappeared_frames: 追跡を諦めるまでのフレーム数
            similarity_threshold: マッチング類似度閾値
        """
        super().__init__(model_path, confidence, device)

        self.appearance_weight = appearance_weight
        self.position_weight = position_weight
        self.iou_threshold = iou_threshold
        self.max_disappeared_frames = max_disappeared_frames
        self.similarity_threshold = similarity_threshold

        # 追跡状態管理
        self.active_tracks: list[Track] = []
        self.track_features: dict[int, np.ndarray] = {}
        self.disappeared_tracks: dict[int, int] = {}

        # 現在のフレーム（特徴量抽出用）
        self.current_frame = None

    def _extract_simple_features(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        簡易的な特徴量抽出（色ヒストグラム + エッジ）

        Args:
            frame: 入力フレーム
            bbox: バウンディングボックス (x1, y1, x2, y2)

        Returns:
            特徴量ベクトル
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]

        # 範囲チェック
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        # ROI抽出
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros(32)  # 固定サイズの特徴量ベクトル

        try:
            # RGB色ヒストグラム（簡易版）
            roi_resized = cv2.resize(roi, (32, 32))
            hist_b = cv2.calcHist([roi_resized], [0], None, [8], [0, 256])
            hist_g = cv2.calcHist([roi_resized], [1], None, [8], [0, 256])
            hist_r = cv2.calcHist([roi_resized], [2], None, [8], [0, 256])

            # ヒストグラムを正規化
            hist_features = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
            hist_features = hist_features / (np.sum(hist_features) + 1e-7)

            # エッジ特徴量
            gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (32 * 32)

            # 基本統計量
            mean_val = np.mean(gray)
            std_val = np.std(gray)

            # 特徴量結合（24 + 2 + 2 = 28次元、パディングで32次元）
            features = np.concatenate([hist_features, [edge_density, mean_val, std_val]])

            # 32次元に調整
            if len(features) < 32:
                padding = np.zeros(32 - len(features))
                features = np.concatenate([features, padding])
            else:
                features = features[:32]

            # 正規化
            if np.linalg.norm(features) > 0:
                features = features / np.linalg.norm(features)

            return features

        except Exception:
            return np.zeros(32)

    def _calculate_similarity_matrix(
        self, detections: list[Detection], tracks: list[Track]
    ) -> np.ndarray:
        """
        検出とトラック間の類似度行列を計算

        Args:
            detections: 検出リスト
            tracks: トラックリスト

        Returns:
            類似度行列
        """
        if len(detections) == 0 or len(tracks) == 0:
            return np.array([])

        similarity_matrix = np.zeros((len(detections), len(tracks)))

        for i, detection in enumerate(detections):
            # 検出の特徴量を抽出
            if self.current_frame is not None:
                det_features = self._extract_simple_features(self.current_frame, detection.bbox)
            else:
                det_features = np.zeros(32)

            for j, track in enumerate(tracks):
                # 外観類似度の計算
                if track.track_id in self.track_features:
                    track_features = self.track_features[track.track_id]
                    appearance_sim = float(
                        cosine_similarity(
                            det_features.reshape(1, -1), track_features.reshape(1, -1)
                        )[0][0]
                    )
                else:
                    appearance_sim = 0.0

                # 位置類似度の計算（IoU）
                iou = self.calculate_iou(detection.bbox, track.bbox)
                position_sim = iou

                # 総合類似度の計算
                total_similarity = (
                    self.appearance_weight * appearance_sim + self.position_weight * position_sim
                )

                similarity_matrix[i, j] = total_similarity

        return similarity_matrix

    def _hungarian_assignment(self, similarity_matrix: np.ndarray):
        """
        ハンガリアン法による最適マッチング

        Args:
            similarity_matrix: 類似度行列

        Returns:
            (マッチしたペア, マッチしなかった検出, マッチしなかったトラック)
        """
        if similarity_matrix.size == 0:
            num_detections = 0
            num_tracks = len(self.active_tracks)
            return [], list(range(num_detections)), list(range(num_tracks))

        num_detections, num_tracks = similarity_matrix.shape

        # コスト行列に変換
        cost_matrix = 1.0 - similarity_matrix

        # ハンガリアン法実行
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # 閾値以上のマッチのみ採用
        matched_pairs = []
        for row, col in zip(row_indices, col_indices, strict=False):
            if similarity_matrix[row, col] >= self.similarity_threshold:
                matched_pairs.append((row, col))

        # マッチしなかった検出とトラックを特定
        matched_detections = set(pair[0] for pair in matched_pairs)
        matched_tracks = set(pair[1] for pair in matched_pairs)

        unmatched_detections = [i for i in range(num_detections) if i not in matched_detections]
        unmatched_tracks = [i for i in range(num_tracks) if i not in matched_tracks]

        return matched_pairs, unmatched_detections, unmatched_tracks

    def update_tracks(self, detections: list[Detection]) -> list[Track]:
        """
        検出結果を基に追跡を更新

        Args:
            detections: 現在フレームの検出結果

        Returns:
            更新された追跡結果のリスト
        """
        if len(self.active_tracks) == 0:
            # 初回の場合
            new_tracks = []
            for detection in detections:
                track_id = self.get_new_id()
                track = Track(
                    track_id=track_id,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    frame_id=detection.frame_id,
                    age=1,
                    time_since_update=0,
                )
                new_tracks.append(track)

                # 特徴量を抽出・保存
                if self.current_frame is not None:
                    features = self._extract_simple_features(self.current_frame, detection.bbox)
                    self.track_features[track_id] = features

            self.active_tracks = new_tracks
            return new_tracks

        # 類似度行列の計算
        similarity_matrix = self._calculate_similarity_matrix(detections, self.active_tracks)

        # ハンガリアン法によるマッチング
        matched_pairs, unmatched_detections, unmatched_tracks = self._hungarian_assignment(
            similarity_matrix
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

            # 特徴量の更新（移動平均）
            if self.current_frame is not None:
                new_features = self._extract_simple_features(self.current_frame, detection.bbox)
                if track.track_id in self.track_features:
                    old_features = self.track_features[track.track_id]
                    # 移動平均で特徴量を更新
                    alpha = 0.7
                    updated_features = alpha * new_features + (1 - alpha) * old_features
                    self.track_features[track.track_id] = updated_features
                else:
                    self.track_features[track.track_id] = new_features

            # disappeared_tracksからも削除
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
            track_id = self.get_new_id()
            new_track = Track(
                track_id=track_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                class_id=detection.class_id,
                frame_id=detection.frame_id,
                age=1,
                time_since_update=0,
            )
            updated_tracks.append(new_track)

            # 新しいトラックの特徴量を保存
            if self.current_frame is not None:
                features = self._extract_simple_features(self.current_frame, detection.bbox)
                self.track_features[track_id] = features

        # 古い消失トラックの削除
        tracks_to_remove = []
        for track_id, disappeared_frames in self.disappeared_tracks.items():
            if disappeared_frames > self.max_disappeared_frames:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.disappeared_tracks[track_id]
            if track_id in self.track_features:
                del self.track_features[track_id]

        # アクティブトラックのリストを更新
        self.active_tracks = [track for track in updated_tracks if track.time_since_update == 0]

        return updated_tracks

    def detect_and_track(self, frame: np.ndarray):
        """
        検出と追跡を実行（フレーム保存付き）

        Args:
            frame: 入力フレーム

        Returns:
            (追跡結果, 性能メトリクス)
        """
        # フレームを保存（特徴量抽出用）
        self.current_frame = frame

        # 基底クラスのメソッドを呼び出し
        return super().detect_and_track(frame)

    def reset(self):
        """
        トラッカーの状態をリセット
        """
        super().reset()
        self.active_tracks = []
        self.track_features = {}
        self.disappeared_tracks = {}
        self.current_frame = None

    def get_algorithm_info(self) -> dict[str, Any]:
        """
        アルゴリズム情報を取得

        Returns:
            アルゴリズム情報辞書
        """
        return {
            "method": "YOLO Advanced Tracker",
            "algorithm": "Appearance + Position feature matching with Hungarian assignment",
            "appearance_weight": self.appearance_weight,
            "position_weight": self.position_weight,
            "similarity_threshold": self.similarity_threshold,
            "max_disappeared_frames": self.max_disappeared_frames,
            "features": [
                "Color histogram",
                "Edge density",
                "Statistical moments",
                "Cosine similarity matching",
                "Hungarian optimization",
            ],
            "pros": [
                "Better re-identification capability",
                "Robust to temporary occlusions",
                "Optimal assignment algorithm",
                "Appearance modeling",
            ],
            "cons": [
                "Higher computational cost than simple",
                "Feature extraction overhead",
                "Memory usage for feature storage",
            ],
        }
