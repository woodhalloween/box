"""
ByteTrackラッパー
既存のByteTrackライブラリを基底クラスインターフェースに適合させる
"""

import os
import sys
from typing import Any

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from boxmot.trackers.bytetrack.bytetrack import ByteTrack

from comparison.id_methods.base_tracker import BaseTracker, Detection, Track


class ByteTrackWrapper(BaseTracker):
    """
    ByteTrackライブラリのラッパークラス
    BaseTrackerインターフェースに適合させる
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.3,
        device: str = "",
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
    ):
        """
        ByteTrackラッパーの初期化

        Args:
            model_path: YOLOモデルのパス
            confidence: 検出信頼度閾値
            device: 実行デバイス
            track_thresh: 追跡開始信頼度閾値
            track_buffer: 追跡バッファサイズ
            match_thresh: マッチング閾値
            frame_rate: フレームレート
        """
        super().__init__(model_path, confidence, device)

        # ByteTrackパラメータ
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate

        # ByteTrackトラッカーの初期化
        self.bytetrack = ByteTrack(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate,
        )

    def update_tracks(self, detections: list[Detection]) -> list[Track]:
        """
        ByteTrackを使用して追跡を更新

        Args:
            detections: 現在フレームの検出結果

        Returns:
            更新された追跡結果のリスト
        """
        # 検出結果をByteTrack形式に変換
        bytetrack_detections = self._convert_to_bytetrack_format(detections)

        # ByteTrackで更新
        # フレームは必要だが、実際の画像データは使わないのでダミーを作成
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bytetrack_tracks = self.bytetrack.update(bytetrack_detections, dummy_frame)

        # 結果をTrack形式に変換
        tracks = self._convert_from_bytetrack_format(bytetrack_tracks, detections)

        return tracks

    def _convert_to_bytetrack_format(self, detections: list[Detection]) -> np.ndarray:
        """
        検出結果をByteTrack形式に変換

        Args:
            detections: 検出結果リスト

        Returns:
            ByteTrack形式の検出結果 [x1, y1, x2, y2, conf, class]
        """
        if len(detections) == 0:
            return np.empty((0, 6))

        bytetrack_detections = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            conf = detection.confidence
            cls = detection.class_id

            bytetrack_detections.append([x1, y1, x2, y2, conf, cls])

        return np.array(bytetrack_detections)

    def _convert_from_bytetrack_format(
        self, bytetrack_tracks: np.ndarray, original_detections: list[Detection]
    ) -> list[Track]:
        """
        ByteTrack結果をTrack形式に変換

        Args:
            bytetrack_tracks: ByteTrackの追跡結果
            original_detections: 元の検出結果（信頼度取得用）

        Returns:
            Track形式の追跡結果リスト
        """
        tracks = []

        if len(bytetrack_tracks) == 0:
            return tracks

        for track_data in bytetrack_tracks:
            # ByteTrackの出力形式: [x1, y1, x2, y2, track_id, conf, cls_id, ...]
            if len(track_data) >= 7:
                x1, y1, x2, y2, track_id, conf, cls_id = track_data[:7]

                track = Track(
                    track_id=int(track_id),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    frame_id=self.frame_count,
                    age=1,  # ByteTrackでは詳細な年齢情報は取得困難
                    time_since_update=0,
                )
                tracks.append(track)

        return tracks

    def reset(self):
        """
        トラッカーの状態をリセット
        """
        super().reset()
        # ByteTrackを再初期化
        self.bytetrack = ByteTrack(
            track_thresh=self.track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            frame_rate=self.frame_rate,
        )

    def get_algorithm_info(self) -> dict[str, Any]:
        """
        アルゴリズム情報を取得

        Returns:
            アルゴリズム情報辞書
        """
        return {
            "method": "ByteTrack",
            "algorithm": "Multi-object tracking with byte association",
            "track_thresh": self.track_thresh,
            "track_buffer": self.track_buffer,
            "match_thresh": self.match_thresh,
            "frame_rate": self.frame_rate,
            "features": [
                "BYTE association algorithm",
                "High and low threshold tracking",
                "Kalman filter prediction",
                "IoU-based matching",
            ],
            "pros": [
                "State-of-the-art tracking performance",
                "Robust to occlusions",
                "Handles crowded scenes well",
                "Proven in competitions",
            ],
            "cons": [
                "More complex implementation",
                "Higher computational cost",
                "More parameters to tune",
                "Dependency on external library",
            ],
        }

    def get_tracking_statistics(self) -> dict[str, Any]:
        """
        ByteTrack特有の統計情報を取得

        Returns:
            統計情報辞書
        """
        # ByteTrackから統計情報を取得
        # （実際のByteTrackライブラリの実装に依存）
        stats = {}

        try:
            # ByteTrackの内部状態から統計を取得
            if hasattr(self.bytetrack, "tracked_stracks"):
                stats["active_tracks"] = len(self.bytetrack.tracked_stracks)
            if hasattr(self.bytetrack, "lost_stracks"):
                stats["lost_tracks"] = len(self.bytetrack.lost_stracks)
            if hasattr(self.bytetrack, "removed_stracks"):
                stats["removed_tracks"] = len(self.bytetrack.removed_stracks)
        except Exception as e:
            stats["error"] = str(e)

        return stats
