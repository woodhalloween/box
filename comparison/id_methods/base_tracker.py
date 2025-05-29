"""
基底トラッカークラス
すべてのID付与手法の共通インターフェースを定義
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """検出結果を表すデータクラス"""

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    frame_id: int


@dataclass
class Track:
    """追跡結果を表すデータクラス"""

    track_id: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    frame_id: int
    age: int = 0  # 追跡継続フレーム数
    time_since_update: int = 0  # 最後の更新からのフレーム数


@dataclass
class PerformanceMetrics:
    """性能メトリクスを表すデータクラス"""

    processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    fps: float = 0.0
    objects_detected: int = 0
    objects_tracked: int = 0
    frame_count: int = 0


class BaseTracker(ABC):
    """
    ID付与トラッカーの基底クラス
    すべてのトラッカー実装が継承すべき抽象クラス
    """

    def __init__(self, model_path: str, confidence: float = 0.3, device: str = ""):
        """
        基底トラッカーの初期化

        Args:
            model_path: YOLOモデルのパス
            confidence: 検出信頼度閾値
            device: 実行デバイス（"", "cpu", "mps", "0"など）
        """
        self.model_path = model_path
        self.confidence = confidence
        self.device = device

        # YOLOモデルの読み込み
        self.model = YOLO(model_path)
        if device:
            self.model.to(device)

        # 追跡状態
        self.tracks: list[Track] = []
        self.next_id = 1
        self.frame_count = 0

        # 性能メトリクス
        self.metrics = PerformanceMetrics()
        self.processing_times = []

    def detect_objects(self, frame: np.ndarray) -> list[Detection]:
        """
        YOLOを使用してオブジェクトを検出

        Args:
            frame: 入力フレーム

        Returns:
            検出結果のリスト
        """
        # 人物クラス(0)のみを検出
        results = self.model.predict(frame, verbose=False, classes=[0], conf=self.confidence)
        result = results[0]

        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                bbox_array = box.xyxy.cpu().numpy()[0]  # type: ignore
                x1, y1, x2, y2 = bbox_array
                conf = float(box.conf.cpu().numpy()[0])  # type: ignore
                cls = int(box.cls.cpu().numpy()[0])  # type: ignore

                detection = Detection(
                    bbox=(x1, y1, x2, y2), confidence=conf, class_id=cls, frame_id=self.frame_count
                )
                detections.append(detection)

        return detections

    @abstractmethod
    def update_tracks(self, detections: list[Detection]) -> list[Track]:
        """
        検出結果を基に追跡を更新（各サブクラスで実装）

        Args:
            detections: 現在フレームの検出結果

        Returns:
            更新された追跡結果のリスト
        """
        pass

    def detect_and_track(self, frame: np.ndarray) -> tuple[list[Track], PerformanceMetrics]:
        """
        検出と追跡を実行

        Args:
            frame: 入力フレーム

        Returns:
            (追跡結果, 性能メトリクス)
        """
        start_time = time.time()

        # メモリ使用量測定（簡易版）
        import psutil

        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # オブジェクト検出
        detection_start = time.time()
        detections = self.detect_objects(frame)
        detection_time = (time.time() - detection_start) * 1000

        # 追跡更新
        tracking_start = time.time()
        tracks = self.update_tracks(detections)
        tracking_time = (time.time() - tracking_start) * 1000

        # 性能メトリクス更新
        total_time = (time.time() - start_time) * 1000
        memory_after = process.memory_info().rss / 1024 / 1024  # MB

        self.processing_times.append(total_time)
        self.frame_count += 1

        # メトリクス計算
        avg_processing_time = np.mean(self.processing_times[-30:])  # 直近30フレームの平均
        fps = float(1000.0 / avg_processing_time) if avg_processing_time > 0 else 0.0

        current_metrics = PerformanceMetrics(
            processing_time_ms=total_time,
            memory_usage_mb=memory_after,
            fps=fps,
            objects_detected=len(detections),
            objects_tracked=len(tracks),
            frame_count=self.frame_count,
        )

        self.metrics = current_metrics
        self.tracks = tracks

        return tracks, current_metrics

    def get_performance_summary(self) -> dict[str, Any]:
        """
        累積性能サマリーを取得

        Returns:
            性能サマリー辞書
        """
        if not self.processing_times:
            return {}

        return {
            "avg_processing_time_ms": np.mean(self.processing_times),
            "min_processing_time_ms": np.min(self.processing_times),
            "max_processing_time_ms": np.max(self.processing_times),
            "avg_fps": 1000.0 / np.mean(self.processing_times),
            "total_frames": self.frame_count,
            "avg_objects_detected": np.mean([m.objects_detected for m in [self.metrics]]),
            "avg_objects_tracked": np.mean([m.objects_tracked for m in [self.metrics]]),
        }

    def reset(self):
        """
        トラッカーの状態をリセット
        """
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
        self.processing_times = []
        self.metrics = PerformanceMetrics()

    def get_new_id(self) -> int:
        """
        新しいIDを取得

        Returns:
            新しいID番号
        """
        new_id = self.next_id
        self.next_id += 1
        return new_id

    @staticmethod
    def calculate_iou(
        bbox1: tuple[float, float, float, float], bbox2: tuple[float, float, float, float]
    ) -> float:
        """
        2つのバウンディングボックスのIoUを計算

        Args:
            bbox1: バウンディングボックス1 (x1, y1, x2, y2)
            bbox2: バウンディングボックス2 (x1, y1, x2, y2)

        Returns:
            IoU値
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # 交差領域の計算
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # 和集合の計算
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    @staticmethod
    def calculate_center_distance(
        bbox1: tuple[float, float, float, float], bbox2: tuple[float, float, float, float]
    ) -> float:
        """
        2つのバウンディングボックスの中心点間距離を計算

        Args:
            bbox1: バウンディングボックス1 (x1, y1, x2, y2)
            bbox2: バウンディングボックス2 (x1, y1, x2, y2)

        Returns:
            中心点間距離
        """
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center1_y = (bbox1[1] + bbox1[3]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2
        center2_y = (bbox2[1] + bbox2[3]) / 2

        return np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)

    def get_method_name(self) -> str:
        """
        トラッカー手法名を取得

        Returns:
            手法名
        """
        return self.__class__.__name__
 