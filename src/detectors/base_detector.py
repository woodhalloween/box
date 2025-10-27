"""
base_detector.py

姿勢・動作検出器の基底クラスを提供するモジュール。
共通の機能とインターフェースを定義する。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Any

import numpy as np


class PostureAndMotionDetectorBase(ABC):
    """
    姿勢・動作検出器の抽象基底クラス。

    ランドマークベースの検出器に共通する機能を提供し、
    サブクラスで実装すべきインターフェースを定義する。

    Attributes:
        confidence_threshold (float): ランドマークの信頼度閾値
    """

    def __init__(self, confidence_threshold: float = 0.5):
        """
        基底クラスを初期化する。

        Args:
            confidence_threshold (float): ランドマークが有効と見なされる最小信頼度
        """
        self.confidence_threshold = confidence_threshold
        self._history_deques: dict[str, deque] = {}

    # ==================== History Management ====================

    def _create_history_deque(self, name: str, maxlen: int) -> deque:
        """
        時系列分析用の履歴dequeを作成・登録する。

        Args:
            name (str): dequeの識別名
            maxlen (int): dequeの最大長

        Returns:
            deque: 作成されたdeque
        """
        self._history_deques[name] = deque(maxlen=maxlen)
        return self._history_deques[name]

    def _clear_all_history(self):
        """登録されているすべての履歴dequeをクリアする。"""
        for dq in self._history_deques.values():
            dq.clear()

    def _clear_history(self, name: str):
        """
        指定された履歴dequeをクリアする。

        Args:
            name (str): クリアするdequeの識別名
        """
        if name in self._history_deques:
            self._history_deques[name].clear()

    # ==================== Landmark Utilities ====================

    @staticmethod
    def _check_landmark_visibility(landmark: np.ndarray, threshold: float) -> bool:
        """
        ランドマークが信頼度閾値を満たしているか確認する。

        Args:
            landmark (np.ndarray): ランドマークデータ [x, y, z, visibility]
            threshold (float): 信頼度閾値

        Returns:
            bool: 閾値を満たしていればTrue
        """
        try:
            return float(landmark[3]) >= threshold
        except (IndexError, TypeError, ValueError):
            return False

    @staticmethod
    def _extract_landmark_2d(landmark: np.ndarray, frame_shape: tuple[int, int]) -> tuple[float, float]:
        """
        正規化されたランドマーク座標をピクセル座標に変換する。

        Args:
            landmark (np.ndarray): 正規化されたランドマーク [x, y, z, visibility]
            frame_shape (tuple[int, int]): フレームの形状 (height, width)

        Returns:
            tuple[float, float]: ピクセル座標 (x_px, y_px)
        """
        height, width = frame_shape[:2]
        return float(landmark[0] * width), float(landmark[1] * height)

    @staticmethod
    def _calculate_distance_2d(point1: tuple[float, float], point2: tuple[float, float]) -> float:
        """
        2点間のユークリッド距離を計算する。

        Args:
            point1 (tuple[float, float]): 最初の点 (x, y)
            point2 (tuple[float, float]): 2番目の点 (x, y)

        Returns:
            float: 2点間の距離
        """
        return float(np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2))

    def _validate_landmarks(self, landmarks: np.ndarray | None) -> bool:
        """
        ランドマークが存在し適切にフォーマットされているか検証する。

        Args:
            landmarks (np.ndarray | None): 検証するランドマーク

        Returns:
            bool: ランドマークが有効ならTrue
        """
        return landmarks is not None and len(landmarks) > 0

    # ==================== Pattern Detection Utilities ====================

    @staticmethod
    def _detect_oscillation_pattern(values: list[float], threshold: float, min_extrema: int = 4) -> bool:
        """
        値の時系列データから周期的な振動パターンを検出する。

        Args:
            values (list[float]): 分析対象の時系列データ
            threshold (float): 有意な極値と見なす最小値（絶対値）
            min_extrema (int): 振動と判定する最小極値数（デフォルト: 4 = 2往復）

        Returns:
            bool: 振動パターンが検出された場合True
        """
        if len(values) < min_extrema * 2:
            return False

        peaks = []  # 極大値のインデックス
        valleys = []  # 極小値のインデックス

        # 極大・極小値を検出
        for i in range(1, len(values) - 1):
            if abs(values[i]) > threshold:
                if values[i] > values[i - 1] and values[i] > values[i + 1]:
                    peaks.append(i)
                elif values[i] < values[i - 1] and values[i] < values[i + 1]:
                    valleys.append(i)

        if not peaks or not valleys:
            return False

        # 十分な数の極値があるかチェック
        total_extrema = len(peaks) + len(valleys)
        return total_extrema >= min_extrema

    # ==================== Abstract Methods ====================

    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        """
        検出器の現在の状態を取得する。

        サブクラスで実装必須。

        Returns:
            dict[str, Any]: 検出器の状態を示す辞書
        """
        pass
