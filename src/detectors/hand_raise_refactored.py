"""
hand_raise_refactored.py

姿勢ランドマークから手の挙上状態を検出するモジュール。
連続フレーム分析と履歴管理を用いて誤検出を抑制する。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark

from .base_detector import PostureAndMotionDetectorBase


@dataclass
class _HandState:
    """単一の手に対する内部状態を保持するデータクラス。"""

    consecutive_frames: int = 0
    is_raised: bool = False
    last_vertical_delta: float = 0.0
    visibility_score: float = 0.0
    visible: bool = False


@dataclass(frozen=True)
class _HandEvaluation:
    """ランドマーク評価の結果を表す不変データ構造。"""

    visible: bool
    criteria_met: bool
    vertical_delta: float
    visibility_score: float


class HandRaiseDetector(PostureAndMotionDetectorBase):
    """
    姿勢ランドマークから手の挙上状態を検出するクラス。

    連続フレームの合格数に基づきチャタリングを抑えつつ、
    ランドマークの可視性と手首・肩の位置関係から手挙げを判定する。
    """

    DEFAULT_VISIBILITY_THRESHOLD: float = 0.5
    DEFAULT_MIN_CONSECUTIVE_FRAMES: int = 3

    _HAND_LANDMARK_PAIRS: dict[str, tuple[int, int]] = {
        "left": (int(PoseLandmark.LEFT_SHOULDER), int(PoseLandmark.LEFT_WRIST)),
        "right": (int(PoseLandmark.RIGHT_SHOULDER), int(PoseLandmark.RIGHT_WRIST)),
    }

    def __init__(
        self,
        visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
        min_consecutive_frames: int = DEFAULT_MIN_CONSECUTIVE_FRAMES,
        vertical_margin: float = 0.0,
        history_window: int | None = None,
    ):
        """
        HandRaiseDetectorを初期化する。

        Args:
            visibility_threshold (float):
                ランドマークが有効と見なされる最小visibility値。
            min_consecutive_frames (int):
                手挙げ判定に必要な連続合格フレーム数。
            vertical_margin (float):
                肩と手首のy座標差に要求する最小マージン（肩 - 手首）。
            history_window (int | None):
                履歴dequeの長さ。省略時はmin_consecutive_framesと同じ。
        """
        if min_consecutive_frames < 1:
            raise ValueError("min_consecutive_frames must be >= 1")
        if vertical_margin < 0:
            raise ValueError("vertical_margin must be >= 0.0")

        super().__init__(confidence_threshold=visibility_threshold)

        self.min_consecutive_frames = int(min_consecutive_frames)
        self.vertical_margin = float(vertical_margin)
        self._history_window = max(1, history_window or self.min_consecutive_frames)

        self._hand_states: dict[str, _HandState] = {hand: _HandState() for hand in self._HAND_LANDMARK_PAIRS}
        self._hand_histories: dict[str, deque[bool]] = {
            hand: self._create_history_deque(f"{hand}_hand_history", maxlen=self._history_window)
            for hand in self._HAND_LANDMARK_PAIRS
        }

    def detect(self, landmarks: np.ndarray | None) -> dict[str, bool]:
        """
        与えられたランドマークから左右の手が挙がっているかを判定する。

        Args:
            landmarks (np.ndarray | None):
                Mediapipe Poseから得られるランドマーク配列 (33, 4)。

        Returns:
            dict[str, bool]:
                左右の手の挙上状態を示す辞書。
        """
        if not self._validate_landmarks(landmarks):
            self._reset_all_states()
            return self._export_raise_flags()

        # 新しいフレームに対し可視性フラグを初期化
        for state in self._hand_states.values():
            state.visible = False

        for hand, (shoulder_idx, wrist_idx) in self._HAND_LANDMARK_PAIRS.items():
            evaluation = self._evaluate_hand(landmarks, shoulder_idx, wrist_idx)

            if not evaluation.visible:
                self._reset_hand_state(hand)
                continue

            state = self._hand_states[hand]
            history = self._hand_histories[hand]

            history.append(evaluation.criteria_met)
            state.visible = True
            state.visibility_score = evaluation.visibility_score
            state.last_vertical_delta = evaluation.vertical_delta

            if evaluation.criteria_met:
                state.consecutive_frames += 1
            else:
                state.consecutive_frames = 0

            state.is_raised = state.consecutive_frames >= self.min_consecutive_frames

        return self._export_raise_flags()

    def reset(self) -> None:
        """検出状態と履歴をリセットする。"""
        self._reset_all_states()

    def get_status(self) -> dict[str, Any]:
        """
        検出器の現在状態を辞書形式で返す。

        Returns:
            dict[str, Any]:
                左右の手の状態と検出設定を含む辞書。
        """
        left_state = self._hand_states["left"]
        right_state = self._hand_states["right"]

        return {
            "left_hand_raised": left_state.is_raised,
            "right_hand_raised": right_state.is_raised,
            "any_hand_raised": left_state.is_raised or right_state.is_raised,
            "min_consecutive_frames": self.min_consecutive_frames,
            "vertical_margin": self.vertical_margin,
            "visibility_threshold": self.confidence_threshold,
            "history_window": self._history_window,
            "hands": {
                "left": self._serialize_hand_state("left"),
                "right": self._serialize_hand_state("right"),
            },
        }

    # ==================== 後方互換性プロパティ ====================

    @property
    def _left_hand_consecutive_frames(self) -> int:
        """左手の連続検出フレーム数（後方互換性のため）。"""
        return self._hand_states["left"].consecutive_frames

    @property
    def _right_hand_consecutive_frames(self) -> int:
        """右手の連続検出フレーム数（後方互換性のため）。"""
        return self._hand_states["right"].consecutive_frames

    @property
    def _is_left_hand_raised(self) -> bool:
        """左手が挙がっているか（後方互換性のため）。"""
        return self._hand_states["left"].is_raised

    @property
    def _is_right_hand_raised(self) -> bool:
        """右手が挙がっているか（後方互換性のため）。"""
        return self._hand_states["right"].is_raised

    # ==================== 内部ユーティリティ ====================

    def _evaluate_hand(
        self,
        landmarks: np.ndarray,
        shoulder_idx: int,
        wrist_idx: int,
    ) -> _HandEvaluation:
        """肩と手首のランドマークから手挙げ基準を評価する。"""
        try:
            shoulder_landmark = landmarks[shoulder_idx]
            wrist_landmark = landmarks[wrist_idx]
        except (IndexError, TypeError, ValueError):
            return _HandEvaluation(False, False, 0.0, 0.0)

        visibility_score = self._compute_visibility_score(shoulder_landmark, wrist_landmark)

        if not (
            self._check_landmark_visibility(shoulder_landmark, self.confidence_threshold)
            and self._check_landmark_visibility(wrist_landmark, self.confidence_threshold)
        ):
            return _HandEvaluation(False, False, 0.0, visibility_score)

        try:
            shoulder_y = float(shoulder_landmark[1])
            wrist_y = float(wrist_landmark[1])
        except (IndexError, TypeError, ValueError):
            return _HandEvaluation(False, False, 0.0, visibility_score)

        vertical_delta = shoulder_y - wrist_y
        criteria_met = vertical_delta >= self.vertical_margin

        return _HandEvaluation(True, criteria_met, vertical_delta, visibility_score)

    def _export_raise_flags(self) -> dict[str, bool]:
        """外部公開用の挙手フラグを生成する。"""
        return {
            "left_hand_raised": self._hand_states["left"].is_raised,
            "right_hand_raised": self._hand_states["right"].is_raised,
        }

    def _serialize_hand_state(self, hand: str) -> dict[str, Any]:
        """ハンド状態をシリアライズ可能な辞書に変換する。"""
        state = self._hand_states[hand]
        history = list(self._hand_histories[hand])
        history_len = len(history)
        raised_ratio = float(sum(history)) / history_len if history_len else 0.0

        return {
            "is_raised": state.is_raised,
            "visible": state.visible,
            "consecutive_frames": state.consecutive_frames,
            "visibility_score": state.visibility_score,
            "vertical_delta": state.last_vertical_delta,
            "history": history,
            "raised_ratio": raised_ratio,
        }

    def _reset_all_states(self) -> None:
        """全ての手の状態と履歴をリセットする。"""
        self._clear_all_history()
        for hand in self._HAND_LANDMARK_PAIRS:
            self._reset_hand_state(hand, clear_history=False)

    def _reset_hand_state(self, hand: str, *, clear_history: bool = True) -> None:
        """指定した手の状態と履歴をリセットする。"""
        state = self._hand_states[hand]
        state.consecutive_frames = 0
        state.is_raised = False
        state.last_vertical_delta = 0.0
        state.visibility_score = 0.0
        state.visible = False

        if clear_history:
            self._hand_histories[hand].clear()

    @staticmethod
    def _compute_visibility_score(shoulder: np.ndarray, wrist: np.ndarray) -> float:
        """肩と手首ランドマークのvisibilityスコアを計算する。"""
        try:
            shoulder_visibility = float(shoulder[3])
            wrist_visibility = float(wrist[3])
        except (IndexError, TypeError, ValueError):
            return 0.0
        return float(min(shoulder_visibility, wrist_visibility))
