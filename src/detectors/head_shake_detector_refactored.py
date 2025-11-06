"""
head_shake_detector.py (Architecture C Refactor)

Refactored by: 🦸‍♀️ Saki Shirasawa, 🖥️ Daisy Hé, 🗽 Megan Dressel
Date: 2025-11-04
Original Author: Ryotaro (AI Engineer)

姿勢ランドマークから頭部の動きを検出し、首振りやうなずきを判定するモジュール。
PostureAndMotionDetectorBase を適切に継承し、構成パターンに従った実装。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark

from ..definitions import Angle, MovementState
from .base_detector import PostureAndMotionDetectorBase


@dataclass(frozen=True)
class HeadAngles:
    """
    単一フレームにおける頭部角度情報を表すイミュータブルなデータクラス。

    Attributes:
        horizontal (float): 水平方向の回転角度（度）。正=右向き、負=左向き。
        vertical (float): 垂直方向のうなずき角度（度）。正=下向き、負=上向き。
        confidence (float): この角度計算の元となったランドマークの平均信頼度。
        timestamp (float): データが記録されたタイムスタンプ（秒）。
        frame_number (int): 対応する動画のフレーム番号。
    """

    horizontal: float
    vertical: float
    confidence: float
    timestamp: float
    frame_number: int


class OscillationAnalyzer:
    """
    時系列角度データから周期的な振動パターンを検出する純粋関数型アナライザー。

    状態を持たず、分析対象の履歴データと閾値を受け取り、
    振動パターンの有無を判定する。
    """

    @staticmethod
    def detect(
        angle_history: list[float],
        threshold: float,
        min_oscillations: int = 2,
    ) -> bool:
        """
        角度の時系列データから周期的な振動パターンを検出する。

        改善点（Ryotaro版からの変更）:
        - 極値の絶対値ではなく、peak-to-valley振幅を検証
        - peak/valleyの交互出現を確認
        - 最小データ長の要件を明確化

        Args:
            angle_history (list[float]): 分析対象の角度データ列。
            threshold (float): 振動として認識するための最小peak-to-valley振幅（度）。
            min_oscillations (int): 振動と判定するために必要な最小往復回数。

        Returns:
            bool: 振動パターンが検出された場合True。
        """
        required_points = min_oscillations * 4  # 1往復=2peaks+2valleys
        if len(angle_history) < required_points:
            return False

        peaks = []  # (index, value)
        valleys = []  # (index, value)

        # 極大・極小値を検出
        for i in range(1, len(angle_history) - 1):
            if angle_history[i] > angle_history[i - 1] and angle_history[i] > angle_history[i + 1]:
                peaks.append((i, angle_history[i]))
            elif angle_history[i] < angle_history[i - 1] and angle_history[i] < angle_history[i + 1]:
                valleys.append((i, angle_history[i]))

        if not peaks or not valleys:
            return False

        # 振幅検証: 各peakとその前後のvalleyとの差がthreshold以上か
        valid_oscillations = 0
        all_extrema = sorted(peaks + valleys, key=lambda x: x[0])  # indexでソート

        for i in range(1, len(all_extrema)):
            current_idx, current_val = all_extrema[i]
            prev_idx, prev_val = all_extrema[i - 1]

            # peak→valley または valley→peak の遷移を確認
            amplitude = abs(current_val - prev_val)
            if amplitude >= threshold:
                valid_oscillations += 1

        # 必要な往復回数を満たしているか
        # 1往復=2回の有意な遷移（peak→valley→peak または valley→peak→valley）
        return valid_oscillations >= min_oscillations * 2


class HeadShakeDetector(PostureAndMotionDetectorBase):
    """
    姿勢ランドマークから頭部の動きを分析し、首振りやうなずきを検出するクラス。

    PostureAndMotionDetectorBaseを継承し、HandRaiseDetectorと同様の
    パターンに従った実装。履歴管理、状態管理、ヒステリシスを統合。

    主な改善点（Ryotaro版からの変更）:
    - 単一の履歴dequeのみ使用（angle_history）
    - 死んだコード（_analyze_*メソッド）を削除
    - 更新フローの修正: append→analyze→update（クリアなし）
    - ヒステリシスによる状態遷移の安定化
    - detect()メソッドをprimary interfaceとして確立
    """

    DEFAULT_HORIZONTAL_THRESHOLD: float = 15.0
    DEFAULT_VERTICAL_THRESHOLD: float = 10.0
    DEFAULT_CYCLE_DETECTION_WINDOW: int = 60
    DEFAULT_MIN_OSCILLATIONS: int = 2
    DEFAULT_HYSTERESIS_FRAMES: int = 3

    def __init__(
        self,
        horizontal_threshold: float = DEFAULT_HORIZONTAL_THRESHOLD,
        vertical_threshold: float = DEFAULT_VERTICAL_THRESHOLD,
        cycle_detection_window: int = DEFAULT_CYCLE_DETECTION_WINDOW,
        min_oscillations: int = DEFAULT_MIN_OSCILLATIONS,
        confidence_threshold: float = 0.5,
        hysteresis_frames: int = DEFAULT_HYSTERESIS_FRAMES,
    ):
        """
        HeadShakeDetectorを初期化する。

        Args:
            horizontal_threshold (float): 水平首振りを判定するための角度閾値（度）。
            vertical_threshold (float): 垂直うなずきを判定するための角度閾値（度）。
            cycle_detection_window (int): 振動パターンを検出するための履歴の長さ（フレーム数）。
            min_oscillations (int): 振動と判定するために必要な最小往復回数。
            confidence_threshold (float): ランドマークの信頼度閾値。
            hysteresis_frames (int): 状態遷移に必要な連続確認フレーム数。
        """
        super().__init__(confidence_threshold=confidence_threshold)

        self.horizontal_threshold = float(horizontal_threshold)
        self.vertical_threshold = float(vertical_threshold)
        self.min_oscillations = int(min_oscillations)
        self.hysteresis_frames = int(hysteresis_frames)

        # 単一の履歴dequeのみ使用（PostureAndMotionDetectorBaseのパターンに従う）
        self._angle_history = self._create_history_deque(
            "angle_history",
            maxlen=cycle_detection_window,
        )

        # 状態管理（private）
        self._current_horizontal_state: MovementState = MovementState.HEAD_STATIC
        self._current_vertical_state: MovementState = MovementState.HEAD_STATIC

        # ヒステリシスカウンタ
        self._h_state_candidate: MovementState = MovementState.HEAD_STATIC
        self._v_state_candidate: MovementState = MovementState.HEAD_STATIC
        self._h_consecutive_frames: int = 0
        self._v_consecutive_frames: int = 0

        # アラート管理（後方互換性のため）
        self.last_horizontal_alert_time: float = float("-inf")
        self.last_vertical_alert_time: float = float("-inf")
        # self.alert_cooldown: float = 10.0  # アラート間隔（秒）
        self.alert_cooldown: float = 1.0  # アラート間隔（秒）

    def detect(
        self,
        landmarks: np.ndarray | None,
        timestamp: float,
        frame_number: int,
    ) -> dict[str, Any]:
        """
        新しいフレームのランドマークから頭部の動きを検出する。

        これがprimary interfaceです。update()は後方互換性のため残します。

        Args:
            landmarks (np.ndarray | None): ポーズランドマーク配列。
            timestamp (float): 現在のタイムスタンプ（秒）。
            frame_number (int): 現在のフレーム番号。

        Returns:
            dict[str, Any]: 水平・垂直それぞれの動き検出結果。
                {
                    "horizontal_state": MovementState,
                    "vertical_state": MovementState,
                    "horizontal_angle": float,
                    "vertical_angle": float,
                    "confidence": float,
                }
        """
        if not self._validate_landmarks(landmarks):
            return self._build_null_result()

        # 角度計算
        angles = self._calculate_head_angles(landmarks, timestamp, frame_number)

        if angles.confidence < self.confidence_threshold:
            return self._build_null_result()

        # 履歴に追加（Ryotaro版の間違い修正: 分析前に追加する）
        self._angle_history.append(angles)

        # 状態分析と更新
        self._update_horizontal_state()
        self._update_vertical_state()

        return {
            "horizontal_state": self._current_horizontal_state,
            "vertical_state": self._current_vertical_state,
            "horizontal_angle": angles.horizontal,
            "vertical_angle": angles.vertical,
            "confidence": angles.confidence,
        }

    def update(
        self,
        landmarks: np.ndarray | None,
        timestamp: float,
        frame_number: int,
    ) -> dict[Angle, dict[str, Any]]:
        """
        後方互換性のためのラッパーメソッド。

        Ryotaro版のAPIを維持しつつ、内部的にはdetect()を呼び出す。

        Args:
            landmarks (np.ndarray | None): ポーズランドマーク配列。
            timestamp (float): タイムスタンプ（秒）。
            frame_number (int): フレーム番号。

        Returns:
            dict[Angle, dict[str, Any]]: Ryotaro版互換の結果フォーマット。
        """
        result = self.detect(landmarks, timestamp, frame_number)

        # Ryotaro版のフォーマットに変換
        return {
            Angle.HEAD_HORIZONTAL_ROTATION: {
                "angle": result["horizontal_angle"],
                "state": result["horizontal_state"],
                "confidence": result["confidence"],
            },
            Angle.HEAD_VERTICAL_NOD: {
                "angle": result["vertical_angle"],
                "state": result["vertical_state"],
                "confidence": result["confidence"],
            },
        }

    def check_alerts(self, timestamp: float) -> list[str]:
        """
        現在の頭部状態に基づき、アラートを発行すべきかチェックします。

        周期的な首振り（`HORIZONTAL_SHAKE`）またはうなずき（`VERTICAL_NOD`）が
        検出された場合にアラートメッセージを生成します。
        連続してアラートが発生するのを防ぐため、クールダウン期間（`alert_cooldown`）を設けています。

        Args:
            timestamp (float): 現在のタイムスタンプ（秒）。

        Returns:
            list[str]: 生成されたアラートメッセージのリスト。アラートがない場合は空のリスト。
        """
        alerts = []

        # 水平首振りアラート
        if (
            self._current_horizontal_state == MovementState.HORIZONTAL_SHAKE
            and timestamp - self.last_horizontal_alert_time > self.alert_cooldown
        ):
            alerts.append("[!] Horizontal Head Shake Detected")
            self.last_horizontal_alert_time = timestamp

        # 垂直うなずきアラート
        if (
            self._current_vertical_state == MovementState.VERTICAL_NOD
            and timestamp - self.last_vertical_alert_time > self.alert_cooldown
        ):
            alerts.append("[!] Vertical Head Nod Detected")
            self.last_vertical_alert_time = timestamp

        return alerts

    def get_status(self) -> dict[str, Any]:
        """
        検出器の現在の状態を取得する。

        デバッグ・ロギング用。状態の真の情報源はdetect()の戻り値。

        Returns:
            dict[str, Any]: 検出器の現在の状態。
        """
        if not self._angle_history:
            return {
                "horizontal_state": MovementState.HEAD_STATIC.name,
                "vertical_state": MovementState.HEAD_STATIC.name,
                "horizontal_angle": 0.0,
                "vertical_angle": 0.0,
                "confidence": 0.0,
                "sample_count": 0,
                "hysteresis_h_frames": 0,
                "hysteresis_v_frames": 0,
            }

        latest = self._angle_history[-1]
        return {
            "horizontal_state": self._current_horizontal_state.name,
            "vertical_state": self._current_vertical_state.name,
            "horizontal_angle": latest.horizontal,
            "vertical_angle": latest.vertical,
            "confidence": latest.confidence,
            "sample_count": len(self._angle_history),
            "hysteresis_h_frames": self._h_consecutive_frames,
            "hysteresis_v_frames": self._v_consecutive_frames,
        }

    def reset(self) -> None:
        """検出器の状態と履歴をリセットする。"""
        self._clear_all_history()
        self._current_horizontal_state = MovementState.HEAD_STATIC
        self._current_vertical_state = MovementState.HEAD_STATIC
        self._h_state_candidate = MovementState.HEAD_STATIC
        self._v_state_candidate = MovementState.HEAD_STATIC
        self._h_consecutive_frames = 0
        self._v_consecutive_frames = 0
        self.last_horizontal_alert_time = float("-inf")
        self.last_vertical_alert_time = float("-inf")

    # ==================== Internal Methods ====================

    def _calculate_head_angles(
        self,
        landmarks: np.ndarray,
        timestamp: float,
        frame_number: int,
    ) -> HeadAngles:
        """
        首の水平・垂直角度と信頼度を計算する。

        Ryotaro版と同じロジックだが、HeadAnglesデータクラスを返す。

        Args:
            landmarks (np.ndarray): ポーズランドマーク配列。
            timestamp (float): タイムスタンプ（秒）。
            frame_number (int): フレーム番号。

        Returns:
            HeadAngles: 計算された角度情報。
        """
        try:
            # 必要なランドマークを取得
            nose = landmarks[PoseLandmark.NOSE.value]
            left_ear = landmarks[PoseLandmark.LEFT_EAR.value]
            right_ear = landmarks[PoseLandmark.RIGHT_EAR.value]
            left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER.value]

            # 信頼度チェック
            confidences = [nose[3], left_ear[3], right_ear[3], left_shoulder[3], right_shoulder[3]]
            avg_confidence = float(sum(confidences) / len(confidences))

            if avg_confidence < self.confidence_threshold:
                return HeadAngles(0.0, 0.0, avg_confidence, timestamp, frame_number)

            # 水平角度 (Yaw) の計算
            ear_dist = float(np.linalg.norm(right_ear[:2] - left_ear[:2]))
            if ear_dist < 1e-6:
                horizontal_angle = 0.0
            else:
                ear_midpoint_x = (left_ear[0] + right_ear[0]) / 2
                offset = nose[0] - ear_midpoint_x
                horizontal_angle = float((offset / ear_dist) * 180)

            # 垂直角度 (Pitch) の計算
            shoulder_midpoint_y = (left_shoulder[1] + right_shoulder[1]) / 2
            ear_midpoint_y = (left_ear[1] + right_ear[1]) / 2
            neck_length_approx = abs(shoulder_midpoint_y - ear_midpoint_y)

            if neck_length_approx < 1e-6:
                vertical_angle = 0.0
            else:
                offset_y = nose[1] - ear_midpoint_y
                vertical_angle = float((offset_y / neck_length_approx) * 90)

            return HeadAngles(
                horizontal=horizontal_angle,
                vertical=vertical_angle,
                confidence=avg_confidence,
                timestamp=timestamp,
                frame_number=frame_number,
            )

        except (IndexError, TypeError, ZeroDivisionError):
            return HeadAngles(0.0, 0.0, 0.0, timestamp, frame_number)

    def _update_horizontal_state(self) -> None:
        """
        水平方向の頭部状態を更新する。

        ヒステリシスを適用し、安定した状態遷移を実現。
        履歴は決してクリアしない（Ryotaro版の間違い修正）。
        """
        if len(self._angle_history) < 10:
            self._current_horizontal_state = MovementState.HEAD_STATIC
            return

        latest_angle = self._angle_history[-1].horizontal
        h_angles = [a.horizontal for a in self._angle_history]

        # 状態候補を決定
        if OscillationAnalyzer.detect(h_angles, self.horizontal_threshold, self.min_oscillations):
            candidate = MovementState.HORIZONTAL_SHAKE
        elif latest_angle > self.horizontal_threshold:
            candidate = MovementState.HEAD_RIGHT_TURN
        elif latest_angle < -self.horizontal_threshold:
            candidate = MovementState.HEAD_LEFT_TURN
        else:
            candidate = MovementState.HEAD_STATIC

        # ヒステリシス適用
        if candidate == self._h_state_candidate:
            self._h_consecutive_frames += 1
            if self._h_consecutive_frames >= self.hysteresis_frames:
                self._current_horizontal_state = candidate
        else:
            self._h_state_candidate = candidate
            self._h_consecutive_frames = 1

    def _update_vertical_state(self) -> None:
        """
        垂直方向の頭部状態を更新する。

        ヒステリシスを適用し、安定した状態遷移を実現。
        履歴は決してクリアしない（Ryotaro版の間違い修正）。
        """
        if len(self._angle_history) < 10:
            self._current_vertical_state = MovementState.HEAD_STATIC
            return

        latest_angle = self._angle_history[-1].vertical
        v_angles = [a.vertical for a in self._angle_history]

        # 状態候補を決定
        if OscillationAnalyzer.detect(v_angles, self.vertical_threshold, self.min_oscillations):
            candidate = MovementState.VERTICAL_NOD
        elif latest_angle > self.vertical_threshold:
            candidate = MovementState.HEAD_DOWN_NOD
        elif latest_angle < -self.vertical_threshold:
            candidate = MovementState.HEAD_UP_NOD
        else:
            candidate = MovementState.HEAD_STATIC

        # ヒステリシス適用
        if candidate == self._v_state_candidate:
            self._v_consecutive_frames += 1
            if self._v_consecutive_frames >= self.hysteresis_frames:
                self._current_vertical_state = candidate
        else:
            self._v_state_candidate = candidate
            self._v_consecutive_frames = 1

    @staticmethod
    def _build_null_result() -> dict[str, Any]:
        """ランドマークが無効な場合の結果を構築する。"""
        return {
            "horizontal_state": MovementState.HEAD_STATIC,
            "vertical_state": MovementState.HEAD_STATIC,
            "horizontal_angle": 0.0,
            "vertical_angle": 0.0,
            "confidence": 0.0,
        }
