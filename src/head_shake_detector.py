"""
src/head_shake_detector.py

このモジュールは、骨格検知モデルから得られるポーズランドマークを利用して、
人間の頭部の動きを分析する機能を提供します。

主な機能は以下の通りです：
- 水平方向の首振り（左右の振り）の検出
- 垂直方向のうなずき（上下の動き）の検出
- 動きの状態（静止、右向き、左向き、上向き、下向き、振動）の判定

`HeadShakeDetector` クラスが中心的な役割を担い、連続したフレームの
頭部角度データを分析して、周期的な動きや特定の向きを識別します。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark  # PoseLandmarkをインポート

from src.definitions import Angle, MovementState


@dataclass
class HeadAngleSnapshot:
    """各フレームにおける頭部の角度情報を保持するデータクラス。

    タイムスタンプ、フレーム番号と共に、計算された水平・垂直角度、
    およびその計算の信頼度を格納します。

    Attributes:
        timestamp (float): データが記録されたタイムスタンプ（秒）。
        frame_number (int): 対応する動画のフレーム番号。
        horizontal_angle (float): 水平方向の回転角度（度）。正の値は右向き、負の値は左向きを示す。
        vertical_angle (float): 垂直方向のうなずき角度（度）。正の値は下向き、負の値は上向きを示す。
        confidence (float): この角度計算の元となったランドマークの平均信頼度。
    """

    timestamp: float
    frame_number: int
    horizontal_angle: float  # 水平回転角度（度）
    vertical_angle: float  # 垂直うなずき角度（度）
    confidence: float  # 検出信頼度


class HeadShakeDetector:
    """ポーズランドマークから頭部の動きを分析し、首振りやうなずきを検出するクラス。

    連続したフレーム間の頭部角度の変化を追跡し、水平方向の首振り（"いやいや"）、
    垂直方向のうなずき（"うんうん"）、および頭が特定の方向を向いている状態を判定します。

    主な機能:
    - 鼻と両耳、両肩のランドマークから頭の水平・垂直角度を計算。
    - 角度の履歴を保持し、周期的な振動パターンを検出。
    - リアルタイムで現在の頭部の動きの状態を更新。
    - 閾値に基づいたアラート生成機能。

    Attributes:
        horizontal_threshold (float): 水平方向の動きを「右向き」または「左向き」と判定するための角度閾値（度）。
        vertical_threshold (float): 垂直方向の動きを「上向き」または「下向き」と判定するための角度閾値（度）。
        cycle_detection_window (int): 周期的な動き（振動）を検出するために使用するフレーム数（履歴のサイズ）。
        min_oscillations (int): 振動と判定するために必要な最小往復回数。
        confidence_threshold (float): ランドマークの信頼度がこの値を下回る場合、計算をスキップするための閾値。
        angle_history (deque[HeadAngleSnapshot]): 頭部角度の履歴データ。
        current_horizontal_state (MovementState): 現在の水平方向の動きの状態。
        current_vertical_state (MovementState): 現在の垂直方向の動きの状態。
    """

    def __init__(
        self,
        horizontal_threshold: float = 15.0,  # 水平首振り判定の角度閾値（度）
        vertical_threshold: float = 10.0,  # 垂直うなずき判定の角度閾値（度）
        cycle_detection_window: int = 60,  # 周期検出の窓サイズ（フレーム）
        min_oscillations: int = 2,  # 最低振動回数
        confidence_threshold: float = 0.5,  # 検出信頼度閾値
    ):
        """HeadShakeDetectorを初期化します。

        Args:
            horizontal_threshold (float): 水平首振りを判定するための角度閾値（度）。
                この値を超えると頭が左右どちらかを向いていると判断されます。
            vertical_threshold (float): 垂直うなずきを判定するための角度閾値（度）。
                この値を超えると頭が上下どちらかを向いていると判断されます。
            cycle_detection_window (int): 振動パターンを検出するための履歴の長さ（フレーム数）。
                例えば30fpsの動画で60フレームを指定すると、約2秒間のデータを分析します。
            min_oscillations (int): `cycle_detection_window` 内で首振りやうなずきと
                判定するために必要な最小往復回数。
            confidence_threshold (float): 計算に使用するランドマークの平均信頼度の下限値。
        """
        self.horizontal_threshold = horizontal_threshold
        self.vertical_threshold = vertical_threshold
        self.cycle_detection_window = cycle_detection_window
        self.min_oscillations = min_oscillations
        self.confidence_threshold = confidence_threshold

        self.angle_history: deque[HeadAngleSnapshot] = deque(maxlen=cycle_detection_window)
        self.horizontal_angles = deque(maxlen=cycle_detection_window)
        self.vertical_angles = deque(maxlen=cycle_detection_window)
        self.timestamps = deque(maxlen=cycle_detection_window)

        self.current_horizontal_state: MovementState = MovementState.HEAD_STATIC
        self.current_vertical_state: MovementState = MovementState.HEAD_STATIC
        self.previous_horizontal_angle: float | None = 0.0
        self.previous_vertical_angle: float | None = 0.0

        self.last_horizontal_alert_time: float = 0
        self.last_vertical_alert_time: float = 0
        self.alert_cooldown: float = 10.0  # アラート間隔（秒）

    def _calculate_head_angles(self, landmarks: np.ndarray) -> tuple[float, float, float]:
        """首の水平・垂直角度と信頼度を計算 (リファクタリング版)"""
        try:
            # --- 必要なランドマークを取得 ---
            nose = landmarks[PoseLandmark.NOSE.value]
            left_ear = landmarks[PoseLandmark.LEFT_EAR.value]
            right_ear = landmarks[PoseLandmark.RIGHT_EAR.value]
            left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER.value]

            # --- 信頼度チェック ---
            confidences = [nose[3], left_ear[3], right_ear[3], left_shoulder[3], right_shoulder[3]]
            avg_confidence = sum(confidences) / len(confidences)
            if avg_confidence < self.confidence_threshold:
                return 0.0, 0.0, avg_confidence

            # --- 水平角度 (Yaw) の計算 ---
            ear_dist = np.linalg.norm(right_ear[:2] - left_ear[:2])
            if ear_dist < 1e-6:
                horizontal_angle = 0.0
            else:
                ear_midpoint_x = (left_ear[0] + right_ear[0]) / 2
                offset = nose[0] - ear_midpoint_x
                # 鼻が右にずれるとoffset > 0。これを右向き(正)の角度とする。
                # 正規化されたオフセットを角度に変換 (例: オフセットが耳間距離の半分なら90度)
                horizontal_angle = (offset / ear_dist) * 180

            # --- 垂直角度 (Pitch) の計算 ---
            shoulder_midpoint_y = (left_shoulder[1] + right_shoulder[1]) / 2
            ear_midpoint_y = (left_ear[1] + right_ear[1]) / 2
            neck_length_approx = abs(shoulder_midpoint_y - ear_midpoint_y)
            if neck_length_approx < 1e-6:
                vertical_angle = 0.0
            else:
                offset_y = nose[1] - ear_midpoint_y
                # 鼻が耳より下にあるとoffset_y > 0。これを下向き(正)の角度とする。
                vertical_angle = (offset_y / neck_length_approx) * 90

            return float(horizontal_angle), float(vertical_angle), avg_confidence

        except (IndexError, TypeError, ZeroDivisionError):
            return 0.0, 0.0, 0.0

    def _detect_oscillation_pattern(self, values: list[float], threshold: float) -> bool:
        """角度の時系列データから、周期的で顕著な振動パターンを検出します。

        時系列データ内の極大値（ピーク）と極小値（谷）を検出し、その数が
        設定された最小振動回数（`min_oscillations`）を満たすかどうかを判定します。
        これにより、単なるノイズではなく、意図的な首振りやうなずきを識別します。

        Args:
            values (list[float]): 分析対象の角度データのリスト。
            threshold (float): 振動として認識するための最小角度（絶対値）。

        Returns:
            bool: 振動パターンが検出された場合は True, そうでなければ False。
        """
        if len(values) < self.min_oscillations * 4:  # 最低でも2往復分のデータが必要
            return False

        peaks = []  # 極大値のインデックス
        valleys = []  # 極小値のインデックス

        # 極大・極小値を検出
        for i in range(1, len(values) - 1):
            if values[i] > values[i - 1] and values[i] > values[i + 1]:
                if abs(values[i]) > threshold:  # 閾値以上の極大値
                    peaks.append(i)
            elif values[i] < values[i - 1] and values[i] < values[i + 1]:
                if abs(values[i]) > threshold:  # 閾値以上の極小値
                    valleys.append(i)

        # 十分な数の極値があるかチェック
        total_extremes = len(peaks) + len(valleys)
        return total_extremes >= self.min_oscillations * 2  # 往復には極大・極小が必要

    def update(self, landmarks: np.ndarray, timestamp: float, frame_number: int) -> dict[Angle, dict[str, Any]]:
        """新しいフレームのランドマーク情報で、頭部の動き検出器の状態を更新します。

        このメソッドは、外部からフレームごとに呼び出されることを想定しています。
        内部で角度の計算、履歴の更新、状態分析を行い、結果を返します。

        Args:
            landmarks (np.ndarray): 最新フレームのポーズランドマーク。
            timestamp (float): 最新フレームのタイムスタンプ（秒）。
            frame_number (int): 最新フレームのフレーム番号。

        Returns:
            dict[Angle, dict[str, Any]]: 水平・垂直それぞれの動きに関する分析結果の辞書。
            各キー (`Angle.HEAD_HORIZONTAL_ROTATION`, `Angle.HEAD_VERTICAL_NOD`) には、
            角度、状態、信頼度が含まれる。ランドマークがNoneの場合は空の辞書を返す。
        """
        if landmarks is None:
            return {}

        # 角度計算
        horizontal_angle, vertical_angle, confidence = self._calculate_head_angles(landmarks)

        # スナップショット作成
        snapshot = HeadAngleSnapshot(
            timestamp=timestamp,
            frame_number=frame_number,
            horizontal_angle=horizontal_angle,
            vertical_angle=vertical_angle,
            confidence=confidence,
        )

        # 履歴に追加
        self.angle_history.append(snapshot)

        # --- 状態分析 ---
        # 履歴が溜まっている場合、まず振動パターンをチェック
        horizontal_shake_detected = False
        if len(self.horizontal_angles) == self.cycle_detection_window:
            is_shake = self._detect_oscillation_pattern(list(self.horizontal_angles), self.horizontal_threshold)
            if is_shake:
                horizontal_shake_detected = True

        vertical_nod_detected = False
        if len(self.vertical_angles) == self.cycle_detection_window:
            is_nod = self._detect_oscillation_pattern(list(self.vertical_angles), self.vertical_threshold)
            if is_nod:
                vertical_nod_detected = True

        # 履歴を更新
        self.horizontal_angles.append(horizontal_angle)
        self.vertical_angles.append(vertical_angle)
        self.timestamps.append(timestamp)

        # 状態を決定（振動が検出されたら最優先）
        if horizontal_shake_detected:
            self.current_horizontal_state = MovementState.HORIZONTAL_SHAKE
            self.horizontal_angles.clear()
        elif horizontal_angle > self.horizontal_threshold:
            self.current_horizontal_state = MovementState.HEAD_RIGHT_TURN
        elif horizontal_angle < -self.horizontal_threshold:
            self.current_horizontal_state = MovementState.HEAD_LEFT_TURN
        else:
            self.current_horizontal_state = MovementState.HEAD_STATIC

        if vertical_nod_detected:
            self.current_vertical_state = MovementState.VERTICAL_NOD
            self.vertical_angles.clear()
        elif vertical_angle > self.vertical_threshold:
            self.current_vertical_state = MovementState.HEAD_DOWN_NOD
        elif vertical_angle < -self.vertical_threshold:
            self.current_vertical_state = MovementState.HEAD_UP_NOD
        else:
            self.current_vertical_state = MovementState.HEAD_STATIC

        self.previous_horizontal_angle = horizontal_angle
        self.previous_vertical_angle = vertical_angle

        results = {}
        results[Angle.HEAD_HORIZONTAL_ROTATION] = {
            "angle": horizontal_angle,
            "state": self.current_horizontal_state,
            "confidence": confidence,
        }

        # 垂直うなずきの結果
        results[Angle.HEAD_VERTICAL_NOD] = {
            "angle": vertical_angle,
            "state": self.current_vertical_state,
            "confidence": confidence,
        }

        return results

    def _analyze_horizontal_movement(self) -> MovementState:
        """角度の履歴データに基づき、水平方向の頭部の動きの状態を分析・判定します。

        最新の角度が閾値を超えている場合は、即座に「右向き」または「左向き」と判定します。
        閾値内に収まっている場合は、履歴データ全体で振動パターンが見られるかをチェックし、
        周期的な首振り（`HORIZONTAL_SHAKE`）か、単なる静止状態（`HEAD_STATIC`）かを判断します。

        Returns:
            MovementState: 分析された現在の水平方向の動きの状態。
        """
        if len(self.angle_history) < 10:  # 最低限のデータが必要
            return MovementState.HEAD_STATIC

        # 最新の角度
        latest_angle = self.angle_history[-1].horizontal_angle

        # 即座の向き判定
        if latest_angle > self.horizontal_threshold:
            self.current_horizontal_state = MovementState.HEAD_RIGHT_TURN
        elif latest_angle < -self.horizontal_threshold:
            self.current_horizontal_state = MovementState.HEAD_LEFT_TURN
        # 振動パターンをチェック
        elif self._detect_oscillation_pattern(
            [snapshot.horizontal_angle for snapshot in self.angle_history],
            self.horizontal_threshold,
        ):
            self.current_horizontal_state = MovementState.HORIZONTAL_SHAKE
        else:
            self.current_horizontal_state = MovementState.HEAD_STATIC

        return self.current_horizontal_state

    def _analyze_vertical_movement(self) -> MovementState:
        """角度の履歴データに基づき、垂直方向の頭部の動きの状態を分析・判定します。

        最新の角度が閾値を超えている場合は、即座に「上向き」または「下向き」と判定します。
        閾値内に収まっている場合は、履歴データ全体で振動パターンが見られるかをチェックし、
        周期的なうなずき（`VERTICAL_NOD`）か、単なる静止状態（`HEAD_STATIC`）かを判断します。

        Returns:
            MovementState: 分析された現在の垂直方向の動きの状態。
        """
        if len(self.angle_history) < 10:  # 最低限のデータが必要
            return MovementState.HEAD_STATIC

        # 最新の角度
        latest_angle = self.angle_history[-1].vertical_angle

        # 即座の向き判定
        if latest_angle > self.vertical_threshold:
            self.current_vertical_state = MovementState.HEAD_DOWN_NOD
        elif latest_angle < -self.vertical_threshold:
            self.current_vertical_state = MovementState.HEAD_UP_NOD
        # 振動パターンをチェック
        elif self._detect_oscillation_pattern(
            [snapshot.vertical_angle for snapshot in self.angle_history], self.vertical_threshold
        ):
            self.current_vertical_state = MovementState.VERTICAL_NOD
        else:
            self.current_vertical_state = MovementState.HEAD_STATIC

        return self.current_vertical_state

    def check_alerts(self, timestamp: float) -> list[str]:
        """現在の頭部状態に基づき、アラートを発行すべきかチェックします。

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
            self.current_horizontal_state == MovementState.HORIZONTAL_SHAKE
            and timestamp - self.last_horizontal_alert_time > self.alert_cooldown
        ):
            alerts.append("[!] Horizontal Head Shake Detected")
            self.last_horizontal_alert_time = timestamp

        # 垂直うなずきアラート
        if (
            self.current_vertical_state == MovementState.VERTICAL_NOD
            and timestamp - self.last_vertical_alert_time > self.alert_cooldown
        ):
            alerts.append("[!] Vertical Head Nod Detected")
            self.last_vertical_alert_time = timestamp

        return alerts

    def get_status(self) -> dict[str, Any]:
        """現在の検出器の内部状態を要約して返します。

        デバッグや外部への状態通知に使用できます。最新の角度、状態、信頼度、
        および履歴として保持しているサンプル数が含まれます。

        Returns:
            dict[str, Any]: 検出器の現在の状態を示すキーと値のペアを含む辞書。
        """
        if not self.angle_history:
            return {
                "horizontal_state": MovementState.HEAD_STATIC.name,
                "vertical_state": MovementState.HEAD_STATIC.name,
                "horizontal_angle": 0.0,
                "vertical_angle": 0.0,
                "confidence": 0.0,
                "sample_count": 0,
            }

        latest = self.angle_history[-1]
        return {
            "horizontal_state": self.current_horizontal_state.name,
            "vertical_state": self.current_vertical_state.name,
            "horizontal_angle": latest.horizontal_angle,
            "vertical_angle": latest.vertical_angle,
            "confidence": latest.confidence,
            "sample_count": len(self.angle_history),
        }
