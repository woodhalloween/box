"""src/pose/movement_analyzer.py

This module contains the JointMovementAnalyzer class, which is responsible for
analyzing the pose landmarks to calculate joint angles and eventually detect
specific movements.
"""

from collections import deque
from typing import Any

import numpy as np

from .definitions import BodyPart, Movement
from .utils import calculate_angle


class JointMovementAnalyzer:
    """関節の動きを分析し、運動の種類を検出するクラス"""

    # 角度を計算するための関節の組み合わせを定義
    ANGLE_DEFINITIONS: dict[str, tuple[BodyPart, BodyPart, BodyPart]] = {
        "right_elbow": (BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST),
        "left_elbow": (BodyPart.LEFT_SHOULDER, BodyPart.LEFT_ELBOW, BodyPart.LEFT_WRIST),
        "right_shoulder": (BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_HIP),
        "left_shoulder": (BodyPart.LEFT_ELBOW, BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP),
        "right_hip": (BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE),
        "left_hip": (BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE),
        "right_knee": (BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE),
        "left_knee": (BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE),
    }

    def __init__(self, history_size: int = 5, flexion_threshold: float = -2, extension_threshold: float = 2):
        self.angle_history: dict[str, deque[float]] = {
            name: deque(maxlen=history_size) for name in self.ANGLE_DEFINITIONS
        }
        self.flexion_threshold = flexion_threshold
        self.extension_threshold = extension_threshold

    def analyze_frame(self, landmarks: Any) -> dict[str, tuple[float, Movement]]:
        if not landmarks:
            return {}

        analysis_results: dict[str, tuple[float, Movement]] = {}

        for name, (p1_idx, p2_idx, p3_idx) in self.ANGLE_DEFINITIONS.items():
            try:
                p1 = np.array(
                    [landmarks.landmark[p1_idx].x, landmarks.landmark[p1_idx].y, landmarks.landmark[p1_idx].z]
                )
                p2 = np.array(
                    [landmarks.landmark[p2_idx].x, landmarks.landmark[p2_idx].y, landmarks.landmark[p2_idx].z]
                )
                p3 = np.array(
                    [landmarks.landmark[p3_idx].x, landmarks.landmark[p3_idx].y, landmarks.landmark[p3_idx].z]
                )

                angle = calculate_angle(p1, p2, p3)

                movement = self._detect_movement(name, angle)
                analysis_results[name] = (angle, movement)

                self.angle_history[name].append(angle)

            except (IndexError, AttributeError):
                continue  # ランドマークが検出できない場合はスキップ

        return analysis_results

    def _detect_movement(self, joint_name: str, current_angle: float) -> Movement:
        history = self.angle_history[joint_name]
        if len(history) < 2:
            return Movement.NONE

        # 角度の変化率を簡易的に計算
        angle_change = current_angle - np.mean(list(history))

        if angle_change < self.flexion_threshold:
            return Movement.FLEXION
        if angle_change > self.extension_threshold:
            return Movement.EXTENSION
        return Movement.NONE
