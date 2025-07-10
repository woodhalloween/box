from __future__ import annotations

import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark

from src.definitions import Angle, MovementState
from src.utils import calculate_angle, calculate_midpoint


class MovementAnalyzer:
    """関節の角度と動きの状態を分析するクラス"""

    ANGLE_DEFINITIONS: dict[Angle, tuple[PoseLandmark, PoseLandmark, PoseLandmark]] = {
        Angle.RIGHT_ELBOW: (
            PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.RIGHT_ELBOW,
            PoseLandmark.RIGHT_WRIST,
        ),
        Angle.LEFT_ELBOW: (
            PoseLandmark.LEFT_SHOULDER,
            PoseLandmark.LEFT_ELBOW,
            PoseLandmark.LEFT_WRIST,
        ),
        Angle.RIGHT_SHOULDER: (
            PoseLandmark.RIGHT_ELBOW,
            PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.RIGHT_HIP,
        ),
        Angle.LEFT_SHOULDER: (
            PoseLandmark.LEFT_ELBOW,
            PoseLandmark.LEFT_SHOULDER,
            PoseLandmark.LEFT_HIP,
        ),
        Angle.RIGHT_HIP: (
            PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.RIGHT_HIP,
            PoseLandmark.RIGHT_KNEE,
        ),
        Angle.LEFT_HIP: (
            PoseLandmark.LEFT_SHOULDER,
            PoseLandmark.LEFT_HIP,
            PoseLandmark.LEFT_KNEE,
        ),
        Angle.RIGHT_KNEE: (
            PoseLandmark.RIGHT_HIP,
            PoseLandmark.RIGHT_KNEE,
            PoseLandmark.RIGHT_ANKLE,
        ),
        Angle.LEFT_KNEE: (
            PoseLandmark.LEFT_HIP,
            PoseLandmark.LEFT_KNEE,
            PoseLandmark.LEFT_ANKLE,
        ),
    }

    def __init__(self, angle_threshold: float = 5.0):
        self.angle_threshold = angle_threshold
        self.previous_angles: dict[Angle, float] = {}

    def analyze(self, landmarks: np.ndarray) -> dict[Angle, dict[str, float | MovementState]]:
        """ランドマークデータから各関節の角度と状態を分析する"""
        analysis_results = {}

        # 各関節の角度と運動状態を計算
        for angle_name, points in self.ANGLE_DEFINITIONS.items():
            p1 = landmarks[points[0].value]
            p2 = landmarks[points[1].value]
            p3 = landmarks[points[2].value]

            angle = calculate_angle(p1, p2, p3)
            previous_angle = self.previous_angles.get(angle_name)

            state = MovementState.UNKNOWN
            if previous_angle is not None:
                if angle < previous_angle - self.angle_threshold:
                    state = MovementState.FLEXION  # 屈曲
                elif angle > previous_angle + self.angle_threshold:
                    state = MovementState.EXTENSION  # 伸展
                else:
                    state = MovementState.STATIC
            else:
                state = MovementState.STATIC

            analysis_results[angle_name] = {"angle": angle, "state": state}
            self.previous_angles[angle_name] = angle

        # 体幹の傾き計算
        p_left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER.value]
        p_right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER.value]
        p_left_hip = landmarks[PoseLandmark.LEFT_HIP.value]
        p_right_hip = landmarks[PoseLandmark.RIGHT_HIP.value]

        p_shoulder_mid = calculate_midpoint(p_left_shoulder, p_right_shoulder)
        p_hip_mid = calculate_midpoint(p_left_hip, p_right_hip)

        # 3次元座標のみを使用する (x, y, z)
        p_shoulder_mid_3d = p_shoulder_mid[:3]
        p_hip_mid_3d = p_hip_mid[:3]

        # 腰の中心から真下に仮想の点を設定（垂直ベクトル用）
        p_hip_vertical = p_hip_mid_3d + np.array([0, 1, 0])

        # 体幹の傾き角度を計算
        body_tilt_angle = calculate_angle(p_shoulder_mid_3d, p_hip_mid_3d, p_hip_vertical)

        # 傾きの状態を判定
        tilt_state = MovementState.FORWARD_TILT if body_tilt_angle > 15 else MovementState.UPRIGHT

        analysis_results[Angle.BODY_TILT] = {
            "angle": body_tilt_angle,
            "state": tilt_state,
        }

        # --- 頸部-体幹角度（うつむき）計算 ---
        p_nose = landmarks[PoseLandmark.NOSE.value]

        # 体幹の傾き計算で使った中心点を再利用
        # 3次元座標のみを使用
        p_hip_mid_3d = p_hip_mid[:3]
        p_shoulder_mid_3d = p_shoulder_mid[:3]
        p_nose_3d = p_nose[:3]

        # 肩を中心に、腰、肩、鼻がなす角度を計算
        neck_trunk_angle = calculate_angle(p_hip_mid_3d, p_shoulder_mid_3d, p_nose_3d)

        # 状態を判定
        neck_state = (
            MovementState.HUNCH
            if neck_trunk_angle < 165  # 165度未満なら猫背とみなす
            else MovementState.STRAIGHT
        )

        analysis_results[Angle.NECK_TRUNK_ANGLE] = {
            "angle": neck_trunk_angle,
            "state": neck_state,
        }
        # --- ここまで ---

        return analysis_results
