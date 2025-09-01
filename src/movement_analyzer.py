from __future__ import annotations

import math

import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark

from .definitions import Angle, MovementState
from .pose.utils import calculate_angle, calculate_midpoint


class MovementAnalyzer:
    """mediapipeのlandmarksから関節の角度や動きの状態を分析する"""

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

            analysis_results[angle_name] = {
                "angle": angle,
                "state": state,
                "p1_confidence": float(p1[3]),
                "p2_confidence": float(p2[3]),
                "p3_confidence": float(p3[3]),
            }
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

        # 傾きの状態を判定（150度以下で前傾判定）
        tilt_state = MovementState.FORWARD_TILT if body_tilt_angle <= 150 else MovementState.UPRIGHT

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
            if neck_trunk_angle <= 150  # 150度以下なら猫背とみなす
            else MovementState.STRAIGHT
        )

        analysis_results[Angle.NECK_TRUNK_ANGLE] = {
            "angle": neck_trunk_angle,
            "state": neck_state,
        }
        # --- ここまで ---

        # --- 側屈（LATERAL_TILT）計算 ---
        # 肩線と腰線の傾斜（画面座標のx,yを使用）
        # ランドマークは正規化座標 [x, y, z, visibility]
        shoulder_dx = float(p_right_shoulder[0] - p_left_shoulder[0])
        shoulder_dy = float(p_right_shoulder[1] - p_left_shoulder[1])
        hip_dx = float(p_right_hip[0] - p_left_hip[0])
        hip_dy = float(p_right_hip[1] - p_left_hip[1])

        # arctan2で各線分の傾き（ラジアン）を求め、平均を側屈傾斜とする
        shoulder_slope = float(np.arctan2(shoulder_dy, shoulder_dx))
        hip_slope = float(np.arctan2(hip_dy, hip_dx))
        lateral_rad = (shoulder_slope + hip_slope) / 2.0
        lateral_deg = abs(float(np.degrees(lateral_rad)))

        # 閾値（度）。この値以上で側屈として扱う
        lateral_threshold_deg = 10.0

        if lateral_rad > 0 and lateral_deg >= lateral_threshold_deg:
            lateral_state = MovementState.RIGHT_TILT
        elif lateral_rad < 0 and lateral_deg >= lateral_threshold_deg:
            lateral_state = MovementState.LEFT_TILT
        else:
            lateral_state = MovementState.UPRIGHT

        analysis_results[Angle.LATERAL_TILT] = {
            "angle": lateral_deg,
            "state": lateral_state,
        }
        # --- 側屈 計算 ここまで ---

        # --- 首振り角度計算 ---
        head_horizontal_angle, head_vertical_angle = self._calculate_head_angles(landmarks)

        # 水平首振り（左右）
        horizontal_state = MovementState.HEAD_STATIC
        if abs(head_horizontal_angle) > 15.0:  # 15度閾値
            if head_horizontal_angle > 0:
                horizontal_state = MovementState.HEAD_RIGHT_TURN
            else:
                horizontal_state = MovementState.HEAD_LEFT_TURN

        analysis_results[Angle.HEAD_HORIZONTAL_ROTATION] = {
            "angle": head_horizontal_angle,
            "state": horizontal_state,
        }

        # 垂直うなずき（上下）
        vertical_state = MovementState.HEAD_STATIC
        if abs(head_vertical_angle) > 10.0:  # 10度閾値
            if head_vertical_angle > 0:
                vertical_state = MovementState.HEAD_DOWN_NOD
            else:
                vertical_state = MovementState.HEAD_UP_NOD

        analysis_results[Angle.HEAD_VERTICAL_NOD] = {
            "angle": head_vertical_angle,
            "state": vertical_state,
        }
        # --- 首振り角度計算 ここまで ---

        return analysis_results

    def _calculate_head_angles(self, landmarks: np.ndarray) -> tuple[float, float]:
        """
        首の水平・垂直角度を計算

        Returns:
            (horizontal_angle, vertical_angle): 水平角度、垂直角度（度）
        """
        try:
            # 鼻、左耳、右耳、肩の座標を取得
            nose = landmarks[PoseLandmark.NOSE.value]
            left_ear = landmarks[PoseLandmark.LEFT_EAR.value]
            right_ear = landmarks[PoseLandmark.RIGHT_EAR.value]
            left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER.value]

            # 水平角度計算（鼻と耳の関係）
            ear_vector = np.array([right_ear[0] - left_ear[0], right_ear[1] - left_ear[1]])
            ear_midpoint = np.array([(left_ear[0] + right_ear[0]) / 2, (left_ear[1] + right_ear[1]) / 2])
            nose_vector = np.array([nose[0] - ear_midpoint[0], nose[1] - ear_midpoint[1]])

            # 水平面での回転角度
            horizontal_angle_rad = math.atan2(nose_vector[1], nose_vector[0]) - math.atan2(ear_vector[1], ear_vector[0])
            horizontal_angle = math.degrees(horizontal_angle_rad)

            # 角度を -180° ～ +180° の範囲に正規化
            while horizontal_angle > 180:
                horizontal_angle -= 360
            while horizontal_angle < -180:
                horizontal_angle += 360

            # 垂直角度計算（鼻と肩の関係）
            shoulder_midpoint = np.array(
                [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]
            )
            nose_shoulder_vector = np.array([nose[0] - shoulder_midpoint[0], nose[1] - shoulder_midpoint[1]])
            vertical_vector = np.array([0, -1])  # 上向きベクトル

            vertical_angle_rad = math.atan2(nose_shoulder_vector[1], nose_shoulder_vector[0]) - math.atan2(
                vertical_vector[1], vertical_vector[0]
            )
            vertical_angle = math.degrees(vertical_angle_rad)

            # 角度を -180° ～ +180° の範囲に正規化
            while vertical_angle > 180:
                vertical_angle -= 360
            while vertical_angle < -180:
                vertical_angle += 360

            return float(horizontal_angle), float(vertical_angle)

        except (IndexError, TypeError, ZeroDivisionError):
            return 0.0, 0.0
