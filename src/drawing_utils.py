from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark

from src.definitions import Angle

CONNECTIONS = [
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
    (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
    (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
    (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
    (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
    (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
]


def draw_landmarks(image: np.ndarray, landmarks: np.ndarray) -> None:
    """骨格を描画する"""
    h, w, _ = image.shape
    for landmark in landmarks:
        x, y = int(landmark[0] * w), int(landmark[1] * h)
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    for connection in CONNECTIONS:
        start_idx = connection[0].value
        end_idx = connection[1].value
        start_point = (
            int(landmarks[start_idx][0] * w),
            int(landmarks[start_idx][1] * h),
        )
        end_point = int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h)
        cv2.line(image, start_point, end_point, (255, 255, 255), 2)


def draw_analysis_results(
    image: np.ndarray,
    results: dict[Angle, dict[str, Any]],
    landmarks: np.ndarray,
) -> None:
    """分析結果を画像に描画する。"""
    h, w, _ = image.shape
    y_offset = 30
    for angle, data in results.items():
        angle_val = data["angle"]
        state = data["state"]
        text = f"{angle.value}: {angle_val:.1f} deg, {state.value}"
        cv2.putText(
            image,
            text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y_offset += 30

    # --- デバッグ用の描画: 体幹の中心線と垂直線 ---
    if Angle.BODY_TILT in results and landmarks is not None:
        p_left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER.value]
        p_right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER.value]
        p_left_hip = landmarks[PoseLandmark.LEFT_HIP.value]
        p_right_hip = landmarks[PoseLandmark.RIGHT_HIP.value]

        p_shoulder_mid = (
            int(((p_left_shoulder[0] + p_right_shoulder[0]) / 2) * w),
            int(((p_left_shoulder[1] + p_right_shoulder[1]) / 2) * h),
        )
        p_hip_mid = (
            int(((p_left_hip[0] + p_right_hip[0]) / 2) * w),
            int(((p_left_hip[1] + p_right_hip[1]) / 2) * h),
        )

        # 体幹の中心線 (緑)
        cv2.line(image, p_hip_mid, p_shoulder_mid, (0, 255, 0), 2)
        # 垂直線 (青)
        cv2.line(
            image,
            p_hip_mid,
            (p_hip_mid[0], p_hip_mid[1] + 100),
            (255, 0, 0),
            2,
        )
