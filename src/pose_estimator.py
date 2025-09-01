from __future__ import annotations

import cv2
import numpy as np
from mediapipe.python.solutions import pose


class PoseEstimator:
    """MediaPipe Poseを使用して姿勢を推定するクラス"""

    def __init__(self, model_complexity: int = 1, device: str = "cpu"):
        self.pose = pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # device引数は現状のmediapipeでは直接使用されませんが、
        # 将来の拡張性のために残しています。

    def estimate(self, image: np.ndarray) -> np.ndarray | None:
        """画像から姿勢のランドマークを推定する"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return None

        return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])

    def close(self):
        self.pose.close()
