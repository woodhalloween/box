"""src/pose/pose_estimator.py

This module contains the PoseEstimator class, which serves as a wrapper around
the MediaPipe Pose solution. It simplifies the process of detecting pose
landmarks from an image by handling the initialization of the MediaPipe model
and the processing of image frames.
"""

from typing import Any

import cv2
import mediapipe as mp
import numpy as np


class PoseEstimator:
    """
    A wrapper for the MediaPipe Pose model to detect human pose landmarks.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """
        Initializes the PoseEstimator.

        Args:
            static_image_mode: Whether to treat the input images as a batch of static,
                possibly unrelated, images.
            model_complexity: Complexity of the pose landmark model: 0, 1, or 2.
            min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for the
                person detection to be considered successful.
            min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
                pose landmarks to be considered tracked successfully.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame: np.ndarray) -> Any | None:
        """フレームを処理して、骨格ランドマークを抽出します。

        Args:
            frame (np.ndarray): 入力フレーム。

        Returns:
            Any | None: 検出されたランドマーク。検出されない場合はNone。
                実際には mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
                を返しますが、型ヒントの互換性のためにAnyを使用します。
        """
        # MediaPipeはRGB画像を処理するため、BGRからRGBに変換します。
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # メモリ上でC連続な配列に変換します。
        image_rgb = np.ascontiguousarray(image_rgb)

        # 画像を処理して結果を取得します。
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return None

        return results.pose_landmarks

    def close(self) -> None:
        """
        Releases the MediaPipe Pose resources.
        """
        self.pose.close()
