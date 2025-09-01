"""src/pose_estimator.pyのテスト"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pose_estimator import PoseEstimator


@pytest.fixture
def mock_mediapipe_pose():
    """mediapipe.python.solutions.pose.Poseをモック化するフィクスチャ"""
    with patch("mediapipe.python.solutions.pose.Pose") as mock_pose_class:
        mock_pose_instance = MagicMock()
        mock_pose_class.return_value = mock_pose_instance
        yield mock_pose_instance


def test_pose_estimator_init(mock_mediapipe_pose):
    """PoseEstimatorの初期化時にmediapipe.pose.Poseが正しく呼ばれるかテストする"""
    with patch("mediapipe.python.solutions.pose.Pose") as mock_pose_class:
        PoseEstimator(model_complexity=2)
        mock_pose_class.assert_called_once_with(
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )


def test_pose_estimator_estimate_landmarks_found(mock_mediapipe_pose):
    """estimateメソッドでランドマークが検出された場合のテスト"""
    # ダミーのランドマークデータを作成
    mock_landmark = MagicMock()
    mock_landmark.x, mock_landmark.y, mock_landmark.z, mock_landmark.visibility = 0.1, 0.2, 0.3, 0.9
    mock_results = MagicMock()
    mock_results.pose_landmarks.landmark = [mock_landmark] * 33
    mock_mediapipe_pose.process.return_value = mock_results

    estimator = PoseEstimator()
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    landmarks = estimator.estimate(dummy_image)

    mock_mediapipe_pose.process.assert_called_once()
    assert isinstance(landmarks, np.ndarray)
    assert landmarks.shape == (33, 4)
    np.testing.assert_allclose(landmarks[0], [0.1, 0.2, 0.3, 0.9])


def test_pose_estimator_estimate_no_landmarks(mock_mediapipe_pose):
    """estimateメソッドでランドマークが検出されなかった場合のテスト"""
    mock_results = MagicMock()
    mock_results.pose_landmarks = None
    mock_mediapipe_pose.process.return_value = mock_results

    estimator = PoseEstimator()
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    landmarks = estimator.estimate(dummy_image)

    mock_mediapipe_pose.process.assert_called_once()
    assert landmarks is None


def test_pose_estimator_close(mock_mediapipe_pose):
    """closeメソッドが正しく内部のposeインスタンスのcloseを呼び出すかテストする"""
    estimator = PoseEstimator()
    estimator.close()
    mock_mediapipe_pose.close.assert_called_once()
