"""src/head_shake_detector.pyのテスト"""

import numpy as np
import pytest

from src.definitions import Angle, MovementState
from src.head_shake_detector import HeadShakeDetector
from src.pose.definitions import BodyPart


@pytest.fixture
def detector():
    """テスト用のHeadShakeDetectorインスタンスを生成するフィクスチャ"""
    return HeadShakeDetector(
        cycle_detection_window=20, min_oscillations=2, horizontal_threshold=15, vertical_threshold=15
    )


@pytest.fixture
def stable_landmarks():
    """テスト用の安定したダミーランドマークを生成するフィクスチャ（正面向き）"""
    landmarks = np.zeros((33, 4), dtype=np.float32)
    landmarks[:, 3] = 1.0  # visibility
    landmarks[BodyPart.NOSE] = [0.5, 0.5, 0, 1]
    landmarks[BodyPart.LEFT_EAR] = [0.4, 0.5, 0, 1]
    landmarks[BodyPart.RIGHT_EAR] = [0.6, 0.5, 0, 1]
    landmarks[BodyPart.LEFT_SHOULDER] = [0.4, 0.7, 0, 1]
    landmarks[BodyPart.RIGHT_SHOULDER] = [0.6, 0.7, 0, 1]
    return landmarks


def test_detect_oscillation_pattern_positive(detector):
    """振動パターンが正しく検出されるかテストする"""
    angles = [30 * np.sin(np.pi * i / 5) for i in range(20)]
    assert detector._detect_oscillation_pattern(angles, threshold=20) is True


def test_detect_oscillation_pattern_negative(detector):
    """振動がない場合に正しく検出されないことをテストする"""
    assert detector._detect_oscillation_pattern([5] * 20, threshold=20) is False


def test_update_static(detector, stable_landmarks):
    """頭が静止している場合にHEAD_STATICと判定されるかテストする"""
    for i in range(15):
        results = detector.update(stable_landmarks, timestamp=i * 0.1, frame_number=i)
    assert results[Angle.HEAD_HORIZONTAL_ROTATION]["state"] == MovementState.HEAD_STATIC
    assert results[Angle.HEAD_VERTICAL_NOD]["state"] == MovementState.HEAD_STATIC


def test_head_turn(detector, stable_landmarks):
    """頭の向きに応じて状態が変わるかテストする"""
    # 右向き
    landmarks_right = stable_landmarks.copy()
    landmarks_right[BodyPart.NOSE][0] = 0.55
    results = detector.update(landmarks_right, timestamp=0.1, frame_number=1)
    assert results[Angle.HEAD_HORIZONTAL_ROTATION]["state"] == MovementState.HEAD_RIGHT_TURN

    # 左向き
    landmarks_left = stable_landmarks.copy()
    landmarks_left[BodyPart.NOSE][0] = 0.45
    results = detector.update(landmarks_left, timestamp=0.2, frame_number=2)
    assert results[Angle.HEAD_HORIZONTAL_ROTATION]["state"] == MovementState.HEAD_LEFT_TURN


def test_horizontal_shake(detector, stable_landmarks):
    """水平の首振りがHORIZONTAL_SHAKEと判定されるかテストする"""
    landmarks_right = stable_landmarks.copy()
    landmarks_right[BodyPart.NOSE][0] = 0.55  # 角度 ~45
    landmarks_left = stable_landmarks.copy()
    landmarks_left[BodyPart.NOSE][0] = 0.45  # 角度 ~-45
    landmarks_center = stable_landmarks.copy()  # 角度 ~0

    # R -> C -> L -> C のシーケンスで「山頂」と「谷底」を意図的に作る
    sequence = [landmarks_right, landmarks_center, landmarks_left, landmarks_center] * 5  # 20 frames

    for i in range(20):
        detector.update(sequence[i], timestamp=i * 0.1, frame_number=i)

    # 最後のフレームで振動が検出されるはず
    results = detector.update(stable_landmarks, timestamp=2.0, frame_number=20)
    assert results[Angle.HEAD_HORIZONTAL_ROTATION]["state"] == MovementState.HORIZONTAL_SHAKE


def test_vertical_nod(detector, stable_landmarks):
    """垂直のうなずきがVERTICAL_NODと判定されるかテストする"""
    landmarks_down = stable_landmarks.copy()
    landmarks_down[BodyPart.NOSE][1] = 0.55  # 下向き
    landmarks_up = stable_landmarks.copy()
    landmarks_up[BodyPart.NOSE][1] = 0.45  # 上向き
    landmarks_center = stable_landmarks.copy()

    # D -> C -> U -> C のシーケンスで「山頂」と「谷底」を意図的に作る
    sequence = [landmarks_down, landmarks_center, landmarks_up, landmarks_center] * 5  # 20 frames

    for i in range(20):
        detector.update(sequence[i], timestamp=i * 0.1, frame_number=i)

    # 最後のフレームで振動が検出されるはず
    results = detector.update(stable_landmarks, timestamp=2.0, frame_number=20)
    assert results[Angle.HEAD_VERTICAL_NOD]["state"] == MovementState.VERTICAL_NOD
