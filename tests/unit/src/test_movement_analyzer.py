"""src/movement_analyzer.pyのテスト"""

import numpy as np
import pytest
from mediapipe.python.solutions.pose import PoseLandmark

from src.definitions import Angle, MovementState
from src.movement_analyzer import MovementAnalyzer


@pytest.fixture
def dummy_landmarks():
    """テスト用の安定したダミーランドマークを生成するフィクスチャ"""
    landmarks = np.zeros((33, 4), dtype=np.float32)
    landmarks[:, 3] = 1.0  # visibilityを1に設定

    # 直立姿勢を基本とする
    # 頭
    landmarks[PoseLandmark.NOSE.value] = [0.5, 0.1, 0, 1]
    landmarks[PoseLandmark.LEFT_EAR.value] = [0.45, 0.1, -0.05, 1]
    landmarks[PoseLandmark.RIGHT_EAR.value] = [0.55, 0.1, -0.05, 1]
    # 肩 (x座標: left < right)
    landmarks[PoseLandmark.LEFT_SHOULDER.value] = [0.4, 0.2, 0, 1]
    landmarks[PoseLandmark.RIGHT_SHOULDER.value] = [0.6, 0.2, 0, 1]
    # 肘
    landmarks[PoseLandmark.LEFT_ELBOW.value] = [0.35, 0.3, 0, 1]
    landmarks[PoseLandmark.RIGHT_ELBOW.value] = [0.65, 0.3, 0, 1]
    # 手首
    landmarks[PoseLandmark.LEFT_WRIST.value] = [0.3, 0.4, 0, 1]
    landmarks[PoseLandmark.RIGHT_WRIST.value] = [0.7, 0.4, 0, 1]
    # 腰 (x座標: left < right)
    landmarks[PoseLandmark.LEFT_HIP.value] = [0.45, 0.5, 0, 1]
    landmarks[PoseLandmark.RIGHT_HIP.value] = [0.55, 0.5, 0, 1]
    # 膝
    landmarks[PoseLandmark.LEFT_KNEE.value] = [0.45, 0.7, 0, 1]
    landmarks[PoseLandmark.RIGHT_KNEE.value] = [0.55, 0.7, 0, 1]
    # 足首
    landmarks[PoseLandmark.LEFT_ANKLE.value] = [0.45, 0.9, 0, 1]
    landmarks[PoseLandmark.RIGHT_ANKLE.value] = [0.55, 0.9, 0, 1]

    return landmarks


def test_movement_analyzer_initial_state(dummy_landmarks):
    """MovementAnalyzerの初回analyze呼び出しで状態がSTATICになることをテストする"""
    analyzer = MovementAnalyzer()
    results = analyzer.analyze(dummy_landmarks)

    # 初回はprevious_anglesがないため、主要な関節はSTATICになるはず
    main_angles = analyzer.ANGLE_DEFINITIONS.keys()
    for angle_name in main_angles:
        assert results[angle_name]["state"] == MovementState.STATIC


def test_movement_analyzer_state_change(dummy_landmarks):
    """ランドマークの変化に応じて関節の状態が正しく変化するかテストする"""
    analyzer = MovementAnalyzer(angle_threshold=5.0)

    # 1. 初期状態（直立）
    analyzer.analyze(dummy_landmarks)

    # 2. 屈曲状態にする
    flexion_landmarks = dummy_landmarks.copy()
    flexion_landmarks[PoseLandmark.RIGHT_WRIST.value] = [0.75, 0.3, 0, 1]  # 手首を水平に移動させ、肘を90度に
    results_flexion = analyzer.analyze(flexion_landmarks)
    assert results_flexion[Angle.RIGHT_ELBOW]["state"] == MovementState.FLEXION

    # 3. 伸展テスト（屈曲状態から直立状態に戻す）
    # `dummy_landmarks` は初期の直立状態
    results_extension = analyzer.analyze(dummy_landmarks)
    assert results_extension[Angle.RIGHT_ELBOW]["state"] == MovementState.EXTENSION


def test_body_tilt_forward(dummy_landmarks):
    """前傾姿勢が正しく検出されるかテストする"""
    analyzer = MovementAnalyzer()
    tilted_landmarks = dummy_landmarks.copy()
    # 肩を前に倒す (z座標を変化させる)
    tilted_landmarks[PoseLandmark.LEFT_SHOULDER.value][2] = -0.3
    tilted_landmarks[PoseLandmark.RIGHT_SHOULDER.value][2] = -0.3
    results = analyzer.analyze(tilted_landmarks)
    assert results[Angle.BODY_TILT]["state"] == MovementState.FORWARD_TILT


def test_hunch_detection(dummy_landmarks):
    """うつむき（猫背）姿勢が正しく検出されるかテストする"""
    analyzer = MovementAnalyzer()
    hunch_landmarks = dummy_landmarks.copy()
    # 鼻を肩より下に（前に）出す
    hunch_landmarks[PoseLandmark.NOSE.value] = [0.5, 0.3, 0.1, 1]
    shoulder_mid_y = (
        dummy_landmarks[PoseLandmark.LEFT_SHOULDER.value][1] + dummy_landmarks[PoseLandmark.RIGHT_SHOULDER.value][1]
    ) / 2
    hunch_landmarks[PoseLandmark.NOSE.value][1] = shoulder_mid_y + 0.1

    results = analyzer.analyze(hunch_landmarks)
    assert results[Angle.NECK_TRUNK_ANGLE]["state"] == MovementState.HUNCH


def test_lateral_tilt_detection(dummy_landmarks):
    """側屈が正しく検出されるかテストする"""
    analyzer = MovementAnalyzer()

    # 右に傾ける
    right_tilt_landmarks = dummy_landmarks.copy()
    right_tilt_landmarks[PoseLandmark.RIGHT_SHOULDER.value][1] += 0.1  # 右肩を下げる
    right_tilt_landmarks[PoseLandmark.LEFT_SHOULDER.value][1] -= 0.1  # 左肩を上げる
    results_right = analyzer.analyze(right_tilt_landmarks)
    assert results_right[Angle.LATERAL_TILT]["state"] == MovementState.RIGHT_TILT

    # 左に傾ける
    analyzer = MovementAnalyzer()  # アナライザーを初期化
    left_tilt_landmarks = dummy_landmarks.copy()
    left_tilt_landmarks[PoseLandmark.LEFT_SHOULDER.value][1] += 0.1  # 左肩を下げる
    left_tilt_landmarks[PoseLandmark.RIGHT_SHOULDER.value][1] -= 0.1  # 右肩を上げる
    results_left = analyzer.analyze(left_tilt_landmarks)
    assert results_left[Angle.LATERAL_TILT]["state"] == MovementState.LEFT_TILT
