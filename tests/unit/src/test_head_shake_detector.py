"""src/head_shake_detector.pyのテスト"""

import numpy as np
import pytest

from src.definitions import Angle, MovementState
from src.head_shake_detector import HeadAngleSnapshot, HeadShakeDetector
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


# --- Add below to the end of test_head_shake_detector.py ---


def _fill_history_for_angles(detector, horiz_values, vert_values, conf=1.0, start_ts=0.0):
    """Utility: populate angle_history with synthetic snapshots of given angles."""
    detector.angle_history.clear()
    detector.horizontal_angles.clear()
    detector.vertical_angles.clear()
    detector.timestamps.clear()

    # Ensure same length for both sequences
    n = max(len(horiz_values), len(vert_values))
    for i in range(n):
        h = horiz_values[i] if i < len(horiz_values) else 0.0
        v = vert_values[i] if i < len(vert_values) else 0.0
        snap = HeadAngleSnapshot(
            timestamp=start_ts + i * 0.033,
            frame_number=i,
            horizontal_angle=h,
            vertical_angle=v,
            confidence=conf,
        )
        detector.angle_history.append(snap)
        detector.horizontal_angles.append(h)
        detector.vertical_angles.append(v)
        detector.timestamps.append(start_ts + i * 0.033)


# ------------------------------
# _analyze_horizontal_movement
# ------------------------------
def test__analyze_horizontal_movement_right_left_static(detector):
    """Directly exercise threshold branches: RIGHT, LEFT, STATIC."""
    thr = detector.horizontal_threshold

    # Case: latest > +threshold -> HEAD_RIGHT_TURN
    _fill_history_for_angles(
        detector,
        horiz_values=[0.0] * 9 + [thr + 5.0],  # >=10 samples required internally
        vert_values=[0.0] * 10,
    )
    state = detector._analyze_horizontal_movement()
    assert state == MovementState.HEAD_RIGHT_TURN

    # Case: latest < -threshold -> HEAD_LEFT_TURN
    _fill_history_for_angles(
        detector,
        horiz_values=[0.0] * 9 + [-(thr + 3.0)],
        vert_values=[0.0] * 10,
    )
    state = detector._analyze_horizontal_movement()
    assert state == MovementState.HEAD_LEFT_TURN

    # Case: within threshold, no oscillation -> HEAD_STATIC
    _fill_history_for_angles(
        detector,
        horiz_values=[0.0] * 12,  # flat series
        vert_values=[0.0] * 12,
    )
    state = detector._analyze_horizontal_movement()
    assert state == MovementState.HEAD_STATIC


def test__analyze_horizontal_movement_oscillation(detector):
    """Within threshold at the latest sample, but oscillatory pattern across history -> HORIZONTAL_SHAKE."""
    thr = detector.horizontal_threshold
    # Build peaks/valleys above threshold to satisfy internal peak/valley counting.
    # Sequence: 0, +A, 0, -A, 0, +A, 0, -A, ...  (A > threshold)
    amplitude = thr + 5.0
    seq = [0.0, +amplitude, 0.0, -amplitude] * 5  # 20 samples, enough for min_oscillations=2
    # Make the latest point neutral (within threshold) to avoid the immediate RIGHT/LEFT branch.
    if seq[-1] != 0.0:
        seq[-1] = 0.0
    _fill_history_for_angles(detector, horiz_values=seq, vert_values=[0.0] * len(seq))
    state = detector._analyze_horizontal_movement()
    assert state == MovementState.HORIZONTAL_SHAKE


# ------------------------------
# _analyze_vertical_movement
# ------------------------------
def test__analyze_vertical_movement_up_down_static(detector):
    """Directly exercise threshold branches: HEAD_DOWN_NOD, HEAD_UP_NOD, HEAD_STATIC."""
    thr = detector.vertical_threshold

    # Case: latest > +threshold -> HEAD_DOWN_NOD
    _fill_history_for_angles(
        detector,
        horiz_values=[0.0] * 10,
        vert_values=[0.0] * 9 + [thr + 2.0],
    )
    state = detector._analyze_vertical_movement()
    assert state == MovementState.HEAD_DOWN_NOD

    # Case: latest < -threshold -> HEAD_UP_NOD
    _fill_history_for_angles(
        detector,
        horiz_values=[0.0] * 10,
        vert_values=[0.0] * 9 + [-(thr + 4.0)],
    )
    state = detector._analyze_vertical_movement()
    assert state == MovementState.HEAD_UP_NOD

    # Case: within threshold, no oscillation -> HEAD_STATIC
    _fill_history_for_angles(detector, horiz_values=[0.0] * 12, vert_values=[0.0] * 12)
    state = detector._analyze_vertical_movement()
    assert state == MovementState.HEAD_STATIC


def test__analyze_vertical_movement_oscillation(detector):
    """Within threshold at the latest sample, but oscillatory pattern across history -> VERTICAL_NOD."""
    thr = detector.vertical_threshold
    amplitude = thr + 6.0
    seq = [0.0, +amplitude, 0.0, -amplitude] * 5  # 20 samples
    if seq[-1] != 0.0:
        seq[-1] = 0.0
    _fill_history_for_angles(detector, horiz_values=[0.0] * len(seq), vert_values=seq)
    state = detector._analyze_vertical_movement()
    assert state == MovementState.VERTICAL_NOD


# ------------------------------
# check_alerts
# ------------------------------
def test_check_alerts_horizontal_with_cooldown(detector):
    """Emit horizontal alert once, then suppress within cooldown, then emit again after cooldown."""
    # Prepare state as if the detector already concluded horizontal shake.
    detector.current_horizontal_state = MovementState.HORIZONTAL_SHAKE
    detector.current_vertical_state = MovementState.HEAD_STATIC
    detector.last_horizontal_alert_time = 0.0
    detector.last_vertical_alert_time = 0.0
    detector.alert_cooldown = 5.0

    # First call -> should produce one horizontal alert
    alerts1 = detector.check_alerts(timestamp=10.0)
    assert "[!] Horizontal Head Shake Detected" in alerts1

    # Within cooldown -> no horizontal alert
    alerts2 = detector.check_alerts(timestamp=12.0)
    assert "[!] Horizontal Head Shake Detected" not in alerts2

    # After cooldown -> alert again
    alerts3 = detector.check_alerts(timestamp=16.0)
    assert "[!] Horizontal Head Shake Detected" in alerts3


def test_check_alerts_vertical_with_cooldown(detector):
    """Emit vertical alert once, suppress within cooldown, then emit again after cooldown."""
    detector.current_horizontal_state = MovementState.HEAD_STATIC
    detector.current_vertical_state = MovementState.VERTICAL_NOD
    detector.last_horizontal_alert_time = float("-inf")
    detector.last_vertical_alert_time = float("-inf")
    detector.alert_cooldown = 3.0

    alerts1 = detector.check_alerts(timestamp=1.0)
    assert "[!] Vertical Head Nod Detected" in alerts1

    alerts2 = detector.check_alerts(timestamp=2.5)  # within cooldown
    assert "[!] Vertical Head Nod Detected" not in alerts2

    alerts3 = detector.check_alerts(timestamp=4.1)  # after cooldown
    assert "[!] Vertical Head Nod Detected" in alerts3


# ------------------------------
# get_status
# ------------------------------
def test_get_status_when_empty(detector):
    """When no samples exist, get_status returns zeroed angles and HEAD_STATIC states."""
    # Ensure empty
    detector.angle_history.clear()
    detector.current_horizontal_state = MovementState.HEAD_STATIC
    detector.current_vertical_state = MovementState.HEAD_STATIC

    status = detector.get_status()
    assert status["horizontal_state"] == MovementState.HEAD_STATIC.name
    assert status["vertical_state"] == MovementState.HEAD_STATIC.name
    assert status["horizontal_angle"] == 0.0
    assert status["vertical_angle"] == 0.0
    assert status["confidence"] == 0.0
    assert status["sample_count"] == 0


def test_get_status_when_filled(detector):
    """When samples exist, get_status reflects the latest snapshot and current states."""
    # Populate with a few snapshots and set current states explicitly
    _fill_history_for_angles(
        detector,
        horiz_values=[1.0, 2.0, 3.0, 20.0],
        vert_values=[-1.0, -2.0, -3.0, -25.0],
        conf=0.85,
        start_ts=100.0,
    )
    detector.current_horizontal_state = MovementState.HEAD_RIGHT_TURN
    detector.current_vertical_state = MovementState.HEAD_UP_NOD

    status = detector.get_status()
    assert status["horizontal_state"] == MovementState.HEAD_RIGHT_TURN.name
    assert status["vertical_state"] == MovementState.HEAD_UP_NOD.name
    # Latest angles must match the last snapshot inserted above
    assert status["horizontal_angle"] == 20.0
    assert status["vertical_angle"] == -25.0
    assert status["confidence"] == 0.85
    assert status["sample_count"] >= 4  # exact count depends on window, but >= 4 here
