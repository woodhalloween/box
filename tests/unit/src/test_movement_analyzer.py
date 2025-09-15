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


# === Additional tests for full coverage (all comments in English) ===


def _make_base_landmarks(conf: float = 1.0):
    """Create a simple landmarks array (33x4) with confidence `conf`."""
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[:, 3] = conf
    # Basic geometry for head/shoulders/hips/ears to be valid
    lm[PoseLandmark.LEFT_EAR.value, :2] = [0.4, 0.4]
    lm[PoseLandmark.RIGHT_EAR.value, :2] = [0.6, 0.4]
    lm[PoseLandmark.NOSE.value, :2] = [0.5, 0.35]
    lm[PoseLandmark.LEFT_SHOULDER.value, :2] = [0.45, 0.6]
    lm[PoseLandmark.RIGHT_SHOULDER.value, :2] = [0.55, 0.6]
    lm[PoseLandmark.LEFT_HIP.value, :2] = [0.46, 0.9]
    lm[PoseLandmark.RIGHT_HIP.value, :2] = [0.54, 0.9]
    # Elbow/wrist points to avoid unexpected ZeroDiv in angle calc paths
    lm[PoseLandmark.RIGHT_ELBOW.value, :2] = [0.55, 0.7]
    lm[PoseLandmark.RIGHT_WRIST.value, :2] = [0.56, 0.8]
    lm[PoseLandmark.LEFT_ELBOW.value, :2] = [0.45, 0.7]
    lm[PoseLandmark.LEFT_WRIST.value, :2] = [0.44, 0.8]
    lm[PoseLandmark.LEFT_KNEE.value, :2] = [0.46, 1.2]
    lm[PoseLandmark.RIGHT_KNEE.value, :2] = [0.54, 1.2]
    lm[PoseLandmark.LEFT_ANKLE.value, :2] = [0.46, 1.5]
    lm[PoseLandmark.RIGHT_ANKLE.value, :2] = [0.54, 1.5]
    return lm


def test_analyze_skips_angle_when_low_confidence_in_loop():
    """Angles with any keypoint confidence < threshold must be skipped (not stored in previous_angles)."""
    analyzer = MovementAnalyzer(confidence_threshold=0.7)
    lm = _make_base_landmarks(conf=1.0)
    # Force RIGHT_ELBOW triplet below threshold to trigger `continue`
    lm[PoseLandmark.RIGHT_SHOULDER.value, 3] = 0.1
    lm[PoseLandmark.RIGHT_ELBOW.value, 3] = 0.1
    lm[PoseLandmark.RIGHT_WRIST.value, 3] = 0.1

    analyzer.previous_angles.clear()
    _ = analyzer.analyze(lm)
    # The skipped angle should not be recorded into previous_angles
    assert Angle.RIGHT_ELBOW not in analyzer.previous_angles


def test_analyze_early_return_when_trunk_keypoints_low_confidence():
    """If trunk keypoints (shoulders/hips) are below threshold, analyzer should early-return without BODY_TILT."""
    analyzer = MovementAnalyzer(confidence_threshold=0.7)
    lm = _make_base_landmarks(conf=1.0)
    # Make left shoulder low confidence to trigger the early return
    lm[PoseLandmark.LEFT_SHOULDER.value, 3] = 0.2

    results = analyzer.analyze(lm)
    assert Angle.BODY_TILT not in results  # BODY_TILT is computed after the guard


def test_analyze_early_return_when_nose_low_confidence():
    """If nose confidence is below threshold, analyzer should return results without NECK_TRUNK_ANGLE."""
    analyzer = MovementAnalyzer(confidence_threshold=0.7)
    lm = _make_base_landmarks(conf=1.0)
    # Ensure trunk stage passes
    lm[PoseLandmark.LEFT_SHOULDER.value, 3] = 1.0
    lm[PoseLandmark.RIGHT_SHOULDER.value, 3] = 1.0
    lm[PoseLandmark.LEFT_HIP.value, 3] = 1.0
    lm[PoseLandmark.RIGHT_HIP.value, 3] = 1.0
    # Now make nose low confidence to trigger the early return at nose check
    lm[PoseLandmark.NOSE.value, 3] = 0.2

    results = analyzer.analyze(lm)
    assert Angle.BODY_TILT in results  # computed before the nose guard
    assert Angle.NECK_TRUNK_ANGLE not in results


def test_analyze_horizontal_left_turn_via_monkeypatch(monkeypatch):
    """When head_horizontal_angle < -15, state must be HEAD_LEFT_TURN."""
    analyzer = MovementAnalyzer()
    lm = _make_base_landmarks(conf=1.0)

    # Monkeypatch _calculate_head_angles to force a strong left turn
    monkeypatch.setattr(MovementAnalyzer, "_calculate_head_angles", lambda self, landmarks: (-20.0, 0.0), raising=True)
    results = analyzer.analyze(lm)
    assert results[Angle.HEAD_HORIZONTAL_ROTATION]["state"] == MovementState.HEAD_LEFT_TURN


def test_analyze_vertical_nod_up_and_down(monkeypatch):
    """Vertical nod block runs only if the key exists in previous_angles; assert both UP and DOWN states."""
    analyzer = MovementAnalyzer()
    lm = _make_base_landmarks(conf=1.0)

    # Prime the flag so the block executes
    analyzer.previous_angles[Angle.HEAD_VERTICAL_NOD] = 0.0

    # Create the method dynamically if absent; return +20 (DOWN nod)
    monkeypatch.setattr(MovementAnalyzer, "_calculate_head_vertical_angle", lambda self, landmarks: 20.0, raising=False)
    results = analyzer.analyze(lm)
    assert Angle.HEAD_VERTICAL_NOD in results
    assert results[Angle.HEAD_VERTICAL_NOD]["state"] == MovementState.HEAD_DOWN_NOD

    # Now test negative angle (UP nod)
    analyzer.previous_angles[Angle.HEAD_VERTICAL_NOD] = 0.0
    monkeypatch.setattr(
        MovementAnalyzer, "_calculate_head_vertical_angle", lambda self, landmarks: -25.0, raising=False
    )
    results = analyzer.analyze(lm)
    assert results[Angle.HEAD_VERTICAL_NOD]["state"] == MovementState.HEAD_UP_NOD


def test__calculate_head_angles_normalizes_angles_positive(monkeypatch):
    """Normalization: when degrees returns +540, while-loops must reduce to +180."""
    analyzer = MovementAnalyzer()
    lm = _make_base_landmarks(conf=1.0)

    # Patch math.degrees to always return 540 so both horizontal and vertical angles exceed +180.
    monkeypatch.setattr("src.movement_analyzer.math.degrees", lambda r: 540.0)
    h, v = analyzer._calculate_head_angles(lm)
    assert h == pytest.approx(180.0)
    assert v == pytest.approx(180.0)


def test__calculate_head_angles_normalizes_angles_negative(monkeypatch):
    """Normalization: when degrees returns -540, while-loops must increase to -180."""
    analyzer = MovementAnalyzer()
    lm = _make_base_landmarks(conf=1.0)

    monkeypatch.setattr("src.movement_analyzer.math.degrees", lambda r: -540.0)
    h, v = analyzer._calculate_head_angles(lm)
    assert h == pytest.approx(-180.0)
    assert v == pytest.approx(-180.0)


def test__calculate_head_angles_exception_returns_zeros():
    """If indexing fails (e.g., empty landmarks), method must return (0.0, 0.0)."""
    analyzer = MovementAnalyzer()
    empty = np.empty((0, 4), dtype=np.float32)
    h, v = analyzer._calculate_head_angles(empty)
    assert (h, v) == (0.0, 0.0)
