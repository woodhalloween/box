import sys
from pathlib import Path

# プロジェクトルートをsys.pathに追加
sys.path.append(str(Path(__file__).resolve().parents[4]))

import numpy as np
import pytest
from mediapipe.python.solutions.pose import PoseLandmark

from src.detectors.hand_raise_refactored import HandRaiseDetector

# --- テスト用の設定 ---
VISIBILITY_THRESHOLD = 0.5
MIN_CONSECUTIVE_FRAMES = 3


# --- テスト用のランドマーク生成ヘルパー ---
def create_landmarks(
    wrist_y: float,
    shoulder_y: float,
    wrist_visibility: float = 1.0,
    shoulder_visibility: float = 1.0,
) -> np.ndarray:
    """テスト用のランドマークデータを生成する。"""
    landmarks = np.zeros((33, 4), dtype=np.float32)
    # 左手
    landmarks[PoseLandmark.LEFT_WRIST] = [0.5, wrist_y, 0, wrist_visibility]
    landmarks[PoseLandmark.LEFT_SHOULDER] = [0.5, shoulder_y, 0, shoulder_visibility]
    # 右手
    landmarks[PoseLandmark.RIGHT_WRIST] = [0.5, wrist_y, 0, wrist_visibility]
    landmarks[PoseLandmark.RIGHT_SHOULDER] = [0.5, shoulder_y, 0, shoulder_visibility]
    return landmarks


@pytest.fixture
def detector() -> HandRaiseDetector:
    """HandRaiseDetectorのインスタンスを生成するpytestフィクスチャ。"""
    return HandRaiseDetector(
        visibility_threshold=VISIBILITY_THRESHOLD,
        min_consecutive_frames=MIN_CONSECUTIVE_FRAMES,
    )


def test_initial_state(detector: HandRaiseDetector):
    """初期状態では手が挙がっていないことをテストする。"""
    landmarks = create_landmarks(wrist_y=0.6, shoulder_y=0.4)  # 手を下げている
    result = detector.detect(landmarks)
    assert not result["left_hand_raised"]
    assert not result["right_hand_raised"]


def test_hand_raised_below_threshold(detector: HandRaiseDetector):
    """手が挙がっているが、連続フレーム数が閾値未満の場合をテストする。"""
    landmarks_up = create_landmarks(wrist_y=0.2, shoulder_y=0.4)  # 手を挙げている

    for _ in range(MIN_CONSECUTIVE_FRAMES - 1):
        result = detector.detect(landmarks_up)
        assert not result["left_hand_raised"]
        assert not result["right_hand_raised"]


def test_hand_raised_above_threshold(detector: HandRaiseDetector):
    """手が挙がっており、連続フレーム数が閾値に達した場合をテストする。"""
    landmarks_up = create_landmarks(wrist_y=0.2, shoulder_y=0.4)

    # 閾値未満のフレーム
    for _ in range(MIN_CONSECUTIVE_FRAMES - 1):
        detector.detect(landmarks_up)

    # 閾値に達したフレーム
    result = detector.detect(landmarks_up)
    assert result["left_hand_raised"]
    assert result["right_hand_raised"]


def test_hand_lowered_after_raised(detector: HandRaiseDetector):
    """一度手を挙げたと判定された後、手を下げると状態がリセットされることをテストする。"""
    landmarks_up = create_landmarks(wrist_y=0.2, shoulder_y=0.4)
    landmarks_down = create_landmarks(wrist_y=0.6, shoulder_y=0.4)

    # 手を挙げる
    for _ in range(MIN_CONSECUTIVE_FRAMES):
        detector.detect(landmarks_up)

    result = detector.detect(landmarks_up)
    assert result["left_hand_raised"]
    assert result["right_hand_raised"]

    # 手を下げる
    result_lowered = detector.detect(landmarks_down)
    assert not result_lowered["left_hand_raised"]
    assert not result_lowered["right_hand_raised"]


def test_no_landmarks_resets_counter(detector: HandRaiseDetector):
    """ランドマークがNoneの場合にカウンターがリセットされることをテストする。"""
    landmarks_up = create_landmarks(wrist_y=0.2, shoulder_y=0.4)

    detector.detect(landmarks_up)  # カウンターを1にする
    detector.detect(None)  # リセット
    result = detector.detect(landmarks_up)  # 再度検出

    assert detector._left_hand_consecutive_frames == 1
    assert detector._right_hand_consecutive_frames == 1
    assert not result["left_hand_raised"]
    assert not result["right_hand_raised"]


def test_visibility_below_threshold(detector: HandRaiseDetector):
    """ランドマークの信頼度が閾値未満の場合、手が挙がっていても検出されないことをテストする。"""
    landmarks_low_visibility = create_landmarks(
        wrist_y=0.2, shoulder_y=0.4, wrist_visibility=VISIBILITY_THRESHOLD - 0.1
    )

    for _ in range(MIN_CONSECUTIVE_FRAMES):
        result = detector.detect(landmarks_low_visibility)
        assert not result["left_hand_raised"]
        assert not result["right_hand_raised"]


def test_independent_hand_detection(detector: HandRaiseDetector):
    """左右の手が独立して検出されることをテストする。"""
    # 左手だけが挙がっているランドマーク
    landmarks_left_up = np.zeros((33, 4), dtype=np.float32)
    landmarks_left_up[PoseLandmark.LEFT_WRIST] = [0.5, 0.2, 0, 1.0]
    landmarks_left_up[PoseLandmark.LEFT_SHOULDER] = [0.5, 0.4, 0, 1.0]
    landmarks_left_up[PoseLandmark.RIGHT_WRIST] = [0.5, 0.6, 0, 1.0]  # 右手は下がっている
    landmarks_left_up[PoseLandmark.RIGHT_SHOULDER] = [0.5, 0.4, 0, 1.0]

    for _ in range(MIN_CONSECUTIVE_FRAMES):
        result = detector.detect(landmarks_left_up)

    assert result["left_hand_raised"]
    assert not result["right_hand_raised"]

    # 右手だけが挙がっているランドマーク
    landmarks_right_up = np.zeros((33, 4), dtype=np.float32)
    landmarks_right_up[PoseLandmark.LEFT_WRIST] = [0.5, 0.6, 0, 1.0]  # 左手は下がっている
    landmarks_right_up[PoseLandmark.LEFT_SHOULDER] = [0.5, 0.4, 0, 1.0]
    landmarks_right_up[PoseLandmark.RIGHT_WRIST] = [0.5, 0.2, 0, 1.0]
    landmarks_right_up[PoseLandmark.RIGHT_SHOULDER] = [0.5, 0.4, 0, 1.0]

    # Detectorをリセットして再テスト
    detector = HandRaiseDetector(
        visibility_threshold=VISIBILITY_THRESHOLD,
        min_consecutive_frames=MIN_CONSECUTIVE_FRAMES,
    )
    for _ in range(MIN_CONSECUTIVE_FRAMES):
        result = detector.detect(landmarks_right_up)

    assert not result["left_hand_raised"]
    assert result["right_hand_raised"]
