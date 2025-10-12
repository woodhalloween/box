"""Torso and dwell-time detection component tests."""

from __future__ import annotations

import sys
import types
from enum import IntEnum

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mediapipe stub (tests only)
# ---------------------------------------------------------------------------


def _ensure_mediapipe_stub() -> None:
    """Install a lightweight mediapipe stub so imports succeed during tests."""
    if "mediapipe" in sys.modules:
        return

    class _PoseLandmark(IntEnum):
        NOSE = 0
        LEFT_EYE_INNER = 1
        LEFT_EYE = 2
        LEFT_EYE_OUTER = 3
        RIGHT_EYE_INNER = 4
        RIGHT_EYE = 5
        RIGHT_EYE_OUTER = 6
        LEFT_EAR = 7
        RIGHT_EAR = 8
        MOUTH_LEFT = 9
        MOUTH_RIGHT = 10
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_PINKY = 17
        RIGHT_PINKY = 18
        LEFT_INDEX = 19
        RIGHT_INDEX = 20
        LEFT_THUMB = 21
        RIGHT_THUMB = 22
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        LEFT_HEEL = 29
        RIGHT_HEEL = 30
        LEFT_FOOT_INDEX = 31
        RIGHT_FOOT_INDEX = 32

    class _Pose:
        """Minimal stub so patching Pose works in unit tests."""

        def __init__(self, *args, **kwargs):
            self._args = args
            self._kwargs = kwargs

        def process(self, _image):
            raise NotImplementedError("This is a test stub. Use mocking instead.")

        def close(self):
            pass

    pose_module = types.ModuleType("mediapipe.solutions.pose")
    pose_module.PoseLandmark = _PoseLandmark
    pose_module.Pose = _Pose

    solutions_module = types.ModuleType("mediapipe.solutions")
    solutions_module.pose = pose_module

    mediapipe_module = types.ModuleType("mediapipe")
    mediapipe_module.solutions = solutions_module

    python_pose_module = types.ModuleType("mediapipe.python.solutions.pose")
    python_pose_module.PoseLandmark = _PoseLandmark
    python_pose_module.Pose = _Pose

    python_solutions_module = types.ModuleType("mediapipe.python.solutions")
    python_solutions_module.pose = python_pose_module

    python_module = types.ModuleType("mediapipe.python")
    python_module.solutions = python_solutions_module

    sys.modules.update(
        {
            "mediapipe": mediapipe_module,
            "mediapipe.solutions": solutions_module,
            "mediapipe.solutions.pose": pose_module,
            "mediapipe.python": python_module,
            "mediapipe.python.solutions": python_solutions_module,
            "mediapipe.python.solutions.pose": python_pose_module,
        }
    )


_ensure_mediapipe_stub()

from src.analysis.dwell_time_detector import DwellTimeDetector
from src.definitions import Angle, MovementState
from src.movement_analyzer import MovementAnalyzer
from src.pose.definitions import BodyPart


@pytest.fixture
def frame_shape() -> tuple[int, int, int]:
    return (1080, 1920, 3)


@pytest.fixture
def sample_landmarks() -> np.ndarray:
    landmarks = np.zeros((33, 4), dtype=float)
    landmarks[:, 3] = 0.9

    landmarks[BodyPart.LEFT_EAR] = [0.4, 0.25, 0.0, 0.9]
    landmarks[BodyPart.RIGHT_EAR] = [0.6, 0.25, 0.0, 0.9]
    landmarks[BodyPart.LEFT_SHOULDER] = [0.4, 0.3, 0.0, 0.9]
    landmarks[BodyPart.RIGHT_SHOULDER] = [0.6, 0.3, 0.0, 0.9]
    landmarks[BodyPart.LEFT_HIP] = [0.4, 0.6, 0.0, 0.9]
    landmarks[BodyPart.RIGHT_HIP] = [0.6, 0.6, 0.0, 0.9]
    landmarks[BodyPart.NOSE] = [0.5, 0.2, 0.0, 0.9]

    return landmarks


@pytest.fixture
def movement_analyzer() -> MovementAnalyzer:
    return MovementAnalyzer(confidence_threshold=0.7)


@pytest.fixture
def dwell_time_detector() -> DwellTimeDetector:
    return DwellTimeDetector(
        stay_threshold_sec=3.0,
        confidence_threshold=0.5,
        spike_threshold=0.5,
        spike_window_sec=0.2,
        stability_threshold_px=10.0,
        stability_window_sec=0.2,
        grace_period_sec=0.1,
        confirmation_ratio=0.5,
        use_normalization=True,
        normalization_base="torso",
    )


class TestMovementAnalyzer:
    def test_body_tilt_is_upright(self, movement_analyzer, sample_landmarks):
        results = movement_analyzer.analyze(sample_landmarks)
        body_tilt = results[Angle.BODY_TILT]

        assert body_tilt["state"] is MovementState.UPRIGHT
        assert body_tilt["angle"] == pytest.approx(180.0, abs=1e-5)

    def test_neck_trunk_angle_is_straight(self, movement_analyzer, sample_landmarks):
        results = movement_analyzer.analyze(sample_landmarks)
        neck_trunk = results[Angle.NECK_TRUNK_ANGLE]

        assert neck_trunk["state"] is MovementState.STRAIGHT
        assert neck_trunk["angle"] == pytest.approx(180.0, abs=1e-5)

    def test_forward_tilt_detection(self, movement_analyzer, sample_landmarks):
        forward_landmarks = sample_landmarks.copy()
        forward_landmarks[BodyPart.LEFT_SHOULDER] = [0.4, 0.4, 0.2, 0.9]
        forward_landmarks[BodyPart.RIGHT_SHOULDER] = [0.6, 0.4, 0.2, 0.9]
        forward_landmarks[BodyPart.NOSE] = [0.5, 0.3, 0.3, 0.9]

        results = movement_analyzer.analyze(forward_landmarks)
        body_tilt = results[Angle.BODY_TILT]

        assert body_tilt["state"] is MovementState.FORWARD_TILT
        assert body_tilt["angle"] == pytest.approx(135.0, rel=1e-2)

    def test_lateral_right_tilt_detection(self, movement_analyzer, sample_landmarks):
        tilted_landmarks = sample_landmarks.copy()
        tilted_landmarks[BodyPart.LEFT_SHOULDER, 1] = 0.2
        tilted_landmarks[BodyPart.RIGHT_SHOULDER, 1] = 0.6
        tilted_landmarks[BodyPart.LEFT_HIP, 1] = 0.5
        tilted_landmarks[BodyPart.RIGHT_HIP, 1] = 0.9

        results = movement_analyzer.analyze(tilted_landmarks)
        lateral = results[Angle.LATERAL_TILT]

        assert lateral["state"] is MovementState.RIGHT_TILT
        assert lateral["angle"] == pytest.approx(63.4, rel=1e-2)


class TestDwellTimeDetector:
    def test_extract_hip_center(self, dwell_time_detector, sample_landmarks, frame_shape):
        hip_x, hip_y, confidence = dwell_time_detector.extract_hip_center(sample_landmarks, frame_shape)

        assert hip_x == pytest.approx(960.0)
        assert hip_y == pytest.approx(648.0)
        assert confidence == pytest.approx(0.9)

    def test_compute_person_scale_torso(self, dwell_time_detector, sample_landmarks, frame_shape):
        scale = dwell_time_detector._compute_person_scale(sample_landmarks, frame_shape)
        assert scale == pytest.approx(324.0)

    def test_compute_person_scale_requires_visibility(self, dwell_time_detector, sample_landmarks, frame_shape):
        landmarks = sample_landmarks.copy()
        landmarks[BodyPart.LEFT_SHOULDER, 3] = 0.2

        scale = dwell_time_detector._compute_person_scale(landmarks, frame_shape)
        assert scale is None

    def test_long_stay_alert_triggered(self, dwell_time_detector, sample_landmarks, frame_shape):
        alert = None
        for t in range(4):
            alert = dwell_time_detector.update(sample_landmarks, frame_shape, float(t))

        assert alert is not None
        assert alert.startswith("[!] Long Stay Detected")

        status = dwell_time_detector.get_current_status()
        assert status["is_long_stay"]
        assert status["stay_duration"] >= 3.0

    def test_movement_confirmation_resets_state(self, dwell_time_detector, sample_landmarks, frame_shape):
        dwell_time_detector.update(sample_landmarks, frame_shape, 0.0)

        moved = sample_landmarks.copy()
        moved[BodyPart.LEFT_HIP] = [0.05, 0.6, 0.0, 0.9]
        moved[BodyPart.RIGHT_HIP] = [0.25, 0.6, 0.0, 0.9]

        dwell_time_detector.update(moved, frame_shape, 0.05)
        alert = dwell_time_detector.update(moved, frame_shape, 0.2)

        assert alert == "[!] Movement Confirmed"

        status = dwell_time_detector.get_current_status()
        assert status["state"] == "STAYING"
        assert status["stay_duration"] == pytest.approx(0.0)
