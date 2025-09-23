import csv
import io

import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark

from src.analysis.dwell_time_detector import DwellTimeDetector, DwellTimeInfo, DwellTimeState
from src.analysis.posture_monitor import PostureMonitor, PostureSnapshot
from src.definitions import Angle, MovementState
from src.io.csv_writer import setup_csv_writer, write_results_to_csv


def read_csv_rows(buffer: io.StringIO) -> list[dict[str, str]]:
    buffer.seek(0)
    reader = csv.DictReader(io.StringIO(buffer.getvalue()))
    return list(reader)


def test_setup_csv_writer_creates_expected_headers():
    buffer = io.StringIO()

    writer = setup_csv_writer(buffer)

    expected_prefix = [
        "timestamp",
        "frame_number",
        "debug_analysis_results_empty",
        "right_elbow_angle",
        "right_elbow_state",
    ]
    assert writer.fieldnames[: len(expected_prefix)] == expected_prefix

    last_landmark = list(PoseLandmark)[-1]
    expected_suffix = [
        f"{last_landmark.name}_x",
        f"{last_landmark.name}_y",
        f"{last_landmark.name}_z",
        f"{last_landmark.name}_visibility",
    ]
    assert writer.fieldnames[-4:] == expected_suffix

    header_row = buffer.getvalue().splitlines()[0].split(",")
    assert header_row == writer.fieldnames


def test_write_results_to_csv_populates_measurements_with_landmarks():
    buffer = io.StringIO()
    writer = setup_csv_writer(buffer)

    posture_monitor = PostureMonitor()
    snapshot = PostureSnapshot(
        timestamp=1.0,
        frame_number=5,
        analysis_results={},
        is_forward_leaning=True,
        forward_lean_score=0.75,
    )
    posture_monitor.posture_history.append(snapshot)

    dwell_detector = DwellTimeDetector(stay_threshold_sec=5.0)
    dwell_detector.stay_info = DwellTimeInfo(
        last_hip_pos=(123.4, 567.8),
        last_update_time=1.0,
        stay_start_time=0.0,
        stay_duration=6.5,
        notified=True,
        confidence_score=0.9,
    )
    dwell_detector.state = DwellTimeState.STAYING

    analysis_results = {
        Angle.RIGHT_ELBOW: {"angle": 45.0, "state": MovementState.FLEXION},
        Angle.LEFT_ELBOW: {"angle": 30.0, "state": MovementState.EXTENSION},
        Angle.RIGHT_KNEE: {
            "angle": 60.0,
            "state": MovementState.FLEXION,
            "p1_confidence": 0.1,
            "p2_confidence": 0.2,
            "p3_confidence": 0.3,
        },
        Angle.LEFT_KNEE: {
            "angle": 65.0,
            "state": MovementState.EXTENSION,
            "p1_confidence": 0.4,
            "p2_confidence": 0.5,
            "p3_confidence": 0.6,
        },
        Angle.BODY_TILT: {"angle": 120.0, "state": MovementState.FORWARD_TILT},
        Angle.NECK_TRUNK_ANGLE: {"angle": 110.0, "state": MovementState.FORWARD_TILT},
    }

    class DummyHeadShakeDetector:
        def get_status(self):
            return {
                "horizontal_state": MovementState.HEAD_LEFT_TURN.name,
                "vertical_state": MovementState.HEAD_STATIC.name,
            }

    head_shake_detector = DummyHeadShakeDetector()
    head_shake_alerts = ["Horizontal", "Vertical"]

    landmark_count = len(PoseLandmark)
    landmarks = np.zeros((landmark_count, 4), dtype=float)
    for landmark in PoseLandmark:
        idx = landmark.value
        landmarks[idx] = np.array([float(idx), float(idx + 1), float(idx + 2), 0.5], dtype=float)

    write_results_to_csv(
        writer,
        timestamp=1.23,
        frame_number=7,
        analysis_results=analysis_results,
        posture_monitor=posture_monitor,
        dwell_time_detector=dwell_detector,
        dwell_alert="Stay alert",
        head_shake_detector=head_shake_detector,
        head_shake_alerts=head_shake_alerts,
        landmarks=landmarks,
    )

    rows = read_csv_rows(buffer)
    assert len(rows) == 1
    row = rows[0]

    assert row["timestamp"] == "1.23"
    assert row["frame_number"] == "7"
    assert row["right_elbow_angle"] == "45.0"
    assert row["right_elbow_state"] == MovementState.FLEXION.name
    assert row["left_shoulder_state"] == MovementState.STATIC.name
    assert row["right_knee_hip_confidence"] == "0.1"
    assert row["left_knee_ankle_confidence"] == "0.6"
    assert row["is_forward_leaning"] == "True"
    assert row["forward_lean_score"] == "0.75"
    assert row["hip_center_x"] == "123.4"
    assert row["hip_center_y"] == "567.8"
    assert row["stay_duration"] == "6.5"
    assert row["hip_confidence"] == "0.9"
    assert row["is_long_stay"] == "True"
    assert row["long_stay_alert"] == "Stay alert"
    assert row["hip_detector_state"] == DwellTimeState.STAYING.name
    assert row["head_shake_horizontal_detected"] == "True"
    assert row["head_shake_vertical_detected"] == "False"
    assert row["head_shake_alerts"] == "Horizontal; Vertical"
    assert row["NOSE_x"] == "0.0"
    assert row["RIGHT_ANKLE_z"] == str(float(PoseLandmark.RIGHT_ANKLE.value + 2))


def test_write_results_to_csv_handles_missing_optional_inputs():
    buffer = io.StringIO()
    writer = setup_csv_writer(buffer)

    posture_monitor = PostureMonitor()
    dwell_detector = DwellTimeDetector()

    write_results_to_csv(
        writer,
        timestamp=0.0,
        frame_number=0,
        analysis_results={},
        posture_monitor=posture_monitor,
        dwell_time_detector=dwell_detector,
        dwell_alert=None,
        head_shake_detector=None,
        head_shake_alerts=None,
        landmarks=None,
    )

    row = read_csv_rows(buffer)[0]

    assert row["right_elbow_angle"] == "0"
    assert row["right_elbow_state"] == MovementState.STATIC.name
    assert row["hip_center_x"] == "0"
    assert row["hip_center_y"] == "0"
    assert row["stay_duration"] == "0.0"
    assert row["head_shake_horizontal_detected"] == "False"
    assert row["head_shake_alerts"] == ""
    assert row["NOSE_x"] == "0.0"
