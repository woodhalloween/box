# All comments are written in English as requested.

import io
from types import SimpleNamespace

import numpy as np

from src.io_utils import setup_csv_writer, setup_video_writer, write_results_to_csv


def test_setup_video_writer_uses_capture_properties_and_calls_cv2(monkeypatch):
    """setup_video_writer must read FPS/width/height and pass them to cv2.VideoWriter."""

    # Fake VideoCapture with deterministic properties
    class FakeCap:
        def __init__(self):
            self.props = {
                # Use floats to mirror OpenCV return types
                5: 29.97,  # cv2.CAP_PROP_FPS
                3: 640.0,  # cv2.CAP_PROP_FRAME_WIDTH
                4: 360.0,  # cv2.CAP_PROP_FRAME_HEIGHT
            }

        def get(self, prop):
            return self.props[prop]

    # Capture the arguments with which cv2.VideoWriter is constructed
    created = {}

    def fake_vw(path, fourcc, fps, size):
        created["path"] = path
        created["fps"] = fps
        created["size"] = size
        created["fourcc"] = fourcc
        return "FAKE_WRITER"

    # Monkeypatch inside the module under test
    monkeypatch.setattr("src.io_utils.cv2.VideoWriter", fake_vw)
    monkeypatch.setattr("src.io_utils.cv2.CAP_PROP_FPS", 5)
    monkeypatch.setattr("src.io_utils.cv2.CAP_PROP_FRAME_WIDTH", 3)
    monkeypatch.setattr("src.io_utils.cv2.CAP_PROP_FRAME_HEIGHT", 4)
    monkeypatch.setattr("src.io_utils.cv2.VideoWriter_fourcc", lambda *a: 1234)

    cap = FakeCap()
    writer = setup_video_writer(cap, output_path="output/test.mp4")
    assert writer == "FAKE_WRITER"
    assert created["path"] == "output/test.mp4"
    # setup_video_writer casts fps to int and the size tuple to ints
    assert created["fps"] == int(29.97)
    assert created["size"] == (640, 360)
    assert created["fourcc"] == 1234


def test_setup_csv_writer_writes_header_and_landmark_columns(monkeypatch):
    """setup_csv_writer must include core fields and per-landmark columns, and write the header row."""
    # Ensure PoseLandmark enum is reachable
    from mediapipe.python.solutions.pose import PoseLandmark

    buf = io.StringIO()
    _ = setup_csv_writer(buf)

    # The first line in the buffer should be the header
    header_line = buf.getvalue().splitlines()[0]
    # Spot-check a few mandatory columns that appear in the function
    assert "timestamp" in header_line
    assert "frame_number" in header_line
    assert "is_long_stay" in header_line
    assert "hip_detector_state" in header_line
    # And at least one landmark column like "NOSE_x"
    assert f"{PoseLandmark.NOSE.name}_x" in header_line


def _make_landmarks_with_identifiable_values():
    """
    Build a (33,4) landmarks array with identifiable values so we can assert they appear in CSV.
    For index i, store:
      x = i + 0.1, y = i + 0.2, z = i + 0.3, visibility = i + 0.4
    """
    n = 33
    lm = np.zeros((n, 4), dtype=float)
    for i in range(n):
        lm[i, 0] = i + 0.1
        lm[i, 1] = i + 0.2
        lm[i, 2] = i + 0.3
        lm[i, 3] = i + 0.4
    return lm


def _make_minimal_deps():
    """
    Minimal stubs required by write_results_to_csv.
    Provides all keys that the function indexes into.
    """
    posture_monitor = SimpleNamespace(
        get_status=lambda: {
            "hip_detector_state": "",  # not strictly required, but harmless
            "posture_alerts": [],
            "torso_tilt": 0.0,
            "forward_ratio": 0.25,  # required
            "avg_score": 0.10,  # required
        },
        posture_history=[],  # falsy -> branch skipped
    )

    dwell_time_detector = SimpleNamespace(
        get_current_status=lambda: {
            "hip_position": (123.0, 456.0),  # required (tuple or None)
            "stay_duration": 7.5,  # required
            "confidence": 0.9,  # required
            "is_long_stay": False,  # required
            "state": "TRACKING",  # required
        }
    )

    head_shake_detector = None  # ok; function guards with `if head_shake_detector`
    return posture_monitor, dwell_time_detector, head_shake_detector


def test_write_results_to_csv_with_landmarks(monkeypatch):
    """write_results_to_csv should write angles/states plus per-landmark values when landmarks are provided."""
    # Build a minimal but valid analysis_results: include a couple of Angle keys with angle + state.
    from src.definitions import Angle, MovementState

    # Prepare two angles to prove both angle/state columns are populated
    analysis_results = {
        Angle.RIGHT_ELBOW: {"angle": 42.0, "state": MovementState.STATIC},
        Angle.LEFT_SHOULDER: {"angle": 77.7, "state": MovementState.HEAD_LEFT_TURN},
    }

    posture_monitor, dwell_time_detector, head_shake_detector = _make_minimal_deps()
    dwell_alert = "DWELL_ALERT"
    head_shake_alerts = ["H_SHAKE"]

    buf = io.StringIO()
    writer = setup_csv_writer(buf)

    lm = _make_landmarks_with_identifiable_values()
    write_results_to_csv(
        csv_writer=writer,
        timestamp=12.34,
        frame_number=5,
        analysis_results=analysis_results,
        posture_monitor=posture_monitor,
        dwell_time_detector=dwell_time_detector,
        dwell_alert=dwell_alert,
        head_shake_detector=head_shake_detector,
        head_shake_alerts=head_shake_alerts,
        landmarks=lm,
    )

    # Grab the last CSV row and assert landmark values are present (e.g., NOSE_x == 0 + 0.1)
    last_line = buf.getvalue().splitlines()[-1]
    assert "12.34" in last_line
    # Spot check: the NOSE is index 0; we set x=0.1, y=0.2, z=0.3, visibility=0.4
    assert "0.1" in last_line and "0.2" in last_line and "0.3" in last_line and "0.4" in last_line
    # Our angle fields should also be present in the row
    assert "42.0" in last_line or "42" in last_line
    assert "77.7" in last_line or "77.70" in last_line


def test_write_results_to_csv_without_landmarks(monkeypatch):
    """write_results_to_csv should zero-fill landmark columns when landmarks is None."""
    from mediapipe.python.solutions.pose import PoseLandmark

    posture_monitor, dwell_time_detector, head_shake_detector = _make_minimal_deps()
    buf = io.StringIO()
    writer = setup_csv_writer(buf)

    # Call with landmarks=None to trigger the else-branch (zero-filling)
    write_results_to_csv(
        csv_writer=writer,
        timestamp=0.0,
        frame_number=0,
        analysis_results={},  # empty is fine; branch we care about is landmarks None
        posture_monitor=posture_monitor,
        dwell_time_detector=dwell_time_detector,
        dwell_alert=None,
        head_shake_detector=head_shake_detector,
        head_shake_alerts=None,
        landmarks=None,
    )

    header, last_line = buf.getvalue().splitlines()[0], buf.getvalue().splitlines()[-1]
    # For a few landmark columns, expect "0.0" since we zero-filled
    for suffix in ("_x", "_y", "_z", "_visibility"):
        col = f"{PoseLandmark.NOSE.name}{suffix}"
        # Ensure the column exists in header and its value in the row includes "0.0"
        assert col in header
        assert "0.0" in last_line  # at least some zeroes present across landmark columns


def test_write_results_to_csv_uses_latest_posture_history_entry():
    """
    When posture_monitor.posture_history is non-empty, the function must read the latest entry
    and write 'is_forward_leaning' and 'forward_lean_score' from that object into the CSV.
    """
    import io
    from types import SimpleNamespace

    # Build a posture_monitor stub:
    # - get_status returns required keys used later in row.update
    # - posture_history contains objects with the needed attributes; we put two entries and
    #   expect the function to read the *last* one.
    history_first = SimpleNamespace(is_forward_leaning=False, forward_lean_score=0.12)
    history_latest = SimpleNamespace(is_forward_leaning=True, forward_lean_score=0.73)  # should be used
    posture_monitor = SimpleNamespace(
        get_status=lambda: {
            "forward_ratio": 0.33,
            "avg_score": 0.11,
            "hip_detector_state": "",  # harmless extras
            "posture_alerts": [],
            "torso_tilt": 0.0,
        },
        posture_history=[history_first, history_latest],
    )

    # dwell_time_detector must provide all indexed keys
    dwell_time_detector = SimpleNamespace(
        get_current_status=lambda: {
            "hip_position": (10.0, 20.0),
            "stay_duration": 5.0,
            "confidence": 0.9,
            "is_long_stay": False,
            "state": "TRACKING",
        }
    )

    # head_shake_detector can be None; function guards it
    head_shake_detector = None

    # Minimal analysis_results is fine; the branch we care about is posture_history handling
    analysis_results = {}

    buf = io.StringIO()
    writer = setup_csv_writer(buf)

    write_results_to_csv(
        csv_writer=writer,
        timestamp=1.23,
        frame_number=7,
        analysis_results=analysis_results,
        posture_monitor=posture_monitor,
        dwell_time_detector=dwell_time_detector,
        dwell_alert=None,
        head_shake_detector=head_shake_detector,
        head_shake_alerts=None,
        landmarks=None,  # keep simple; zero-fill path
    )

    last_line = buf.getvalue().splitlines()[-1]
    # Assert values sourced from the *latest* posture_history entry
    assert "True" in last_line  # is_forward_leaning from history_latest
    assert "0.73" in last_line  # forward_lean_score from history_latest
    # Also ensure the get_status-derived fields made it in (sanity checks)
    assert "0.33" in last_line  # forward_lean_ratio
    assert "0.11" in last_line  # avg_forward_lean_score
