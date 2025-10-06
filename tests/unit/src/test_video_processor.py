import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, mock_open, patch

import numpy as np
import pytest

from src.video_processor import VideoProcessor


class TestVideoProcessor(unittest.TestCase):
    @patch("src.video_processor.cv2.VideoCapture")
    @patch("src.video_processor.setup_video_writer")
    @patch("src.video_processor.open", new_callable=mock_open)
    @patch("src.video_processor.setup_csv_writer")
    @patch("src.video_processor.PoseEstimator")
    @patch("src.video_processor.MovementAnalyzer")
    @patch("src.video_processor.UserClassifier")
    @patch("src.video_processor.DwellTimeDetector")
    @patch("src.video_processor.PostureMonitor")
    @patch("src.video_processor.HeadShakeDetector")
    def test_initialization_and_context_management(
        self,
        mock_head_shake_detector,
        mock_posture_monitor,
        mock_dwell_time_detector,
        mock_user_classifier,
        mock_movement_analyzer,
        mock_pose_estimator,
        mock_setup_csv_writer,
        mock_file_open,
        mock_setup_video_writer,
        mock_video_capture,
    ):
        """
        Tests if VideoProcessor initializes correctly and manages resources
        using the with statement.
        """
        # Arrange
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap_instance
        video_path = "dummy.mp4"

        # Act
        with VideoProcessor(
            video_path=video_path,
            output_csv_path="dummy.csv",
            output_video_path="dummy_out.mp4",
            disable_japanese=False,
            stay_threshold_sec=10.0,
            spike_threshold=2.0,
            stability_threshold_px=100.0,
            grace_period_sec=2.5,
            pm_monitoring_duration_sec=60.0,
            pm_alert_threshold_ratio=0.7,
            uc_threshold_deg=90.0,
            uc_moving_window_seconds=5,
        ):
            # Assert inside __enter__
            mock_video_capture.assert_called_once_with(video_path)
            mock_file_open.assert_called_once_with("dummy.csv", "w", newline="", encoding="utf-8")
            mock_setup_video_writer.assert_called_once()
            mock_setup_csv_writer.assert_called_once()
            # Check if all analyzers are instantiated
            mock_pose_estimator.assert_called_once()
            mock_movement_analyzer.assert_called_once()
            mock_user_classifier.assert_called_once_with(threshold_deg=90.0, moving_window_seconds=5)
            mock_dwell_time_detector.assert_called_once_with(
                stay_threshold_sec=10.0,
                spike_threshold=2.0,
                stability_threshold_px=100.0,
                grace_period_sec=2.5,
            )
            mock_posture_monitor.assert_called_once_with(monitoring_duration=60.0, alert_threshold=0.7)
            mock_head_shake_detector.assert_called_once()

        # Assert inside __exit__
        mock_cap_instance.release.assert_called_once()
        mock_setup_video_writer.return_value.release.assert_called_once()
        mock_file_open.return_value.close.assert_called_once()

    @patch("src.video_processor.cv2.VideoCapture")
    @patch("src.video_processor.setup_video_writer")
    @patch("src.video_processor.open", new_callable=mock_open)
    @patch("src.video_processor.setup_csv_writer")
    @patch("src.video_processor.PoseEstimator")
    @patch("src.video_processor.MovementAnalyzer")
    @patch("src.video_processor.UserClassifier")
    @patch("src.video_processor.DwellTimeDetector")
    @patch("src.video_processor.PostureMonitor")
    @patch("src.video_processor.HeadShakeDetector")
    @patch("src.video_processor.HandRaiseDetector")
    @patch("src.video_processor.write_results_to_csv")
    @patch("src.video_processor.draw_landmarks")
    @patch("src.video_processor.draw_analysis_results")
    @patch("src.video_processor.draw_detection_info")
    @patch("src.video_processor.cv2.imshow")
    @patch("src.video_processor.cv2.waitKey")
    def test_run_loop_and_frame_processing(
        self,
        mock_wait_key,
        mock_imshow,
        mock_draw_detection_info,
        mock_draw_analysis_results,
        mock_draw_landmarks,
        mock_write_csv,
        mock_hand_raise_detector,
        mock_head_shake_detector,
        mock_posture_monitor,
        mock_dwell_time_detector,
        mock_user_classifier,
        mock_movement_analyzer,
        mock_pose_estimator,
        mock_setup_csv_writer,
        mock_file_open,
        mock_setup_video_writer,
        mock_video_capture,
    ):
        """
        Tests the main run loop and ensures frame processing calls the correct
        sub-modules and functions.
        """
        # Arrange
        # Mock video capture to return 2 frames then stop
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        # Simulate 2 frames then end of video
        mock_cap_instance.read.side_effect = [(True, dummy_frame), (True, dummy_frame), (False, None)]
        mock_cap_instance.get.return_value = 1000.0  # timestamp
        mock_video_capture.return_value = mock_cap_instance

        # Mock waitKey to stop the loop after 3 calls (2 frames + 1 for exit)
        mock_wait_key.side_effect = [0, 0, ord("q")]

        # Mock PoseEstimator to return dummy landmarks for the first frame, None for the second
        mock_pose_estimator_instance = mock_pose_estimator.return_value
        mock_pose_estimator_instance.estimate.side_effect = ["dummy_landmarks", None]

        # Mock drawing functions to just return the frame
        mock_draw_landmarks.return_value = dummy_frame
        mock_draw_analysis_results.return_value = dummy_frame
        mock_draw_detection_info.return_value = dummy_frame

        # Mock HandRaiseDetector
        mock_hand_raise_detector_instance = mock_hand_raise_detector.return_value
        mock_hand_raise_detector_instance.detect.return_value = {"left_hand_raised": True, "right_hand_raised": False}

        # Act
        with VideoProcessor(
            video_path="dummy.mp4",
            output_csv_path="dummy.csv",
            output_video_path="dummy_out.mp4",
            disable_japanese=False,
            stay_threshold_sec=10.0,
            spike_threshold=2.0,
            stability_threshold_px=100.0,
            grace_period_sec=2.5,
            pm_monitoring_duration_sec=60.0,
            pm_alert_threshold_ratio=0.7,
            uc_threshold_deg=90.0,
            uc_moving_window_seconds=5,
        ) as processor:
            processor.run()

        # Assert
        self.assertEqual(mock_cap_instance.read.call_count, 3)

        # Assert imshow was called for each frame
        self.assertEqual(mock_imshow.call_count, 2)

        # --- Frame 1 (with landmarks) ---
        # Assert analysis modules were called
        mock_pose_estimator_instance.estimate.assert_any_call(dummy_frame)
        mock_movement_analyzer.return_value.analyze.assert_called_once_with("dummy_landmarks")
        mock_user_classifier.return_value.update.assert_called_once()
        mock_dwell_time_detector.return_value.update.assert_called_once()
        mock_posture_monitor.return_value.update.assert_called_once()
        mock_head_shake_detector.return_value.update.assert_called_once()
        mock_head_shake_detector.return_value.check_alerts.assert_called_once()
        mock_hand_raise_detector_instance.detect.assert_called_once_with("dummy_landmarks")

        # Assert output functions were called
        mock_write_csv.assert_called_once()
        mock_draw_landmarks.assert_called_once_with(dummy_frame, "dummy_landmarks")
        mock_draw_analysis_results.assert_called_once()
        mock_draw_analysis_results.assert_called_with(
            dummy_frame,
            ANY,
            {"left_hand_raised": True, "right_hand_raised": False},
            "dummy_landmarks",
            disable_japanese=False,
        )
        mock_draw_detection_info.assert_called_once()

        # Assert video writer was called for the processed frame
        mock_video_writer_instance = mock_setup_video_writer.return_value
        self.assertEqual(mock_video_writer_instance.write.call_count, 2)

        # --- Frame 2 (no landmarks) ---
        # No more calls to analyzers or drawing functions that need landmarks
        self.assertEqual(mock_movement_analyzer.return_value.analyze.call_count, 1)
        self.assertEqual(mock_write_csv.call_count, 1)


# -------------------------
# Additional tests for video_processor.py
# -------------------------


def _make_video_processor():
    """Helper: construct VideoProcessor with safe default numeric parameters."""
    return VideoProcessor(
        video_path="videos/sample.mp4",
        output_csv_path=None,
        output_video_path=None,
        disable_japanese=True,
        stay_threshold_sec=5.0,
        spike_threshold=2.0,
        stability_threshold_px=10.0,
        grace_period_sec=1.0,
        pm_monitoring_duration_sec=5.0,
        pm_alert_threshold_ratio=0.5,
        uc_threshold_deg=15.0,
        uc_moving_window_seconds=2.0,
    )


def test___enter___raises_when_video_cannot_open(monkeypatch):
    """__enter__ must raise OSError when cv2.VideoCapture cannot open the video file."""
    vp = _make_video_processor()

    class FakeCap:
        def __init__(self, path):
            self.path = path

        def isOpened(self):  # noqa: N802
            return False

        def release(self):
            pass

    # Monkeypatch the VideoCapture constructor used inside the module
    monkeypatch.setattr("src.video_processor.cv2.VideoCapture", lambda p: FakeCap(p))

    with pytest.raises(OSError), vp:
        # Entering the context triggers __enter__ which should raise
        pass


def test__setup_paths_sets_defaults_when_none(tmp_path):
    """When output paths are None, _setup_paths should populate default output paths that include the file stem."""
    # Use a fake video path under a tmp location to exercise Path(p).stem logic
    video_path = str(tmp_path / "somefolder" / "myvideo.mp4")
    vp = _make_video_processor()
    vp.video_path = video_path
    vp.output_csv_path = None
    vp.output_video_path = None

    vp._setup_paths()

    # Both should be set and include the video stem
    assert vp.output_csv_path is not None
    assert "myvideo_analysis_" in Path(vp.output_csv_path).name
    assert vp.output_video_path is not None
    assert "myvideo_output_" in Path(vp.output_video_path).name

    # They should be placed under the 'output' directory per implementation
    assert Path(vp.output_csv_path).parts[0] == "output"
    assert Path(vp.output_video_path).parts[0] == "output"


def test__process_frame_prints_notification_when_conditions_met(monkeypatch, capsys):
    """
    _process_frame prints a notification when:
      - user_classifier.get_current_alert() is not None (user_is_classified)
      - dwell_time_detector.get_current_status()['is_long_stay'] is True (is_long_stay)
      - dwell_time_detector.stay_info exists and stay_info.notified is False
    This test monkeypatches drawing / I/O helpers into no-ops so we only assert the printed text.
    """
    vp = _make_video_processor()

    # Minimal dummy frame
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    # Monkeypatch drawing and CSV functions to be no-ops so the test focuses on the print branch
    monkeypatch.setattr("src.video_processor.draw_landmarks", lambda f, landmarks: f)
    monkeypatch.setattr(
        "src.video_processor.draw_analysis_results",
        lambda f, a, hand_statuses, landmarks, disable_japanese: f,
    )
    monkeypatch.setattr("src.video_processor.draw_detection_info", lambda f, *a, **k: f)
    monkeypatch.setattr("src.video_processor.write_results_to_csv", lambda **kw: None)

    # Attach simple components expected by _process_frame
    # Pose estimator returns a landmarks array (non-None) so processing continues
    vp.pose_estimator = SimpleNamespace(estimate=lambda fr: np.zeros((33, 4)))
    # Analyzer returns an analysis_results dict
    vp.analyzer = SimpleNamespace(analyze=lambda lm: {})
    # user_classifier: get_current_alert returning non-None -> classified
    vp.user_classifier = SimpleNamespace(update=lambda *a, **k: None, get_current_alert=lambda: "SOME_ALERT")

    # Dwell detector: get_current_status says is_long_stay True, and stay_info.notified == False
    stay_info = SimpleNamespace(notified=False)
    vp.dwell_time_detector = SimpleNamespace(
        update=lambda lm, shape, ts: None,
        get_current_status=lambda: {"is_long_stay": True},
        stay_info=stay_info,
    )

    # Other modules: no-op implementations
    vp.posture_monitor = SimpleNamespace(update=lambda *a, **k: [])
    vp.head_shake_detector = SimpleNamespace(update=lambda *a, **k: {}, check_alerts=lambda ts: [])
    vp.hand_raise_detector = SimpleNamespace(detect=lambda lm: {"left_hand_raised": False, "right_hand_raised": False})
    vp.video_writer = SimpleNamespace(write=lambda fr: None)
    vp.csv_writer = None  # write_results_to_csv was monkeypatched to no-op

    # Call the method and capture stdout
    ts = 12.34
    _ = vp._process_frame(frame, frame_count=0, timestamp=ts)
    captured = capsys.readouterr()

    expected = f"[{ts:.1f}s] 通知: 指定エリアのお客様対応をお願いします。"
    assert expected in captured.out


def test___enter___fps_fallback_on_invalid(monkeypatch):
    """__enter__ should fallback to fps=30.0 when CAP_PROP_FPS is invalid/non-castable."""
    vp = _make_video_processor()

    class FakeCap:
        def __init__(self, path):
            self._opened = True

        def isOpened(self):  # noqa: N802
            return True

        def get(self, prop):
            # Return a string that cannot be cast to float → triggers except and fallback branch
            return "not-a-number"

        def release(self):
            pass

    # Patch dependencies invoked in __enter__
    monkeypatch.setattr("src.video_processor.cv2.VideoCapture", lambda p: FakeCap(p))
    monkeypatch.setattr("src.video_processor.setup_video_writer", lambda *a, **k: SimpleNamespace(release=lambda: None))
    monkeypatch.setattr("src.video_processor.setup_csv_writer", lambda f: object())
    # open is a built-in, patch there (module does not expose open)
    monkeypatch.setattr("builtins.open", mock_open())

    with vp:
        # After fallback, fps should be 30.0
        assert getattr(vp, "fps", None) == 30.0


def test__process_frame_initializes_writer_when_no_landmarks(monkeypatch):
    """If landmarks is None and output_video_path is set, _process_frame must lazy-init writer with frame shape."""
    vp = _make_video_processor()
    vp.output_video_path = "out.mp4"
    vp.fps = 24.0

    # Pose returns None → triggers landmarks None path
    vp.pose_estimator = SimpleNamespace(estimate=lambda fr: None)

    # Spy for setup_video_writer
    writes = {"called": False, "args": None, "write_count": 0}

    class DummyWriter:
        def write(self, frame):
            writes["write_count"] += 1

        def release(self):
            pass

    def fake_setup_video_writer(shape_hw, out_path, fps):
        writes["called"] = True
        writes["args"] = (shape_hw, out_path, fps)
        return DummyWriter()

    monkeypatch.setattr("src.video_processor.setup_video_writer", fake_setup_video_writer)

    frame = np.zeros((100, 200, 3), dtype=np.uint8)  # (h, w)
    _ = vp._process_frame(frame, frame_count=0, timestamp=0.0)

    assert writes["called"] is True
    assert writes["args"] == ((100, 200), "out.mp4", 24.0)
    # In the None-landmarks path, frame is written once
    assert writes["write_count"] == 1


def test__process_frame_initializes_and_writes_when_landmarks_present(monkeypatch):
    """
    When landmarks exist and writer is None,
    _process_frame should create writer with actual frame shape and write once.
    """
    vp = _make_video_processor()
    vp.output_video_path = "out2.mp4"
    vp.fps = 30.0

    # Minimal components to get through the main path
    landmarks = np.zeros((33, 4))
    vp.pose_estimator = SimpleNamespace(estimate=lambda fr: landmarks)
    vp.analyzer = SimpleNamespace(analyze=lambda lm: {})
    vp.user_classifier = SimpleNamespace(update=lambda *a, **k: [], get_current_alert=lambda: None)
    vp.dwell_time_detector = SimpleNamespace(
        update=lambda *a, **k: None, get_current_status=lambda: {"is_long_stay": False}, stay_info=None
    )
    vp.posture_monitor = SimpleNamespace(update=lambda *a, **k: [])
    vp.head_shake_detector = SimpleNamespace(update=lambda *a, **k: {}, check_alerts=lambda *a, **k: [])
    vp.hand_raise_detector = SimpleNamespace(detect=lambda lm: {"left_hand_raised": False, "right_hand_raised": False})
    monkeypatch.setattr("src.video_processor.draw_landmarks", lambda f, lm: f)
    monkeypatch.setattr(
        "src.video_processor.draw_analysis_results",
        lambda f, r, hand_statuses, lm, disable_japanese: f,
    )
    monkeypatch.setattr("src.video_processor.draw_detection_info", lambda f, *a, **k: f)
    monkeypatch.setattr("src.video_processor.write_results_to_csv", lambda **kw: None)

    calls = {"setup": None, "writes": 0}

    class DummyWriter2:
        def write(self, frame):
            calls["writes"] += 1

        def release(self):
            pass

    def fake_setup_video_writer2(shape_hw, out_path, fps):
        calls["setup"] = (shape_hw, out_path, fps)
        return DummyWriter2()

    monkeypatch.setattr("src.video_processor.setup_video_writer", fake_setup_video_writer2)

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    _ = vp._process_frame(frame, frame_count=1, timestamp=0.5)

    assert calls["setup"] == ((120, 160), "out2.mp4", 30.0)
    assert calls["writes"] == 1
