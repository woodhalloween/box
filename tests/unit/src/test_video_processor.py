import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np

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

        # Assert output functions were called
        mock_write_csv.assert_called_once()
        mock_draw_landmarks.assert_called_once_with(dummy_frame, "dummy_landmarks")
        mock_draw_analysis_results.assert_called_once()
        mock_draw_detection_info.assert_called_once()

        # Assert video writer was called for the processed frame
        mock_video_writer_instance = mock_setup_video_writer.return_value
        self.assertEqual(mock_video_writer_instance.write.call_count, 2)

        # --- Frame 2 (no landmarks) ---
        # No more calls to analyzers or drawing functions that need landmarks
        self.assertEqual(mock_movement_analyzer.return_value.analyze.call_count, 1)
        self.assertEqual(mock_write_csv.call_count, 1)
