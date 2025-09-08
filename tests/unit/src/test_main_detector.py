import sys
import unittest
from unittest.mock import MagicMock, patch

from src import main_detector


def setup_mock_config():
    """Helper function to create a mock config that returns fallback values."""
    mock_config = MagicMock()
    # Make getfloat return the fallback value provided in the call
    mock_config.getfloat.side_effect = lambda key, fallback: fallback
    mock_config.getint.side_effect = lambda key, fallback: fallback
    return mock_config


@patch("src.main_detector.config", setup_mock_config())
class TestMainDetector(unittest.TestCase):
    @patch("src.main_detector.process_video")
    def test_main_with_all_args(self, mock_process_video):
        """
        Tests if main function parses all arguments correctly and calls process_video.
        """
        # Arrange
        test_args = [
            "main_detector.py",
            "--video",
            "test.mp4",
            "--output-csv",
            "out.csv",
            "--output-video",
            "out.mp4",
            "--disable-japanese",
            "--stay-threshold",
            "5.0",
            "--spike-threshold",
            "2.0",
            "--stability-threshold",
            "100.0",
            "--grace-period",
            "2.5",
            "--pm-duration",
            "30.0",
            "--pm-threshold",
            "0.6",
            "--uc-threshold",
            "80.0",
            "--uc-window",
            "3",
        ]

        # Act
        with patch.object(sys, "argv", test_args):
            main_detector.main()

        # Assert
        # NOTE: The advanced args are parsed but not passed to process_video yet.
        # This test reflects the current implementation.
        mock_process_video.assert_called_once_with(
            video_path="test.mp4",
            output_csv_path="out.csv",
            output_video_path="out.mp4",
            disable_japanese=True,
            stay_threshold_sec=5.0,
            spike_threshold=2.0,
            stability_threshold_px=100.0,
            grace_period_sec=2.5,
            pm_monitoring_duration_sec=30.0,
            pm_alert_threshold_ratio=0.6,
            uc_threshold_deg=80.0,
            uc_moving_window_seconds=3,
        )

    @patch("src.main_detector.process_video")
    def test_main_with_defaults(self, mock_process_video):
        """
        Tests if main function uses default values correctly.
        """
        # Arrange
        test_args = [
            "main_detector.py",
            "--video",
            "test.mp4",
        ]

        # Act
        with patch.object(sys, "argv", test_args):
            main_detector.main()

        # Assert
        mock_process_video.assert_called_once_with(
            video_path="test.mp4",
            output_csv_path=None,
            output_video_path=None,
            disable_japanese=False,
            stay_threshold_sec=10.0,  # Default value
            spike_threshold=1.5,  # Default value
            stability_threshold_px=50.0,  # Default value
            grace_period_sec=1.5,  # Default value
            pm_monitoring_duration_sec=60.0,
            pm_alert_threshold_ratio=0.7,
            uc_threshold_deg=90.0,
            uc_moving_window_seconds=5,
        )

    @patch("src.main_detector.VideoProcessor")
    def test_process_video_calls_videoprocessor(self, mock_video_processor):
        """
        Tests if the process_video function correctly initializes and runs VideoProcessor.
        """
        # Arrange
        mock_processor_instance = mock_video_processor.return_value.__enter__.return_value

        # Act
        main_detector.process_video(
            video_path="test.mp4",
            output_csv_path="out.csv",
            output_video_path="out.mp4",
            disable_japanese=True,
            stay_threshold_sec=5.0,
            spike_threshold=2.0,
            stability_threshold_px=100.0,
            grace_period_sec=2.5,
            pm_monitoring_duration_sec=60.0,
            pm_alert_threshold_ratio=0.7,
            uc_threshold_deg=90.0,
            uc_moving_window_seconds=5,
        )

        # Assert
        mock_video_processor.assert_called_once_with(
            video_path="test.mp4",
            output_csv_path="out.csv",
            output_video_path="out.mp4",
            disable_japanese=True,
            stay_threshold_sec=5.0,
            spike_threshold=2.0,
            stability_threshold_px=100.0,
            grace_period_sec=2.5,
            pm_monitoring_duration_sec=60.0,
            pm_alert_threshold_ratio=0.7,
            uc_threshold_deg=90.0,
            uc_moving_window_seconds=5,
        )
        mock_processor_instance.run.assert_called_once()
