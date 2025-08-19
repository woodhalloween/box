from unittest.mock import MagicMock, patch

from src.tracking.bytetrack_utils import initialize_bytetrack


@patch("src.tracking.bytetrack_utils.ByteTrack")  # Patch the class used in the function
class TestInitializeByteTrack:
    def test_default_frame_rate(self, mock_bytetrack):
        """Should initialize ByteTrack with default frame_rate=30"""
        tracker_instance = MagicMock()
        mock_bytetrack.return_value = tracker_instance

        result = initialize_bytetrack()

        mock_bytetrack.assert_called_once_with(track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30)
        assert result == tracker_instance

    def test_custom_frame_rate(self, mock_bytetrack):
        """Should initialize ByteTrack with custom frame_rate"""
        tracker_instance = MagicMock()
        mock_bytetrack.return_value = tracker_instance

        result = initialize_bytetrack(frame_rate=60)

        mock_bytetrack.assert_called_once_with(track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=60)
        assert result == tracker_instance
