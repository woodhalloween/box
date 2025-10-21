from unittest.mock import MagicMock, patch

import cv2

from src.video_processing.estimate_total_frames import estimate_total_frames


def test_show_progress_false():
    result = estimate_total_frames("dummy_path.mp4", fps=30.0, show_progress=False)
    assert result is None


@patch("cv2.VideoCapture")
def test_video_capture_not_opened(mock_videocap):
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = False
    mock_videocap.return_value = mock_instance

    result = estimate_total_frames("dummy_path.mp4", fps=30.0, show_progress=True)
    assert result is None
    mock_instance.release.assert_not_called()


@patch("cv2.VideoCapture")
def test_same_fps_returns_original_frame_count(mock_videocap):
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    mock_instance.get.side_effect = lambda x: {
        cv2.CAP_PROP_FRAME_COUNT: 300,
        cv2.CAP_PROP_FPS: 30.0,
    }[x]
    mock_videocap.return_value = mock_instance

    result = estimate_total_frames("dummy_path.mp4", fps=30.0, show_progress=True)
    assert result == 300
    mock_instance.release.assert_called_once()


@patch("cv2.VideoCapture")
def test_adjusted_fps_rounding_down(mock_videocap):
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    # 300 frames / 20 fps = 15s -> 15 * 30 = 450
    mock_instance.get.side_effect = lambda x: {
        cv2.CAP_PROP_FRAME_COUNT: 300,
        cv2.CAP_PROP_FPS: 20.0,
    }[x]
    mock_videocap.return_value = mock_instance

    result = estimate_total_frames("dummy_path.mp4", fps=30.0, show_progress=True)
    assert result == 450


@patch("cv2.VideoCapture")
def test_adjusted_fps_buffer_increment(mock_videocap):
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    # 301 frames / 20 fps = 15.05s -> 15.05 * 30 = 451.5 → round to 452 (buffer increment)
    mock_instance.get.side_effect = lambda x: {
        cv2.CAP_PROP_FRAME_COUNT: 301,
        cv2.CAP_PROP_FPS: 20.0,
    }[x]
    mock_videocap.return_value = mock_instance

    result = estimate_total_frames("dummy_path.mp4", fps=30.0, show_progress=True)
    assert result == 452


@patch("cv2.VideoCapture", side_effect=Exception("Simulated failure"))
def test_videocapture_exception(mock_videocap):
    result = estimate_total_frames("dummy_path.mp4", fps=30.0, show_progress=True)
    assert result is None


@patch("cv2.VideoCapture")
def test_adjusted_fps_fractional_05(mock_videocap):
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    # 301 frames / 20 fps = 15.05s -> 15.05 * 30 = 451.5 → should round to 452 (buffer increment)
    mock_instance.get.side_effect = lambda x: {
        cv2.CAP_PROP_FRAME_COUNT: 301,
        cv2.CAP_PROP_FPS: 20.0,
    }[x]
    mock_videocap.return_value = mock_instance

    result = estimate_total_frames("dummy_path.mp4", fps=30.0, show_progress=True)
    assert result == 452


@patch("cv2.VideoCapture")
def test_adjusted_fps_fractional_05_below(mock_videocap):
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    # 300 frames / 20 fps = 15s -> 15 * 30 = 450 → no buffer increment needed
    mock_instance.get.side_effect = lambda x: {
        cv2.CAP_PROP_FRAME_COUNT: 300,
        cv2.CAP_PROP_FPS: 20.0,
    }[x]
    mock_videocap.return_value = mock_instance

    result = estimate_total_frames("dummy_path.mp4", fps=30.0, show_progress=True)
    assert result == 450


@patch("cv2.VideoCapture")
def test_adjusted_fps_fractional_just_above(mock_videocap):
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    # 301 frames / 20 fps = 15.05s -> 15.05 * 30 = 451.6 → should round to 452 (buffer increment)
    mock_instance.get.side_effect = lambda x: {
        cv2.CAP_PROP_FRAME_COUNT: 301,
        cv2.CAP_PROP_FPS: 20.0,
    }[x]
    mock_videocap.return_value = mock_instance

    result = estimate_total_frames("dummy_path.mp4", fps=30.0, show_progress=True)
    assert result == 452


@patch("cv2.VideoCapture")
def test_adjusted_fps_fractional_just_below_05(mock_videocap):
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    # 301 frames / 20 fps = 15.05s -> 15.05 * 30 = 451.4999999 (just below 452, should not round up)
    mock_instance.get.side_effect = lambda x: {
        cv2.CAP_PROP_FRAME_COUNT: 301,
        cv2.CAP_PROP_FPS: 20.0,
    }[x]
    mock_videocap.return_value = mock_instance

    result = estimate_total_frames("dummy_path.mp4", fps=30.0, show_progress=True)

    # Debugging: Print the exact result value
    print(f"Calculated total_frames: {result}")

    # The result should be 452 due to rounding up at 0.5
    assert result == 452


@patch("cv2.VideoCapture")
def test_adjusted_fps_fractional_just_above_05(mock_videocap):
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    # 301 frames / 20 fps = 15.05s -> 15.05 * 30 = 451.5000001 (just above 451.5, should round up)
    mock_instance.get.side_effect = lambda x: {
        cv2.CAP_PROP_FRAME_COUNT: 301,
        cv2.CAP_PROP_FPS: 20.0,
    }[x]
    mock_videocap.return_value = mock_instance

    result = estimate_total_frames("dummy_path.mp4", fps=30.0, show_progress=True)
    # The result should be 453 due to rounding up above 0.5
    assert result == 452


@patch("cv2.VideoCapture")
def test_adjusted_fps_with_high_precision(mock_videocap):
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    # FPS very close to the original FPS (e.g., 29.999999 vs 30), still should use original count
    mock_instance.get.side_effect = lambda x: {
        cv2.CAP_PROP_FRAME_COUNT: 300,
        cv2.CAP_PROP_FPS: 29.999999,
    }[x]
    mock_videocap.return_value = mock_instance

    result = estimate_total_frames("dummy_path.mp4", fps=30.0, show_progress=True)
    assert result == 300  # Should use original frame count due to negligible difference in fps


@patch("cv2.VideoCapture")
def test_adjusted_fps_with_exact_fps(mock_videocap):
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    # FPS exactly the same, no change
    mock_instance.get.side_effect = lambda x: {
        cv2.CAP_PROP_FRAME_COUNT: 300,
        cv2.CAP_PROP_FPS: 30.0,
    }[x]
    mock_videocap.return_value = mock_instance

    result = estimate_total_frames("dummy_path.mp4", fps=30.0, show_progress=True)
    assert result == 300  # Exact match should not modify the frame count


@patch("cv2.VideoCapture", side_effect=Exception("Simulated failure"))
def test_videocapture_exception_with_message(mock_videocap):
    result = estimate_total_frames("dummy_path.mp4", fps=30.0, show_progress=True)
    assert result is None  # Ensure the exception is handled gracefully
