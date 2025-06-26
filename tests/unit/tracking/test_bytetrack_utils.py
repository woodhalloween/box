import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import cv2

from src.tracking.bytetrack_utils import (
    get_system_info,
    initialize_perf_log,
    process_frame_for_tracking,
    draw_tracking_info,
    SKELETON_ULTRA,
    KEYPOINT_COLOR,
    SKELETON_COLOR
)

# テスト用のダミー入力ファイル名とモデルパス
DUMMY_INPUT_FILE = "test_video.mp4"
DUMMY_MODEL_PATH = "test_model.pt"


def test_initialize_perf_log_enabled(tmp_path):
    """Test case when performance log is enabled"""

    perf_log_file = initialize_perf_log(
        enable_perf_log=True,
        input_file=DUMMY_INPUT_FILE,
        model_path=DUMMY_MODEL_PATH,
        log_type="test_log",
    )

    assert perf_log_file is not None

    log_file_path = Path(perf_log_file)
    assert log_file_path.exists()
    assert log_file_path.name.startswith(f"log_{Path(DUMMY_INPUT_FILE).stem}_test_log_{Path(DUMMY_MODEL_PATH).stem}_")
    assert log_file_path.parent.name == "logs"
    assert log_file_path.parent.parent.name == "output"

    # ✅ Now read after flush/close
    with open(perf_log_file) as f:
        lines = f.readlines()
        assert lines  # ensure file is not empty
        assert "# System Information" in lines[0]
        assert "Frame" in lines[-1]

    # 🧹 Cleanup
    if Path("output/logs").exists() and Path(perf_log_file).is_relative_to(Path("output/logs")):
        os.remove(perf_log_file)
        if not os.listdir("output/logs"):
            os.rmdir("output/logs")
        if not os.listdir("output"):
            os.rmdir("output")


def test_initialize_perf_log_disabled(tmp_path):
    """パフォーマンスログが無効な場合のテスト"""
    # tmp_path はこのテストでは直接使われないが、pytestの慣習として引数に含める
    perf_log_file = initialize_perf_log(
        enable_perf_log=False,
        input_file=DUMMY_INPUT_FILE,
        model_path=DUMMY_MODEL_PATH,
        log_type="test_log",
    )
    assert perf_log_file is None


def test_initialize_perf_log_file_creation_error(mocker):
    # Patch get_system_info to avoid psutil usage (prevent collateral errors)
    mocker.patch(
        "src.tracking.bytetrack_utils.get_system_info",
        return_value={
            "os": "Linux",
            "os_version": "5.10",
            "python_version": "3.10.0",
            "cpu": "FakeCPU",
            "cpu_cores": 4,
            "cpu_threads": 8,
            "ram_total": 16,
        },
    )

    # Patch ONLY the built-in open inside the specific context
    with patch("builtins.open", side_effect=OSError("Test error: Cannot open file")):
        result = initialize_perf_log(
            enable_perf_log=True,
            input_file="test_video.mp4",
            model_path="test_model.pt",
            log_type="test_log",
        )

    assert result is None


def test_initialize_perf_log_with_long_stay_column(mocker):
    """Should insert 'Stay_Check_Time_ms' column when log_type is 'long_stay'."""

    # Patch system info
    mocker.patch(
        "src.tracking.bytetrack_utils.get_system_info",
        return_value={
            "os": "Linux",
            "os_version": "5.10",
            "python_version": "3.10.0",
            "cpu": "FakeCPU",
            "cpu_cores": 4,
            "cpu_threads": 8,
            "ram_total": 16,
        },
    )

    perf_log_file = initialize_perf_log(
        enable_perf_log=True,
        input_file=DUMMY_INPUT_FILE,
        model_path=DUMMY_MODEL_PATH,
        log_type="long_stay",
    )

    assert perf_log_file is not None
    log_file_path = Path(perf_log_file)
    assert log_file_path.exists()

    with open(log_file_path, encoding="utf-8") as f:
        lines = f.readlines()
        header = [col.strip() for col in lines[-1].strip().split(",")]
        assert "Stay_Check_Time_ms" in header
        assert header.index("Stay_Check_Time_ms") == 4  # Confirm correct insertion point

    # 🧹 Cleanup
    try:
        os.remove(log_file_path)
        logs_dir = log_file_path.parent
        if logs_dir.exists() and not any(logs_dir.iterdir()):
            logs_dir.rmdir()
        output_dir = logs_dir.parent
        if output_dir.exists() and not any(output_dir.iterdir()):
            output_dir.rmdir()
    except Exception:
        pass


# `get_system_info` は外部ライブラリに依存しているため、簡単な呼び出しテストのみ
def test_get_system_info():
    """get_system_info関数の基本的な動作テスト"""
    info = get_system_info()
    assert "os" in info
    assert "python_version" in info
    assert "cpu_cores" in info
    assert "ram_total" in info


def test_process_frame_for_tracking_no_detections(mocker):
    """process_frame_for_tracking: 検出がない場合のテスト"""
    mock_model = MagicMock()
    mock_model.predict.return_value = [MagicMock(boxes=[], keypoints=None)] # No detections

    mock_tracker = MagicMock()
    mock_tracker.update.return_value = np.array([])

    frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)

    tracks, det_ms, track_ms, num_det, num_track, keypoints = process_frame_for_tracking(
        frame_rgb, mock_model, mock_tracker, conf=0.5, enable_pose=False
    )

    assert len(tracks) == 0
    assert num_det == 0
    assert num_track == 0
    assert keypoints is None
    # Use mocker.ANY for the frame_rgb argument
    mock_tracker.update.assert_called_once_with(mocker.ANY, mocker.ANY)


def test_draw_tracking_info_with_pose(mocker):
    """draw_tracking_info: 姿勢推定が有効でキーポイントがある場合のテスト"""
    mock_rectangle = mocker.patch("cv2.rectangle")
    mock_line = mocker.patch("cv2.line")
    mock_circle = mocker.patch("cv2.circle")
    mock_put_text = mocker.patch("cv2.putText")

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    tracks = np.array([[10, 20, 60, 80, 1, 0.9, 0]]) # x1, y1, x2, y2, track_id, conf, cls_id

    # Dummy keypoints for one person (17 keypoints, each with x,y coords)
    mock_keypoints_obj = MagicMock()
    mock_keypoints_obj.xy.cpu.return_value.numpy.return_value = np.array([
        [[15, 25], [20, 20], [30, 20], [25, 15], [35, 15], # 0-4
        [20, 40], [30, 40], [15, 50], [35, 50], [10, 60], # 5-9
        [40, 60], [25, 70], [35, 70], [20, 80], [40, 80], # 10-14
        [15, 90], [45, 90]] # 15-16
    ], dtype=np.float32)

    draw_tracking_info(
        frame,
        tracks,
        keypoints=mock_keypoints_obj,
        enable_pose=True,
        show_duration=False,
        stay_info=None
    )

    # Assert that rectangle and text for ID are drawn
    mock_rectangle.assert_called()
    mock_put_text.assert_called()

    # Assert that lines for skeleton are drawn
    assert mock_line.call_count > 0
    for call in mock_line.call_args_list:
        # Check if the color is SKELETON_COLOR
        assert call.args[3] == SKELETON_COLOR

    # Assert that circles for keypoints are drawn
    assert mock_circle.call_count > 0
    for call in mock_circle.call_args_list:
        # Check if the color is KEYPOINT_COLOR
        assert call.args[3] == KEYPOINT_COLOR
