# test_detect_long_stay_core.py
# 🧪 Pytest test suite for Daisy's detect_long_stay_core.py
import csv
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.detect_long_stay_core import run_long_stay_detection


# === Fixtures ===
@pytest.fixture
def dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def dummy_video(tmp_path, dummy_frame):
    path = tmp_path / "dummy.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
    for _ in range(60):
        out.write(dummy_frame)
    out.release()
    return str(path)


# === Dummy Components ===
def dummy_draw_tracking_info(frame, tracks, keypoints=None, enable_pose=False, show_duration=False, stay_info=None):
    return frame


def dummy_load_yolo_model(path, device):
    return "dummy_model"


def dummy_initialize_bytetrack():
    return "dummy_tracker"


def dummy_initialize_perf_log(enable, input_file, model_path, log_type="long_stay"):
    return "/dev/null"


def dummy_process_frame_for_tracking(frame, model, tracker, conf, enable_pose):
    tracks = [np.array([100, 100, 200, 200, 1, 0.9, 0])]
    keypoints = MagicMock()
    keypoints.xy.cpu.return_value.numpy.return_value = np.array([[[0, 0]] * 17])  # Dummy keypoints
    return tracks, 1.0, 1.0, 1, 1, keypoints


def dummy_update_stay_times(tracks, stay_info, current_time, move_threshold, stay_threshold):
    stay_info = {1: {"stay_duration": stay_threshold + 1.0}}
    return stay_info, ["Long stay detected"], 0.5


def dummy_process_frame(frame, model, tracker, conf, enable_pose):
    return [], 0.1, 0.1, 1, 1, None


# === Mocked Classes ===
class InterruptingCapture:
    def __init__(self, path):
        self.counter = 0

    def isOpened(self):  # noqa: N802
        return True

    def read(self):
        if self.counter == 0:
            self.counter += 1
            return True, np.zeros((480, 640, 3), dtype=np.uint8)
        raise KeyboardInterrupt

    def get(self, prop_id):
        return {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 10.0,
            cv2.CAP_PROP_FRAME_COUNT: 2,
        }.get(prop_id, 0)

    def release(self):
        pass


class MockVideoCapture:
    def __init__(self, path):
        self.frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        self.index = 0
        self.opened = True

    def isOpened(self):  # noqa: N802
        return self.opened

    def read(self):
        if self.index < len(self.frames):
            frame = self.frames[self.index]
            self.index += 1
            return True, frame
        return False, None

    def get(self, prop_id):
        return {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 10.0,
            cv2.CAP_PROP_FRAME_COUNT: len(self.frames),
        }.get(prop_id, 0)

    def release(self):
        self.opened = False


# === Tests ===
def test_run_long_stay_detection(dummy_video, tmp_path):
    output_file = tmp_path / "output.mp4"
    with patch("cv2.VideoWriter") as mock_video_writer:
        run_long_stay_detection(
            input_path=dummy_video,
            output_path=str(output_file),
            model_path="dummy_model.pt",
            device="cpu",
            stay_threshold=2.0,
            move_threshold=10.0,
            conf=0.3,
            enable_perf_log=False,
            enable_video_display=False,
            enable_pose=False,  # Add enable_pose
            draw_fn=dummy_draw_tracking_info,
            load_model_fn=dummy_load_yolo_model,
            tracker_init_fn=dummy_initialize_bytetrack,
            perf_log_fn=dummy_initialize_perf_log,
            process_frame_fn=dummy_process_frame_for_tracking,
            update_stay_fn=dummy_update_stay_times,
        )
        mock_video_writer.assert_called_once()  # Ensure VideoWriter is called


def test_run_long_stay_detection_video_writer_fallback(dummy_video, tmp_path):
    output_file = tmp_path / "output_fallback.mp4"
    mock_writer_instance_fail = MagicMock()
    mock_writer_instance_fail.isOpened.return_value = False

    mock_writer_instance_success = MagicMock()
    mock_writer_instance_success.isOpened.return_value = True

    with patch(
        "cv2.VideoWriter", side_effect=[mock_writer_instance_fail, mock_writer_instance_success]
    ) as mock_video_writer_class:
        run_long_stay_detection(
            input_path=dummy_video,
            output_path=str(output_file),
            model_path="dummy_model.pt",
            device="cpu",
            stay_threshold=2.0,
            move_threshold=10.0,
            conf=0.3,
            enable_perf_log=False,
            enable_video_display=False,
            enable_pose=False,
            draw_fn=dummy_draw_tracking_info,
            load_model_fn=dummy_load_yolo_model,
            tracker_init_fn=dummy_initialize_bytetrack,
            perf_log_fn=dummy_initialize_perf_log,
            process_frame_fn=dummy_process_frame_for_tracking,
            update_stay_fn=dummy_update_stay_times,
        )
        # Assert that VideoWriter was called twice (once for avc1, once for mp4v)
        assert mock_video_writer_class.call_count == 2
        # Assert that the second call used 'mp4v'
        assert mock_video_writer_class.call_args_list[1].args[1] == cv2.VideoWriter_fourcc(*"mp4v")


def test_input_file_not_found():
    with pytest.raises(FileNotFoundError) as excinfo:
        run_long_stay_detection(
            input_path="nonexistent.mp4",
            output_path=None,
            model_path="dummy_model.pt",
            device="cpu",
            stay_threshold=2.0,
            move_threshold=10.0,
            conf=0.3,
            enable_perf_log=False,
            enable_video_display=False,
            enable_pose=False,  # Add enable_pose
            draw_fn=None,
            load_model_fn=None,
            tracker_init_fn=None,
            perf_log_fn=None,
            process_frame_fn=None,
            update_stay_fn=None,
        )
    assert "Input video file not found" in str(excinfo.value)


def test_video_file_cannot_be_opened(tmp_path):
    dummy_path = tmp_path / "dummy_invalid.mp4"
    dummy_path.write_text("not a real video")
    with patch("cv2.VideoCapture") as mock_cv:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv.return_value = mock_cap
        with pytest.raises(OSError) as excinfo:
            run_long_stay_detection(
                input_path=str(dummy_path),
                output_path=None,
                model_path="dummy_model.pt",
                device="cpu",
                stay_threshold=2.0,
                move_threshold=10.0,
                conf=0.3,
                enable_perf_log=False,
                enable_video_display=False,
                enable_pose=False,  # Add enable_pose
                draw_fn=None,
                load_model_fn=lambda p, d: None,
                tracker_init_fn=lambda: None,
                perf_log_fn=lambda *args, **kwargs: "/dev/null",
                process_frame_fn=lambda *args, **kwargs: ([], 0, 0, 0, 0, None),
                update_stay_fn=lambda *args, **kwargs: ({}, [], 0),
            )
        assert "Cannot open video file" in str(excinfo.value)


def test_perf_log_video_metadata_written(tmp_path, dummy_video):
    dummy_log = tmp_path / "perf_log.csv"

    def dummy_perf_log_fn(enable, input_file, model_path, log_type="long_stay"):
        return str(dummy_log)

    with patch("cv2.imshow"), patch("cv2.waitKey", return_value=27):
        run_long_stay_detection(
            input_path=dummy_video,
            output_path=None,
            model_path="dummy_model.pt",
            device="cpu",
            stay_threshold=2.0,
            move_threshold=10.0,
            conf=0.3,
            enable_perf_log=True,
            enable_video_display=True,
            enable_pose=False,  # Add enable_pose
            draw_fn=lambda frame, tracks, keypoints, enable_pose, show_duration, stay_info: frame,  # Update lambda
            load_model_fn=lambda p, d: "model",
            tracker_init_fn=lambda: "tracker",
            perf_log_fn=dummy_perf_log_fn,
            process_frame_fn=lambda f, m, t, c, e: ([], 0.1, 0.1, 0, 0, None),  # Update lambda
            update_stay_fn=lambda t, s, c, m, st: ({}, [], 0.05),
        )
    with open(dummy_log, newline="") as f:
        rows = list(csv.reader(f))
        assert ["# Video Properties", "640x480", "10.0fps"] in rows


@patch("cv2.VideoCapture", new=MockVideoCapture)
def test_perf_log_frame_level_metrics_written(tmp_path):
    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH: 320,
        cv2.CAP_PROP_FRAME_HEIGHT: 240,
        cv2.CAP_PROP_FPS: 1.0,
        cv2.CAP_PROP_FRAME_COUNT: 2,
    }[prop]
    mock_cap.read.side_effect = [(True, dummy_frame), (True, dummy_frame), (False, None)]
    log_path = tmp_path / "perf_log.csv"

    def dummy_perf_log_fn(enable, input_file, model_path, log_type="long_stay"):
        return str(log_path)

    with (
        patch("cv2.VideoCapture", return_value=mock_cap),
        patch("cv2.imshow"),
        patch("cv2.waitKey", side_effect=[-1, -1, 27]),
        patch("os.path.exists", return_value=True),
    ):
        run_long_stay_detection(
            input_path="any.mp4",
            output_path=None,
            model_path="model.pt",
            device="cpu",
            stay_threshold=2.0,
            move_threshold=10.0,
            conf=0.3,
            enable_perf_log=True,
            enable_video_display=True,
            enable_pose=False,  # Add enable_pose
            draw_fn=dummy_draw_tracking_info,
            load_model_fn=dummy_load_yolo_model,
            tracker_init_fn=dummy_initialize_bytetrack,
            perf_log_fn=dummy_perf_log_fn,
            process_frame_fn=lambda f, m, t, c, e: ([], 0.1, 0.1, 0, 0, None),  # Update lambda
            update_stay_fn=dummy_update_stay_times,
        )
    assert log_path.exists()
    with open(log_path, newline="") as f:
        rows = list(csv.reader(f))
        data_rows = [r for r in rows if r and not r[0].strip().startswith("#")]
        assert any(r[0].strip() == "1" for r in data_rows)
        assert any(r[0].strip() == "2" for r in data_rows)


@pytest.mark.usefixtures("tmp_path")
def test_keyboard_interrupt_path(tmp_path):
    dummy_input = tmp_path / "dummy_interrupt.mp4"
    dummy_input.write_text("fake")
    with (
        patch("cv2.VideoCapture", new=InterruptingCapture),
        patch("os.path.exists", return_value=True),
        patch("builtins.print") as mock_print,
    ):
        run_long_stay_detection(
            input_path=str(dummy_input),
            output_path=None,
            model_path="dummy_model.pt",
            device="cpu",
            stay_threshold=1.0,
            move_threshold=1.0,
            conf=0.3,
            enable_perf_log=False,
            enable_video_display=False,
            enable_pose=False,  # Add enable_pose
            draw_fn=lambda f, tracks, keypoints, enable_pose, show_duration, stay_info: f,  # Update lambda
            load_model_fn=lambda p, d: "model",
            tracker_init_fn=lambda: "tracker",
            perf_log_fn=lambda *a, **k: "/dev/null",
            process_frame_fn=lambda f, m, t, c, e: ([], 0.1, 0.1, 0, 0, None),  # Update lambda
            update_stay_fn=lambda t, s, c, m, st: ({}, [], 0.05),
        )
    printed = [call.args[0] for call in mock_print.call_args_list]
    assert any("処理がユーザーによって中断されました。" in line for line in printed)
