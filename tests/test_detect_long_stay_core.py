# test_detect_long_stay_core.py
# 🧪 Pytest test suite for Daisy's detect_long_stay_core.py
import csv
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.detect_long_stay_core import run_long_stay_detection


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
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return 640
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return 480
        if prop_id == cv2.CAP_PROP_FPS:
            return 10.0
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        return 0

    def release(self):
        self.opened = False


@pytest.fixture
def dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def dummy_video(tmp_path, dummy_frame):
    # Create a dummy video file for testing
    path = tmp_path / "dummy.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
    for _ in range(60):
        out.write(dummy_frame)
    out.release()
    return str(path)


# def dummy_draw_tracking_info(frame, *args, **kwargs):
#     return frame
def dummy_draw_tracking_info(frame, tracks, show_duration=False, stay_info=None):
    return frame


def dummy_load_yolo_model(path, device):
    return "dummy_model"


def dummy_initialize_bytetrack():
    return "dummy_tracker"


def dummy_initialize_perf_log(enable, input_file, model_path, log_type="long_stay"):
    return "/dev/null"


def dummy_process_frame_for_tracking(frame, model, tracker):
    tracks = [{"id": 1, "bbox": [100, 100, 200, 200]}]
    return tracks, 1.0, 1.0, 1, 1, None


def dummy_update_stay_times(tracks, stay_info, current_time, move_threshold, stay_threshold):
    stay_info = {1: {"stay_duration": stay_threshold + 1.0}}
    return stay_info, ["Long stay detected"], 0.5


def dummy_process_frame(frame, model, tracker):
    return [], 0.1, 0.1, 1, 1, None


def test_run_long_stay_detection(dummy_video, tmp_path):
    output_file = tmp_path / "output.mp4"

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
        draw_fn=dummy_draw_tracking_info,
        load_model_fn=dummy_load_yolo_model,
        tracker_init_fn=dummy_initialize_bytetrack,
        perf_log_fn=dummy_initialize_perf_log,
        process_frame_fn=dummy_process_frame_for_tracking,
        update_stay_fn=dummy_update_stay_times,
    )

    assert output_file.exists()


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

        with pytest.raises(IOError) as excinfo:
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

    with patch("cv2.imshow"), patch("cv2.waitKey", return_value=27):  # Prevent window rendering
        run_long_stay_detection(
            input_path=dummy_video,
            output_path=None,
            model_path="dummy_model.pt",
            device="cpu",
            stay_threshold=2.0,
            move_threshold=10.0,
            conf=0.3,
            enable_perf_log=True,
            enable_video_display=True,  # Force avg_fps to be computed
            draw_fn=lambda frame, *args, **kwargs: frame,
            load_model_fn=lambda p, d: "model",
            tracker_init_fn=lambda: "tracker",
            perf_log_fn=dummy_perf_log_fn,
            process_frame_fn=lambda f, m, t: ([], 0.1, 0.1, 0, 0, None),
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

    # 🔧 Generate log_path *before* patching os.path.exists
    log_path = tmp_path / "perf_log.csv"

    def dummy_perf_log_fn(enable, input_file, model_path, log_type="long_stay"):
        return str(log_path)

    with (
        patch("cv2.VideoCapture", return_value=mock_cap),
        patch("cv2.imshow"),
        patch("cv2.waitKey", side_effect=[-1, -1, 27]),
        patch("os.path.exists", return_value=True),
    ):  # only affects inside run_long_stay_detection
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
            draw_fn=dummy_draw_tracking_info,
            load_model_fn=dummy_load_yolo_model,
            tracker_init_fn=dummy_initialize_bytetrack,
            perf_log_fn=dummy_perf_log_fn,
            process_frame_fn=dummy_process_frame,
            update_stay_fn=dummy_update_stay_times,
        )

    # ✅ Now this log_path is real and refers to the real file
    assert log_path.exists()
    with open(log_path, newline="") as f:
        rows = list(csv.reader(f))
        print(f"rows: {rows}")
        data_rows = [r for r in rows if r and not r[0].strip().startswith("#")]
        print(f"data_rows: {data_rows}")
        assert any(r[0].strip() == "1" for r in data_rows), f"Rows: {rows}"
        assert any(r[0].strip() == "2" for r in data_rows), f"Rows: {rows}"
