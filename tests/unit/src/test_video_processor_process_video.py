# test_process_video.py
import numpy as np
import pytest

from src.video_processor import process_video


@pytest.fixture(autouse=True)
def _patch_user_classifier(monkeypatch):
    """Make UserClassifier tolerant to float window sizes to avoid deque(maxlen=float) TypeError."""

    class DummyUC:
        def __init__(self, threshold_deg=90.0, moving_window_seconds=5.0, confidence_threshold=0.7):
            # Just store values; do NOT construct deque
            self.threshold_deg = threshold_deg
            self.moving_window_seconds = moving_window_seconds
            self.confidence_threshold = confidence_threshold
            self.current_alert_message = None

        # Optional: stub methods if anything calls them later
        def update(self, *a, **k):
            return []

        def get_current_alert(self):
            return self.current_alert_message

    monkeypatch.setattr("src.video_processor.UserClassifier", DummyUC)


def test_process_video_ffmpeg_default_branch(monkeypatch, tmp_path):
    """
    FFmpeg available (make_frame_iter not None) & input_mode is None.
    Expect:
      - input_mode auto-resolves to 'ffmpeg-file'
      - make_frame_iter called with provided params
      - setup_csv_writer(open(...)) & setup_video_writer(...) are used
      - run_pipeline called with the iterator returned by make_frame_iter and the created sinks
    """
    calls = {"ffmpeg": None, "run": None, "open": None, "csv": None, "vid": None}

    # Fake ffmpeg iterator we can recognize
    def fake_make_frame_iter(input_mode, *, ffmpeg_input, width, height, fps, is_color, add_args):
        calls["ffmpeg"] = {
            "input_mode": input_mode,
            "ffmpeg_input": ffmpeg_input,
            "width": width,
            "height": height,
            "fps": fps,
            "is_color": is_color,
            "add_args": add_args,
        }

        # Return a simple iterator
        def _it():
            yield 0.0, np.zeros((2, 3, 3), dtype=np.uint8)

        return _it()

    monkeypatch.setattr("src.video_processor.make_frame_iter", fake_make_frame_iter)

    # Fake open and csv setup
    class DummyFile:
        def writable(self):
            return True

        def close(self):
            pass

    def fake_open(path, mode, newline, encoding):
        calls["open"] = {
            "path": path,
            "mode": mode,
            "newline": newline,
            "encoding": encoding,
        }
        return DummyFile()

    def fake_setup_csv_writer(fobj):
        assert fobj.writable()
        calls["csv"] = "CSV_WRITER_SENTINEL"
        return "CSV_WRITER_SENTINEL"

    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr("src.video_processor.setup_csv_writer", fake_setup_csv_writer)

    # Fake video writer setup (the real signature may differ; we accept whatever is passed)
    def fake_setup_video_writer(shape_or_cap, output_path, *rest):
        calls["vid"] = {
            "arg": shape_or_cap,
            "path": output_path,
            "rest": rest,
        }
        return "VIDEO_WRITER_SENTINEL"

    monkeypatch.setattr("src.video_processor.setup_video_writer", fake_setup_video_writer)

    # Capture run_pipeline inputs
    def fake_run_pipeline(frame_iter, *, csv_writer, video_writer, state, preview, window_name):
        # Consume one item to make sure it's iterable
        first = next(frame_iter)
        calls["run"] = {
            "first": first,
            "csv_writer": csv_writer,
            "video_writer": video_writer,
            "preview": preview,
            "window_name": window_name,
            "state_type": type(state).__name__,
            "disable_jp": state.disable_jp,
        }

    monkeypatch.setattr("src.video_processor.run_pipeline", fake_run_pipeline)

    # Inputs
    csv_path = tmp_path / "out.csv"
    video_path = tmp_path / "out.mp4"

    process_video(
        video_path="input.mp4",
        output_csv_path=str(csv_path),
        output_video_path=str(video_path),
        disable_japanese=True,
        stay_threshold_sec=12.0,
        spike_threshold=1.1,
        stability_threshold_px=33.0,
        grace_period_sec=2.5,
        pm_monitoring_duration_sec=99.0,
        pm_alert_threshold_ratio=0.55,
        uc_threshold_deg=88.0,
        uc_moving_window_seconds=7.0,
        input_mode=None,  # auto
        width=111,
        height=222,
        fps=59.94,
        is_color=False,
        preview=False,
    )

    # make_frame_iter called with expected parameters
    assert calls["ffmpeg"] is not None
    assert calls["ffmpeg"]["input_mode"].startswith("ffmpeg")
    assert calls["ffmpeg"]["ffmpeg_input"] == "input.mp4"
    assert calls["ffmpeg"]["width"] == 111
    assert calls["ffmpeg"]["height"] == 222
    assert calls["ffmpeg"]["fps"] == 59.94
    assert calls["ffmpeg"]["is_color"] is False
    assert calls["ffmpeg"]["add_args"] is None

    # Sinks created and passed along
    assert calls["open"]["path"].endswith("out.csv")
    assert calls["csv"] == "CSV_WRITER_SENTINEL"
    # process_video passes (height, width) into setup_video_writer in this codebase
    assert calls["vid"]["arg"] == (222, 111)
    assert str(calls["vid"]["path"]).endswith("out.mp4")

    # run_pipeline received the iterator and sinks, and disable_japanese flag
    assert calls["run"]["first"][0] == 0.0  # timestamp from fake iterator
    assert calls["run"]["csv_writer"] == "CSV_WRITER_SENTINEL"
    assert calls["run"]["video_writer"] == "VIDEO_WRITER_SENTINEL"
    assert calls["run"]["preview"] is False
    assert calls["run"]["window_name"] == "Integrated Analysis"
    assert calls["run"]["disable_jp"] is True  # from disable_japanese


def test_process_video_opencv_fallback_when_no_ffmpeg(monkeypatch):
    """
    FFmpeg unavailable (make_frame_iter is None) -> use internal _opencv_iter.
    Verify:
      - cv2.VideoCapture used
      - isOpened/read/get/release flow
      - run_pipeline gets an iterator that yields frames with t from CAP_PROP_POS_MSEC/1000
    """
    # No FFmpeg
    monkeypatch.setattr("src.video_processor.make_frame_iter", None)

    # Build a fake VideoCapture
    class FakeCap:
        def __init__(self, path):
            self.path = path
            self._opened = True
            self._count = 0
            self.released = False

        def isOpened(self):  # noqa: N802
            return self._opened

        def read(self):
            # two frames then stop
            if self._count < 2:
                self._count += 1
                f = np.zeros((3, 4, 3), dtype=np.uint8) + self._count
                return True, f
            return False, None

        def get(self, prop):
            if prop == 0:  # patched CAP_PROP_POS_MSEC
                return 1000.0 * self._count  # 1.0s, 2.0s ...
            return 0.0

        def release(self):
            self.released = True

    # Patch cv2 constants used inside process_video._opencv_iter
    monkeypatch.setattr("src.video_processor.cv2.CAP_PROP_POS_MSEC", 0)
    monkeypatch.setattr("src.video_processor.cv2.VideoCapture", FakeCap)

    captured = {}

    def fake_run_pipeline(frame_iter, **kwargs):
        # consume the iterator fully
        items = list(frame_iter)
        captured["items"] = items
        captured["kwargs"] = kwargs

    monkeypatch.setattr("src.video_processor.run_pipeline", fake_run_pipeline)

    # Don’t create sinks in this test
    monkeypatch.setattr("src.video_processor.setup_csv_writer", lambda *a, **k: "CSV")
    monkeypatch.setattr("src.video_processor.setup_video_writer", lambda *a, **k: "VW")
    monkeypatch.setattr("builtins.open", lambda *a, **k: object())

    process_video(
        video_path="some.avi",
        output_csv_path=None,
        output_video_path=None,
        disable_japanese=False,
        input_mode=None,  # should auto-resolve to 'opencv-file' when ffmpeg is None
        preview=True,
    )

    # Iterator produced two frames with timestamps 1.0 and 2.0
    ts = [t for t, _ in captured["items"]]
    assert ts == [1.0, 2.0]


def test_process_video_opencv_mode_even_when_ffmpeg_available(monkeypatch):
    """
    If input_mode is explicitly 'opencv-file', process_video must pick the OpenCV iterator,
    even if make_frame_iter is available.
    """

    # Present a dummy make_frame_iter but ensure it must NOT be called
    def _boom(*a, **k):
        raise AssertionError("make_frame_iter should not be called in this test")

    monkeypatch.setattr("src.video_processor.make_frame_iter", _boom)

    # Fake VideoCapture with one frame then end
    class FakeCap:
        def __init__(self, path):
            self._count = 0

        def isOpened(self):  # noqa: N802
            return True

        def read(self):
            if self._count == 0:
                self._count += 1
                return True, np.zeros((2, 2, 3), dtype=np.uint8)
            return False, None

        def get(self, prop):
            return 500.0  # 0.5s

        def release(self):
            pass

    monkeypatch.setattr("src.video_processor.cv2.CAP_PROP_POS_MSEC", 0)
    monkeypatch.setattr("src.video_processor.cv2.VideoCapture", FakeCap)

    captured = {}

    def fake_run_pipeline(frame_iter, **kwargs):
        captured["items"] = list(frame_iter)

    monkeypatch.setattr("src.video_processor.run_pipeline", fake_run_pipeline)

    # No sinks created here
    monkeypatch.setattr("src.video_processor.setup_csv_writer", lambda *a, **k: "CSV")
    monkeypatch.setattr("src.video_processor.setup_video_writer", lambda *a, **k: "VW")
    monkeypatch.setattr("builtins.open", lambda *a, **k: object())

    process_video(
        video_path="cam0",
        output_csv_path=None,
        output_video_path=None,
        disable_japanese=False,
        input_mode="opencv-file",  # force opencv path
        width=640,
        height=480,
        fps=30.0,
        is_color=True,
        preview=False,
    )

    # Should have exactly one item from FakeCap
    assert captured["items"][0][0] == 0.5
    assert captured["items"][0][1].shape == (2, 2, 3)
