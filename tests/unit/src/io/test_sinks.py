# tests/test_sinks.py


# Adjust this import to your actual module location if needed.
# The tested module must expose: Sinks, setup_video_writer_shape, build_sinks,
# safe_imshow, close_windows, and import setup_csv_writer from src.io_utils.
from src.io import sinks as mod


def test_sinks_dataclass_defaults():
    s = mod.Sinks(csv=None, video=None)
    assert s.csv is None
    assert s.video is None
    assert s.preview is True


def test_setup_video_writer_shape_calls_cv2_with_correct_params(monkeypatch):
    # Arrange
    frame_shape = (480, 640)  # h, w
    out_path = "dummy.mp4"
    fps = 29.97

    called = {}

    def fake_fourcc(*args):
        called["fourcc_args"] = args
        return 0x12345678

    class FakeWriter:
        pass

    def fake_vw(path, fourcc, ffps, size_tuple):
        called["vw_args"] = (path, fourcc, ffps, size_tuple)
        return FakeWriter()

    monkeypatch.setattr(mod.cv2, "VideoWriter_fourcc", fake_fourcc)
    monkeypatch.setattr(mod.cv2, "VideoWriter", fake_vw)

    # Act
    writer = mod.setup_video_writer_shape(frame_shape, out_path, fps)

    # Assert
    assert isinstance(writer, FakeWriter)
    # fourcc called with unpacked "mp4v"
    assert called["fourcc_args"] == tuple("mp4v")
    # VideoWriter called with (w, h) — note width/height order
    assert called["vw_args"] == (out_path, 0x12345678, fps, (640, 480))


def test_build_sinks_none_paths(monkeypatch):
    sinks = mod.build_sinks(csv_path=None, video_path=None, frame_shape=(10, 20), fps=30.0)
    assert sinks.csv is None
    assert sinks.video is None
    assert sinks.preview is True


def test_build_sinks_with_paths(tmp_path, monkeypatch):
    csv_path = tmp_path / "out.csv"
    video_path = tmp_path / "out.mp4"

    # Capture the file object passed to setup_csv_writer and return a sentinel
    csv_called = {}

    def fake_setup_csv_writer(fobj):
        # Ensure it's an open file-like object
        assert fobj.writable()
        csv_called["got"] = True
        return "CSV_WRITER_SENTINEL"

    # Replace the function imported into the module under test
    monkeypatch.setattr(mod, "setup_csv_writer", fake_setup_csv_writer)

    # Stub video writer creator in this module to avoid touching cv2 for this test
    def fake_setup_video_writer_shape(shape, path, fps):
        assert shape == (108, 192)
        assert str(path) == str(video_path)
        assert fps == 24.0
        return "VIDEO_WRITER_SENTINEL"

    monkeypatch.setattr(mod, "setup_video_writer_shape", fake_setup_video_writer_shape)

    # Act
    sinks = mod.build_sinks(
        csv_path=str(csv_path),
        video_path=str(video_path),
        frame_shape=(108, 192),
        fps=24.0,
    )

    # Assert
    assert csv_called.get("got") is True
    assert sinks.csv == "CSV_WRITER_SENTINEL"
    assert sinks.video == "VIDEO_WRITER_SENTINEL"
    assert sinks.preview is True


def test_safe_imshow_true_when_not_q(monkeypatch):
    shown = {}

    def fake_imshow(win, frame):
        shown["ok"] = (win, frame)

    # Return a key code different from 'q'
    def fake_waitKey(delay):  # noqa: N802
        assert delay == 1
        return ord("a")  # not 'q'

    monkeypatch.setattr(mod.cv2, "imshow", fake_imshow)
    monkeypatch.setattr(mod.cv2, "waitKey", fake_waitKey)

    result = mod.safe_imshow("w", frame="FRAME")
    assert shown["ok"] == ("w", "FRAME")
    assert result is True  # not 'q' => keep running


def test_safe_imshow_false_when_q(monkeypatch):
    def fake_imshow(win, frame):
        pass

    # Simulate pressing 'q'
    def fake_waitKey(delay):  # noqa: N802
        return ord("q")

    monkeypatch.setattr(mod.cv2, "imshow", fake_imshow)
    monkeypatch.setattr(mod.cv2, "waitKey", fake_waitKey)

    result = mod.safe_imshow("main", frame="F")
    assert result is False  # 'q' pressed => stop


def test_safe_imshow_returns_true_on_exception(monkeypatch):
    def fake_imshow(win, frame):
        raise RuntimeError("no display")

    monkeypatch.setattr(mod.cv2, "imshow", fake_imshow)

    # waitKey should not be called; but set anyway to fail if it is
    def fake_waitKey(delay):  # noqa: N802
        raise AssertionError("waitKey should not be called after exception")

    monkeypatch.setattr(mod.cv2, "waitKey", fake_waitKey)

    assert mod.safe_imshow("w", frame="F") is True  # headless path returns True


def test_close_windows_normal(monkeypatch):
    called = {}

    def fake_destroyAllWindows():  # noqa: N802
        called["ok"] = True

    monkeypatch.setattr(mod.cv2, "destroyAllWindows", fake_destroyAllWindows)
    # Should not raise
    mod.close_windows()
    assert called["ok"] is True


def test_close_windows_swallow_exception(monkeypatch):
    def fake_destroyAllWindows():  # noqa: N802
        raise RuntimeError("headless")

    monkeypatch.setattr(mod.cv2, "destroyAllWindows", fake_destroyAllWindows)
    # Should not raise even if cv2 destroys fail
    mod.close_windows()
