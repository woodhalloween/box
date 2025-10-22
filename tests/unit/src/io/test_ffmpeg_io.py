# test_ffmpeg_io.py
import io

import numpy as np
import pytest

# ⬇️ CHANGE THIS to your actual module name
import src.io.ffmpeg_io as m


class _FakeStream(io.RawIOBase):
    """A tiny readable stream that returns predefined byte chunks from .read()."""

    def __init__(self, chunks, raise_on_close: bool = False):
        super().__init__()
        self._chunks = list(chunks)  # list of bytes objects to return in order
        self.closed_flag = False
        self._raise_on_close = raise_on_close

    def read(self, n: int = -1):
        if not self._chunks:
            return b""
        # Return exactly the next chunk regardless of n; tests craft chunk sizes.
        return self._chunks.pop(0)

    def close(self):
        # Mark closed and optionally raise to exercise the except branch in _ffmpeg_frames
        self.closed_flag = True
        if self._raise_on_close:
            raise RuntimeError("close failed intentionally")


class _FakeProc:
    """Minimal stand-in for subprocess.Popen return obj used by _ffmpeg_frames."""

    def __init__(self, stdout_chunks, raise_on_wait=False, raise_on_close=False):
        self.stdout = _FakeStream(stdout_chunks, raise_on_close=raise_on_close)
        self.stderr = _FakeStream([b"err"], raise_on_close=raise_on_close)
        self._terminated = False
        self._killed = False
        self._waited = False
        self._raise_on_wait = raise_on_wait

    # Methods used by the code under test
    def terminate(self):
        self._terminated = True

    def kill(self):
        self._killed = True

    def wait(self, timeout=None):
        self._waited = True
        if self._raise_on_wait:
            # Trigger the except: proc.kill() branch
            raise TimeoutError("simulated wait timeout")


def _install_popen(monkeypatch, fake_proc):
    """Monkeypatch subprocess.Popen to return our fake proc."""

    def _fake_popen(cmd, stdout=None, stderr=None, bufsize=None):
        # Basic sanity: the builder should pass a list-like command
        assert isinstance(cmd, list | tuple) and cmd, "ffmpeg command should be a non-empty list"
        return fake_proc

    monkeypatch.setattr(m.subprocess, "Popen", _fake_popen)


def _indexes(cmd, token):
    # helper: return all indexes where token appears
    return [i for i, t in enumerate(cmd) if t == token]


# -------- Tests for _build_ffmpeg_cmd --------


def test_build_ffmpeg_cmd_color_with_scaling_and_fps():
    cmd = m._build_ffmpeg_cmd(
        ffmpeg_input="input.mp4",
        width=320,
        height=240,
        fps=25.0,
        is_color=True,
        additional_input_args=["-re"],  # arbitrary extra
    )
    # Structure expectations - cmd[0] should be the FFmpeg executable path
    assert cmd[0].endswith("ffmpeg") or cmd[0] == "ffmpeg"
    assert "-hide_banner" in cmd and "-loglevel" in cmd
    # Input and extras present in correct order
    idx_i = cmd.index("-i")
    assert cmd[idx_i + 1] == "input.mp4"
    assert "-re" in cmd  # carried through
    # Video filter joined correctly
    vf_idx = cmd.index("-vf")
    assert cmd[vf_idx + 1] == "scale=320:240,fps=25.0"
    # Pixel format for color
    pix_idx = cmd.index("-pix_fmt")
    assert cmd[pix_idx + 1] == "bgr24"
    # Raw pipe out
    assert cmd[-2] == "-f" and cmd[-1] == "rawvideo" or cmd[-1].startswith("pipe:")
    assert "pipe:1" in cmd


def test_build_ffmpeg_cmd_gray_no_scaling_no_fps():
    cmd = m._build_ffmpeg_cmd(
        ffmpeg_input="rtsp://camera/stream", width=0, height=0, fps=0.0, is_color=False, additional_input_args=None
    )
    # With no width/height/fps, vf should be 'null'
    vf_idx = cmd.index("-vf")
    assert cmd[vf_idx + 1] == "null"
    # Pixel format for mono
    pix_idx = cmd.index("-pix_fmt")
    assert cmd[pix_idx + 1] == "gray"
    # Input URL preserved
    idx_i = cmd.index("-i")
    assert cmd[idx_i + 1].startswith("rtsp://")


# ——— Updated tests for _build_ffmpeg_cmd under the new behavior ———


def test_build_ffmpeg_cmd_no_implicit_cam_flags_and_correct_vf(monkeypatch):
    """No implicit -framerate/-video_size are injected anymore.
    Only -vf scale/fps are managed here; any input-side flags must come via additional_input_args.
    """
    monkeypatch.setattr(m.sys, "platform", "darwin")  # platform no longer matters for cmd content

    cmd = m._build_ffmpeg_cmd(
        ffmpeg_input="0",  # camera token, but now treated the same as any input for this function
        width=640,
        height=480,
        fps=29.6,
        is_color=True,
        additional_input_args=None,
    )

    # Nothing implicit before -i except the fixed boilerplate
    assert "-framerate" not in cmd
    assert "-video_size" not in cmd

    # -vf remains responsible for scale and fps (uses original float)
    vf = cmd[cmd.index("-vf") + 1]
    assert vf == "scale=640:480,fps=29.6"


def test_build_ffmpeg_cmd_no_implicit_framerate_when_no_size(monkeypatch):
    monkeypatch.setattr(m.sys, "platform", "darwin")

    cmd = m._build_ffmpeg_cmd(
        ffmpeg_input="0",
        width=0,
        height=480,
        fps=25.0,
        is_color=False,
        additional_input_args=None,
    )

    # No injected input-side flags
    assert "-framerate" not in cmd
    assert "-video_size" not in cmd

    # vf only has fps filter
    vf = cmd[cmd.index("-vf") + 1]
    assert vf == "fps=25.0"
    # pixel format still correct
    assert cmd[cmd.index("-pix_fmt") + 1] == "gray"


def test_build_ffmpeg_cmd_no_implicit_video_size_when_fps_leq_zero(monkeypatch):
    monkeypatch.setattr(m.sys, "platform", "darwin")

    cmd = m._build_ffmpeg_cmd(
        ffmpeg_input="0",
        width=320,
        height=240,
        fps=0.0,  # no fps filter
        is_color=True,
        additional_input_args=None,
    )

    # No injected input-side flags
    assert "-framerate" not in cmd
    assert "-video_size" not in cmd

    # vf only has scale
    vf = cmd[cmd.index("-vf") + 1]
    assert vf == "scale=320:240"
    # pixel format still correct
    assert cmd[cmd.index("-pix_fmt") + 1] == "bgr24"


def test_build_ffmpeg_cmd_non_darwin_or_non_digit_device_adds_no_capture_options(monkeypatch):
    # Case A: non-macOS platform → no capture options even if input is digit
    monkeypatch.setattr(m.sys, "platform", "linux")
    cmd_linux = m._build_ffmpeg_cmd(ffmpeg_input="0", width=640, height=480, fps=30.0, is_color=True)
    assert "-framerate" not in cmd_linux
    assert "-video_size" not in cmd_linux

    # Case B: macOS but input is NOT a pure digit → no capture options
    monkeypatch.setattr(m.sys, "platform", "darwin")
    cmd_nondigit = m._build_ffmpeg_cmd(ffmpeg_input="camera0", width=640, height=480, fps=30.0, is_color=True)
    assert "-framerate" not in cmd_nondigit
    assert "-video_size" not in cmd_nondigit


# -------- Tests for _ffmpeg_frames --------


def test_ffmpeg_frames_color_two_frames_and_cleanup(monkeypatch):
    # 2x2 color => 12 bytes per frame
    width, height, fps, is_color = 2, 2, 30.0, True
    frame_bytes = width * height * 3

    # Prepare two frames (increasing values) and then EOF
    f0 = bytes(range(0, frame_bytes))  # 0..11
    f1 = bytes([(x + 1) % 256 for x in range(frame_bytes)])  # 1..12 (wrapped)
    fake_proc = _FakeProc(stdout_chunks=[f0, f1, b""])

    _install_popen(monkeypatch, fake_proc)

    cmd = ["ffmpeg", "-i", "dummy", "-f", "rawvideo", "pipe:1"]
    out = list(m._ffmpeg_frames(cmd, width, height, is_color, fps))

    # Two frames yielded
    assert len(out) == 2
    # Timestamps from frame_count / fps
    assert out[0][0] == pytest.approx(0.0)
    assert out[1][0] == pytest.approx(1.0 / fps)

    # Frame shapes and exact content (BGR already)
    assert out[0][1].shape == (height, width, 3)
    assert out[1][1].shape == (height, width, 3)
    # Flatten and compare first frame content
    assert out[0][1].ravel().tolist() == list(range(frame_bytes))
    # Ensure cleanup was invoked
    assert fake_proc._terminated is True
    assert fake_proc._waited is True
    assert fake_proc._killed is False  # wait succeeded by default


def test_ffmpeg_frames_gray_promotes_to_bgr_and_zero_fps_timestamp(monkeypatch):
    # 2x2 gray => 4 bytes per frame
    width, height, fps, is_color = 2, 2, 0.0, False
    frame_bytes = width * height * 1

    # One gray frame with simple pattern, then EOF
    g0 = bytes([0, 64, 128, 255])
    assert len(g0) == frame_bytes
    fake_proc = _FakeProc(stdout_chunks=[g0, b""])

    _install_popen(monkeypatch, fake_proc)

    cmd = ["ffmpeg", "-i", "dummy", "-f", "rawvideo", "pipe:1"]
    out = list(m._ffmpeg_frames(cmd, width, height, is_color, fps))

    # Single frame yielded; timestamp forced to 0.0 when fps <= 0
    assert len(out) == 1
    t0, frame0 = out[0]
    assert t0 == 0.0
    # Should be promoted to BGR for downstream compatibility
    assert frame0.shape == (height, width, 3)
    # All channels equal original gray
    # Expected arrangement:
    expected_gray = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    expected_bgr = np.dstack([expected_gray, expected_gray, expected_gray])
    assert np.array_equal(frame0, expected_bgr)


def test_ffmpeg_frames_partial_read_breaks(monkeypatch):
    # Create a partial buffer smaller than a full frame to trigger the 'break' path
    width, height, fps, is_color = 3, 3, 10.0, True
    partial = bytes([1, 2, 3])  # definitely smaller than 27

    fake_proc = _FakeProc(stdout_chunks=[partial])  # no full frames -> loop exits immediately
    _install_popen(monkeypatch, fake_proc)

    cmd = ["ffmpeg", "-i", "dummy", "-f", "rawvideo", "pipe:1"]
    out = list(m._ffmpeg_frames(cmd, width, height, is_color, fps))

    assert out == []  # no frames yielded
    # Cleanup still runs
    assert fake_proc._terminated is True
    assert fake_proc._waited is True


def test_ffmpeg_frames_wait_timeout_triggers_kill(monkeypatch):
    width, height, fps, is_color = 2, 2, 30.0, True
    frame_bytes = width * height * 3
    f0 = bytes(range(frame_bytes))

    # Configure wait to raise -> exercise except branch that calls proc.kill()
    fake_proc = _FakeProc(stdout_chunks=[f0, b""], raise_on_wait=True)
    _install_popen(monkeypatch, fake_proc)

    cmd = ["ffmpeg", "-i", "dummy", "-f", "rawvideo", "pipe:1"]
    # Exhaust generator
    _ = list(m._ffmpeg_frames(cmd, width, height, is_color, fps))

    assert fake_proc._terminated is True
    assert fake_proc._waited is True
    assert fake_proc._killed is True  # kill path taken


def test_ffmpeg_frames_close_raises_is_swallowed(monkeypatch):
    """stdout/stderr.close() raising should be swallowed by the try/except in finally."""
    width, height, fps, is_color = 2, 2, 30.0, True
    frame_bytes = width * height * 3
    f0 = bytes(range(frame_bytes))

    # close() will raise for both stdout/stderr to hit the except block
    fake_proc = _FakeProc(stdout_chunks=[f0, b""], raise_on_wait=False, raise_on_close=True)
    _install_popen(monkeypatch, fake_proc)

    cmd = ["ffmpeg", "-i", "dummy", "-f", "rawvideo", "pipe:1"]
    # Should not propagate the close exception
    out = list(m._ffmpeg_frames(cmd, width, height, is_color, fps))
    assert len(out) == 1
    assert fake_proc._terminated is True


# --------------------------------
# make_frame_iter coverage (all branches)
# --------------------------------


def test_make_frame_iter_file_mode_calls_build_and_frames(monkeypatch):
    captured = {}

    def fake_build_ffmpeg_cmd(ffmpeg_input, width, height, fps, is_color, add_args):
        # ensure ffmpeg_input becomes "" when None
        captured["build"] = {
            "ffmpeg_input": ffmpeg_input,
            "width": width,
            "height": height,
            "fps": fps,
            "is_color": is_color,
            "add_args": add_args,
        }
        return ["ffmpeg", "-y"]  # dummy cmd

    def fake_ffmpeg_frames(cmd, width, height, is_color, fps):
        captured["frames"] = {
            "cmd": cmd,
            "width": width,
            "height": height,
            "is_color": is_color,
            "fps": fps,
        }
        return "FRAMES_SENTINEL"

    monkeypatch.setattr(m, "_build_ffmpeg_cmd", fake_build_ffmpeg_cmd)
    monkeypatch.setattr(m, "_ffmpeg_frames", fake_ffmpeg_frames)

    # Pass None to verify it becomes "" inside make_frame_iter for file mode,
    # and that add_args is forwarded as-is.
    res = m.make_frame_iter(
        "ffmpeg-file",
        ffmpeg_input=None,
        width=640,
        height=360,
        fps=29.97,
        is_color=False,
        add_args=["-nostdin", "-hide_banner"],
    )

    assert res == "FRAMES_SENTINEL"
    assert captured["build"]["ffmpeg_input"] == ""  # None -> ""
    assert captured["build"]["width"] == 640
    assert captured["build"]["height"] == 360
    assert captured["build"]["fps"] == 29.97
    assert captured["build"]["is_color"] is False
    assert captured["build"]["add_args"] == ["-nostdin", "-hide_banner"]

    assert captured["frames"]["cmd"] == ["ffmpeg", "-y"]
    assert captured["frames"]["width"] == 640
    assert captured["frames"]["height"] == 360
    assert captured["frames"]["is_color"] is False
    assert captured["frames"]["fps"] == 29.97


def test_make_frame_iter_camera_mode_combines_args_and_uses_platform(monkeypatch):
    captured = {}

    # Force linux branch in camera-arg builder
    monkeypatch.setattr(m.sys, "platform", "linux")

    def fake_build_ffmpeg_cmd(*args, **kwargs):
        # signature changed: additional_input_args kwarg
        captured["build"] = {
            "ffmpeg_input": args[0],
            "width": args[1],
            "height": args[2],
            "fps": args[3],
            "is_color": args[4],
            "add_args": (kwargs.get("additional_input_args") if "additional_input_args" in kwargs else args[5])[:],
        }
        return ["ffmpeg", "-camera"]

    def fake_ffmpeg_frames(cmd, width, height, is_color, fps):
        captured["frames"] = {"cmd": cmd, "width": width, "height": height, "is_color": is_color, "fps": fps}
        return "FRAMES_CAMERA"

    monkeypatch.setattr(m, "_build_ffmpeg_cmd", fake_build_ffmpeg_cmd)
    monkeypatch.setattr(m, "_ffmpeg_frames", fake_ffmpeg_frames)

    res = m.make_frame_iter(
        "ffmpeg-camera",
        ffmpeg_input="/dev/video2",
        width=1280,
        height=720,
        fps=60.0,
        is_color=True,
        add_args=["-re", "-nostdin"],  # should appear before camera args
    )

    assert res == "FRAMES_CAMERA"
    b = captured["build"]
    assert b["ffmpeg_input"] == "/dev/video2"
    assert (b["width"], b["height"], b["fps"], b["is_color"]) == (1280, 720, 60.0, True)

    add_args = b["add_args"]
    # user args must be present and in the original relative order
    u0 = add_args.index("-re")
    u1 = add_args.index("-nostdin")
    assert u0 < u1

    # linux camera args should be included (order among them can vary by implementation)
    required_subset = {"-f", "v4l2", "-input_format", "mjpeg", "-thread_queue_size", "4096"}
    assert required_subset.issubset(set(add_args))

    # ensure user args precede at least one known camera token to preserve precedence
    assert u0 < add_args.index("-f")

    # frames used the command returned by fake_build
    assert captured["frames"]["cmd"] == ["ffmpeg", "-camera"]


def test_make_frame_iter_camera_mode_defaults_input_and_empty_add_args(monkeypatch):
    # When ffmpeg_input=None → default "0"; add_args=None → just camera args.
    monkeypatch.setattr(m.sys, "platform", "darwin")  # avfoundation branch

    captured = {}

    def fake_build_ffmpeg_cmd(*args, **kwargs):
        captured["build"] = (
            args[0],
            kwargs.get("additional_input_args") if "additional_input_args" in kwargs else args[5],
        )
        return ["ffmpeg", "-cam-defaults"]

    def fake_ffmpeg_frames(cmd, width, height, is_color, fps):
        return "FRAMES_DEFAULTS"

    monkeypatch.setattr(m, "_build_ffmpeg_cmd", fake_build_ffmpeg_cmd)
    monkeypatch.setattr(m, "_ffmpeg_frames", fake_ffmpeg_frames)

    res = m.make_frame_iter("ffmpeg-camera", ffmpeg_input=None)
    assert res == "FRAMES_DEFAULTS"

    ff_in, add_args = captured["build"]
    assert ff_in == "0"  # defaulted device
    # darwin camera args should include avfoundation and queue sizing; allow flexible order
    must_have = {"-f", "avfoundation", "-thread_queue_size", "4096"}
    assert must_have.issubset(set(add_args))


def test_make_frame_iter_invalid_mode_raises():
    with pytest.raises(ValueError) as ei:
        m.make_frame_iter("unknown-mode", ffmpeg_input="x")
    assert "unsupported input_mode" in str(ei.value)


# -------- Tests for _norm_os_key --------


def test_norm_os_key_windows():
    """Test _norm_os_key function lines 15-16 (Windows case)."""
    # Test various Windows platform strings
    assert m._norm_os_key("win") == "windows"
    assert m._norm_os_key("windows") == "windows"
    assert m._norm_os_key("win32") == "windows"
    assert m._norm_os_key("win64") == "windows"
    assert m._norm_os_key("WIN") == "windows"
    assert m._norm_os_key("Windows") == "windows"


def test_norm_os_key_fallback():
    """Test _norm_os_key function line 19 (fallback case)."""
    # Test fallback case for unrecognized OS names
    assert m._norm_os_key("unknown") == "unknown"
    assert m._norm_os_key("freebsd") == "freebsd"
    assert m._norm_os_key("") == ""
    assert m._norm_os_key(None) == ""
    assert m._norm_os_key("some_random_os") == "some_random_os"


# -------- Tests for _camera_capture_args --------


def test_camera_capture_args_windows():
    """Test _camera_capture_args function lines 55-60 (Windows case)."""
    args = m._camera_capture_args("windows", "Integrated Camera", 1920, 1080, 30.0)

    # Check that Windows-specific args are present
    assert "-f" in args
    assert "dshow" in args
    assert "-thread_queue_size" in args
    assert "4096" in args
    assert "-framerate" in args
    assert "30" in args
    assert "-video_size" in args
    assert "1920x1080" in args

    # Verify the order: -f dshow should come before other args
    f_idx = args.index("-f")
    dshow_idx = args.index("dshow")
    assert f_idx + 1 == dshow_idx


def test_camera_capture_args_fallback():
    """Test _camera_capture_args function lines 59-60 (fallback case)."""
    # Test fallback case for unrecognized OS
    args = m._camera_capture_args("unknown_os", "device", 640, 480, 25.0)

    # Should only return common args (thread_queue_size)
    assert args == ["-thread_queue_size", "4096"]

    # Test with None OS name
    args_none = m._camera_capture_args(None, "device", 640, 480, 25.0)
    assert args_none == ["-thread_queue_size", "4096"]

    # Test with empty OS name
    args_empty = m._camera_capture_args("", "device", 640, 480, 25.0)
    assert args_empty == ["-thread_queue_size", "4096"]


# -------- Tests for _camera_backend_name --------


def test_camera_backend_name_darwin():
    """Test _camera_backend_name function for macOS."""
    assert m._camera_backend_name("darwin") == "avfoundation"
    assert m._camera_backend_name("Darwin") == "avfoundation"


def test_camera_backend_name_linux():
    """Test _camera_backend_name function for Linux."""
    assert m._camera_backend_name("linux") == "v4l2"
    assert m._camera_backend_name("Linux") == "v4l2"
    assert m._camera_backend_name("linux2") == "v4l2"


def test_camera_backend_name_windows():
    """Test _camera_backend_name function for Windows."""
    assert m._camera_backend_name("win") == "dshow"
    assert m._camera_backend_name("windows") == "dshow"
    assert m._camera_backend_name("win32") == "dshow"
    assert m._camera_backend_name("win64") == "dshow"


def test_camera_backend_name_fallback():
    """Test _camera_backend_name function fallback case."""
    # Test fallback case for unrecognized OS names
    assert m._camera_backend_name("unknown") == "unknown"
    assert m._camera_backend_name("freebsd") == "unknown"
    assert m._camera_backend_name("") == "unknown"
    assert m._camera_backend_name(None) == "unknown"
    assert m._camera_backend_name("some_random_os") == "unknown"


# -------- Tests for _build_ffmpeg_cmd debug functionality --------


def test_build_ffmpeg_cmd_debug_print_enabled(monkeypatch, capsys):
    """Test _build_ffmpeg_cmd function lines 316-317 (debug print when DEBUG_FFMPEG=1)."""
    # Set DEBUG_FFMPEG environment variable to "1"
    monkeypatch.setenv("DEBUG_FFMPEG", "1")

    # Call _build_ffmpeg_cmd
    cmd = m._build_ffmpeg_cmd(  # noqa: F841
        ffmpeg_input="test.mp4", width=640, height=480, fps=30.0, is_color=True, additional_input_args=None
    )

    # Capture the printed output
    captured = capsys.readouterr()

    # Verify the debug message was printed
    assert "FFmpeg CMD:" in captured.out
    assert "test.mp4" in captured.out
    assert "ffmpeg" in captured.out or "/ffmpeg" in captured.out
    assert "-hide_banner" in captured.out
    assert "-loglevel" in captured.out
    assert "error" in captured.out
    assert "-i" in captured.out
    assert "-an" in captured.out
    assert "-vf" in captured.out
    assert "scale=640:480,fps=30.0" in captured.out
    assert "-pix_fmt" in captured.out
    assert "bgr24" in captured.out
    assert "-f" in captured.out
    assert "rawvideo" in captured.out
    assert "pipe:1" in captured.out


def test_build_ffmpeg_cmd_debug_print_disabled(monkeypatch, capsys):
    """Test _build_ffmpeg_cmd function lines 316-317 (no debug print when DEBUG_FFMPEG!=1)."""
    # Set DEBUG_FFMPEG environment variable to "0" (disabled)
    monkeypatch.setenv("DEBUG_FFMPEG", "0")

    # Call _build_ffmpeg_cmd
    cmd = m._build_ffmpeg_cmd(  # noqa: F841
        ffmpeg_input="test.mp4", width=640, height=480, fps=30.0, is_color=True, additional_input_args=None
    )

    # Capture the printed output
    captured = capsys.readouterr()

    # Verify no debug message was printed
    assert "FFmpeg CMD:" not in captured.out
    assert captured.out == ""


def test_build_ffmpeg_cmd_debug_print_unset(monkeypatch, capsys):
    """Test _build_ffmpeg_cmd function lines 316-317 (no debug print when DEBUG_FFMPEG unset)."""
    # Ensure DEBUG_FFMPEG environment variable is not set
    monkeypatch.delenv("DEBUG_FFMPEG", raising=False)

    # Call _build_ffmpeg_cmd
    cmd = m._build_ffmpeg_cmd(  # noqa: F841
        ffmpeg_input="test.mp4", width=640, height=480, fps=30.0, is_color=True, additional_input_args=None
    )

    # Capture the printed output
    captured = capsys.readouterr()

    # Verify no debug message was printed
    assert "FFmpeg CMD:" not in captured.out
    assert captured.out == ""


def test_build_ffmpeg_cmd_debug_print_with_additional_args(monkeypatch, capsys):
    """Test _build_ffmpeg_cmd debug print with additional input arguments."""
    # Set DEBUG_FFMPEG environment variable to "1"
    monkeypatch.setenv("DEBUG_FFMPEG", "1")

    # Call _build_ffmpeg_cmd with additional arguments
    cmd = m._build_ffmpeg_cmd(  # noqa: F841
        ffmpeg_input="rtsp://camera/stream",
        width=1280,
        height=720,
        fps=25.0,
        is_color=False,
        additional_input_args=["-re", "-nostdin", "-thread_queue_size", "4096"],
    )

    # Capture the printed output
    captured = capsys.readouterr()

    # Verify the debug message was printed with additional args
    assert "FFmpeg CMD:" in captured.out
    assert "rtsp://camera/stream" in captured.out
    assert "-re" in captured.out
    assert "-nostdin" in captured.out
    assert "-thread_queue_size" in captured.out
    assert "4096" in captured.out
    assert "scale=1280:720,fps=25.0" in captured.out
    assert "gray" in captured.out  # pixel format for is_color=False
