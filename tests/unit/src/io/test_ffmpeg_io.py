# test_ffmpeg_io.py
import io

import numpy as np
import pytest

# ⬇️ CHANGE THIS to your actual module name
import src.io.ffmpeg_io as m  # e.g., `import ffmpeg_io as m`


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
    # Structure expectations
    assert cmd[0] == "ffmpeg"
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
