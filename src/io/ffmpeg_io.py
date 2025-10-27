import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

# ---------- NEW: portable helpers for camera capture ----------


def _find_ffmpeg_executable() -> str:
    """
    Find the FFmpeg executable path, handling cases where GUI apps don't have full PATH access.

    This function tries multiple strategies to locate FFmpeg:
    1. Check common installation paths (especially for macOS Homebrew)
    2. Use shutil.which() to find in PATH
    3. Fall back to 'ffmpeg' if nothing else works

    Returns
    -------
    str
        Path to FFmpeg executable, or 'ffmpeg' as fallback
    """
    # Common FFmpeg installation paths
    common_paths = [
        "/opt/homebrew/bin/ffmpeg",  # macOS Apple Silicon Homebrew
        "/usr/local/bin/ffmpeg",  # macOS Intel Homebrew / Linux
        "/usr/bin/ffmpeg",  # Linux system package
        "C:\\ffmpeg\\bin\\ffmpeg.exe",  # Windows common installation
    ]

    # Check common paths first
    for path in common_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    # Try to find in PATH
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    # Fallback to 'ffmpeg' (will work if it's in PATH)
    return "ffmpeg"


def _norm_os_key(os_name: str) -> str:
    key = (os_name or "").lower()
    if key.startswith("linux"):
        return "linux"
    if key.startswith("win"):
        return "windows"
    if key == "darwin":
        return "darwin"
    return key


def _camera_capture_args(
    os_name: str,
    device: str,
    width: int,
    height: int,
    fps: float,
) -> list[str]:
    """
    Build FFmpeg *input-side* arguments for a webcam/capture device.

    Always returned in the correct order to appear BEFORE `-i <device>`.

    macOS (avfoundation):
        ffmpeg -f avfoundation -thread_queue_size 4096 -framerate 30 -video_size 1280x720 -i "0"
    Linux (v4l2):
        ffmpeg -f v4l2 -thread_queue_size 4096 -input_format mjpeg -framerate 30 -video_size 1280x720 -i /dev/video0
    Windows (dshow):
        ffmpeg -f dshow -thread_queue_size 4096 -framerate 30 -video_size 1280x720 -i video="Integrated Camera"
    """
    key = _norm_os_key(os_name)
    fr = str(int(round(fps))) if fps and fps > 0 else "30"
    sz = f"{max(1, int(width))}x{max(1, int(height))}"

    common = ["-thread_queue_size", "4096"]

    if key == "darwin":  # macOS
        # device: "0", "0:", or a device name like "FaceTime HD Camera (Built-in)"
        return ["-f", "avfoundation", *common, "-framerate", fr, "-video_size", sz]

    if key == "linux":  # Linux
        # Prefer MJPEG to lower USB/CPU load; change to yuyv422 if needed.
        return ["-f", "v4l2", "-input_format", "mjpeg", *common, "-framerate", fr, "-video_size", sz]

    if key == "windows":  # Windows
        # dshow requires a *named* device: video="Integrated Camera"
        return ["-f", "dshow", *common, "-framerate", fr, "-video_size", sz]

    # Fallback: just queue size (better than nothing)
    return [*common]


def _camera_backend_name(os_name: str) -> str:
    key = _norm_os_key(os_name)
    return {"darwin": "avfoundation", "linux": "v4l2", "windows": "dshow"}.get(key, "unknown")


def make_frame_iter(
    input_mode: str,  # "ffmpeg-file" | "ffmpeg-camera"
    *,
    ffmpeg_input: str | None,  # path/URL or device ("0", "/dev/video0")
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    is_color: bool = True,
    add_args: list[str] | None = None,
):
    """
    Factory for a frame iterator backed by FFmpeg.

    Saki’s rule of engagement:
      "Say where your pixels come from. I’ll shape them and stream them. Minimal knobs, maximal clarity."

    Modes
    -----
    - `"ffmpeg-file"`:
        * Treat `ffmpeg_input` as a file/URL.
        * Build the command with any caller-supplied `add_args` (input-side flags).
    - `"ffmpeg-camera"`:
        * Treat `ffmpeg_input` as a capture device token.
        * Auto-select platform backend via `_camera_input_args(sys.platform, ...)`
          and *prepend* those flags to `add_args`.

    Parameters
    ----------
    input_mode : {"ffmpeg-file", "ffmpeg-camera"}
        How to interpret `ffmpeg_input`.
    ffmpeg_input : str | None
        File/URL/device. For camera, examples:
          - Linux: "/dev/video0"
          - macOS: "0" or "0:0"
          - Windows: "video=Integrated Camera"
        If `None`, defaults to "0" for camera mode and "" for file mode.
    width, height : int
        Output size for the video pipe. Skip scaling if non-positive.
    fps : float
        Target FPS for the pipe and nominal timestamps. Skip if `<= 0`.
    is_color : bool
        BGR24 (`True`) or GRAY8→BGR (`False`).
    add_args : list[str] | None
        Extra *input-side* args to inject before `-i`.

    Returns
    -------
    Iterator[tuple[float, np.ndarray]]
        `(timestamp_s, frame_bgr_uint8)` yielded by `_ffmpeg_frames`.

    Raises
    ------
    ValueError
        If `input_mode` is not one of the supported values.

    Examples
    --------
    Read from a file:
    >>> it = make_frame_iter("ffmpeg-file", ffmpeg_input="demo.mp4", width=640, height=360, fps=24)
    >>> t, frame = next(it)

    Read from a camera (Linux):
    >>> it = make_frame_iter("ffmpeg-camera", ffmpeg_input="/dev/video0", width=1280, height=720, fps=30)
    >>> for t, frame in it:
    ...     pass  # consume frames

    Design Notes
    ------------
    - Keep camera backend logic isolated in `_camera_input_args` to stay portable.
    - Keep output shaping (`scale`, `fps`, `pix_fmt`) inside `_build_ffmpeg_cmd`.
    """
    if input_mode == "ffmpeg-file":
        cmd = _build_ffmpeg_cmd(ffmpeg_input or "", width, height, fps, is_color, add_args)
        return _ffmpeg_frames(cmd, width, height, is_color, fps)

    if input_mode == "ffmpeg-camera":
        dev = ffmpeg_input or "0"
        cam_args = _camera_capture_args(sys.platform, dev, width, height, fps)
        # Ensure camera-specific args come BEFORE -i <device>
        input_args = (add_args or []) + cam_args
        cmd = _build_ffmpeg_cmd(dev, width, height, fps, is_color, input_args)
        return _ffmpeg_frames(cmd, width, height, is_color, fps)

    raise ValueError(f"unsupported input_mode={input_mode!r}")


def _ffmpeg_frames(cmd: list[str], width: int, height: int, is_color: bool, fps: float):
    """
    Spawn FFmpeg and yield `(timestamp_s, frame)` tuples from its rawvideo stdout.

    Saki's promise:
      "You push frames at me; I stamp time and hand them over—clean, fast, predictable."

    Contract
    --------
    - `cmd` must write raw frames to `stdout` with:
        * `-f rawvideo`
        * `-pix_fmt bgr24` (color) or `gray`/`gray8` (mono)
        * exact frame size `width x height`
    - We read `width*height*(3 or 1)` bytes per frame.
    - If `is_color` is `False`, frames are promoted to BGR with `cv2.cvtColor`, so downstream
      code can assume BGR arrays either way.

    Timing
    ------
    Timestamps are nominal: `frame_index / fps` when `fps > 0`, else `0.0`.
    They reflect configured cadence, not wall-clock. If you need wall-time, stamp it yourself
    at consumption.

    Parameters
    ----------
    cmd : list[str]
        Full FFmpeg command (see `_build_ffmpeg_cmd`).
    width, height : int
        Expected frame geometry.
    is_color : bool
        `True` for BGR24. `False` for GRAY8 → promoted to BGR.
    fps : float
        Used for nominal timestamps only.

    Yields
    ------
    tuple[float, np.ndarray]
        `(t_seconds, frame_bgr_uint8)`.

    Robustness & Cleanup
    --------------------
    - Stops when `stdout` ends or delivers a short read (EOF / upstream error).
    - Closes pipes and terminates/kills the process on exit paths.
    - Leaves `stderr` connected for post-mortem debugging but does not read it to avoid blocking.

    Pitfalls
    --------
    - If your command doesn't match `is_color`/size, reshaping will fail or frames will look wrong.
    - If `fps <= 0`, all timestamps will be `0.0`. Don't do that unless you really mean it.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # keep for debugging; not read to avoid blocking
        bufsize=10**8,
    )
    bytes_per_pix = 3 if is_color else 1
    frame_bytes = width * height * bytes_per_pix
    frame_count = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes) if proc.stdout else None
            if not buf or len(buf) < frame_bytes:
                break
            arr = np.frombuffer(buf, dtype=np.uint8)
            if is_color:
                frame = arr.reshape((height, width, 3))
            else:
                gray = arr.reshape((height, width))
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # type: ignore

            t = frame_count / fps if fps > 0 else 0.0
            yield (t, frame)
            frame_count += 1
    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except Exception:
            proc.kill()


def _build_ffmpeg_cmd(
    ffmpeg_input: str,
    width: int,
    height: int,
    fps: float,
    is_color: bool,
    additional_input_args: list[str] | None = None,
) -> list[str]:
    """
    Build the `ffmpeg` command that turns an input (file/URL/device) into a raw frame pipe.

    TL;DR (Saki style):
      "No drama. Give me pixels at the size and speed I asked for. To stdout."

    What this does
    --------------
    - Accepts any FFmpeg-readable input: file path, RTSP/HTTP URL, or a device token.
    - Appends optional *input-side* flags (e.g., backend selectors, buffers) before `-i`.
    - Applies a minimal `-vf` chain:
        * `scale=<width>:<height>` when both are positive.
        * `fps=<fps>` when `fps > 0`.
      If neither is requested, uses `vf=null`.
    - Forces an OpenCV-friendly pixel format:
        * Color  → `-pix_fmt bgr24`
        * Monochrome → `-pix_fmt gray`
    - Writes raw frames to `stdout` (`-f rawvideo pipe:1`) and mutes audio (`-an`).

    Parameters
    ----------
    ffmpeg_input : str
        File path, network URL, or a capture device string.
    width, height : int
        Output frame size. If either is non-positive, scaling is skipped.
    fps : float
        Target frame rate. If `<= 0`, FPS filter is skipped.
    is_color : bool
        `True` for BGR24 output, `False` for GRAY8 output.
    additional_input_args : list[str] | None
        Extra flags to place *before* `-i` (e.g., `["-f","v4l2","-thread_queue_size","4096"]`).

    Returns
    -------
    list[str]
        A complete `ffmpeg` command ready for `subprocess.Popen`.

    Notes
    -----
    - Keep input tuning (buffers/backends) in `additional_input_args`. Keep output shaping here.
    - This function does not validate the existence of `ffmpeg_input`; FFmpeg will report errors.
    - Pair with `_ffmpeg_frames(...)` to actually read the pipe.

    Example
    -------
    >>> cmd = _build_ffmpeg_cmd("sample.mp4", 1280, 720, 30.0, True)
    >>> cmd[:6]
    ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', 'sample.mp4']
    """

    pix_fmt = "bgr24" if is_color else "gray"

    vf_parts = []
    if width > 0 and height > 0:
        vf_parts.append(f"scale={width}:{height}")
    if fps > 0:
        vf_parts.append(f"fps={fps}")
    vf = ",".join(vf_parts) if vf_parts else "null"

    ffmpeg_executable = _find_ffmpeg_executable()
    cmd = [ffmpeg_executable, "-hide_banner", "-loglevel", "error"]
    if additional_input_args:
        cmd += additional_input_args
    cmd += ["-i", ffmpeg_input, "-an", "-vf", vf, "-pix_fmt", pix_fmt, "-f", "rawvideo", "pipe:1"]

    # Optional: print the exact command when debugging
    if os.environ.get("DEBUG_FFMPEG") == "1":
        print(f"FFmpeg executable: {ffmpeg_executable}")
        print("FFmpeg CMD:", " ".join(str(x) for x in cmd))

    return cmd
