import subprocess
import sys

import cv2
import numpy as np


# --- NEW: small helper to build/read from FFmpeg pipe ---
def _camera_input_args(os_name: str, device: str) -> list[str]:
    """
    Build FFmpeg *input-side* arguments for a webcam/capture device.

    These flags are intended to appear BEFORE the token `-i <device>`.
    They only define the input demuxer and buffering behavior. Resolution and
    frame rate should be handled on the output side (e.g., via `-vf scale=...,fps=...`)
    inside `_build_ffmpeg_cmd()`.

    Parameters
    ----------
    os_name : str
        Typically `sys.platform` (e.g., "linux", "linux2", "darwin", "win32").
        Used to select the appropriate FFmpeg capture backend.
    device : str
        The exact token that will follow `-i`.
        - Linux (v4l2): prefer an absolute path such as "/dev/video0".
        - macOS (avfoundation): an index string like "0" or "0:0".
        - Windows (dshow): a *named* device like "video=Integrated Camera"
          (numeric indices are not supported by dshow).

    Returns
    -------
    list[str]
        Arguments to place BEFORE `-i <device>`.

    Rationale
    ---------
    Keep input arguments minimal and portable:
    - Select the capture backend (`-f ...`).
    - Choose a sane default transport (e.g., `-input_format mjpeg` on v4l2 to reduce USB/CPU load).
    - Increase input-side queues/buffers to reduce frame drops.
    """
    key = (os_name or "").lower()

    # Common: enlarge the input thread queue to mitigate drops when downstream is busy.
    common = ["-thread_queue_size", "4096"]

    # Linux: v4l2 backend. MJPEG is a good default; switch to "yuyv422" if your device requires it.
    if key.startswith("linux"):
        return ["-f", "v4l2", "-input_format", "mjpeg", *common]

    # macOS: avfoundation backend. If you need to drive capture rate on input,
    # pass `-framerate` from the caller; keep this function minimal.
    if key == "darwin":
        return ["-f", "avfoundation", *common]

    # Windows: dshow backend. Use named devices (e.g., "video=..."). Increase realtime buffer.
    if key.startswith("win"):
        return ["-f", "dshow", "-rtbufsize", "256M", *common]

    # Fallback for unknown platforms: return only common safety flags.
    return [*common]


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
        cam_args = _camera_input_args(sys.platform, ffmpeg_input or "0")
        cmd = _build_ffmpeg_cmd(ffmpeg_input or "0", width, height, fps, is_color, (add_args or []) + cam_args)
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

    # Input side: include framerate/size for camera
    input_args = []
    if additional_input_args:
        input_args += additional_input_args
    if sys.platform == "darwin" and ffmpeg_input.isdigit():
        # Add capture-specific options
        if fps > 0:
            input_args += ["-framerate", str(int(round(fps)))]
        if width > 0 and height > 0:
            input_args += ["-video_size", f"{width}x{height}"]

    vf_parts = []
    if width > 0 and height > 0:
        vf_parts.append(f"scale={width}:{height}")
    if fps > 0:
        vf_parts.append(f"fps={fps}")
    vf = ",".join(vf_parts) if vf_parts else "null"

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    cmd += input_args
    cmd += ["-i", ffmpeg_input, "-an", "-vf", vf, "-pix_fmt", pix_fmt, "-f", "rawvideo", "pipe:1"]
    return cmd
