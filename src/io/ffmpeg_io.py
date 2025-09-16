import subprocess

import cv2
import numpy as np

# --- NEW: small helper to build/read from FFmpeg pipe ---


def _ffmpeg_frames(cmd: list[str], width: int, height: int, is_color: bool, fps: float):
    """
    Spawn ffmpeg and yield (timestamp_s, frame ndarray[BGR]) tuples.
    Assumes ffmpeg outputs rawvideo to stdout with pix_fmt=bgr24 (color) or gray8 (mono).
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
                # keep downstream drawing/pose code happy: promote to BGR
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # type: ignore

            # nominal timestamp derived from frame index and configured FPS
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
    Build an ffmpeg command that reads from `ffmpeg_input` and writes raw frames to stdout.
    - `ffmpeg_input` can be any FFmpeg input (file path, RTSP/HTTP URL, device name, etc.)
    - Set pixel format to bgr24 (color) or gray (mono) for OpenCV-friendly output.
    """
    pix_fmt = "bgr24" if is_color else "gray"
    vf_parts = []
    if width > 0 and height > 0:
        vf_parts.append(f"scale={width}:{height}")
    if fps > 0:
        vf_parts.append(f"fps={fps}")
    vf = ",".join(vf_parts) if vf_parts else "null"

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if additional_input_args:
        cmd += additional_input_args
    cmd += ["-i", ffmpeg_input, "-an", "-vf", vf, "-pix_fmt", pix_fmt, "-f", "rawvideo", "pipe:1"]
    return cmd
