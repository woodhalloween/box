import contextlib
import csv
from dataclasses import dataclass

import cv2

from src.io_utils import setup_csv_writer


@dataclass
class Sinks:
    csv: csv.DictWriter | None
    video: cv2.VideoWriter | None
    preview: bool = True


def setup_video_writer_shape(frame_shape: tuple[int, int], out_path: str, fps: float) -> cv2.VideoWriter:
    h, w = frame_shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 既存ポリシーに合わせて
    return cv2.VideoWriter(out_path, fourcc, fps, (w, h))


def build_sinks(csv_path: str | None, video_path: str | None, frame_shape: tuple[int, int], fps: float) -> Sinks:
    csv_writer = setup_csv_writer(open(csv_path, "w", newline="", encoding="utf-8")) if csv_path else None  # noqa: SIM115
    video = setup_video_writer_shape(frame_shape, video_path, fps) if video_path else None
    return Sinks(csv=csv_writer, video=video, preview=True)


def safe_imshow(window: str, frame) -> bool:
    try:
        cv2.imshow(window, frame)
        return (cv2.waitKey(1) & 0xFF) != ord("q")
    except Exception:
        return True  # headless環境では無視


def close_windows():
    with contextlib.suppress(Exception):
        cv2.destroyAllWindows()
