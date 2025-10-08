import cv2


def estimate_total_frames(
    video_path: str,
    fps: float,
    show_progress: bool,
) -> int | None:
    """
    Estimate total frame count for progress bar.
    Adjusts for FPS changes if FFmpeg is involved.

    Returns
    -------
    int | None
        Estimated total frame count, or None if estimation fails.
    """
    if not show_progress:
        return None

    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            original_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            if original_fps > 0 and fps > 0 and abs(original_fps - fps) > 0.1:
                duration_seconds = original_frame_count / original_fps
                calculated_frames = duration_seconds * fps
                total_frames = round(calculated_frames)
                if calculated_frames - int(calculated_frames) >= 0.5:
                    total_frames += 1
                print(
                    f"Progress bar: Adjusted frame count from {original_frame_count} to {total_frames} "
                    f"(fps: {original_fps} -> {fps}, calculated: {calculated_frames:.2f})"
                )
                return total_frames

            print(f"Progress bar: Using original frame count {original_frame_count} (fps: {original_fps})")
            return original_frame_count
    except Exception:
        pass  # If we can't get frame count, progress bar will work without total

    return None
