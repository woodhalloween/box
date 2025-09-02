import os

import cv2
import numpy as np


def create_dummy_video(
    output_path="tests/assets/dummy_video.mp4",
    width=100,
    height=100,
    duration_sec=5,
    fps=30,
):
    """
    テスト用のダミー動画ファイルを生成する.
    動画にはフレーム番号が描画される.
    """
    if os.path.exists(output_path):
        print(f"Dummy video already exists at {output_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise OSError(f"Cannot open video writer for {output_path}")

    total_frames = duration_sec * fps
    for i in range(total_frames):
        # 黒い背景を作成
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # フレーム番号を描画
        text = f"F:{i}"
        cv2.putText(
            frame,
            text,
            (10, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        out.write(frame)

    out.release()
    print(f"Dummy video created at {output_path}")


if __name__ == "__main__":
    os.makedirs("tests/assets", exist_ok=True)
    create_dummy_video()
