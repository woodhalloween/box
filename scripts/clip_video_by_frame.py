import argparse
import os
import sys

import cv2


def clip_video_opencv(video_path, output_path, start_frame, end_frame):
    # 入力ファイルの存在チェック
    if not os.path.exists(video_path):
        print(f"Error: Input file does not exist: {video_path}", file=sys.stderr)
        sys.exit(1)

    # フレーム範囲の妥当性チェック
    if start_frame < 0:
        print("Error: Start frame must be a non-negative number.", file=sys.stderr)
        sys.exit(1)
    if end_frame <= start_frame:
        print("Error: End frame must be greater than start frame.", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # フレーム範囲が動画の長さを超えていないかチェック
    if start_frame >= total_frames:
        print(
            f"Error: Start frame ({start_frame}) is beyond the total number of frames ({total_frames}).",
            file=sys.stderr,
        )
        sys.exit(1)

    # 調整された終了フレーム
    end_frame = min(end_frame, total_frames - 1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()
    print(f"動画は正常に切り取られ、'{output_path}'として保存されました。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="指定されたフレーム範囲で動画を切り取ります。")
    parser.add_argument("--input", type=str, help="入力動画ファイルのパス")
    parser.add_argument("--output", type=str, help="出力動画ファイルのパス")
    parser.add_argument("--start", type=int, help="切り取り開始フレーム番号")
    parser.add_argument("--end", type=int, help="切り取り終了フレーム番号")

    args = parser.parse_args()

    clip_video_opencv(args.input, args.output, args.start, args.end)
