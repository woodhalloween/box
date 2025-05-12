"""create_landmark_video.py
入力ビデオからMediaPipe Poseを使って骨格のランドマークだけを描画した新しい動画を出力します。
Usage:
    python analysis/create_landmark_video.py input.mp4 --output landmarks.mp4 --codec mp4v
"""

import argparse
import os

import cv2
import mediapipe as mp

# C++側（glog）とTensorFlow/Mediapipeのログを抑制
os.environ["GLOG_MINLOGLEVEL"] = "2"  # INFO以下を隐藏
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # TensorFlowのINFO/WARNINGを抑制
from absl import logging as absl_logging

# abslライブラリの標準エラー出力を無効化し、ログレベルをERRORに設定
# これにより、MediapipeやTensorFlowが出力する大量のINFOレベルのログを抑制
absl_logging._warn_preinit_stderr = False
absl_logging.set_verbosity(absl_logging.ERROR)


def main():
    parser = argparse.ArgumentParser(description="Create video with only stick figure landmarks")
    parser.add_argument("input", help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to output video")
    parser.add_argument("--codec", default="mp4v", help="FourCC codec, e.g. mp4v, XVID")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise OSError(f"Cannot open video {args.input}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            # 空の黒画像作成
            blank = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blank = cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    blank,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

            writer.write(blank)
    finally:
        cap.release()
        writer.release()
        pose.close()


if __name__ == "__main__":
    main()
