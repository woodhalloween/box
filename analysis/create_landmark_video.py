"""create_landmark_video.py
入力ビデオからMediaPipe Poseを使って骨格のランドマークだけを描画した新しい動画を出力します。
Usage:
    python analysis/create_landmark_video.py input.mp4 --output landmarks.mp4 --codec mp4v
"""
import argparse
import os
import cv2
import mediapipe as mp

def main():
    parser = argparse.ArgumentParser(description='Create video with only stick figure landmarks')
    parser.add_argument('input', help='Path to input video')
    parser.add_argument('--output', required=True, help='Path to output video')
    parser.add_argument('--codec', default='mp4v', help='FourCC codec, e.g. mp4v, XVID')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {args.input}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        smooth_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
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
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())

            writer.write(blank)
    finally:
        cap.release()
        writer.release()
        pose.close()

if __name__ == '__main__':
    main() 