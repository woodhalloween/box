import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from src.hand_raise_detector import HandRaiseDetector
from src.pose_estimator import PoseEstimator
from src.video_processor import VideoProcessor


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Detect hand-raising gestures in a video.")
    parser.add_argument(
        "--input_video",
        type=str,
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to the output CSV file to save results.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/yolov8n-pose.pt",
        help="Path to the YOLOv8 pose estimation model.",
    )
    return parser.parse_args()


def main():
    """Main function to run the hand-raise detection process."""
    args = parse_args()

    video_path = Path(args.input_video)
    if not video_path.exists():
        print(f"Error: Input video not found at {video_path}")
        return

    # 1. Initialize detectors
    pose_estimator = PoseEstimator(model_path=args.model_path)

    video_processor = VideoProcessor(str(video_path))
    fps = video_processor.fps
    hand_raise_detector = HandRaiseDetector(fps=fps)

    all_hand_raise_events = []

    # 2. Process video frame by frame
    progress_bar = tqdm(total=video_processor.total_frames, desc="Processing frames")
    for frame_number, frame in enumerate(video_processor.process_video()):
        # Perform pose estimation to get tracks
        tracks = pose_estimator.estimate_pose(frame)

        # Process the current frame for hand raises
        completed_events = hand_raise_detector.process_frame(frame_number, tracks)

        if completed_events:
            all_hand_raise_events.extend(completed_events)
            print(f"Frame {frame_number}: Detected {len(completed_events)} new hand-raise event(s).")

        progress_bar.update(1)

    progress_bar.close()

    # 3. Save results
    save_results_to_csv(args.output_csv, all_hand_raise_events)
    print(f"\nDetection complete. Found {len(all_hand_raise_events)} events. Results saved to {args.output_csv}")


def save_results_to_csv(output_path: str, events: list[dict]):
    """Saves the detected events to a CSV file."""
    if not events:
        print("No events to save.")
        return

    header = events[0].keys()
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(events)


if __name__ == "__main__":
    main()
