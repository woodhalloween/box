"""
Example script demonstrating the usage of main_detector.py for video processing.

This script shows how to use the main_detector module programmatically to process
videos with the integrated detection system (User Classification, Dwell Time,
Posture Monitoring, etc.) using the VideoProcessor class.

Unlike example_debug_csv.py which uses the video_processor.process_video function,
this script uses the main_detector.process_video function which provides a
cleaner interface with configurable parameters.
"""

import os

from src.main_detector import process_video


def main():
    # Example video file
    # video_path = "data/videos/776928198.044893.mp4"
    # video_path = "data/videos/781183819.498211.mp4"

    # Head Shake video file
    video_path = "data/videos/783221473.644636.mp4"
    # video_path = "data/videos/783221566.097142.mp4"
    # video_path = "data/videos/783221473.683117.mp4"

    # Automatically derive base name without extension and directory
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # Standard outputs
    output_csv_path = f"output/{base_name}_results.csv"
    output_video_path = f"output/{base_name}_output.mp4"

    # Process video using main_detector.process_video
    # This function provides a clean interface with configurable parameters
    # for all detection components (DwellTimeDetector, PostureMonitor, UserClassifier)
    process_video(
        video_path=video_path,
        output_csv_path=output_csv_path,
        output_video_path=output_video_path,
        disable_japanese=False,
        # DwellTimeDetector parameters
        stay_threshold_sec=60.0,  # Threshold for "long stay" detection (seconds)
        spike_threshold=1.5,  # Movement spike detection threshold (torso length ratio)
        stability_threshold_px=50.0,  # Detection stability threshold (pixels)
        grace_period_sec=1.5,  # Grace period for movement detection (seconds)
        # PostureMonitor parameters
        pm_monitoring_duration_sec=60.0,  # Posture monitoring duration (seconds)
        pm_alert_threshold_ratio=0.7,  # Alert threshold for forward posture ratio
        # UserClassifier parameters
        uc_threshold_deg=90.0,  # User classification alert threshold (degrees)
        uc_moving_window_seconds=5.0,  # Moving average window for knee angle (seconds)
    )

    print("\n✅ Processing complete!")
    print(f"   Results CSV: {output_csv_path}")
    print(f"   Output video: {output_video_path}")
    print("\n📊 The results CSV contains:")
    print("   - User classification (wheelchair/standing status)")
    print("   - Dwell time detection (long stay status, duration)")
    print("   - Posture monitoring (alerts, forward posture ratio)")
    print("   - Frame-by-frame analysis results")
    print("\n💡 Note: For debug CSV output with frame-by-frame detection states,")
    print("   use example_debug_csv.py instead, which uses video_processor.process_video")


if __name__ == "__main__":
    main()
