"""
Example script demonstrating the debug CSV feature for detection analysis.

This script shows how to use the new debug_csv_path parameter to output
detailed debug information for all detections (Posture, Dwell, Head Shake,
Hand Raise, and User Classification).
"""

import os

from src.video_processor import process_video


def main():
    # Example video file
    # video_path = "data/videos/776928198.044893.mp4"
    video_path = "data/videos/781183819.498211.mp4"

    # Automatically derive base name without extension and directory
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # Standard outputs
    output_csv_path = f"output/{base_name}_results.csv"
    output_video_path = f"output/{base_name}_output.mp4"

    # NEW: Debug CSV output path for all detection states
    debug_csv_path = f"output/{base_name}_debug.csv"

    # Process video with debug CSV enabled
    process_video(
        video_path=video_path,
        output_csv_path=output_csv_path,
        output_video_path=output_video_path,
        disable_japanese=False,
        # Enable debug CSV output
        debug_csv_path=debug_csv_path,
        # Other parameters (using defaults)
        stay_threshold_sec=60.0,
        preview=False,  # Set to True to see real-time preview
        show_progress=True,  # Show progress bar
    )

    print("\n✅ Processing complete!")
    print(f"   Results CSV: {output_csv_path}")
    print(f"   Debug CSV: {debug_csv_path}")
    print(f"   Output video: {output_video_path}")
    print("\n📊 The debug CSV contains frame-by-frame detection states:")
    print("   - Posture monitoring (alerts, bad/total counts)")
    print("   - Dwell time detection (long stay status, duration)")
    print("   - Head shake detection (alerts)")
    print("   - Hand raise detection (left/right/both)")
    print("   - User classification (wheelchair/standing status)")


if __name__ == "__main__":
    main()
