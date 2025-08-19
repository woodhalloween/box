import numpy as np
import pytest

from src.tracking.bytetrack_utils import DetectionResults  # Adjust the import path as needed


class TestDetectionResults:
    def test_initial_state(self):
        """Should initialize with zeroed fields"""
        dr = DetectionResults()
        summary = dr.get_summary()
        assert summary["frame_count"] == 0
        for key, value in summary.items():
            if key != "frame_count":
                assert value == 0

    def test_add_single_frame_without_confidence(self):
        """Should add one frame without detection_conf"""
        dr = DetectionResults()
        dr.add_frame_result(
            detection_time=100,
            tracking_time=50,
            fps=30,
            objects_detected=5,
            objects_tracked=4,
            memory_usage=512,
            detection_conf=None,
        )

        assert dr.frame_count == 1
        assert dr.detection_times == [100]
        assert dr.detection_confidence == []  # Not added
        summary = dr.get_summary()
        assert summary["avg_detection_time"] == 100
        assert summary["avg_detection_conf"] == 0  # Nothing added

    def test_add_single_frame_with_confidence(self):
        """Should add one frame with detection_conf"""
        dr = DetectionResults()
        dr.add_frame_result(
            detection_time=120,
            tracking_time=60,
            fps=25,
            objects_detected=6,
            objects_tracked=6,
            memory_usage=768,
            detection_conf=0.8,
        )

        assert dr.frame_count == 1
        assert dr.detection_confidence == [0.8]
        summary = dr.get_summary()
        assert summary["avg_detection_conf"] == pytest.approx(0.8)

    def test_add_multiple_frames(self):
        """Should correctly compute summary over multiple frames"""
        dr = DetectionResults()
        frames = [
            (100, 50, 30, 5, 4, 512, 0.8),
            (200, 60, 28, 7, 6, 600, 0.9),
            (150, 55, 32, 6, 5, 580, 0.85),
        ]

        for frame in frames:
            dr.add_frame_result(*frame)

        assert dr.frame_count == 3
        summary = dr.get_summary()

        assert summary["avg_detection_time"] == pytest.approx(np.mean([100, 200, 150]))
        assert summary["avg_tracking_time"] == pytest.approx(np.mean([50, 60, 55]))
        assert summary["avg_fps"] == pytest.approx(np.mean([30, 28, 32]))
        assert summary["avg_objects_detected"] == pytest.approx(np.mean([5, 7, 6]))
        assert summary["avg_objects_tracked"] == pytest.approx(np.mean([4, 6, 5]))
        assert summary["avg_memory_usage"] == pytest.approx(np.mean([512, 600, 580]))
        assert summary["avg_detection_conf"] == pytest.approx(np.mean([0.8, 0.9, 0.85]))
        assert summary["max_objects_detected"] == 7
        assert summary["max_objects_tracked"] == 6
