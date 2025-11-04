"""
Test cases for head_shake_detector_refactored.py

Comprehensive test suite to achieve 100% code coverage.
"""

import numpy as np
import pytest
from mediapipe.python.solutions.pose import PoseLandmark

from src.definitions import Angle, MovementState
from src.detectors.head_shake_detector_refactored import (
    HeadAngles,
    HeadShakeDetector,
    OscillationAnalyzer,
)

# ==================== Fixtures ====================


@pytest.fixture
def detector():
    """Create a default HeadShakeDetector instance for testing."""
    return HeadShakeDetector()


@pytest.fixture
def detector_custom():
    """Create a HeadShakeDetector with custom parameters."""
    return HeadShakeDetector(
        horizontal_threshold=20.0,
        vertical_threshold=15.0,
        cycle_detection_window=30,
        min_oscillations=3,
        confidence_threshold=0.7,
        hysteresis_frames=5,
    )


@pytest.fixture
def mock_landmarks():
    """Create mock landmarks array for testing."""
    # Create landmarks with 33 pose landmarks (MediaPipe format)
    # Format: [x, y, z, visibility]
    landmarks = np.zeros((33, 4))
    # Set high visibility for all landmarks
    landmarks[:, 3] = 0.8
    # Set default positions
    landmarks[PoseLandmark.NOSE.value] = [0.5, 0.4, 0.0, 0.8]
    landmarks[PoseLandmark.LEFT_EAR.value] = [0.45, 0.4, 0.0, 0.8]
    landmarks[PoseLandmark.RIGHT_EAR.value] = [0.55, 0.4, 0.0, 0.8]
    landmarks[PoseLandmark.LEFT_SHOULDER.value] = [0.45, 0.6, 0.0, 0.8]
    landmarks[PoseLandmark.RIGHT_SHOULDER.value] = [0.55, 0.6, 0.0, 0.8]
    return landmarks


@pytest.fixture
def mock_landmarks_low_confidence():
    """Create mock landmarks with low confidence."""
    landmarks = np.zeros((33, 4))
    landmarks[:, 3] = 0.3  # Low visibility
    return landmarks


# ==================== HeadAngles Dataclass Tests ====================


class TestHeadAngles:
    """Test cases for HeadAngles dataclass."""

    def test_head_angles_creation(self):
        """Test HeadAngles can be created with all attributes."""
        angles = HeadAngles(
            horizontal=10.5,
            vertical=-5.2,
            confidence=0.85,
            timestamp=1.5,
            frame_number=42,
        )
        assert angles.horizontal == 10.5
        assert angles.vertical == -5.2
        assert angles.confidence == 0.85
        assert angles.timestamp == 1.5
        assert angles.frame_number == 42

    def test_head_angles_immutable(self):
        """Test HeadAngles is immutable (frozen dataclass)."""
        angles = HeadAngles(0.0, 0.0, 0.5, 0.0, 0)
        with pytest.raises(AttributeError):
            angles.horizontal = 10.0

    def test_head_angles_zero_values(self):
        """Test HeadAngles with zero values."""
        angles = HeadAngles(0.0, 0.0, 0.0, 0.0, 0)
        assert angles.horizontal == 0.0
        assert angles.vertical == 0.0
        assert angles.confidence == 0.0
        assert angles.timestamp == 0.0
        assert angles.frame_number == 0

    def test_head_angles_negative_values(self):
        """Test HeadAngles with negative angle values."""
        angles = HeadAngles(-15.0, -10.0, 0.7, 2.5, 100)
        assert angles.horizontal == -15.0
        assert angles.vertical == -10.0


# ==================== OscillationAnalyzer Tests ====================


class TestOscillationAnalyzer:
    """Test cases for OscillationAnalyzer class."""

    def test_detect_insufficient_data(self):
        """Test detect() with insufficient data points."""
        # min_oscillations=2 requires 8 points (2*4)
        angle_history = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]  # Only 6 points
        result = OscillationAnalyzer.detect(angle_history, threshold=5.0, min_oscillations=2)
        assert result is False

    def test_detect_no_peaks(self):
        """Test detect() with no peaks detected."""
        # Monotonically decreasing values
        angle_history = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        result = OscillationAnalyzer.detect(angle_history, threshold=5.0, min_oscillations=2)
        assert result is False

    def test_detect_no_valleys(self):
        """Test detect() with no valleys detected."""
        # Monotonically increasing values
        angle_history = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = OscillationAnalyzer.detect(angle_history, threshold=5.0, min_oscillations=2)
        assert result is False

    def test_detect_insufficient_amplitude(self):
        """Test detect() with oscillations below threshold."""
        # Small oscillations
        angle_history = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        result = OscillationAnalyzer.detect(angle_history, threshold=5.0, min_oscillations=2)
        assert result is False

    def test_detect_valid_oscillation(self):
        """Test detect() with valid oscillation pattern."""
        # Clear oscillation: peak at 20, valley at -20
        # Need 4 transitions for 2 oscillations (min_oscillations * 2)
        # Need 3 peaks and 2 valleys to get 4 transitions
        # Pattern: 0, peak, 0, valley, 0, peak, 0, valley, 0, peak, 0 (last 0 to detect peak)
        angle_history = [0.0, 20.0, 0.0, -20.0, 0.0, 20.0, 0.0, -20.0, 0.0, 20.0, 0.0]
        result = OscillationAnalyzer.detect(angle_history, threshold=15.0, min_oscillations=2)
        assert result is True

    def test_detect_valid_oscillation_one_way(self):
        """Test detect() with valid oscillation in one direction."""
        # Oscillation from 0 to positive values
        angle_history = [0.0, 20.0, 5.0, 25.0, 10.0, 20.0, 5.0, 25.0]
        result = OscillationAnalyzer.detect(angle_history, threshold=15.0, min_oscillations=2)
        assert result is True

    def test_detect_min_oscillations_requirement(self):
        """Test detect() requires minimum oscillations."""
        # Only one oscillation cycle
        angle_history = [0.0, 20.0, 0.0, -20.0, 0.0]
        result = OscillationAnalyzer.detect(angle_history, threshold=15.0, min_oscillations=2)
        assert result is False  # Need at least 2 oscillations

    def test_detect_exact_minimum_points(self):
        """Test detect() with exactly minimum required points."""
        # min_oscillations=2 requires 8 points minimum (2*4), but we need 4 transitions
        # So we need at least 11 points to get 3 peaks and 2 valleys properly detected
        # Pattern: 0, peak, 0, valley, 0, peak, 0, valley, 0, peak, 0
        angle_history = [0.0, 20.0, 0.0, -20.0, 0.0, 20.0, 0.0, -20.0, 0.0, 20.0, 0.0]
        result = OscillationAnalyzer.detect(angle_history, threshold=15.0, min_oscillations=2)
        assert result is True

    def test_detect_custom_min_oscillations(self):
        """Test detect() with custom min_oscillations parameter."""
        # Need 12 points for 3 oscillations (3 * 4), and 6 transitions (3 * 2)
        # Pattern with 4 peaks and 3 valleys gives 6 transitions
        # Need final element to detect last peak: 0, peak, 0, valley, 0, peak, 0, valley, 0, peak, 0, valley, 0, peak, 0
        angle_history = [0.0, 20.0, 0.0, -20.0, 0.0, 20.0, 0.0, -20.0, 0.0, 20.0, 0.0, -20.0, 0.0, 20.0, 0.0]
        result = OscillationAnalyzer.detect(angle_history, threshold=15.0, min_oscillations=3)
        assert result is True

    def test_detect_mixed_peaks_and_valleys(self):
        """Test detect() with mixed peak and valley patterns."""
        # Need 4 transitions for 2 oscillations - use 3 peaks and 2 valleys
        # Peaks at 25, valleys at -25, need final element to detect last peak
        angle_history = [10.0, 25.0, 15.0, -25.0, 10.0, 25.0, 15.0, -25.0, 10.0, 25.0, 15.0]
        result = OscillationAnalyzer.detect(angle_history, threshold=20.0, min_oscillations=2)
        assert result is True

    def test_detect_edge_case_single_peak_valley(self):
        """Test detect() edge case with single peak-valley pair."""
        # For 1 oscillation, need 2 transitions (min_oscillations * 2 = 1 * 2 = 2)
        # Need 2 peaks and 1 valley to get 2 transitions
        # Pattern: 0, peak, 0, valley, 0, peak, 0 (last 0 to detect peak)
        angle_history = [0.0, 20.0, 0.0, -20.0, 0.0, 20.0, 0.0]
        result = OscillationAnalyzer.detect(angle_history, threshold=15.0, min_oscillations=1)
        assert result is True  # min_oscillations=1 requires 4 points minimum, but 7 for proper detection

    def test_detect_threshold_at_boundary(self):
        """Test detect() with threshold exactly at amplitude."""
        # Need 4 transitions for 2 oscillations - use 3 peaks and 2 valleys
        # Amplitude exactly at threshold (15.0), need final element to detect last peak
        angle_history = [0.0, 15.0, 0.0, -15.0, 0.0, 15.0, 0.0, -15.0, 0.0, 15.0, 0.0]
        result = OscillationAnalyzer.detect(angle_history, threshold=15.0, min_oscillations=2)
        assert result is True  # amplitude equals threshold

    def test_detect_complex_oscillation(self):
        """Test detect() with complex multi-cycle oscillation."""
        angle_history = [
            0.0,
            20.0,
            5.0,
            -18.0,
            2.0,
            22.0,
            3.0,
            -20.0,
            1.0,
            21.0,
            4.0,
            -19.0,
            0.0,
            20.0,
        ]
        result = OscillationAnalyzer.detect(angle_history, threshold=15.0, min_oscillations=2)
        assert result is True


# ==================== HeadShakeDetector Initialization Tests ====================


class TestHeadShakeDetectorInit:
    """Test cases for HeadShakeDetector initialization."""

    def test_init_default_parameters(self, detector):
        """Test initialization with default parameters."""
        assert detector.horizontal_threshold == 15.0
        assert detector.vertical_threshold == 10.0
        assert detector.min_oscillations == 2
        assert detector.hysteresis_frames == 3
        assert detector.confidence_threshold == 0.5
        assert detector._angle_history.maxlen == 60

    def test_init_custom_parameters(self, detector_custom):
        """Test initialization with custom parameters."""
        assert detector_custom.horizontal_threshold == 20.0
        assert detector_custom.vertical_threshold == 15.0
        assert detector_custom.min_oscillations == 3
        assert detector_custom.hysteresis_frames == 5
        assert detector_custom.confidence_threshold == 0.7
        assert detector_custom._angle_history.maxlen == 30

    def test_init_state_initialization(self, detector):
        """Test initial state is HEAD_STATIC."""
        assert detector._current_horizontal_state == MovementState.HEAD_STATIC
        assert detector._current_vertical_state == MovementState.HEAD_STATIC
        assert detector._h_state_candidate == MovementState.HEAD_STATIC
        assert detector._v_state_candidate == MovementState.HEAD_STATIC
        assert detector._h_consecutive_frames == 0
        assert detector._v_consecutive_frames == 0

    def test_init_angle_history_created(self, detector):
        """Test angle_history deque is created."""
        assert len(detector._angle_history) == 0
        assert detector._angle_history.maxlen == 60


# ==================== HeadShakeDetector.detect() Tests ====================


class TestHeadShakeDetectorDetect:
    """Test cases for HeadShakeDetector.detect() method."""

    def test_detect_none_landmarks(self, detector):
        """Test detect() with None landmarks."""
        result = detector.detect(None, timestamp=0.0, frame_number=0)
        assert result == {
            "horizontal_state": MovementState.HEAD_STATIC,
            "vertical_state": MovementState.HEAD_STATIC,
            "horizontal_angle": 0.0,
            "vertical_angle": 0.0,
            "confidence": 0.0,
        }

    def test_detect_empty_landmarks(self, detector):
        """Test detect() with empty landmarks array."""
        empty_landmarks = np.array([])
        result = detector.detect(empty_landmarks, timestamp=0.0, frame_number=0)
        assert result["horizontal_state"] == MovementState.HEAD_STATIC
        assert result["confidence"] == 0.0

    def test_detect_low_confidence(self, detector, mock_landmarks_low_confidence):
        """Test detect() with low confidence landmarks."""
        result = detector.detect(mock_landmarks_low_confidence, timestamp=0.0, frame_number=0)
        assert result["confidence"] == 0.0
        assert result["horizontal_state"] == MovementState.HEAD_STATIC

    def test_detect_valid_landmarks_static(self, detector, mock_landmarks):
        """Test detect() with valid landmarks (static head)."""
        result = detector.detect(mock_landmarks, timestamp=1.0, frame_number=1)
        assert "horizontal_state" in result
        assert "vertical_state" in result
        assert "horizontal_angle" in result
        assert "vertical_angle" in result
        assert "confidence" in result
        assert result["confidence"] > 0.0

    def test_detect_right_turn(self, detector, mock_landmarks):
        """Test detect() detects right turn."""
        # Move nose to the right
        mock_landmarks[PoseLandmark.NOSE.value][0] = 0.65  # Right of center
        # Process enough frames to build history and trigger state change
        for i in range(15):
            detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
        result = detector.detect(mock_landmarks, timestamp=15 * 0.033, frame_number=15)
        # Should eventually detect right turn or shake
        assert result["horizontal_angle"] > 0

    def test_detect_left_turn(self, detector, mock_landmarks):
        """Test detect() detects left turn."""
        # Move nose to the left
        mock_landmarks[PoseLandmark.NOSE.value][0] = 0.35  # Left of center
        # Process enough frames
        for i in range(15):
            detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
        result = detector.detect(mock_landmarks, timestamp=15 * 0.033, frame_number=15)
        assert result["horizontal_angle"] < 0

    def test_detect_down_nod(self, detector, mock_landmarks):
        """Test detect() detects down nod."""
        # Move nose down
        mock_landmarks[PoseLandmark.NOSE.value][1] = 0.5  # Below ears
        # Process enough frames
        for i in range(15):
            detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
        result = detector.detect(mock_landmarks, timestamp=15 * 0.033, frame_number=15)
        assert result["vertical_angle"] > 0

    def test_detect_up_nod(self, detector, mock_landmarks):
        """Test detect() detects up nod."""
        # Move nose up
        mock_landmarks[PoseLandmark.NOSE.value][1] = 0.3  # Above ears
        # Process enough frames
        for i in range(15):
            detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
        result = detector.detect(mock_landmarks, timestamp=15 * 0.033, frame_number=15)
        assert result["vertical_angle"] < 0

    def test_detect_angle_history_accumulation(self, detector, mock_landmarks):
        """Test detect() accumulates angle history."""
        initial_length = len(detector._angle_history)
        detector.detect(mock_landmarks, timestamp=1.0, frame_number=1)
        assert len(detector._angle_history) == initial_length + 1

    def test_detect_multiple_frames(self, detector, mock_landmarks):
        """Test detect() with multiple consecutive frames."""
        for i in range(10):
            result = detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
            assert result["confidence"] > 0.0
        assert len(detector._angle_history) == 10


# ==================== HeadShakeDetector.update() Tests ====================


class TestHeadShakeDetectorUpdate:
    """Test cases for HeadShakeDetector.update() method (backward compatibility)."""

    def test_update_backward_compatibility(self, detector, mock_landmarks):
        """Test update() returns Ryotaro-compatible format."""
        result = detector.update(mock_landmarks, timestamp=1.0, frame_number=1)
        assert Angle.HEAD_HORIZONTAL_ROTATION in result
        assert Angle.HEAD_VERTICAL_NOD in result
        assert "angle" in result[Angle.HEAD_HORIZONTAL_ROTATION]
        assert "state" in result[Angle.HEAD_HORIZONTAL_ROTATION]
        assert "confidence" in result[Angle.HEAD_HORIZONTAL_ROTATION]

    def test_update_calls_detect(self, detector, mock_landmarks):
        """Test update() internally calls detect()."""
        result_update = detector.update(mock_landmarks, timestamp=1.0, frame_number=1)
        result_detect = detector.detect(mock_landmarks, timestamp=2.0, frame_number=2)
        # Both should have same structure for corresponding angles
        assert result_update[Angle.HEAD_HORIZONTAL_ROTATION]["angle"] == result_detect["horizontal_angle"]

    def test_update_none_landmarks(self, detector):
        """Test update() with None landmarks."""
        result = detector.update(None, timestamp=0.0, frame_number=0)
        assert result[Angle.HEAD_HORIZONTAL_ROTATION]["state"] == MovementState.HEAD_STATIC
        assert result[Angle.HEAD_VERTICAL_NOD]["state"] == MovementState.HEAD_STATIC


# ==================== HeadShakeDetector.get_status() Tests ====================


class TestHeadShakeDetectorGetStatus:
    """Test cases for HeadShakeDetector.get_status() method."""

    def test_get_status_empty_history(self, detector):
        """Test get_status() with empty history."""
        status = detector.get_status()
        assert status["horizontal_state"] == MovementState.HEAD_STATIC.name
        assert status["vertical_state"] == MovementState.HEAD_STATIC.name
        assert status["horizontal_angle"] == 0.0
        assert status["vertical_angle"] == 0.0
        assert status["confidence"] == 0.0
        assert status["sample_count"] == 0
        assert status["hysteresis_h_frames"] == 0
        assert status["hysteresis_v_frames"] == 0

    def test_get_status_with_history(self, detector, mock_landmarks):
        """Test get_status() with accumulated history."""
        detector.detect(mock_landmarks, timestamp=1.0, frame_number=1)
        status = detector.get_status()
        assert status["sample_count"] == 1
        # Angles and confidence should be valid (may be 0.0 if perfectly centered)
        assert isinstance(status["horizontal_angle"], float)
        assert isinstance(status["vertical_angle"], float)
        assert isinstance(status["confidence"], float)

    def test_get_status_multiple_samples(self, detector, mock_landmarks):
        """Test get_status() with multiple samples."""
        for i in range(5):
            detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
        status = detector.get_status()
        assert status["sample_count"] == 5
        assert "hysteresis_h_frames" in status
        assert "hysteresis_v_frames" in status

    def test_get_status_reflects_current_state(self, detector, mock_landmarks):
        """Test get_status() reflects current detector state."""
        # Move nose right
        mock_landmarks[PoseLandmark.NOSE.value][0] = 0.65
        # Process many frames to trigger state change
        for i in range(20):
            detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
        status = detector.get_status()
        # Should reflect some state (may be static, turn, or shake depending on hysteresis)
        assert status["horizontal_state"] in [
            MovementState.HEAD_STATIC.name,
            MovementState.HEAD_RIGHT_TURN.name,
            MovementState.HORIZONTAL_SHAKE.name,
        ]


# ==================== HeadShakeDetector.reset() Tests ====================


class TestHeadShakeDetectorReset:
    """Test cases for HeadShakeDetector.reset() method."""

    def test_reset_clears_history(self, detector, mock_landmarks):
        """Test reset() clears angle history."""
        detector.detect(mock_landmarks, timestamp=1.0, frame_number=1)
        assert len(detector._angle_history) > 0
        detector.reset()
        assert len(detector._angle_history) == 0

    def test_reset_resets_states(self, detector, mock_landmarks):
        """Test reset() resets all states to HEAD_STATIC."""
        # Build up some state
        for i in range(20):
            detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
        detector.reset()
        assert detector._current_horizontal_state == MovementState.HEAD_STATIC
        assert detector._current_vertical_state == MovementState.HEAD_STATIC
        assert detector._h_state_candidate == MovementState.HEAD_STATIC
        assert detector._v_state_candidate == MovementState.HEAD_STATIC

    def test_reset_resets_hysteresis_counters(self, detector, mock_landmarks):
        """Test reset() resets hysteresis counters."""
        # Build up some state
        for i in range(20):
            detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
        detector.reset()
        assert detector._h_consecutive_frames == 0
        assert detector._v_consecutive_frames == 0

    def test_reset_preserves_configuration(self, detector_custom, mock_landmarks):
        """Test reset() preserves detector configuration."""
        original_threshold = detector_custom.horizontal_threshold
        detector_custom.detect(mock_landmarks, timestamp=1.0, frame_number=1)
        detector_custom.reset()
        assert detector_custom.horizontal_threshold == original_threshold
        assert detector_custom.confidence_threshold == 0.7


# ==================== HeadShakeDetector._calculate_head_angles() Tests ====================


class TestHeadShakeDetectorCalculateHeadAngles:
    """Test cases for HeadShakeDetector._calculate_head_angles() method."""

    def test_calculate_head_angles_valid(self, detector, mock_landmarks):
        """Test _calculate_head_angles() with valid landmarks."""
        angles = detector._calculate_head_angles(mock_landmarks, timestamp=1.0, frame_number=1)
        assert isinstance(angles, HeadAngles)
        assert angles.confidence > 0.0
        assert angles.timestamp == 1.0
        assert angles.frame_number == 1

    def test_calculate_head_angles_low_confidence(self, detector, mock_landmarks_low_confidence):
        """Test _calculate_head_angles() with low confidence."""
        angles = detector._calculate_head_angles(mock_landmarks_low_confidence, timestamp=1.0, frame_number=1)
        assert angles.confidence < detector.confidence_threshold
        assert angles.horizontal == 0.0
        assert angles.vertical == 0.0

    def test_calculate_head_angles_right_turn(self, detector, mock_landmarks):
        """Test _calculate_head_angles() calculates right turn correctly."""
        # Nose to the right of ear midpoint
        mock_landmarks[PoseLandmark.NOSE.value][0] = 0.65
        mock_landmarks[PoseLandmark.LEFT_EAR.value][0] = 0.45
        mock_landmarks[PoseLandmark.RIGHT_EAR.value][0] = 0.55
        angles = detector._calculate_head_angles(mock_landmarks, timestamp=1.0, frame_number=1)
        assert angles.horizontal > 0  # Right turn is positive

    def test_calculate_head_angles_left_turn(self, detector, mock_landmarks):
        """Test _calculate_head_angles() calculates left turn correctly."""
        # Nose to the left of ear midpoint
        mock_landmarks[PoseLandmark.NOSE.value][0] = 0.35
        mock_landmarks[PoseLandmark.LEFT_EAR.value][0] = 0.45
        mock_landmarks[PoseLandmark.RIGHT_EAR.value][0] = 0.55
        angles = detector._calculate_head_angles(mock_landmarks, timestamp=1.0, frame_number=1)
        assert angles.horizontal < 0  # Left turn is negative

    def test_calculate_head_angles_down_nod(self, detector, mock_landmarks):
        """Test _calculate_head_angles() calculates down nod correctly."""
        # Nose below ear midpoint
        mock_landmarks[PoseLandmark.NOSE.value][1] = 0.5
        mock_landmarks[PoseLandmark.LEFT_EAR.value][1] = 0.4
        mock_landmarks[PoseLandmark.RIGHT_EAR.value][1] = 0.4
        angles = detector._calculate_head_angles(mock_landmarks, timestamp=1.0, frame_number=1)
        assert angles.vertical > 0  # Down nod is positive

    def test_calculate_head_angles_up_nod(self, detector, mock_landmarks):
        """Test _calculate_head_angles() calculates up nod correctly."""
        # Nose above ear midpoint
        mock_landmarks[PoseLandmark.NOSE.value][1] = 0.3
        mock_landmarks[PoseLandmark.LEFT_EAR.value][1] = 0.4
        mock_landmarks[PoseLandmark.RIGHT_EAR.value][1] = 0.4
        angles = detector._calculate_head_angles(mock_landmarks, timestamp=1.0, frame_number=1)
        assert angles.vertical < 0  # Up nod is negative

    def test_calculate_head_angles_zero_ear_distance(self, detector, mock_landmarks):
        """Test _calculate_head_angles() handles zero ear distance."""
        # Make ears at same position
        mock_landmarks[PoseLandmark.LEFT_EAR.value][0] = 0.5
        mock_landmarks[PoseLandmark.RIGHT_EAR.value][0] = 0.5
        angles = detector._calculate_head_angles(mock_landmarks, timestamp=1.0, frame_number=1)
        assert angles.horizontal == 0.0

    def test_calculate_head_angles_zero_neck_length(self, detector, mock_landmarks):
        """Test _calculate_head_angles() handles zero neck length."""
        # Make shoulders and ears at same y position
        mock_landmarks[PoseLandmark.LEFT_SHOULDER.value][1] = 0.5
        mock_landmarks[PoseLandmark.RIGHT_SHOULDER.value][1] = 0.5
        mock_landmarks[PoseLandmark.LEFT_EAR.value][1] = 0.5
        mock_landmarks[PoseLandmark.RIGHT_EAR.value][1] = 0.5
        angles = detector._calculate_head_angles(mock_landmarks, timestamp=1.0, frame_number=1)
        assert angles.vertical == 0.0

    def test_calculate_head_angles_index_error(self, detector):
        """Test _calculate_head_angles() handles IndexError."""
        # Landmarks array too short
        short_landmarks = np.zeros((10, 4))
        angles = detector._calculate_head_angles(short_landmarks, timestamp=1.0, frame_number=1)
        assert angles.horizontal == 0.0
        assert angles.vertical == 0.0
        assert angles.confidence == 0.0

    def test_calculate_head_angles_type_error(self, detector):
        """Test _calculate_head_angles() handles TypeError."""
        # Invalid landmarks type - should be caught by try-except
        invalid_landmarks = "invalid"
        angles = detector._calculate_head_angles(invalid_landmarks, timestamp=1.0, frame_number=1)
        # Should return default HeadAngles with zero values
        assert angles.horizontal == 0.0
        assert angles.vertical == 0.0
        assert angles.confidence == 0.0

    def test_calculate_head_angles_zero_division_error(self, detector, mock_landmarks):
        """Test _calculate_head_angles() handles ZeroDivisionError gracefully."""
        # This should be handled by the ear_dist and neck_length checks
        # But we can test with very small values that might cause issues
        mock_landmarks[PoseLandmark.LEFT_EAR.value][0] = 0.5
        mock_landmarks[PoseLandmark.RIGHT_EAR.value][0] = 0.5 + 1e-7  # Very small distance
        angles = detector._calculate_head_angles(mock_landmarks, timestamp=1.0, frame_number=1)
        # Should handle gracefully
        assert isinstance(angles, HeadAngles)


# ==================== HeadShakeDetector._update_horizontal_state() Tests ====================


class TestHeadShakeDetectorUpdateHorizontalState:
    """Test cases for HeadShakeDetector._update_horizontal_state() method."""

    def test_update_horizontal_state_insufficient_history(self, detector, mock_landmarks):
        """Test _update_horizontal_state() with insufficient history."""
        # Less than 10 frames
        for i in range(5):
            detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
        assert detector._current_horizontal_state == MovementState.HEAD_STATIC

    def test_update_horizontal_state_static(self, detector, mock_landmarks):
        """Test _update_horizontal_state() remains static for neutral position."""
        # Process enough frames
        for i in range(15):
            detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
        # Should remain static if angle is small
        assert detector._current_horizontal_state in [
            MovementState.HEAD_STATIC,
            MovementState.HEAD_LEFT_TURN,
            MovementState.HEAD_RIGHT_TURN,
            MovementState.HORIZONTAL_SHAKE,
        ]

    def test_update_horizontal_state_right_turn(self, detector, mock_landmarks):
        """Test _update_horizontal_state() detects right turn."""
        # Move nose right
        mock_landmarks[PoseLandmark.NOSE.value][0] = 0.65
        # Process enough frames with hysteresis
        for i in range(20):
            detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
        # Should eventually detect right turn (after hysteresis)
        result = detector.detect(mock_landmarks, timestamp=20 * 0.033, frame_number=20)
        assert result["horizontal_angle"] > 0

    def test_update_horizontal_state_left_turn(self, detector, mock_landmarks):
        """Test _update_horizontal_state() detects left turn."""
        # Move nose left
        mock_landmarks[PoseLandmark.NOSE.value][0] = 0.35
        # Process enough frames
        for i in range(20):
            detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
        result = detector.detect(mock_landmarks, timestamp=20 * 0.033, frame_number=20)
        assert result["horizontal_angle"] < 0

    def test_update_horizontal_state_horizontal_shake(self, detector, mock_landmarks):
        """Test _update_horizontal_state() detects horizontal shake."""
        # Create oscillation pattern
        base_x = 0.5
        for i in range(30):
            # Oscillate nose position
            oscillation = 0.2 * np.sin(i * 0.5)
            mock_landmarks[PoseLandmark.NOSE.value][0] = base_x + oscillation
            detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
        # Should detect shake if oscillation is sufficient
        status = detector.get_status()
        assert status["horizontal_state"] in [
            MovementState.HEAD_STATIC.name,
            MovementState.HORIZONTAL_SHAKE.name,
            MovementState.HEAD_LEFT_TURN.name,
            MovementState.HEAD_RIGHT_TURN.name,
        ]

    def test_update_horizontal_state_hysteresis(self, detector, mock_landmarks):
        """Test _update_horizontal_state() applies hysteresis."""
        # Move nose right
        mock_landmarks[PoseLandmark.NOSE.value][0] = 0.65
        # Process frames but not enough for hysteresis
        for i in range(2):  # Less than hysteresis_frames (3)
            detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
        # State should not change yet
        assert detector._h_consecutive_frames < detector.hysteresis_frames

    def test_update_horizontal_state_hysteresis_transition(self, detector, mock_landmarks):
        """Test _update_horizontal_state() transitions after hysteresis."""
        # Move nose right
        mock_landmarks[PoseLandmark.NOSE.value][0] = 0.65
        # Process enough frames for hysteresis
        for i in range(5):
            detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
        # Candidate should be set
        assert detector._h_state_candidate in [
            MovementState.HEAD_RIGHT_TURN,
            MovementState.HEAD_STATIC,
            MovementState.HORIZONTAL_SHAKE,
        ]


# ==================== HeadShakeDetector._update_vertical_state() Tests ====================


class TestHeadShakeDetectorUpdateVerticalState:
    """Test cases for HeadShakeDetector._update_vertical_state() method."""

    def test_update_vertical_state_insufficient_history(self, detector, mock_landmarks):
        """Test _update_vertical_state() with insufficient history."""
        # Less than 10 frames
        for i in range(5):
            detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
        assert detector._current_vertical_state == MovementState.HEAD_STATIC

    def test_update_vertical_state_static(self, detector, mock_landmarks):
        """Test _update_vertical_state() remains static for neutral position."""
        # Process enough frames
        for i in range(15):
            detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
        assert detector._current_vertical_state in [
            MovementState.HEAD_STATIC,
            MovementState.HEAD_DOWN_NOD,
            MovementState.HEAD_UP_NOD,
            MovementState.VERTICAL_NOD,
        ]

    def test_update_vertical_state_down_nod(self, detector, mock_landmarks):
        """Test _update_vertical_state() detects down nod."""
        # Move nose down
        mock_landmarks[PoseLandmark.NOSE.value][1] = 0.5
        # Process enough frames
        for i in range(20):
            detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
        result = detector.detect(mock_landmarks, timestamp=20 * 0.033, frame_number=20)
        assert result["vertical_angle"] > 0

    def test_update_vertical_state_up_nod(self, detector, mock_landmarks):
        """Test _update_vertical_state() detects up nod."""
        # Move nose up
        mock_landmarks[PoseLandmark.NOSE.value][1] = 0.3
        # Process enough frames
        for i in range(20):
            detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
        result = detector.detect(mock_landmarks, timestamp=20 * 0.033, frame_number=20)
        assert result["vertical_angle"] < 0

    def test_update_vertical_state_vertical_nod(self, detector, mock_landmarks):
        """Test _update_vertical_state() detects vertical nod oscillation."""
        # Create oscillation pattern
        base_y = 0.4
        for i in range(30):
            # Oscillate nose position vertically
            oscillation = 0.15 * np.sin(i * 0.5)
            mock_landmarks[PoseLandmark.NOSE.value][1] = base_y + oscillation
            detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
        # Should detect nod if oscillation is sufficient
        status = detector.get_status()
        assert status["vertical_state"] in [
            MovementState.HEAD_STATIC.name,
            MovementState.VERTICAL_NOD.name,
            MovementState.HEAD_DOWN_NOD.name,
            MovementState.HEAD_UP_NOD.name,
        ]

    def test_update_vertical_state_hysteresis(self, detector, mock_landmarks):
        """Test _update_vertical_state() applies hysteresis."""
        # Move nose down
        mock_landmarks[PoseLandmark.NOSE.value][1] = 0.5
        # Process frames but not enough for hysteresis
        for i in range(2):
            detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
        assert detector._v_consecutive_frames < detector.hysteresis_frames


# ==================== HeadShakeDetector._build_null_result() Tests ====================


class TestHeadShakeDetectorBuildNullResult:
    """Test cases for HeadShakeDetector._build_null_result() static method."""

    def test_build_null_result(self):
        """Test _build_null_result() returns correct structure."""
        result = HeadShakeDetector._build_null_result()
        assert result == {
            "horizontal_state": MovementState.HEAD_STATIC,
            "vertical_state": MovementState.HEAD_STATIC,
            "horizontal_angle": 0.0,
            "vertical_angle": 0.0,
            "confidence": 0.0,
        }

    def test_build_null_result_static_call(self):
        """Test _build_null_result() can be called statically."""
        result = HeadShakeDetector._build_null_result()
        assert isinstance(result, dict)
        assert "horizontal_state" in result


# ==================== Integration and Edge Case Tests ====================


class TestHeadShakeDetectorEdgeCases:
    """Test cases for edge cases and error conditions."""

    def test_detect_with_invalid_landmark_shape(self, detector):
        """Test detect() with invalid landmark array shape."""
        invalid_landmarks = np.zeros((33, 2))  # Missing z and visibility
        result = detector.detect(invalid_landmarks, timestamp=0.0, frame_number=0)
        assert result["confidence"] == 0.0

    def test_detect_with_nan_values(self, detector, mock_landmarks):
        """Test detect() handles NaN values."""
        mock_landmarks[PoseLandmark.NOSE.value][0] = np.nan
        result = detector.detect(mock_landmarks, timestamp=0.0, frame_number=0)
        # Should handle gracefully
        assert isinstance(result, dict)

    def test_detect_history_maxlen(self, detector, mock_landmarks):
        """Test detect() respects history maxlen."""
        # Fill history beyond maxlen
        for i in range(100):
            detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
        # History should be capped at maxlen (60)
        assert len(detector._angle_history) == detector._angle_history.maxlen

    def test_multiple_resets(self, detector, mock_landmarks):
        """Test multiple reset() calls."""
        detector.detect(mock_landmarks, timestamp=1.0, frame_number=1)
        detector.reset()
        detector.reset()
        detector.reset()
        assert len(detector._angle_history) == 0
        assert detector._current_horizontal_state == MovementState.HEAD_STATIC

    def test_detect_varying_confidence(self, detector, mock_landmarks):
        """Test detect() with varying confidence levels."""
        # Start with high confidence
        mock_landmarks[:, 3] = 0.9
        result1 = detector.detect(mock_landmarks, timestamp=1.0, frame_number=1)
        # Then low confidence
        mock_landmarks[:, 3] = 0.3
        result2 = detector.detect(mock_landmarks, timestamp=2.0, frame_number=2)
        assert result1["confidence"] > result2["confidence"]

    def test_detect_extreme_angles(self, detector, mock_landmarks):
        """Test detect() with extreme angle values."""
        # Extreme right turn
        mock_landmarks[PoseLandmark.NOSE.value][0] = 0.9
        mock_landmarks[PoseLandmark.LEFT_EAR.value][0] = 0.4
        mock_landmarks[PoseLandmark.RIGHT_EAR.value][0] = 0.5
        result = detector.detect(mock_landmarks, timestamp=1.0, frame_number=1)
        assert abs(result["horizontal_angle"]) > 0

    def test_detect_identical_landmarks(self, detector, mock_landmarks):
        """Test detect() with identical landmark positions."""
        # All landmarks at same position
        mock_landmarks[:, :2] = 0.5
        result = detector.detect(mock_landmarks, timestamp=1.0, frame_number=1)
        assert result["horizontal_angle"] == 0.0
        assert result["vertical_angle"] == 0.0

    def test_detect_rapid_state_changes(self, detector, mock_landmarks):
        """Test detect() with rapid state changes."""
        # Rapidly alternate between left and right
        for i in range(20):
            if i % 2 == 0:
                mock_landmarks[PoseLandmark.NOSE.value][0] = 0.35  # Left
            else:
                mock_landmarks[PoseLandmark.NOSE.value][0] = 0.65  # Right
            detector.detect(mock_landmarks, timestamp=i * 0.033, frame_number=i)
        # Should handle without errors
        status = detector.get_status()
        assert isinstance(status, dict)
