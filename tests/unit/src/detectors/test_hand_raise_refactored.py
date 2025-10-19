"""
Test cases for hand_raise_refactored.py
"""

import numpy as np
import pytest

from src.detectors.hand_raise_refactored import HandRaiseDetector


class TestHandRaiseDetector:
    """Test cases for HandRaiseDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a default HandRaiseDetector instance for testing."""
        return HandRaiseDetector()

    @pytest.fixture
    def detector_custom(self):
        """Create a HandRaiseDetector with custom parameters."""
        return HandRaiseDetector(
            visibility_threshold=0.7, min_consecutive_frames=5, vertical_margin=0.1, history_window=10
        )

    @pytest.fixture
    def mock_landmarks(self):
        """Create mock landmarks array for testing."""
        # Create landmarks with 33 pose landmarks (MediaPipe format)
        landmarks = np.zeros((33, 4))
        # Set visibility scores for all landmarks
        landmarks[:, 3] = 0.8  # High visibility
        return landmarks

    @pytest.fixture
    def mock_landmarks_low_visibility(self):
        """Create mock landmarks with low visibility."""
        landmarks = np.zeros((33, 4))
        landmarks[:, 3] = 0.3  # Low visibility
        return landmarks

    def test_get_status_default_state(self, detector):
        """Test get_status() returns correct default state."""
        status = detector.get_status()

        # Check main status fields
        assert status["left_hand_raised"] is False
        assert status["right_hand_raised"] is False
        assert status["any_hand_raised"] is False

        # Check configuration fields
        assert status["min_consecutive_frames"] == 3
        assert status["vertical_margin"] == 0.0
        assert status["visibility_threshold"] == 0.5
        assert status["history_window"] == 3

        # Check hands structure
        assert "hands" in status
        assert "left" in status["hands"]
        assert "right" in status["hands"]

    def test_get_status_custom_configuration(self, detector_custom):
        """Test get_status() with custom configuration."""
        status = detector_custom.get_status()

        assert status["min_consecutive_frames"] == 5
        assert status["vertical_margin"] == 0.1
        assert status["visibility_threshold"] == 0.7
        assert status["history_window"] == 10

    def test_get_status_after_detection(self, detector, mock_landmarks):
        """Test get_status() after hand detection."""
        # Set up landmarks for hand raising
        # Left shoulder (11) and left wrist (15)
        mock_landmarks[11] = [0.5, 0.3, 0.0, 0.8]  # shoulder
        mock_landmarks[15] = [0.5, 0.1, 0.0, 0.8]  # wrist (above shoulder)

        # Right shoulder (12) and right wrist (16)
        mock_landmarks[12] = [0.5, 0.3, 0.0, 0.8]  # shoulder
        mock_landmarks[16] = [0.5, 0.1, 0.0, 0.8]  # wrist (above shoulder)

        # Process multiple frames to trigger hand raising
        for _ in range(5):
            detector.detect(mock_landmarks)

        status = detector.get_status()

        # Both hands should be raised
        assert status["left_hand_raised"] is True
        assert status["right_hand_raised"] is True
        assert status["any_hand_raised"] is True

    def test_get_status_partial_detection(self, detector, mock_landmarks):
        """Test get_status() with only one hand raised."""
        # Set up only left hand raising
        mock_landmarks[11] = [0.5, 0.3, 0.0, 0.8]  # left shoulder
        mock_landmarks[15] = [0.5, 0.1, 0.0, 0.8]  # left wrist (above shoulder)

        # Right hand not visible
        mock_landmarks[12] = [0.5, 0.3, 0.0, 0.3]  # right shoulder (low visibility)
        mock_landmarks[16] = [0.5, 0.1, 0.0, 0.3]  # right wrist (low visibility)

        # Process frames
        for _ in range(5):
            detector.detect(mock_landmarks)

        status = detector.get_status()

        assert status["left_hand_raised"] is True
        assert status["right_hand_raised"] is False
        assert status["any_hand_raised"] is True

    def test_get_status_no_detection(self, detector, mock_landmarks_low_visibility):
        """Test get_status() with no hand detection."""
        # Process frames with low visibility
        for _ in range(5):
            detector.detect(mock_landmarks_low_visibility)

        status = detector.get_status()

        assert status["left_hand_raised"] is False
        assert status["right_hand_raised"] is False
        assert status["any_hand_raised"] is False

    def test_get_status_hands_structure(self, detector):
        """Test get_status() hands structure contains expected fields."""
        status = detector.get_status()

        for hand in ["left", "right"]:
            hand_data = status["hands"][hand]

            # Check all expected fields are present
            expected_fields = [
                "is_raised",
                "visible",
                "consecutive_frames",
                "visibility_score",
                "vertical_delta",
                "history",
                "raised_ratio",
            ]
            for field in expected_fields:
                assert field in hand_data

    def test_serialize_hand_state_default(self, detector):
        """Test _serialize_hand_state() with default state."""
        left_state = detector._serialize_hand_state("left")
        right_state = detector._serialize_hand_state("right")

        # Both hands should have same default state
        for state in [left_state, right_state]:
            assert state["is_raised"] is False
            assert state["visible"] is False
            assert state["consecutive_frames"] == 0
            assert state["visibility_score"] == 0.0
            assert state["vertical_delta"] == 0.0
            assert state["history"] == []
            assert state["raised_ratio"] == 0.0

    def test_serialize_hand_state_after_detection(self, detector, mock_landmarks):
        """Test _serialize_hand_state() after hand detection."""
        # Set up left hand raising
        mock_landmarks[11] = [0.5, 0.3, 0.0, 0.8]  # left shoulder
        mock_landmarks[15] = [0.5, 0.1, 0.0, 0.8]  # left wrist (above shoulder)

        # Process frames to build up state
        for _ in range(5):
            detector.detect(mock_landmarks)

        left_state = detector._serialize_hand_state("left")

        assert left_state["is_raised"] is True
        assert left_state["visible"] is True
        assert left_state["consecutive_frames"] == 5
        assert left_state["visibility_score"] > 0
        assert left_state["vertical_delta"] > 0
        assert len(left_state["history"]) == 3  # maxlen is 3 (min_consecutive_frames)
        assert left_state["raised_ratio"] == 1.0  # All frames raised

    def test_serialize_hand_state_mixed_history(self, detector, mock_landmarks):
        """Test _serialize_hand_state() with mixed detection history."""
        # Set up landmarks for intermittent detection
        mock_landmarks[11] = [0.5, 0.3, 0.0, 0.8]  # left shoulder
        mock_landmarks[15] = [0.5, 0.1, 0.0, 0.8]  # left wrist (above shoulder)

        # Process frames with alternating visibility
        for i in range(6):
            if i % 2 == 0:
                # Even frames: hand raised
                mock_landmarks[11] = [0.5, 0.3, 0.0, 0.8]
                mock_landmarks[15] = [0.5, 0.1, 0.0, 0.8]  # above shoulder
            else:
                # Odd frames: hand not raised (wrist below shoulder)
                mock_landmarks[11] = [0.5, 0.3, 0.0, 0.8]
                mock_landmarks[15] = [0.5, 0.5, 0.0, 0.8]  # below shoulder

            detector.detect(mock_landmarks)

        left_state = detector._serialize_hand_state("left")

        # Should not be raised due to inconsistent detection
        assert left_state["is_raised"] is False
        assert left_state["consecutive_frames"] < detector.min_consecutive_frames
        assert len(left_state["history"]) == 3  # maxlen is 3 (min_consecutive_frames)
        assert 0.0 < left_state["raised_ratio"] < 1.0

    def test_serialize_hand_state_empty_history(self, detector):
        """Test _serialize_hand_state() with empty history."""
        # Clear history manually
        detector._hand_histories["left"].clear()

        left_state = detector._serialize_hand_state("left")

        assert left_state["history"] == []
        assert left_state["raised_ratio"] == 0.0

    def test_serialize_hand_state_invalid_hand(self, detector):
        """Test _serialize_hand_state() with invalid hand name."""
        with pytest.raises(KeyError):
            detector._serialize_hand_state("invalid_hand")

    def test_reset_default_state(self, detector):
        """Test reset() returns detector to default state."""
        # Modify state first
        detector._hand_states["left"].is_raised = True
        detector._hand_states["left"].consecutive_frames = 5
        detector._hand_states["left"].visible = True
        detector._hand_states["left"].visibility_score = 0.8
        detector._hand_states["left"].last_vertical_delta = 0.2

        detector._hand_histories["left"].extend([True, True, True])

        # Reset
        detector.reset()

        # Check left hand state
        left_state = detector._hand_states["left"]
        assert left_state.is_raised is False
        assert left_state.consecutive_frames == 0
        assert left_state.visible is False
        assert left_state.visibility_score == 0.0
        assert left_state.last_vertical_delta == 0.0

        # Check left hand history
        assert len(detector._hand_histories["left"]) == 0

        # Check right hand state (should also be reset)
        right_state = detector._hand_states["right"]
        assert right_state.is_raised is False
        assert right_state.consecutive_frames == 0
        assert right_state.visible is False
        assert right_state.visibility_score == 0.0
        assert right_state.last_vertical_delta == 0.0

        # Check right hand history
        assert len(detector._hand_histories["right"]) == 0

    def test_reset_after_detection(self, detector, mock_landmarks):
        """Test reset() after successful detection."""
        # Set up both hands raising
        mock_landmarks[11] = [0.5, 0.3, 0.0, 0.8]  # left shoulder
        mock_landmarks[15] = [0.5, 0.1, 0.0, 0.8]  # left wrist (above shoulder)
        mock_landmarks[12] = [0.5, 0.3, 0.0, 0.8]  # right shoulder
        mock_landmarks[16] = [0.5, 0.1, 0.0, 0.8]  # right wrist (above shoulder)

        # Process frames to build up state
        for _ in range(5):
            detector.detect(mock_landmarks)

        # Verify hands are raised
        status_before = detector.get_status()
        assert status_before["left_hand_raised"] is True
        assert status_before["right_hand_raised"] is True

        # Reset
        detector.reset()

        # Verify reset
        status_after = detector.get_status()
        assert status_after["left_hand_raised"] is False
        assert status_after["right_hand_raised"] is False
        assert status_after["any_hand_raised"] is False

    def test_reset_custom_detector(self, detector_custom, mock_landmarks):
        """Test reset() with custom detector configuration."""
        # Set up detection
        mock_landmarks[11] = [0.5, 0.3, 0.0, 0.8]  # left shoulder
        mock_landmarks[15] = [0.5, 0.1, 0.0, 0.8]  # left wrist (above shoulder)

        # Process frames
        for _ in range(7):  # More than min_consecutive_frames (5)
            detector_custom.detect(mock_landmarks)

        # Verify detection
        status_before = detector_custom.get_status()
        assert status_before["left_hand_raised"] is True

        # Reset
        detector_custom.reset()

        # Verify reset
        status_after = detector_custom.get_status()
        assert status_after["left_hand_raised"] is False

        # Configuration should remain unchanged
        assert status_after["min_consecutive_frames"] == 5
        assert status_after["vertical_margin"] == 0.1
        assert status_after["visibility_threshold"] == 0.7
        assert status_after["history_window"] == 10

    def test_reset_multiple_times(self, detector, mock_landmarks):
        """Test reset() can be called multiple times safely."""
        # Set up detection
        mock_landmarks[11] = [0.5, 0.3, 0.0, 0.8]
        mock_landmarks[15] = [0.5, 0.5, 0.0, 0.8]

        # Process frames
        for _ in range(5):
            detector.detect(mock_landmarks)

        # Reset multiple times
        detector.reset()
        detector.reset()
        detector.reset()

        # Should still be in reset state
        status = detector.get_status()
        assert status["left_hand_raised"] is False
        assert status["right_hand_raised"] is False

    def test_reset_with_none_landmarks(self, detector):
        """Test reset() behavior when landmarks are None."""
        # Process with None landmarks (should trigger reset)
        detector.detect(None)

        # Verify reset state
        status = detector.get_status()
        assert status["left_hand_raised"] is False
        assert status["right_hand_raised"] is False

        # Manual reset should also work
        detector.reset()
        status_after = detector.get_status()
        assert status_after["left_hand_raised"] is False
        assert status_after["right_hand_raised"] is False

    def test_reset_preserves_configuration(self, detector_custom):
        """Test reset() preserves detector configuration."""
        original_config = {
            "min_consecutive_frames": detector_custom.min_consecutive_frames,
            "vertical_margin": detector_custom.vertical_margin,
            "visibility_threshold": detector_custom.confidence_threshold,
            "history_window": detector_custom._history_window,
        }

        detector_custom.reset()

        # Configuration should be unchanged
        assert detector_custom.min_consecutive_frames == original_config["min_consecutive_frames"]
        assert detector_custom.vertical_margin == original_config["vertical_margin"]
        assert detector_custom.confidence_threshold == original_config["visibility_threshold"]
        assert detector_custom._history_window == original_config["history_window"]

    def test_reset_clears_all_hand_states(self, detector):
        """Test reset() clears all hand states completely."""
        # Modify all hand states
        for hand in ["left", "right"]:
            state = detector._hand_states[hand]
            state.is_raised = True
            state.consecutive_frames = 10
            state.visible = True
            state.visibility_score = 0.9
            state.last_vertical_delta = 0.3

            # Add history
            detector._hand_histories[hand].extend([True] * 5)

        # Reset
        detector.reset()

        # Verify all states are cleared
        for hand in ["left", "right"]:
            state = detector._hand_states[hand]
            assert state.is_raised is False
            assert state.consecutive_frames == 0
            assert state.visible is False
            assert state.visibility_score == 0.0
            assert state.last_vertical_delta == 0.0

            assert len(detector._hand_histories[hand]) == 0

    def test_reset_with_empty_landmarks(self, detector):
        """Test reset() behavior with empty landmarks array."""
        empty_landmarks = np.array([])

        # Process with empty landmarks
        detector.detect(empty_landmarks)

        # Verify reset state
        status = detector.get_status()
        assert status["left_hand_raised"] is False
        assert status["right_hand_raised"] is False

        # Manual reset should also work
        detector.reset()
        status_after = detector.get_status()
        assert status_after["left_hand_raised"] is False
        assert status_after["right_hand_raised"] is False

    def test_reset_hand_state_internal_method(self, detector):
        """Test _reset_hand_state() internal method."""
        # Set up state
        detector._hand_states["left"].is_raised = True
        detector._hand_states["left"].consecutive_frames = 5
        detector._hand_states["left"].visible = True
        detector._hand_states["left"].visibility_score = 0.8
        detector._hand_states["left"].last_vertical_delta = 0.2

        detector._hand_histories["left"].extend([True, True, True])

        # Reset left hand only
        detector._reset_hand_state("left")

        # Check left hand is reset
        left_state = detector._hand_states["left"]
        assert left_state.is_raised is False
        assert left_state.consecutive_frames == 0
        assert left_state.visible is False
        assert left_state.visibility_score == 0.0
        assert left_state.last_vertical_delta == 0.0

        # Check left hand history is cleared
        assert len(detector._hand_histories["left"]) == 0

        # Right hand should be unchanged
        right_state = detector._hand_states["right"]
        assert right_state.is_raised is False  # Default state
        assert right_state.consecutive_frames == 0  # Default state

    def test_reset_hand_state_without_clearing_history(self, detector):
        """Test _reset_hand_state() with clear_history=False."""
        # Set up state and history
        detector._hand_states["left"].is_raised = True
        detector._hand_states["left"].consecutive_frames = 5
        detector._hand_histories["left"].extend([True, True, True])

        # Reset without clearing history
        detector._reset_hand_state("left", clear_history=False)

        # State should be reset
        left_state = detector._hand_states["left"]
        assert left_state.is_raised is False
        assert left_state.consecutive_frames == 0

        # History should remain
        assert len(detector._hand_histories["left"]) == 3

    # ==================== Backward Compatibility Properties ====================

    def test_backward_compatibility_properties(self, detector, mock_landmarks):
        """Test backward compatibility properties."""
        # Set up only left hand raising
        mock_landmarks[11] = [0.5, 0.3, 0.0, 0.8]  # left shoulder
        mock_landmarks[15] = [0.5, 0.1, 0.0, 0.8]  # left wrist (above shoulder)

        # Right hand not visible
        mock_landmarks[12] = [0.5, 0.3, 0.0, 0.3]  # right shoulder (low visibility)
        mock_landmarks[16] = [0.5, 0.1, 0.0, 0.3]  # right wrist (low visibility)

        # Process frames
        for _ in range(5):
            detector.detect(mock_landmarks)

        # Test backward compatibility properties
        assert detector._left_hand_consecutive_frames == 5
        assert detector._right_hand_consecutive_frames == 0
        assert detector._is_left_hand_raised is True
        assert detector._is_right_hand_raised is False

    def test_backward_compatibility_properties_default_state(self, detector):
        """Test backward compatibility properties in default state."""
        assert detector._left_hand_consecutive_frames == 0
        assert detector._right_hand_consecutive_frames == 0
        assert detector._is_left_hand_raised is False
        assert detector._is_right_hand_raised is False

    # ==================== Error Conditions ====================

    def test_init_invalid_min_consecutive_frames(self):
        """Test initialization with invalid min_consecutive_frames."""
        with pytest.raises(ValueError, match="min_consecutive_frames must be >= 1"):
            HandRaiseDetector(min_consecutive_frames=0)

    def test_init_invalid_vertical_margin(self):
        """Test initialization with invalid vertical_margin."""
        with pytest.raises(ValueError, match="vertical_margin must be >= 0.0"):
            HandRaiseDetector(vertical_margin=-0.1)

    # ==================== Exception Handling Tests ====================

    def test_evaluate_hand_index_error(self, detector):
        """Test _evaluate_hand() with IndexError when accessing landmarks."""
        # Create landmarks with insufficient length
        short_landmarks = np.zeros((10, 4))  # Only 10 landmarks instead of 33

        # This should trigger IndexError when accessing shoulder_idx=11 or wrist_idx=15
        result = detector._evaluate_hand(short_landmarks, 11, 15)

        assert result.visible is False
        assert result.criteria_met is False
        assert result.vertical_delta == 0.0
        assert result.visibility_score == 0.0

    def test_evaluate_hand_type_error(self, detector):
        """Test _evaluate_hand() with TypeError when accessing landmarks."""
        # Create landmarks with wrong type
        invalid_landmarks = "invalid_landmarks"

        # This should trigger TypeError when trying to access by index
        result = detector._evaluate_hand(invalid_landmarks, 11, 15)

        assert result.visible is False
        assert result.criteria_met is False
        assert result.vertical_delta == 0.0
        assert result.visibility_score == 0.0

    def test_evaluate_hand_value_error(self, detector):
        """Test _evaluate_hand() with ValueError when accessing landmarks."""
        # Create landmarks with None values
        invalid_landmarks = np.array([None] * 33)

        # This should trigger ValueError when trying to access by index
        result = detector._evaluate_hand(invalid_landmarks, 11, 15)

        assert result.visible is False
        assert result.criteria_met is False
        assert result.vertical_delta == 0.0
        assert result.visibility_score == 0.0

    def test_evaluate_hand_coordinate_conversion_error(self, detector):
        """Test _evaluate_hand() with coordinate conversion errors."""
        # Create landmarks with invalid coordinate data using object dtype
        landmarks = np.zeros((33, 4), dtype=object)
        landmarks[11] = [0.5, "invalid_y", 0.0, 0.8]  # Invalid y coordinate
        landmarks[15] = [0.5, 0.1, 0.0, 0.8]

        # This should trigger TypeError when converting to float
        result = detector._evaluate_hand(landmarks, 11, 15)

        assert result.visible is False
        assert result.criteria_met is False
        assert result.vertical_delta == 0.0
        assert result.visibility_score > 0  # visibility_score should be computed

    def test_evaluate_hand_coordinate_index_error(self, detector):
        """Test _evaluate_hand() with coordinate index error."""
        # Create landmarks with insufficient coordinate data
        landmarks = np.zeros((33, 2))  # Only x, y coordinates, missing z and visibility

        # This should trigger IndexError when accessing index [1] for y coordinate
        result = detector._evaluate_hand(landmarks, 11, 15)

        assert result.visible is False
        assert result.criteria_met is False
        assert result.vertical_delta == 0.0
        assert result.visibility_score == 0.0

    def test_compute_visibility_score_index_error(self, detector):
        """Test _compute_visibility_score() with IndexError."""
        # Create landmarks with insufficient data
        shoulder = np.array([0.5, 0.3])  # Missing z and visibility
        wrist = np.array([0.5, 0.1])  # Missing z and visibility

        result = detector._compute_visibility_score(shoulder, wrist)
        assert result == 0.0

    def test_compute_visibility_score_type_error(self, detector):
        """Test _compute_visibility_score() with TypeError."""
        # Create landmarks with invalid data types
        shoulder = np.array([0.5, 0.3, 0.0, "invalid_visibility"])
        wrist = np.array([0.5, 0.1, 0.0, 0.8])

        result = detector._compute_visibility_score(shoulder, wrist)
        assert result == 0.0

    def test_compute_visibility_score_value_error(self, detector):
        """Test _compute_visibility_score() with ValueError."""
        # Create landmarks with invalid values that cause ValueError
        shoulder = np.array([0.5, 0.3, 0.0, "invalid"], dtype=object)
        wrist = np.array([0.5, 0.1, 0.0, 0.8])

        result = detector._compute_visibility_score(shoulder, wrist)
        assert result == 0.0

    def test_evaluate_hand_coordinate_conversion_error_second_block(self, detector):
        """Test _evaluate_hand() with coordinate conversion errors in second try-except block."""
        # Create landmarks that pass the first try-except but fail in the second
        landmarks = np.zeros((33, 4), dtype=object)
        landmarks[11] = [0.5, 0.3, 0.0, 0.8]  # Valid shoulder
        landmarks[15] = [0.5, "invalid_y", 0.0, 0.8]  # Invalid wrist y coordinate

        # This should trigger TypeError when converting wrist_y to float (lines 215-216)
        result = detector._evaluate_hand(landmarks, 11, 15)

        assert result.visible is False
        assert result.criteria_met is False
        assert result.vertical_delta == 0.0
        assert result.visibility_score > 0  # visibility_score should be computed
