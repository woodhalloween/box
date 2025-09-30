# tests/test_knee_angle_monitor.py
import importlib
import numpy as np
import pytest

# --- Adjust this import path to your actual module path ---
from src.analysis.knee_angle_monitor import KneeAngleMonitor  # noqa: E402
from src.definitions import Angle


def test_finalize_second_no_data_returns_none():
    mon = KneeAngleMonitor()
    # Ensure lists are empty
    assert mon.current_left_values == []
    assert mon.current_right_values == []
    assert mon._finalize_second(0) is None
    assert len(mon.medians_history) == 0


def test_finalize_second_with_values_both_sides_and_clear():
    mon = KneeAngleMonitor()
    mon.current_left_values = [100.0, 90.0, 110.0]
    mon.current_right_values = [95.0, 105.0]

    out = mon._finalize_second(7)
    assert out == (7, float(np.median([100.0, 90.0, 110.0])), float(np.median([95.0, 105.0])))
    assert len(mon.medians_history) == 1
    # lists must be cleared
    assert mon.current_left_values == []
    assert mon.current_right_values == []


def test_moving_average_empty_history_is_none_none():
    mon = KneeAngleMonitor()
    assert mon._moving_average() == (None, None)


def test_moving_average_ignores_none_values():
    mon = KneeAngleMonitor(moving_window_seconds=5)
    # Append tuples: (second, left_med, right_med)
    mon.medians_history.append((0, 100.0, None))
    mon.medians_history.append((1, None, 80.0))
    mon.medians_history.append((2, 90.0, 100.0))

    lma, rma = mon._moving_average()
    # left average over [100.0, 90.0] = 95
    # right average over [80.0, 100.0] = 90
    assert lma == pytest.approx(95.0)
    assert rma == pytest.approx(90.0)


def test_update_initialization_and_same_second_no_alert():
    mon = KneeAngleMonitor()
    alerts = mon.update(0.12, {})  # first frame initializes current_second
    assert alerts == []
    assert mon.current_second == 0
    # same second, still no finalize
    alerts = mon.update(0.98, {})
    assert alerts == []


def test_update_second_rollover_with_no_data_no_alert():
    mon = KneeAngleMonitor()
    mon.update(0.2, {})        # current_second set to 0, no data accumulated
    alerts = mon.update(1.0, {})  # rollover finalizes second 0 -> no data -> no alert
    assert alerts == []
    assert len(mon.medians_history) == 0
    assert mon.get_current_alert() is None


def test_confidence_gating_and_missing_angle_key():
    mon = KneeAngleMonitor(confidence_threshold=0.7)

    # Low confidence left_knee should be ignored, right_knee missing "angle" is ignored
    analysis = {
        Angle.LEFT_KNEE: {
            "angle": 50.0,
            "p1_confidence": 0.6,
            "p2_confidence": 0.9,
            "p3_confidence": 0.95,
        },
        Angle.RIGHT_KNEE: {
            # no "angle" key on purpose
            "p1_confidence": 0.99,
            "p2_confidence": 0.99,
            "p3_confidence": 0.99,
        },
    }
    mon.update(0.1, analysis)
    # Nothing collected because left under threshold, right missing angle
    assert mon.current_left_values == []
    assert mon.current_right_values == []


def test_alert_median_only_not_ma():
    # We want the current second median to be < threshold but the MA to stay >= threshold.
    # Preload history with high values so MA remains high after adding a single low median.
    mon = KneeAngleMonitor(threshold_deg=90.0, moving_window_seconds=5)

    # Prepopulate last 4 seconds with high left medians (no right data)
    mon.medians_history.append((-4, 100.0, None))
    mon.medians_history.append((-3, 100.0, None))
    mon.medians_history.append((-2, 100.0, None))
    mon.medians_history.append((-1, 100.0, None))

    # Current second 0: collect low left knee angles to make median < threshold; no right values
    analysis_low_left = {
        Angle.LEFT_KNEE: {
            "angle": 80.0,
            "p1_confidence": 0.9,
            "p2_confidence": 0.9,
            "p3_confidence": 0.9,
        }
    }
    # Accumulate a couple frames in the same second
    mon.update(0.05, analysis_low_left)
    mon.update(0.90, analysis_low_left)

    # Rollover to second 1 to finalize second 0
    alerts = mon.update(1.00, {})  # finalize second 0

    assert len(alerts) == 1
    msg = alerts[0]
    # Should include Median but NOT MA5 (MA should be still >= 90: avg of [100,100,100,100,80] = 96)
    assert "[!] Knee Angle Low (0s): " in msg
    assert "Median L:80.0 R:-" in msg
    assert "MA5" not in msg
    # current alert should be set
    assert mon.get_current_alert() == msg


def test_alert_ma_only_not_median():
    # Make the moving average low, but the current second median >= threshold.
    mon = KneeAngleMonitor(threshold_deg=90.0, moving_window_seconds=3)

    # Prepopulate 2 low entries so that after adding the current (>= threshold) entry,
    # the MA over [80, 80, 92] = 84 (< 90) -> MA triggers, median does NOT.
    mon.medians_history.append((-2, 80.0, None))
    mon.medians_history.append((-1, 80.0, None))

    # Current second 0: median high (>= threshold)
    analysis_high_left = {
        Angle.LEFT_KNEE: {
            "angle": 92.0,
            "p1_confidence": 0.95,
            "p2_confidence": 0.95,
            "p3_confidence": 0.95,
        }
    }
    mon.update(0.10, analysis_high_left)
    mon.update(0.90, analysis_high_left)
    alerts = mon.update(1.00, {})  # finalize second 0

    assert len(alerts) == 1
    msg = alerts[0]
    # Should include MA5 but NOT Median
    assert "MA5 L:84.0 R:-" in msg
    assert "Median" not in msg


def test_both_median_and_ma_trigger_together_and_right_side_data():
    # Default window 5. Provide only a single second with low values so
    # both Median and MA are below threshold, and include right side data as well.
    mon = KneeAngleMonitor(threshold_deg=90.0)

    analysis_both_low = {
        Angle.LEFT_KNEE: {
            "angle": 70.0,
            "p1_confidence": 0.9,
            "p2_confidence": 0.9,
            "p3_confidence": 0.9,
        },
        Angle.RIGHT_KNEE: {
            "angle": 85.0,
            "p1_confidence": 0.95,
            "p2_confidence": 0.95,
            "p3_confidence": 0.95,
        },
    }

    mon.update(2.10, analysis_both_low)  # current_second becomes 2
    mon.update(2.50, analysis_both_low)  # still second 2
    alerts = mon.update(3.00, {})        # finalize second 2

    assert len(alerts) == 1
    msg = alerts[0]
    # Both Median and MA (single entry -> same)
    assert "Median L:70.0 R:85.0" in msg
    assert "MA5 L:70.0 R:85.0" in msg
    # alert stored
    assert mon.get_current_alert() == msg


def test_right_and_left_collection_independently_with_conf_threshold():
    mon = KneeAngleMonitor(threshold_deg=90.0, confidence_threshold=0.8)

    # Left passes confidence, right fails (min confidence 0.75)
    analysis = {
        Angle.LEFT_KNEE: {
            "angle": 100.0,
            "p1_confidence": 0.9,
            "p2_confidence": 0.9,
            "p3_confidence": 0.9,
        },
        Angle.RIGHT_KNEE: {
            "angle": 100.0,
            "p1_confidence": 0.75,
            "p2_confidence": 0.95,
            "p3_confidence": 0.95,
        },
    }

    mon.update(0.01, analysis)
    # Left collected, right ignored
    assert mon.current_left_values == [100.0]
    assert mon.current_right_values == []

    # Roll second to finalize and ensure no alert (median 100 >= threshold)
    alerts = mon.update(1.00, {})
    assert alerts == []
    assert mon.get_current_alert() is None
