# test_process_frame.py
import numpy as np

# ⬇️ CHANGE THIS to the actual module containing `process_frame`
# e.g., from src.processing.core import process_frame as fn
from src.video_processor import process_frame as fn  # <-- adjust if needed


class _PoseFake:
    def __init__(self, landmarks):
        self._landmarks = landmarks
        self.calls = []

    def estimate(self, frame):
        # record a tiny observable side effect
        self.calls.append(("estimate", frame.shape))
        return self._landmarks


class _AnalyzerFake:
    def __init__(self, result):
        self._result = result
        self.calls = 0

    def analyze(self, landmarks):
        self.calls += 1
        return dict(self._result)  # return a copy to ensure mutation path covered


class _PostureMonitorFake:
    def __init__(self, alerts):
        self._alerts = list(alerts)
        self.calls = []

    def update(self, t, frame_idx, results):
        self.calls.append((t, frame_idx, bool(results)))
        # return list to ensure .extend path is hit
        return list(self._alerts)


class _DwellFake:
    def __init__(self, dwell_alert):
        self._dwell_alert = dwell_alert
        self.calls = []

    def update(self, landmarks, frame_shape, t):
        self.calls.append((frame_shape, t))
        # may be a string (truthy) or None (falsy)
        return self._dwell_alert


class _HeadShakeFake:
    def __init__(self, update_dict, check_alerts_list):
        self._update_dict = dict(update_dict)
        self._alerts = list(check_alerts_list)
        self.update_calls = 0
        self.check_calls = 0

    def update(self, landmarks, t, frame_idx):
        self.update_calls += 1
        # return dict to ensure results.update(...) is exercised
        return dict(self._update_dict)

    def check_alerts(self, t):
        self.check_calls += 1
        # list so that alerts.extend(...) is exercised
        return list(self._alerts)


class _HandRaiseFake:
    def __init__(self):
        self.detect_calls = 0

    def detect(self, landmarks):
        self.detect_calls += 1
        return {"left_hand_raised": False, "right_hand_raised": False}


def _mk_state(
    *,
    landmarks,
    analyzer_result,
    posture_alerts,
    dwell_alert,
    head_update_dict,
    head_alerts,
    frame_idx=42,
    disable_jp=True,
):
    # Minimal “PipelineState” duck type
    class State:
        pass

    s = State()
    s.pose = _PoseFake(landmarks)
    s.analyzer = _AnalyzerFake(analyzer_result)
    s.posture_monitor = _PostureMonitorFake(posture_alerts)
    s.dwell_time_detector = _DwellFake(dwell_alert)
    s.head_shake_detector = _HeadShakeFake(head_update_dict, head_alerts)
    s.hand_raise_detector = _HandRaiseFake()
    s.user_classifier = object()  # only passed through to draw function (no update attr)
    s.frame_idx = frame_idx
    s.disable_jp = disable_jp
    s.last_landmarks = "UNTOUCHED"
    s.last_head_alerts = "UNTOUCHED"
    s.last_hand_statuses = "UNTOUCHED"
    return s


def test_process_frame_no_landmarks_early_return(monkeypatch):
    """
    When pose.estimate returns None:
      - original frame is returned (unmodified),
      - results dict is empty,
      - alerts list is empty,
      - aux contains dwell_alert None,
      - state.last_landmarks is set to None.
    """

    # stub draw functions should never be called in this branch
    def _boom(*a, **k):
        raise AssertionError("draw functions must not be called when landmarks is None")

    # Patch by dotted path (single string each)
    monkeypatch.setattr("src.video_processor.draw_landmarks", _boom)
    monkeypatch.setattr("src.video_processor.draw_analysis_results", _boom)
    monkeypatch.setattr("src.video_processor.draw_detection_info", _boom)

    state = _mk_state(
        landmarks=None, analyzer_result={}, posture_alerts=[], dwell_alert=None, head_update_dict={}, head_alerts=[]
    )
    frame_in = np.zeros((2, 3, 3), dtype=np.uint8)
    out_frame, results, alerts, aux = fn(frame_in, t=0.0, state=state)

    assert out_frame is frame_in
    assert results == {}
    assert alerts == []
    assert aux == {"dwell_alert": None}
    assert state.last_landmarks is None


def test_process_frame_full_path_with_dwell_and_head_alerts(monkeypatch):
    """
    Full pipeline path:
      - landmarks present
      - analyzer returns dict
      - posture monitor returns alerts
      - dwell detector returns a truthy alert (added to alerts and aux)
      - head shake adds results and alerts
      - draw functions are called in order and fed correct args
      - state.last_landmarks & last_head_alerts are updated
    """
    calls = {"draw": []}

    def fake_draw_landmarks(frame, landmarks):
        calls["draw"].append(("landmarks", frame.shape, landmarks))
        return frame  # pass-through

    def fake_draw_analysis_results(frame, results, hand_statuses, landmarks, *, disable_japanese):
        calls["draw"].append(("analysis", bool(results), bool(hand_statuses), disable_japanese))
        return frame

    def fake_draw_detection_info(
        frame,
        user_classifier,
        dwell_time_detector,
        head_shake_detector,
        posture_monitor,
        posture_alerts,
        landmarks,
        t,
    ):
        calls["draw"].append(("info", type(user_classifier).__name__, bool(posture_alerts), t, landmarks))
        return frame

    # Patch the drawing functions using fully qualified dotted paths
    monkeypatch.setattr("src.video_processor.draw_landmarks", fake_draw_landmarks)
    monkeypatch.setattr("src.video_processor.draw_analysis_results", fake_draw_analysis_results)
    monkeypatch.setattr("src.video_processor.draw_detection_info", fake_draw_detection_info)

    landmarks = object()  # any non-None sentinel
    state = _mk_state(
        landmarks=landmarks,
        analyzer_result={"knee": {"angle": 90.0}},
        posture_alerts=["POSTURE_BAD"],
        dwell_alert="DWELL_ALERT",
        head_update_dict={"head_shake": {"score": 0.8}},
        head_alerts=["HEAD_ALERT_A", "HEAD_ALERT_B"],
        frame_idx=7,
        disable_jp=False,
    )

    frame_in = np.zeros((480, 640, 3), dtype=np.uint8)
    out_frame, results, alerts, aux = fn(frame_in, t=1.23, state=state)

    assert out_frame is frame_in
    assert "knee" in results and "head_shake" in results
    assert results["head_shake"]["score"] == 0.8
    assert alerts == ["POSTURE_BAD", "DWELL_ALERT", "HEAD_ALERT_A", "HEAD_ALERT_B"]
    assert aux["dwell_alert"] == "DWELL_ALERT"
    assert state.last_landmarks is landmarks
    assert state.last_head_alerts == ["HEAD_ALERT_A", "HEAD_ALERT_B"]

    # Verify draw call ordering
    assert calls["draw"][0][0] == "landmarks"
    assert calls["draw"][1] == ("analysis", True, True, False)  # disable_japanese=False
    assert calls["draw"][2][0] == "info"
    assert calls["draw"][2][2] is True and calls["draw"][2][3] == 1.23


def test_process_frame_user_classifier_update_is_optional(monkeypatch):
    """process_frame should not crash if state.user_classifier lacks 'update' (hasattr guard)."""
    # Use state with user_classifier as plain object (no update method)
    state = _mk_state(
        landmarks=np.zeros((33, 4)),
        analyzer_result={},
        posture_alerts=[],
        dwell_alert=None,
        head_update_dict={},
        head_alerts=[],
    )

    # Patch draw functions to simple pass-throughs
    monkeypatch.setattr("src.video_processor.draw_landmarks", lambda f, lm: f)
    monkeypatch.setattr(
        "src.video_processor.draw_analysis_results",
        lambda f, r, hand_statuses, lm, disable_japanese: f,
    )
    monkeypatch.setattr("src.video_processor.draw_detection_info", lambda f, *a, **k: f)

    frame_in = np.zeros((10, 10, 3), dtype=np.uint8)
    out_frame, results, alerts, aux = fn(frame_in, t=0.1, state=state)

    # Should complete without exceptions and return the same frame
    assert out_frame is frame_in


def test_process_frame_posture_alerts_appended_and_called_with_args(monkeypatch):
    """
    Ensure posture_monitor.update is called with (t, frame_idx, results)
    and its returned alerts are appended to the outgoing alerts list.
    """
    # Capture posture call args
    calls = {"pm": []}

    class PM:
        def update(self, t, frame_idx, results):
            calls["pm"].append((t, frame_idx, dict(results)))
            return ["ALERT_A", "ALERT_B"]

    # Minimal state
    class State:
        pass

    s = State()
    s.pose = _PoseFake(landmarks=np.zeros((33, 4)))
    s.analyzer = _AnalyzerFake({"K": {"angle": 12.3}})
    s.posture_monitor = PM()
    s.dwell_time_detector = _DwellFake(None)
    s.head_shake_detector = _HeadShakeFake({}, [])
    s.hand_raise_detector = _HandRaiseFake()
    s.user_classifier = object()
    s.frame_idx = 5
    s.disable_jp = True
    s.last_landmarks = None
    s.last_head_alerts = []
    s.last_hand_statuses = None

    # Patch draw functions to pass-through
    monkeypatch.setattr("src.video_processor.draw_landmarks", lambda f, lm: f)
    monkeypatch.setattr(
        "src.video_processor.draw_analysis_results",
        lambda f, r, hand_statuses, lm, disable_japanese: f,
    )
    monkeypatch.setattr("src.video_processor.draw_detection_info", lambda f, *a, **k: f)

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    out_frame, results, alerts, aux = fn(frame, t=2.0, state=s)

    assert out_frame is frame
    assert alerts[:2] == ["ALERT_A", "ALERT_B"]
    # Verify posture update args
    assert calls["pm"][0][0] == 2.0
    assert calls["pm"][0][1] == 5
    assert "K" in calls["pm"][0][2]


def test_process_frame_user_classifier_with_update_returns_alerts(monkeypatch):
    """
    When state.user_classifier has an 'update' method that returns alerts,
    those alerts should be added to the alerts list.
    This tests lines 294-296 when user_classifier.update() returns a list.
    """
    calls = {"uc": []}

    class UserClassifierWithUpdate:
        def update(self, t, results):
            calls["uc"].append((t, dict(results)))
            return ["UC_ALERT_1", "UC_ALERT_2"]

    # Minimal state
    class State:
        pass

    s = State()
    s.pose = _PoseFake(landmarks=np.zeros((33, 4)))
    s.analyzer = _AnalyzerFake({"knee": {"angle": 45.0}})
    s.user_classifier = UserClassifierWithUpdate()  # Has update method
    s.posture_monitor = _PostureMonitorFake([])
    s.dwell_time_detector = _DwellFake(None)
    s.head_shake_detector = _HeadShakeFake({}, [])
    s.hand_raise_detector = _HandRaiseFake()
    s.frame_idx = 10
    s.disable_jp = True
    s.last_landmarks = None
    s.last_head_alerts = []
    s.last_hand_statuses = None

    # Patch draw functions to pass-through
    monkeypatch.setattr("src.video_processor.draw_landmarks", lambda f, lm: f)
    monkeypatch.setattr(
        "src.video_processor.draw_analysis_results",
        lambda f, r, hand_statuses, lm, disable_japanese: f,
    )
    monkeypatch.setattr("src.video_processor.draw_detection_info", lambda f, *a, **k: f)

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    out_frame, results, alerts, aux = fn(frame, t=3.5, state=s)

    # Verify user_classifier.update was called with correct args
    assert len(calls["uc"]) == 1
    assert calls["uc"][0][0] == 3.5
    assert "knee" in calls["uc"][0][1]
    assert calls["uc"][0][1]["knee"]["angle"] == 45.0

    # Verify alerts from user_classifier were added
    assert "UC_ALERT_1" in alerts
    assert "UC_ALERT_2" in alerts


def test_process_frame_user_classifier_with_update_returns_none(monkeypatch):
    """
    When state.user_classifier has an 'update' method that returns None,
    the 'or []' fallback should be used and no error should occur.
    This tests lines 294-296 when user_classifier.update() returns None.
    """
    calls = {"uc": []}

    class UserClassifierReturnsNone:
        def update(self, t, results):
            calls["uc"].append((t, dict(results)))
            return  # Explicitly return None

    # Minimal state
    class State:
        pass

    s = State()
    s.pose = _PoseFake(landmarks=np.zeros((33, 4)))
    s.analyzer = _AnalyzerFake({"hip": {"y": 100}})
    s.user_classifier = UserClassifierReturnsNone()  # Has update method that returns None
    s.posture_monitor = _PostureMonitorFake([])
    s.dwell_time_detector = _DwellFake(None)
    s.head_shake_detector = _HeadShakeFake({}, [])
    s.hand_raise_detector = _HandRaiseFake()
    s.frame_idx = 20
    s.disable_jp = True
    s.last_landmarks = None
    s.last_head_alerts = []
    s.last_hand_statuses = None

    # Patch draw functions to pass-through
    monkeypatch.setattr("src.video_processor.draw_landmarks", lambda f, lm: f)
    monkeypatch.setattr(
        "src.video_processor.draw_analysis_results",
        lambda f, r, hand_statuses, lm, disable_japanese: f,
    )
    monkeypatch.setattr("src.video_processor.draw_detection_info", lambda f, *a, **k: f)

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    out_frame, results, alerts, aux = fn(frame, t=5.0, state=s)

    # Verify user_classifier.update was called
    assert len(calls["uc"]) == 1
    assert calls["uc"][0][0] == 5.0

    # Should complete without exceptions, alerts should not contain anything from user_classifier
    assert out_frame is frame
    # No assertion errors should have occurred


def test_process_frame_hand_raise_detection_indexerror(monkeypatch, capsys):
    """
    Test lines 324-331: Hand raise detection with IndexError exception.
    When hand_raise_detector.detect() raises IndexError, it should be caught,
    a warning should be printed, hand_statuses should be set to None,
    and state.last_hand_statuses should be updated.
    """

    class HandRaiseDetectorWithIndexError:
        def detect(self, landmarks):
            raise IndexError("Landmark index out of range")

    # Minimal state
    class State:
        pass

    s = State()
    s.pose = _PoseFake(landmarks=np.zeros((33, 4)))
    s.analyzer = _AnalyzerFake({})
    s.posture_monitor = _PostureMonitorFake([])
    s.dwell_time_detector = _DwellFake(None)
    s.head_shake_detector = _HeadShakeFake({}, [])
    s.hand_raise_detector = HandRaiseDetectorWithIndexError()
    s.user_classifier = object()
    s.frame_idx = 0
    s.disable_jp = True
    s.last_landmarks = None
    s.last_head_alerts = []
    s.last_hand_statuses = "UNTOUCHED"

    # Patch draw functions to pass-through
    monkeypatch.setattr("src.video_processor.draw_landmarks", lambda f, lm: f)
    monkeypatch.setattr(
        "src.video_processor.draw_analysis_results",
        lambda f, r, hand_statuses, lm, disable_japanese: f,
    )
    monkeypatch.setattr("src.video_processor.draw_detection_info", lambda f, *a, **k: f)

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    out_frame, results, alerts, aux = fn(frame, t=1.0, state=s)

    # Verify the warning was printed
    captured = capsys.readouterr()
    assert (
        "Warning: Hand raise detection failed due to landmark data issue: Landmark index out of range" in captured.out
    )

    # Verify state was updated correctly
    assert s.last_hand_statuses is None
    assert out_frame is frame


def test_process_frame_hand_raise_detection_typeerror(monkeypatch, capsys):
    """
    Test lines 324-331: Hand raise detection with TypeError exception.
    When hand_raise_detector.detect() raises TypeError, it should be caught,
    a warning should be printed, hand_statuses should be set to None,
    and state.last_hand_statuses should be updated.
    """

    class HandRaiseDetectorWithTypeError:
        def detect(self, landmarks):
            raise TypeError("Invalid landmark data type")

    # Minimal state
    class State:
        pass

    s = State()
    s.pose = _PoseFake(landmarks=np.zeros((33, 4)))
    s.analyzer = _AnalyzerFake({})
    s.posture_monitor = _PostureMonitorFake([])
    s.dwell_time_detector = _DwellFake(None)
    s.head_shake_detector = _HeadShakeFake({}, [])
    s.hand_raise_detector = HandRaiseDetectorWithTypeError()
    s.user_classifier = object()
    s.frame_idx = 0
    s.disable_jp = True
    s.last_landmarks = None
    s.last_head_alerts = []
    s.last_hand_statuses = "UNTOUCHED"

    # Patch draw functions to pass-through
    monkeypatch.setattr("src.video_processor.draw_landmarks", lambda f, lm: f)
    monkeypatch.setattr(
        "src.video_processor.draw_analysis_results",
        lambda f, r, hand_statuses, lm, disable_japanese: f,
    )
    monkeypatch.setattr("src.video_processor.draw_detection_info", lambda f, *a, **k: f)

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    out_frame, results, alerts, aux = fn(frame, t=1.0, state=s)

    # Verify the warning was printed
    captured = capsys.readouterr()
    assert "Warning: Hand raise detection failed due to landmark data issue: Invalid landmark data type" in captured.out

    # Verify state was updated correctly
    assert s.last_hand_statuses is None
    assert out_frame is frame


def test_process_frame_hand_raise_detection_valueerror(monkeypatch, capsys):
    """
    Test lines 324-331: Hand raise detection with ValueError exception.
    When hand_raise_detector.detect() raises ValueError, it should be caught,
    a warning should be printed, hand_statuses should be set to None,
    and state.last_hand_statuses should be updated.
    """

    class HandRaiseDetectorWithValueError:
        def detect(self, landmarks):
            raise ValueError("Invalid landmark values")

    # Minimal state
    class State:
        pass

    s = State()
    s.pose = _PoseFake(landmarks=np.zeros((33, 4)))
    s.analyzer = _AnalyzerFake({})
    s.posture_monitor = _PostureMonitorFake([])
    s.dwell_time_detector = _DwellFake(None)
    s.head_shake_detector = _HeadShakeFake({}, [])
    s.hand_raise_detector = HandRaiseDetectorWithValueError()
    s.user_classifier = object()
    s.frame_idx = 0
    s.disable_jp = True
    s.last_landmarks = None
    s.last_head_alerts = []
    s.last_hand_statuses = "UNTOUCHED"

    # Patch draw functions to pass-through
    monkeypatch.setattr("src.video_processor.draw_landmarks", lambda f, lm: f)
    monkeypatch.setattr(
        "src.video_processor.draw_analysis_results",
        lambda f, r, hand_statuses, lm, disable_japanese: f,
    )
    monkeypatch.setattr("src.video_processor.draw_detection_info", lambda f, *a, **k: f)

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    out_frame, results, alerts, aux = fn(frame, t=1.0, state=s)

    # Verify the warning was printed
    captured = capsys.readouterr()
    assert "Warning: Hand raise detection failed due to landmark data issue: Invalid landmark values" in captured.out

    # Verify state was updated correctly
    assert s.last_hand_statuses is None
    assert out_frame is frame


def test_process_frame_hand_raise_detection_unexpected_exception(monkeypatch, capsys):
    """
    Test lines 324-331: Hand raise detection with unexpected Exception.
    When hand_raise_detector.detect() raises an unexpected exception (not IndexError, TypeError, ValueError),
    it should be caught by the general Exception handler, an error should be printed,
    hand_statuses should be set to None, and state.last_hand_statuses should be updated.
    """

    class HandRaiseDetectorWithUnexpectedError:
        def detect(self, landmarks):
            raise RuntimeError("Unexpected runtime error")

    # Minimal state
    class State:
        pass

    s = State()
    s.pose = _PoseFake(landmarks=np.zeros((33, 4)))
    s.analyzer = _AnalyzerFake({})
    s.posture_monitor = _PostureMonitorFake([])
    s.dwell_time_detector = _DwellFake(None)
    s.head_shake_detector = _HeadShakeFake({}, [])
    s.hand_raise_detector = HandRaiseDetectorWithUnexpectedError()
    s.user_classifier = object()
    s.frame_idx = 0
    s.disable_jp = True
    s.last_landmarks = None
    s.last_head_alerts = []
    s.last_hand_statuses = "UNTOUCHED"

    # Patch draw functions to pass-through
    monkeypatch.setattr("src.video_processor.draw_landmarks", lambda f, lm: f)
    monkeypatch.setattr(
        "src.video_processor.draw_analysis_results",
        lambda f, r, hand_statuses, lm, disable_japanese: f,
    )
    monkeypatch.setattr("src.video_processor.draw_detection_info", lambda f, *a, **k: f)

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    out_frame, results, alerts, aux = fn(frame, t=1.0, state=s)

    # Verify the error was printed
    captured = capsys.readouterr()
    assert "Error: Unexpected error in hand raise detection: Unexpected runtime error" in captured.out

    # Verify state was updated correctly
    assert s.last_hand_statuses is None
    assert out_frame is frame


def test_process_frame_hand_raise_detection_success(monkeypatch):
    """
    Test lines 324-331: Hand raise detection successful case.
    When hand_raise_detector.detect() succeeds, it should return the expected result,
    and state.last_hand_statuses should be updated with the returned value.
    """
    expected_hand_statuses = {"left_hand_raised": True, "right_hand_raised": False}

    class HandRaiseDetectorSuccess:
        def detect(self, landmarks):
            return expected_hand_statuses

    # Minimal state
    class State:
        pass

    s = State()
    s.pose = _PoseFake(landmarks=np.zeros((33, 4)))
    s.analyzer = _AnalyzerFake({})
    s.posture_monitor = _PostureMonitorFake([])
    s.dwell_time_detector = _DwellFake(None)
    s.head_shake_detector = _HeadShakeFake({}, [])
    s.hand_raise_detector = HandRaiseDetectorSuccess()
    s.user_classifier = object()
    s.frame_idx = 0
    s.disable_jp = True
    s.last_landmarks = None
    s.last_head_alerts = []
    s.last_hand_statuses = "UNTOUCHED"

    # Patch draw functions to pass-through
    monkeypatch.setattr("src.video_processor.draw_landmarks", lambda f, lm: f)
    monkeypatch.setattr(
        "src.video_processor.draw_analysis_results",
        lambda f, r, hand_statuses, lm, disable_japanese: f,
    )
    monkeypatch.setattr("src.video_processor.draw_detection_info", lambda f, *a, **k: f)

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    out_frame, results, alerts, aux = fn(frame, t=1.0, state=s)

    # Verify state was updated correctly with the expected result
    assert s.last_hand_statuses == expected_hand_statuses
    assert out_frame is frame


def test_process_frame_hand_raise_detection_multiple_exceptions_coverage(monkeypatch, capsys):
    """
    Test lines 324-331: Ensure all exception types are properly covered.
    This test verifies that the exception handling covers all the specific exception types
    mentioned in the code (IndexError, TypeError, ValueError) and the general Exception handler.
    """
    # Test that all three specific exceptions are handled the same way
    specific_exceptions = [
        (IndexError, "Index out of bounds"),
        (TypeError, "Type mismatch"),
        (ValueError, "Invalid value"),
    ]

    def create_detector_with_exception(exc_type, message):
        """Create a detector that raises the specified exception."""

        class HandRaiseDetectorWithSpecificError:
            def detect(self, landmarks):
                raise exc_type(message)

        return HandRaiseDetectorWithSpecificError()

    for exc_type, message in specific_exceptions:
        # Minimal state
        class State:
            pass

        s = State()
        s.pose = _PoseFake(landmarks=np.zeros((33, 4)))
        s.analyzer = _AnalyzerFake({})
        s.posture_monitor = _PostureMonitorFake([])
        s.dwell_time_detector = _DwellFake(None)
        s.head_shake_detector = _HeadShakeFake({}, [])
        s.hand_raise_detector = create_detector_with_exception(exc_type, message)
        s.user_classifier = object()
        s.frame_idx = 0
        s.disable_jp = True
        s.last_landmarks = None
        s.last_head_alerts = []
        s.last_hand_statuses = "UNTOUCHED"

        # Patch draw functions to pass-through
        monkeypatch.setattr("src.video_processor.draw_landmarks", lambda f, lm: f)
        monkeypatch.setattr(
            "src.video_processor.draw_analysis_results",
            lambda f, r, hand_statuses, lm, disable_japanese: f,
        )
        monkeypatch.setattr("src.video_processor.draw_detection_info", lambda f, *a, **k: f)

        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        out_frame, results, alerts, aux = fn(frame, t=1.0, state=s)

        # Verify the warning was printed for specific exceptions
        captured = capsys.readouterr()
        assert f"Warning: Hand raise detection failed due to landmark data issue: {message}" in captured.out

        # Verify state was updated correctly
        assert s.last_hand_statuses is None
        assert out_frame is frame
