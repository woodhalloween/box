from enum import Enum
from types import SimpleNamespace

import numpy as np
import pytest

# test_process_frame.py
from src.definitions import Angle, MovementState
from src.video_processor import PipelineState, _normalize_head_shake_results, process_frame

# Historical alias used by legacy tests in this module
fn = process_frame


# ==================== _normalize_head_shake_results Tests ====================


def test_normalize_head_shake_results_returns_empty_for_none():
    assert _normalize_head_shake_results(None) == {}


def test_normalize_head_shake_results_pass_through_angle_dict():
    payload = {
        Angle.HEAD_HORIZONTAL_ROTATION: {
            "angle": 1.0,
            "state": MovementState.HEAD_STATIC,
            "confidence": 0.9,
        }
    }
    assert _normalize_head_shake_results(payload) is payload


def test_normalize_head_shake_results_converts_structured_dict():
    raw = {
        "horizontal_state": "HORIZONTAL_SHAKE",
        "vertical_state": MovementState.VERTICAL_NOD,
        "horizontal_angle": 12.3,
        "vertical_angle": -4.5,
        "confidence": 0.8,
    }
    result = _normalize_head_shake_results(raw)

    assert result[Angle.HEAD_HORIZONTAL_ROTATION]["angle"] == pytest.approx(12.3)
    assert result[Angle.HEAD_VERTICAL_NOD]["angle"] == pytest.approx(-4.5)
    assert result[Angle.HEAD_HORIZONTAL_ROTATION]["state"] is MovementState.HORIZONTAL_SHAKE
    assert result[Angle.HEAD_VERTICAL_NOD]["state"] is MovementState.VERTICAL_NOD
    assert result[Angle.HEAD_VERTICAL_NOD]["confidence"] == pytest.approx(0.8)


def test_normalize_head_shake_results_missing_keys_returns_raw():
    raw = {"unexpected": 1}
    assert _normalize_head_shake_results(raw) is raw


def test_normalize_head_shake_results_uses_enum_value_coercion(monkeypatch):
    from src import video_processor as vp_module

    class FakeMovementState(Enum):
        HEAD_STATIC = "head_static"
        VERTICAL_NOD = "vertical_nod"

    monkeypatch.setattr(vp_module, "MovementState", FakeMovementState)

    raw = {
        "horizontal_state": "head_static",  # Not a member name, only matches .value
        "vertical_state": "vertical_nod",
        "horizontal_angle": 5.0,
        "vertical_angle": 3.0,
        "confidence": 0.7,
    }
    result = _normalize_head_shake_results(raw)

    assert result[Angle.HEAD_HORIZONTAL_ROTATION]["state"] is FakeMovementState.HEAD_STATIC
    assert result[Angle.HEAD_VERTICAL_NOD]["state"] is FakeMovementState.VERTICAL_NOD


def test_normalize_head_shake_results_non_dict_returns_empty():
    assert _normalize_head_shake_results(["unexpected"]) == {}


# ==================== process_frame Tests ====================


def _make_pipeline_state(**overrides):
    defaults = {
        "pose": SimpleNamespace(estimate=lambda frame: None),
        "analyzer": SimpleNamespace(analyze=lambda landmarks: {}),
        "user_classifier": SimpleNamespace(update=lambda *a, **k: [], get_current_alert=lambda: None),
        "dwell_time_detector": SimpleNamespace(
            update=lambda *a, **k: None,
            get_current_status=lambda: {"is_long_stay": False, "stay_duration": 0.0},
            stay_info=SimpleNamespace(notified=False),
        ),
        "head_shake_detector": SimpleNamespace(
            detect=lambda *a, **k: {},
            update=lambda *a, **k: {},
            check_alerts=lambda ts: [],
        ),
        "hand_raise_detector": SimpleNamespace(detect=lambda *a, **k: {}),
        "posture_monitor": SimpleNamespace(
            update=lambda *a, **k: [], get_status=lambda: {"forward_ratio": 0.0, "avg_score": 0.0, "sample_count": 0}
        ),
    }
    defaults.update(overrides)
    state = PipelineState(**defaults)
    state.frame_idx = overrides.get("frame_idx", 0)
    return state


def test_process_frame_returns_early_when_no_landmarks():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    state = _make_pipeline_state()

    processed_frame, results, alerts, aux = process_frame(frame, t=1.23, state=state)

    assert processed_frame is frame
    assert results == {}
    assert alerts == []
    assert aux == {"dwell_alert": None}
    assert state.last_landmarks is None


def test_process_frame_normalizes_head_shake_and_collects_alerts(monkeypatch):
    from src import video_processor as vp_module

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    landmarks = np.zeros((33, 4))
    call_log = {"color_calls": 0}

    def fake_draw_color_frame(img, color, alpha=1.0):
        call_log["color_calls"] += 1
        return img

    email_notifications = []

    def fake_basic_notification():
        return SimpleNamespace()

    def fake_email_notification_decorator(base, **kwargs):
        recipient_log = SimpleNamespace(kwargs=kwargs, sent=[])

        def send(msg, recipient):
            recipient_log.sent.append((msg, recipient))

        recipient_log.send = send
        email_notifications.append(recipient_log)
        return recipient_log

    monkeypatch.setattr(vp_module, "draw_landmarks", lambda img, lm: img)
    monkeypatch.setattr(
        vp_module,
        "draw_analysis_results",
        lambda img, analysis, hand_statuses, lm, disable_japanese: img,
    )
    monkeypatch.setattr(
        vp_module,
        "draw_detection_info",
        lambda img, *args, **kwargs: img,
    )
    monkeypatch.setattr(vp_module, "draw_color_frame", fake_draw_color_frame)
    monkeypatch.setattr(
        vp_module,
        "_load_email_config",
        lambda: {
            "username": "u",
            "password": "p",
            "smtp_server": "smtp",
            "smtp_port": "587",
            "subject": "s",
            "recipient": "alert@example.com",
        },
    )
    monkeypatch.setattr(vp_module, "BasicNotification", fake_basic_notification)
    monkeypatch.setattr(vp_module, "EmailNotificationDecorator", fake_email_notification_decorator)

    def detect_head_shake(*args):
        if len(args) == 3:
            raise TypeError("detect expects two args")
        landmarks_arg, timestamp_arg = args
        assert landmarks_arg is landmarks
        assert timestamp_arg == pytest.approx(1.5)
        return {
            "horizontal_state": "HORIZONTAL_SHAKE",
            "vertical_state": MovementState.VERTICAL_NOD,
            "horizontal_angle": 11.1,
            "vertical_angle": -5.5,
            "confidence": 0.85,
        }

    head_shake_detector = SimpleNamespace(
        detect=detect_head_shake,
        update=lambda *a, **k: {},
        check_alerts=lambda ts: [
            "[!] Horizontal Head Shake Detected",
            "[!] Vertical Head Nod Detected",
        ],
    )

    user_alerts = ["USER_ALERT"]
    posture_alerts = ["POSTURE_ALERT"]

    state = _make_pipeline_state(
        pose=SimpleNamespace(estimate=lambda img: landmarks),
        analyzer=SimpleNamespace(analyze=lambda lm: {"base_metric": 0.1}),
        user_classifier=SimpleNamespace(
            update=lambda *a, **k: user_alerts,
            get_current_alert=lambda: "CLASSIFIED",
        ),
        dwell_time_detector=SimpleNamespace(
            update=lambda lm, shape, ts: "DWELL_ALERT",
            get_current_status=lambda: {"is_long_stay": True, "stay_duration": 2.5},
            stay_info=SimpleNamespace(notified=False),
        ),
        head_shake_detector=head_shake_detector,
        hand_raise_detector=SimpleNamespace(
            detect=lambda landmarks=None: {"left_hand_raised": True, "right_hand_raised": True}
        ),
        posture_monitor=SimpleNamespace(
            update=lambda *a, **k: posture_alerts,
            get_status=lambda: {
                "forward_ratio": 0.2,
                "avg_score": 0.3,
                "sample_count": 7,
            },
        ),
        frame_idx=10,
    )

    class DummyWriter:
        def __init__(self):
            self.rows = []

        def writerow(self, row):
            self.rows.append(row)

    debug_writer = DummyWriter()

    processed_frame, results, alerts, aux = process_frame(
        frame=frame,
        t=1.5,
        state=state,
        debug_csv_writer=debug_writer,
    )

    assert processed_frame is frame
    assert results[Angle.HEAD_HORIZONTAL_ROTATION]["state"] is MovementState.HORIZONTAL_SHAKE
    assert results[Angle.HEAD_VERTICAL_NOD]["state"] is MovementState.VERTICAL_NOD
    assert "[!] Horizontal Head Shake Detected" in alerts
    assert "[!] Vertical Head Nod Detected" in alerts
    assert "USER_ALERT" in alerts
    assert "POSTURE_ALERT" in alerts
    assert "DWELL_ALERT" in alerts
    assert aux["dwell_alert"] == "DWELL_ALERT"
    assert state.last_head_alerts == head_shake_detector.check_alerts(0)
    assert state.prev_head_shake_horizontal is True
    assert state.prev_head_shake_vertical is True
    assert state.last_hand_statuses == {"left_hand_raised": True, "right_hand_raised": True}
    assert state.prev_hand_raised_left is True
    assert state.prev_hand_raised_right is True
    assert call_log["color_calls"] == 2  # hand and head blinking overlays
    assert len(email_notifications) == 2  # head shake + hand raise notifications
    assert email_notifications[0].sent[0][1] == "alert@example.com"
    assert email_notifications[1].sent[0][1] == "alert@example.com"
    assert len(debug_writer.rows) == 1
    assert "[!] Horizontal Head Shake Detected" in debug_writer.rows[0]["head_shake_alerts"]


def test_process_frame_uses_update_when_detect_returns_empty_and_handles_no_alerts(monkeypatch):
    from src import video_processor as vp_module

    frame = np.zeros((6, 6, 3), dtype=np.uint8)
    landmarks = np.ones((33, 4))
    call_log = {"color_calls": 0}

    def fake_draw_color_frame(img, color, alpha=1.0):
        call_log["color_calls"] += 1
        return img

    monkeypatch.setattr(vp_module, "draw_landmarks", lambda img, lm: img)
    monkeypatch.setattr(
        vp_module,
        "draw_analysis_results",
        lambda img, analysis, hand_statuses, lm, disable_japanese: img,
    )
    monkeypatch.setattr(
        vp_module,
        "draw_detection_info",
        lambda img, *args, **kwargs: img,
    )
    monkeypatch.setattr(vp_module, "draw_color_frame", fake_draw_color_frame)
    monkeypatch.setattr(vp_module, "write_results_to_csv", lambda **kwargs: None)
    monkeypatch.setattr(
        vp_module,
        "_load_email_config",
        lambda: {
            "username": "",
            "password": "",
            "smtp_server": "",
            "smtp_port": "587",
            "subject": "",
            "recipient": "",
        },
    )
    monkeypatch.setattr(vp_module, "BasicNotification", lambda: SimpleNamespace())
    monkeypatch.setattr(
        vp_module,
        "EmailNotificationDecorator",
        lambda base, **kwargs: SimpleNamespace(send=lambda msg, recipient: None),
    )

    head_shake_detector = SimpleNamespace(
        detect=lambda *a, **k: {},
        update=lambda *a, **k: {
            Angle.HEAD_HORIZONTAL_ROTATION: {
                "angle": 3.0,
                "state": MovementState.HEAD_STATIC,
                "confidence": 0.5,
            },
            Angle.HEAD_VERTICAL_NOD: {
                "angle": -2.0,
                "state": MovementState.HEAD_STATIC,
                "confidence": 0.5,
            },
        },
        check_alerts=lambda ts: [],
    )

    state = _make_pipeline_state(
        pose=SimpleNamespace(estimate=lambda img: landmarks),
        analyzer=SimpleNamespace(analyze=lambda lm: {"baseline": 0.0}),
        user_classifier=SimpleNamespace(update=lambda *a, **k: [], get_current_alert=lambda: None),
        dwell_time_detector=SimpleNamespace(
            update=lambda *a, **k: None,
            get_current_status=lambda: {"is_long_stay": False, "stay_duration": 0.0},
            stay_info=None,
        ),
        head_shake_detector=head_shake_detector,
        hand_raise_detector=SimpleNamespace(
            detect=lambda **kwargs: (_ for _ in ()).throw(ValueError("landmark issue"))
        ),
        posture_monitor=SimpleNamespace(
            update=lambda *a, **k: [],
            get_status=lambda: {
                "forward_ratio": 0.1,
                "avg_score": 0.2,
                "sample_count": 3,
            },
        ),
        frame_idx=4,
    )

    state.prev_head_shake_horizontal = True
    state.prev_head_shake_vertical = True
    state.head_shake_blink_start_time = 0.8
    state.head_shake_blink_last_toggle_time = 0.5
    state.head_shake_blink_is_active = False
    state.head_shake_blink_duration = 2.0

    state.prev_hand_raised_left = True
    state.prev_hand_raised_right = True
    state.blink_start_time = 0.7
    state.blink_last_toggle_time = 0.4
    state.blink_is_active = False
    state.blink_duration = 1.5

    processed_frame, results, alerts, aux = process_frame(
        frame=frame,
        t=1.0,
        state=state,
        debug_csv_writer=None,
    )

    assert processed_frame is frame
    assert alerts == []
    assert results[Angle.HEAD_HORIZONTAL_ROTATION]["angle"] == pytest.approx(3.0)
    assert results[Angle.HEAD_VERTICAL_NOD]["angle"] == pytest.approx(-2.0)
    assert state.prev_head_shake_horizontal is False
    assert state.prev_head_shake_vertical is False
    assert state.prev_hand_raised_left is False
    assert state.prev_hand_raised_right is False
    assert state.last_hand_statuses is None
    assert call_log["color_calls"] == 2  # toggled overlays for hand + head blink


def test_process_frame_ignores_non_dict_head_shake_results(monkeypatch):
    from src import video_processor as vp_module

    frame = np.zeros((5, 5, 3), dtype=np.uint8)
    landmarks = np.zeros((33, 4))

    class HeadShakeStub:
        def __init__(self):
            self.detect_calls = 0
            self.update_calls = 0

        def detect(self, landmarks, t, frame_idx):
            self.detect_calls += 1
            return ["not", "a", "dict"]

        def update(self, landmarks, t, frame_idx):
            self.update_calls += 1
            return ("still", "not", "a", "dict")

        def check_alerts(self, t):
            return []

    monkeypatch.setattr(vp_module, "draw_landmarks", lambda img, lm: img)
    monkeypatch.setattr(
        vp_module,
        "draw_analysis_results",
        lambda img, analysis, hand_statuses, lm, disable_japanese: img,
    )
    monkeypatch.setattr("src.video_processor.draw_detection_info", lambda img, *a, **k: img)
    monkeypatch.setattr("src.video_processor.draw_color_frame", lambda img, color, alpha=1.0: img)
    monkeypatch.setattr("src.video_processor.write_results_to_csv", lambda **kwargs: None)
    monkeypatch.setattr(
        vp_module,
        "BasicNotification",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        vp_module,
        "EmailNotificationDecorator",
        lambda base, **kwargs: SimpleNamespace(send=lambda *a, **k: None),
    )
    monkeypatch.setattr(
        vp_module,
        "_load_email_config",
        lambda: {
            "username": "",
            "password": "",
            "smtp_server": "",
            "smtp_port": "587",
            "subject": "",
            "recipient": "",
        },
    )

    state = _make_pipeline_state(
        pose=SimpleNamespace(estimate=lambda img: landmarks),
        analyzer=SimpleNamespace(analyze=lambda lm: {"baseline": 1.0}),
        user_classifier=SimpleNamespace(update=lambda *a, **k: [], get_current_alert=lambda: None),
        dwell_time_detector=SimpleNamespace(
            update=lambda *a, **k: None,
            get_current_status=lambda: {"is_long_stay": False, "stay_duration": 0.0},
            stay_info=None,
        ),
        head_shake_detector=HeadShakeStub(),
        hand_raise_detector=SimpleNamespace(detect=lambda **k: {}),
        posture_monitor=SimpleNamespace(
            update=lambda *a, **k: [],
            get_status=lambda: {"forward_ratio": 0.0, "avg_score": 0.0, "sample_count": 0},
        ),
        frame_idx=8,
    )

    processed_frame, results, alerts, aux = process_frame(frame, t=0.5, state=state)

    assert processed_frame is frame
    assert results == {"baseline": 1.0}
    assert alerts == []
    assert aux == {"dwell_alert": None}
    assert state.head_shake_detector.detect_calls == 1
    assert state.head_shake_detector.update_calls == 1


def test_process_frame_head_alert_missing_recipient_prints_message(monkeypatch, capsys):
    from src import video_processor as vp_module

    frame = np.zeros((6, 6, 3), dtype=np.uint8)
    landmarks = np.zeros((33, 4))

    def fake_draw(frame, *args, **kwargs):
        return frame

    monkeypatch.setattr("src.video_processor.draw_landmarks", fake_draw)
    monkeypatch.setattr(
        "src.video_processor.draw_analysis_results",
        lambda frame, results, hand_statuses, landmarks, disable_japanese: frame,
    )
    monkeypatch.setattr("src.video_processor.draw_detection_info", fake_draw)
    monkeypatch.setattr("src.video_processor.draw_color_frame", lambda frame, color, alpha=0.4: frame)
    monkeypatch.setattr("src.video_processor.write_results_to_csv", lambda **kwargs: None)

    monkeypatch.setattr(
        vp_module,
        "_load_email_config",
        lambda: {
            "username": "u",
            "password": "p",
            "smtp_server": "smtp",
            "smtp_port": "587",
            "subject": "s",
            "recipient": "",
        },
    )
    monkeypatch.setattr(vp_module, "BasicNotification", lambda: SimpleNamespace())
    monkeypatch.setattr(
        vp_module,
        "EmailNotificationDecorator",
        lambda base, **kwargs: SimpleNamespace(
            send=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("should not send"))
        ),
    )

    class HeadShakeDetectorStub:
        def detect(self, landmarks, t, frame_idx):
            return {
                "horizontal_state": "HORIZONTAL_SHAKE",
                "vertical_state": "VERTICAL_NOD",
                "horizontal_angle": 12.0,
                "vertical_angle": -4.0,
                "confidence": 0.9,
            }

        def update(self, *a, **k):
            return {}

        def check_alerts(self, t):
            return ["[!] Horizontal Head Shake Detected"]

    state = _make_pipeline_state(
        pose=SimpleNamespace(estimate=lambda img: landmarks),
        analyzer=SimpleNamespace(analyze=lambda lm: {}),
        user_classifier=SimpleNamespace(update=lambda *a, **k: [], get_current_alert=lambda: None),
        dwell_time_detector=SimpleNamespace(
            update=lambda *a, **k: None,
            get_current_status=lambda: {"is_long_stay": False, "stay_duration": 0.0},
            stay_info=None,
        ),
        head_shake_detector=HeadShakeDetectorStub(),
        hand_raise_detector=SimpleNamespace(detect=lambda **k: {}),
        posture_monitor=SimpleNamespace(
            update=lambda *a, **k: [],
            get_status=lambda: {"forward_ratio": 0.0, "avg_score": 0.0, "sample_count": 0},
        ),
        frame_idx=3,
    )

    process_frame(frame, t=1.7, state=state)
    captured = capsys.readouterr()
    assert "[Email] Missing EMAIL_RECIPIENT; would send:" in captured.out


def _apply_common_state_defaults(state) -> None:
    """Populate attributes required by process_frame for blinking/head shake state."""

    if not hasattr(state, "prev_hand_raised_left"):
        state.prev_hand_raised_left = False
    if not hasattr(state, "prev_hand_raised_right"):
        state.prev_hand_raised_right = False
    if not hasattr(state, "blink_start_time"):
        state.blink_start_time = None
    if not hasattr(state, "blink_is_active"):
        state.blink_is_active = False
    if not hasattr(state, "blink_color"):
        state.blink_color = "255,0,0"
    if not hasattr(state, "blink_duration"):
        state.blink_duration = 3.0
    if not hasattr(state, "blink_last_toggle_time"):
        state.blink_last_toggle_time = 0.0
    if not hasattr(state, "head_shake_blink_start_time"):
        state.head_shake_blink_start_time = None
    if not hasattr(state, "head_shake_blink_is_active"):
        state.head_shake_blink_is_active = False
    if not hasattr(state, "head_shake_blink_color"):
        state.head_shake_blink_color = "255,0,0"
    if not hasattr(state, "head_shake_blink_duration"):
        state.head_shake_blink_duration = 3.0
    if not hasattr(state, "head_shake_blink_last_toggle_time"):
        state.head_shake_blink_last_toggle_time = 0.0
    if not hasattr(state, "prev_head_shake_horizontal"):
        state.prev_head_shake_horizontal = False
    if not hasattr(state, "prev_head_shake_vertical"):
        state.prev_head_shake_vertical = False


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
    def __init__(self, detect_dict, check_alerts_list):
        self._detect_dict = dict(detect_dict)
        self._alerts = list(check_alerts_list)
        self.detect_calls = 0
        self.update_calls = 0
        self.check_calls = 0

    def detect(self, landmarks, t, frame_idx):
        self.detect_calls += 1
        return dict(self._detect_dict)

    def update(self, landmarks, t, frame_idx):
        self.update_calls += 1
        return _normalize_head_shake_results(self._detect_dict)

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
    _apply_common_state_defaults(s)
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
        head_update_dict={
            "horizontal_state": MovementState.HORIZONTAL_SHAKE,
            "vertical_state": MovementState.HEAD_STATIC,
            "horizontal_angle": 12.5,
            "vertical_angle": -1.0,
            "confidence": 0.9,
        },
        head_alerts=["HEAD_ALERT_A", "HEAD_ALERT_B"],
        frame_idx=7,
        disable_jp=False,
    )

    frame_in = np.zeros((480, 640, 3), dtype=np.uint8)
    out_frame, results, alerts, aux = fn(frame_in, t=1.23, state=state)

    assert out_frame is frame_in
    assert "knee" in results
    assert Angle.HEAD_HORIZONTAL_ROTATION in results
    assert Angle.HEAD_VERTICAL_NOD in results
    assert results[Angle.HEAD_HORIZONTAL_ROTATION]["angle"] == 12.5
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
    _apply_common_state_defaults(s)

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
    _apply_common_state_defaults(s)

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
    _apply_common_state_defaults(s)

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
    _apply_common_state_defaults(s)

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
    _apply_common_state_defaults(s)

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
    _apply_common_state_defaults(s)

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
    _apply_common_state_defaults(s)

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
    _apply_common_state_defaults(s)

    # Patch draw functions to pass-through
    monkeypatch.setattr("src.video_processor.draw_landmarks", lambda f, lm: f)
    monkeypatch.setattr(
        "src.video_processor.draw_analysis_results",
        lambda f, r, hand_statuses, lm, disable_japanese: f,
    )
    monkeypatch.setattr("src.video_processor.draw_detection_info", lambda f, *a, **k: f)
    monkeypatch.setattr("src.video_processor.draw_color_frame", lambda f, *a, **k: f)

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
        _apply_common_state_defaults(s)

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


def test_process_frame_email_notification_right_hand_transition(monkeypatch, capsys):
    """
    Test lines 507-508: Email notification when right hand transition is detected.
    When right_transition is True, "Right hand raised" should be added to parts.
    """
    expected_hand_statuses = {"left_hand_raised": False, "right_hand_raised": True}

    class HandRaiseDetectorRightHand:
        def detect(self, landmarks):
            return expected_hand_statuses

    class State:
        pass

    s = State()
    s.pose = _PoseFake(landmarks=np.zeros((33, 4)))
    s.analyzer = _AnalyzerFake({})
    s.posture_monitor = _PostureMonitorFake([])
    s.dwell_time_detector = _DwellFake(None)
    s.head_shake_detector = _HeadShakeFake({}, [])
    s.hand_raise_detector = HandRaiseDetectorRightHand()
    s.user_classifier = object()
    s.frame_idx = 5
    s.disable_jp = True
    s.last_landmarks = None
    s.last_head_alerts = []
    s.last_hand_statuses = "UNTOUCHED"
    _apply_common_state_defaults(s)
    s.prev_hand_raised_right = False  # Right transition will be True

    # Mock _load_email_config to return config with empty recipient (tests line 514)
    # Also mock EmailNotificationDecorator to avoid initialization issues
    monkeypatch.setattr(
        "src.video_processor._load_email_config",
        lambda: {
            "username": "test@example.com",
            "password": "password",
            "smtp_server": "smtp.example.com",
            "smtp_port": "587",
            "subject": "Test",
            "recipient": None,  # No recipient - should print message instead of sending
        },
    )

    # Mock EmailNotificationDecorator to return a simple mock
    class MockEmailDecorator:
        def __init__(self, *args, **kwargs):
            pass

        def send(self, msg, recipient):
            pass

    monkeypatch.setattr("src.video_processor.EmailNotificationDecorator", MockEmailDecorator)

    # Patch draw functions
    monkeypatch.setattr("src.video_processor.draw_landmarks", lambda f, lm: f)
    monkeypatch.setattr(
        "src.video_processor.draw_analysis_results",
        lambda f, r, hand_statuses, lm, disable_japanese: f,
    )
    monkeypatch.setattr("src.video_processor.draw_detection_info", lambda f, *a, **k: f)
    monkeypatch.setattr("src.video_processor.draw_color_frame", lambda f, *a, **k: f)

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    out_frame, results, alerts, aux = fn(frame, t=2.0, state=s)

    # Verify right hand transition triggered email notification message
    captured = capsys.readouterr()
    assert "Right hand raised at 2.000s" in captured.out
    assert s.last_hand_statuses == expected_hand_statuses
    assert s.prev_hand_raised_right is True


def test_process_frame_email_notification_both_hands_transition(monkeypatch, capsys):
    """
    Test lines 507-508: Email notification when both hands transition is detected.
    When both left_transition and right_transition are True, both parts should be added.
    """
    expected_hand_statuses = {"left_hand_raised": True, "right_hand_raised": True}

    class HandRaiseDetectorBothHands:
        def detect(self, landmarks):
            return expected_hand_statuses

    class State:
        pass

    s = State()
    s.pose = _PoseFake(landmarks=np.zeros((33, 4)))
    s.analyzer = _AnalyzerFake({})
    s.posture_monitor = _PostureMonitorFake([])
    s.dwell_time_detector = _DwellFake(None)
    s.head_shake_detector = _HeadShakeFake({}, [])
    s.hand_raise_detector = HandRaiseDetectorBothHands()
    s.user_classifier = object()
    s.frame_idx = 10
    s.disable_jp = True
    s.last_landmarks = None
    s.last_head_alerts = []
    s.last_hand_statuses = "UNTOUCHED"
    _apply_common_state_defaults(s)
    s.prev_hand_raised_left = False  # Both transitions will be True
    s.prev_hand_raised_right = False

    # Mock _load_email_config to return config with empty recipient
    # Also mock EmailNotificationDecorator to avoid initialization issues
    monkeypatch.setattr(
        "src.video_processor._load_email_config",
        lambda: {
            "username": "test@example.com",
            "password": "password",
            "smtp_server": "smtp.example.com",
            "smtp_port": "587",
            "subject": "Test",
            "recipient": None,  # No recipient - should print message instead of sending
        },
    )

    # Mock EmailNotificationDecorator to return a simple mock
    class MockEmailDecorator:
        def __init__(self, *args, **kwargs):
            pass

        def send(self, msg, recipient):
            pass

    monkeypatch.setattr("src.video_processor.EmailNotificationDecorator", MockEmailDecorator)

    # Patch draw functions
    monkeypatch.setattr("src.video_processor.draw_landmarks", lambda f, lm: f)
    monkeypatch.setattr(
        "src.video_processor.draw_analysis_results",
        lambda f, r, hand_statuses, lm, disable_japanese: f,
    )
    monkeypatch.setattr("src.video_processor.draw_detection_info", lambda f, *a, **k: f)
    monkeypatch.setattr("src.video_processor.draw_color_frame", lambda f, *a, **k: f)

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    out_frame, results, alerts, aux = fn(frame, t=3.0, state=s)

    # Verify both hands transition triggered email notification message
    captured = capsys.readouterr()
    assert "Left hand raised" in captured.out
    assert "Right hand raised" in captured.out
    assert (
        "Left hand raised & Right hand raised" in captured.out or "Right hand raised & Left hand raised" in captured.out
    )
    assert s.prev_hand_raised_left is True
    assert s.prev_hand_raised_right is True


def test_process_frame_email_notification_with_recipient(monkeypatch):
    """
    Test lines 511-512: Email notification when recipient exists.
    When recipient is set, notification.send should be called.
    """
    expected_hand_statuses = {"left_hand_raised": True, "right_hand_raised": False}

    class HandRaiseDetectorWithRecipient:
        def detect(self, landmarks):
            return expected_hand_statuses

    class MockNotification:
        def __init__(self):
            self.send_calls = []

        def send(self, msg, recipient):
            self.send_calls.append((msg, recipient))

    class State:
        pass

    s = State()
    s.pose = _PoseFake(landmarks=np.zeros((33, 4)))
    s.analyzer = _AnalyzerFake({})
    s.posture_monitor = _PostureMonitorFake([])
    s.dwell_time_detector = _DwellFake(None)
    s.head_shake_detector = _HeadShakeFake({}, [])
    s.hand_raise_detector = HandRaiseDetectorWithRecipient()
    s.user_classifier = object()
    s.frame_idx = 7
    s.disable_jp = True
    s.last_landmarks = None
    s.last_head_alerts = []
    s.last_hand_statuses = "UNTOUCHED"
    _apply_common_state_defaults(s)

    mock_notification = MockNotification()

    # Mock _load_email_config to return a recipient
    monkeypatch.setattr(
        "src.video_processor._load_email_config",
        lambda: {
            "username": "test@example.com",
            "password": "password",
            "smtp_server": "smtp.example.com",
            "smtp_port": "587",
            "subject": "Test",
            "recipient": "recipient@example.com",
        },
    )

    # Mock EmailNotificationDecorator to return our mock
    def mock_email_decorator(notification, **kwargs):
        return mock_notification

    monkeypatch.setattr("src.video_processor.EmailNotificationDecorator", mock_email_decorator)

    # Patch draw functions
    monkeypatch.setattr("src.video_processor.draw_landmarks", lambda f, lm: f)
    monkeypatch.setattr(
        "src.video_processor.draw_analysis_results",
        lambda f, r, hand_statuses, lm, disable_japanese: f,
    )
    monkeypatch.setattr("src.video_processor.draw_detection_info", lambda f, *a, **k: f)
    monkeypatch.setattr("src.video_processor.draw_color_frame", lambda f, *a, **k: f)

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    out_frame, results, alerts, aux = fn(frame, t=4.0, state=s)

    # Verify notification.send was called with correct arguments
    assert len(mock_notification.send_calls) == 1
    assert mock_notification.send_calls[0][0] == "Left hand raised at 4.000s (frame 7)"
    assert mock_notification.send_calls[0][1] == "recipient@example.com"


def test_process_frame_email_notification_exception_handling(monkeypatch, capsys):
    """
    Test lines 515-516: Exception handling in email notification.
    When email notification raises an exception, it should be caught and printed.
    """
    expected_hand_statuses = {"left_hand_raised": True, "right_hand_raised": False}

    class HandRaiseDetectorWithException:
        def detect(self, landmarks):
            return expected_hand_statuses

    class State:
        pass

    s = State()
    s.pose = _PoseFake(landmarks=np.zeros((33, 4)))
    s.analyzer = _AnalyzerFake({})
    s.posture_monitor = _PostureMonitorFake([])
    s.dwell_time_detector = _DwellFake(None)
    s.head_shake_detector = _HeadShakeFake({}, [])
    s.hand_raise_detector = HandRaiseDetectorWithException()
    s.user_classifier = object()
    s.frame_idx = 8
    s.disable_jp = True
    s.last_landmarks = None
    s.last_head_alerts = []
    s.last_hand_statuses = "UNTOUCHED"
    _apply_common_state_defaults(s)

    # Mock _load_email_config to raise an exception
    def mock_load_email_config_raises():
        raise ValueError("Config file error")

    monkeypatch.setattr("src.video_processor._load_email_config", mock_load_email_config_raises)

    # Patch draw functions
    monkeypatch.setattr("src.video_processor.draw_landmarks", lambda f, lm: f)
    monkeypatch.setattr(
        "src.video_processor.draw_analysis_results",
        lambda f, r, hand_statuses, lm, disable_japanese: f,
    )
    monkeypatch.setattr("src.video_processor.draw_detection_info", lambda f, *a, **k: f)
    monkeypatch.setattr("src.video_processor.draw_color_frame", lambda f, *a, **k: f)

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    out_frame, results, alerts, aux = fn(frame, t=5.0, state=s)

    # Verify exception was caught and error message was printed
    captured = capsys.readouterr()
    assert "Email notification error: Config file error" in captured.out
    # Function should complete successfully despite the exception
    assert out_frame is frame


def test_process_frame_blink_toggle_logic(monkeypatch):
    """
    Test lines 546-548: Blink toggle logic when time_since_last_toggle >= 0.15.
    When enough time has passed, blink_is_active should toggle and blink_last_toggle_time should update.
    """
    landmarks = np.zeros((33, 4))
    state = _mk_state(
        landmarks=landmarks,
        analyzer_result={},
        posture_alerts=[],
        dwell_alert=None,
        head_update_dict={},
        head_alerts=[],
        frame_idx=0,
    )

    # Set up blinking state - already started, needs to toggle
    state.blink_start_time = 1.0  # Started at t=1.0
    state.blink_is_active = True
    state.blink_last_toggle_time = 1.0  # Last toggle at t=1.0
    state.prev_hand_raised_left = False
    state.prev_hand_raised_right = False

    # Mock draw_color_frame to track calls
    draw_calls = []

    def mock_draw_color_frame(frame, color, alpha=0.4):
        draw_calls.append((color, alpha))
        return frame

    monkeypatch.setattr("src.video_processor.draw_landmarks", lambda f, lm: f)
    monkeypatch.setattr(
        "src.video_processor.draw_analysis_results",
        lambda f, r, hand_statuses, lm, disable_japanese: f,
    )
    monkeypatch.setattr("src.video_processor.draw_detection_info", lambda f, *a, **k: f)
    monkeypatch.setattr("src.video_processor.draw_color_frame", mock_draw_color_frame)

    # Call at t=1.16 (0.16 seconds after last toggle, >= 0.15)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    out_frame, results, alerts, aux = fn(frame, t=1.16, state=state)

    # Verify blink state was toggled
    assert state.blink_is_active is False  # Should toggle from True to False
    assert state.blink_last_toggle_time == 1.16
    # Since blink_is_active is now False, draw_color_frame should not be called
    assert len(draw_calls) == 0


def test_process_frame_blink_toggle_active_calls_draw(monkeypatch):
    """
    Test lines 546-548 and 551-552: When blink is active after toggle, draw_color_frame should be called.
    """
    landmarks = np.zeros((33, 4))
    state = _mk_state(
        landmarks=landmarks,
        analyzer_result={},
        posture_alerts=[],
        dwell_alert=None,
        head_update_dict={},
        head_alerts=[],
        frame_idx=0,
    )

    # Set up blinking state - will toggle to True
    state.blink_start_time = 1.0
    state.blink_is_active = False  # Will toggle to True
    state.blink_last_toggle_time = 1.0
    state.prev_hand_raised_left = False
    state.prev_hand_raised_right = False

    draw_calls = []

    def mock_draw_color_frame(frame, color, alpha=0.4):
        draw_calls.append((color, alpha))
        return frame

    monkeypatch.setattr("src.video_processor.draw_landmarks", lambda f, lm: f)
    monkeypatch.setattr(
        "src.video_processor.draw_analysis_results",
        lambda f, r, hand_statuses, lm, disable_japanese: f,
    )
    monkeypatch.setattr("src.video_processor.draw_detection_info", lambda f, *a, **k: f)
    monkeypatch.setattr("src.video_processor.draw_color_frame", mock_draw_color_frame)

    # Call at t=1.16 (0.16 seconds after last toggle)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    out_frame, results, alerts, aux = fn(frame, t=1.16, state=state)

    # Verify blink state was toggled to True
    assert state.blink_is_active is True
    assert state.blink_last_toggle_time == 1.16
    # Since blink_is_active is now True, draw_color_frame should be called
    assert len(draw_calls) == 1
    assert draw_calls[0] == (state.blink_color, 0.4)


def test_process_frame_blink_duration_expired(monkeypatch):
    """
    Test lines 553-556: Blinking duration expiration.
    When elapsed > blink_duration, blink_start_time and blink_is_active should be reset.
    """
    landmarks = np.zeros((33, 4))
    state = _mk_state(
        landmarks=landmarks,
        analyzer_result={},
        posture_alerts=[],
        dwell_alert=None,
        head_update_dict={},
        head_alerts=[],
        frame_idx=0,
    )

    # Set up blinking state that has expired
    state.blink_start_time = 1.0  # Started at t=1.0
    state.blink_is_active = True
    state.blink_last_toggle_time = 1.0
    state.blink_duration = 3.0  # Duration is 3 seconds
    state.prev_hand_raised_left = False
    state.prev_hand_raised_right = False

    draw_calls = []

    def mock_draw_color_frame(frame, color, alpha=0.4):
        draw_calls.append((color, alpha))
        return frame

    monkeypatch.setattr("src.video_processor.draw_landmarks", lambda f, lm: f)
    monkeypatch.setattr(
        "src.video_processor.draw_analysis_results",
        lambda f, r, hand_statuses, lm, disable_japanese: f,
    )
    monkeypatch.setattr("src.video_processor.draw_detection_info", lambda f, *a, **k: f)
    monkeypatch.setattr("src.video_processor.draw_color_frame", mock_draw_color_frame)

    # Call at t=5.0 (4.0 seconds after start, > 3.0 duration)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    out_frame, results, alerts, aux = fn(frame, t=5.0, state=state)

    # Verify blinking state was reset
    assert state.blink_start_time is None
    assert state.blink_is_active is False
    # draw_color_frame should not be called since blink is inactive
    assert len(draw_calls) == 0


def test_process_frame_head_blink_duration_expired(monkeypatch):
    """
    Test lines 785-788: Head blink duration expiration resets head-specific blink state.
    """
    landmarks = np.zeros((33, 4))
    state = _mk_state(
        landmarks=landmarks,
        analyzer_result={},
        posture_alerts=[],
        dwell_alert=None,
        head_update_dict={},
        head_alerts=[],
        frame_idx=0,
    )

    state.head_shake_blink_start_time = 0.5
    state.head_shake_blink_is_active = True
    state.head_shake_blink_last_toggle_time = 0.5
    state.head_shake_blink_duration = 1.5

    draw_calls = []

    def mock_draw_color_frame(frame, color, alpha=0.4):
        draw_calls.append((color, alpha))
        return frame

    monkeypatch.setattr("src.video_processor.draw_landmarks", lambda f, lm: f)
    monkeypatch.setattr(
        "src.video_processor.draw_analysis_results",
        lambda f, r, hand_statuses, lm, disable_japanese: f,
    )
    monkeypatch.setattr("src.video_processor.draw_detection_info", lambda f, *a, **k: f)
    monkeypatch.setattr("src.video_processor.draw_color_frame", mock_draw_color_frame)

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    out_frame, results, alerts, aux = fn(frame, t=3.0, state=state)

    assert state.head_shake_blink_start_time is None
    assert state.head_shake_blink_is_active is False
    assert len(draw_calls) == 0
