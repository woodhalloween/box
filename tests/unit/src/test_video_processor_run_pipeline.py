# test_run_pipeline.py
import numpy as np

from src.video_processor import run_pipeline


# --- Minimal PipelineState duck type for the tests ---
class _StubState:
    def __init__(self, frame_idx=0):
        self.frame_idx = frame_idx
        self.posture_monitor = object()
        self.dwell_time_detector = object()
        self.head_shake_detector = object()
        self.last_head_alerts = None
        self.last_landmarks = None
        self.user_classifier = object()  # not used by run_pipeline but often part of state


def _make_frame_iter(n):
    """Iterator yielding (t, frame) n times."""
    for i in range(n):
        yield float(i) + 0.5, np.zeros((4, 5, 3), dtype=np.uint8)


def test_run_pipeline_happy_path_no_break(monkeypatch):
    """
    - 2 frames
    - CSV & video writers provided
    - preview True with non-'q' key
    - verify CSV/video/UI calls and frame_idx increments
    """
    calls = {"process": [], "csv": [], "video": [], "imshow": 0, "waitKey": 0, "destroy": 0}

    # Fake process_frame that also sets state's last_* so CSV path uses non-None values
    def fake_process_frame(frame, t, state):
        calls["process"].append((t, state.frame_idx))
        state.last_landmarks = "LM-" + str(state.frame_idx)
        state.last_head_alerts = ["HS-" + str(state.frame_idx)]
        # annotated frame (same), minimal results, alerts, aux
        return frame, {"knee": {"angle": 90.0}}, ["ALERT"], {"dwell_alert": "DWELL"}

    monkeypatch.setattr("src.video_processor.process_frame", fake_process_frame)

    def fake_write_results_to_csv(csv_writer, **kwargs):
        calls["csv"].append((csv_writer, kwargs))

    monkeypatch.setattr("src.video_processor.write_results_to_csv", fake_write_results_to_csv)

    class FakeVW:
        def write(self, frame):
            calls["video"].append(frame.shape)

    # UI: show + waitKey (not 'q')
    def fake_imshow(name, frame):
        calls["imshow"] += 1

    def fake_waitKey(ms):  # noqa: N802
        calls["waitKey"] += 1
        return ord("a")  # continue

    def fake_destroy():
        calls["destroy"] += 1

    monkeypatch.setattr("src.video_processor.cv2.imshow", fake_imshow)
    monkeypatch.setattr("src.video_processor.cv2.waitKey", fake_waitKey)
    monkeypatch.setattr("src.video_processor.cv2.destroyAllWindows", fake_destroy)

    state = _StubState(frame_idx=10)
    csv_writer = object()
    video_writer = FakeVW()

    run_pipeline(
        _make_frame_iter(2),
        csv_writer=csv_writer,
        video_writer=video_writer,
        state=state,
        preview=True,
        window_name="Win",
    )

    # process called twice, frame_idx advanced twice
    assert calls["process"] == [(0.5, 10), (1.5, 11)]
    assert state.frame_idx == 12

    # CSV called twice with expected dwell/head/landmarks
    assert len(calls["csv"]) == 2
    for i, (_, kwargs) in enumerate(calls["csv"]):
        assert kwargs["timestamp"] == float(i) + 0.5
        assert kwargs["frame_number"] == 10 + i
        assert kwargs["dwell_alert"] == "DWELL"
        assert kwargs["head_shake_alerts"] == [f"HS-{10 + i}"]
        assert kwargs["landmarks"] == f"LM-{10 + i}"

    # Video writer used twice
    assert calls["video"] == [(4, 5, 3), (4, 5, 3)]

    # UI used twice, then cleaned up once
    assert calls["imshow"] == 2
    assert calls["waitKey"] == 2
    assert calls["destroy"] == 0


def test_run_pipeline_break_on_q(monkeypatch):
    """
    - break on 'q' after first frame
    - ensure only one iteration worth of sinks/UI
    """
    calls = {"process": 0, "csv": 0, "video": 0, "destroy": 0}

    def fake_process_frame(frame, t, state):
        calls["process"] += 1
        state.last_landmarks = "LM"
        state.last_head_alerts = ["HS"]
        return frame, {}, [], {"dwell_alert": None}

    monkeypatch.setattr("src.video_processor.process_frame", fake_process_frame)
    monkeypatch.setattr(
        "src.video_processor.write_results_to_csv", lambda *a, **k: calls.__setitem__("csv", calls["csv"] + 1)
    )

    class FakeVW:
        def write(self, frame):
            calls["video"] += 1

    # UI: press 'q'
    monkeypatch.setattr("src.video_processor.cv2.imshow", lambda *a, **k: None)
    monkeypatch.setattr("src.video_processor.cv2.waitKey", lambda *_: ord("q"))
    monkeypatch.setattr(
        "src.video_processor.cv2.destroyAllWindows", lambda: calls.__setitem__("destroy", calls["destroy"] + 1)
    )

    state = _StubState(frame_idx=0)
    run_pipeline(_make_frame_iter(5), csv_writer=object(), video_writer=FakeVW(), state=state, preview=True)

    assert calls["process"] == 1
    assert calls["csv"] == 1
    assert calls["video"] == 1
    assert state.frame_idx == 0
    assert calls["destroy"] == 0


def test_run_pipeline_preview_false_skips_ui(monkeypatch):
    """
    - preview=False → no imshow/waitKey
    """
    calls = {"process": 0, "csv": 0, "video": 0, "imshow": 0, "waitKey": 0, "destroy": 0}

    def fake_process_frame(frame, t, state):
        calls["process"] += 1
        state.last_landmarks = "LM"
        state.last_head_alerts = []
        return frame, {}, [], {"dwell_alert": None}

    monkeypatch.setattr("src.video_processor.process_frame", fake_process_frame)
    monkeypatch.setattr(
        "src.video_processor.write_results_to_csv", lambda *a, **k: calls.__setitem__("csv", calls["csv"] + 1)
    )

    class FakeVW:
        def write(self, frame):
            calls["video"] += 1

    # If these are accidentally called, bump counters (so we can assert 0)
    monkeypatch.setattr(
        "src.video_processor.cv2.imshow", lambda *a, **k: calls.__setitem__("imshow", calls["imshow"] + 1)
    )
    monkeypatch.setattr(
        "src.video_processor.cv2.waitKey", lambda *_: calls.__setitem__("waitKey", calls["waitKey"] + 1) or 0
    )
    monkeypatch.setattr(
        "src.video_processor.cv2.destroyAllWindows", lambda: calls.__setitem__("destroy", calls["destroy"] + 1)
    )

    state = _StubState(frame_idx=3)
    run_pipeline(_make_frame_iter(3), csv_writer=object(), video_writer=FakeVW(), state=state, preview=False)

    assert calls["process"] == 3
    assert calls["csv"] == 3
    assert calls["video"] == 3
    assert calls["imshow"] == 0
    assert calls["waitKey"] == 0
    assert state.frame_idx == 6
    assert calls["destroy"] == 0


def test_run_pipeline_headless_ui_exception_path(monkeypatch):
    """
    - cv2.imshow raises → caught and ignored
    - ensure loop continues and no waitKey call is required
    """
    calls = {"process": 0, "csv": 0, "video": 0, "imshow": 0, "destroy": 0}

    def fake_process_frame(frame, t, state):
        calls["process"] += 1
        return frame, {}, [], {"dwell_alert": None}

    monkeypatch.setattr("src.video_processor.process_frame", fake_process_frame)
    monkeypatch.setattr(
        "src.video_processor.write_results_to_csv", lambda *a, **k: calls.__setitem__("csv", calls["csv"] + 1)
    )

    class FakeVW:
        def write(self, frame):
            calls["video"] += 1

    def boom(*_a, **_k):
        calls["imshow"] += 1
        raise RuntimeError("headless")

    # imshow throws; waitKey should not be needed
    monkeypatch.setattr("src.video_processor.cv2.imshow", boom)
    monkeypatch.setattr(
        "src.video_processor.cv2.waitKey",
        lambda *_: (_ for _ in ()).throw(AssertionError("waitKey should not be called")),
    )
    monkeypatch.setattr(
        "src.video_processor.cv2.destroyAllWindows", lambda: calls.__setitem__("destroy", calls["destroy"] + 1)
    )

    state = _StubState(frame_idx=0)
    run_pipeline(_make_frame_iter(2), csv_writer=object(), video_writer=FakeVW(), state=state, preview=True)

    assert calls["process"] == 2
    assert calls["csv"] == 2
    assert calls["video"] == 2
    assert calls["imshow"] == 2
    assert calls["destroy"] == 0


def test_run_pipeline_destroy_windows_suppressed(monkeypatch):
    """
    - Ensure the finally block's contextlib.suppress swallows destroyAllWindows errors.
    """
    calls = {"destroy": 0}

    monkeypatch.setattr("src.video_processor.process_frame", lambda f, t, s: (f, {}, [], {"dwell_alert": None}))
    monkeypatch.setattr("src.video_processor.write_results_to_csv", lambda *a, **k: None)
    monkeypatch.setattr("src.video_processor.cv2.imshow", lambda *a, **k: None)
    monkeypatch.setattr("src.video_processor.cv2.waitKey", lambda *_: ord("a"))

    def boom_destroy():
        calls["destroy"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr("src.video_processor.cv2.destroyAllWindows", boom_destroy)

    state = _StubState(frame_idx=0)
    # If suppression didn't work, this would raise
    run_pipeline(_make_frame_iter(1), csv_writer=None, video_writer=None, state=state, preview=True)

    assert calls["destroy"] == 1
