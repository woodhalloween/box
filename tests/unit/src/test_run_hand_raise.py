import builtins
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pytest

if "cv2" not in sys.modules:  # pragma: no cover - テスト環境でOpenCVが無い場合のフォールバック

    def _circle(image, center, radius, color, thickness):
        cx, cy = center
        yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
        distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = np.abs(distance - radius) <= max(thickness, 1)
        image[mask] = color
        return image

    cv2_stub = types.SimpleNamespace(
        CAP_PROP_FPS=5,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FRAME_COUNT=7,
        VideoCapture=None,
        VideoWriter=None,
        VideoWriter_fourcc=lambda *args, **kwargs: 0,
        circle=_circle,
        imshow=lambda *args, **kwargs: None,
        waitKey=lambda *args, **kwargs: 0,
        destroyAllWindows=lambda: None,
    )
    sys.modules["cv2"] = cv2_stub

if "tqdm" not in sys.modules:  # pragma: no cover - テスト環境でtqdmが無い場合のフォールバック
    tqdm_stub = types.ModuleType("tqdm")

    def _tqdm_placeholder(*args, **kwargs):
        class _Dummy:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def update(self, *args, **kwargs):  # pragma: no cover - 置き換え前のフォールバック
                return None

        return _Dummy()

    tqdm_stub.tqdm = _tqdm_placeholder
    sys.modules["tqdm"] = tqdm_stub

if "yaml" not in sys.modules:  # pragma: no cover - テスト環境でPyYAMLが無い場合のフォールバック
    yaml_stub = types.ModuleType("yaml")

    def _safe_load(stream):  # pragma: no cover - 置き換え前のフォールバック
        return {}

    yaml_stub.safe_load = _safe_load
    sys.modules["yaml"] = yaml_stub

import src.drawing_utils as drawing_utils
import src.run_hand_raise as run_hand_raise


class DummyProgress:
    def __init__(self, *args, **kwargs):
        self.updates: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, value):
        self.updates.append(value)


def test_draw_dwell_status_draws_circle_and_text(monkeypatch):
    base_image = np.zeros((120, 160, 3), dtype=np.uint8)

    calls: list[tuple[str, tuple[int, int], int, tuple[int, int, int]]] = []

    def fake_draw(image, text, position, font_size, color):
        calls.append((text, position, font_size, color))
        return image

    monkeypatch.setattr(drawing_utils, "draw_japanese_text", fake_draw)

    dwell_status = {
        "stay_duration": 12.3,
        "state": "STAYING",
        "is_long_stay": True,
        "confidence": 0.75,
        "hip_position": (50, 60),
    }

    result = drawing_utils.draw_dwell_status(
        base_image,
        dwell_status,
        dwell_alert="警告",
        font_size=12,
        position=(10, 20),
    )

    # draw_japanese_text が滞在情報・状態・アラートの 3 行で呼び出される
    assert len(calls) == 3
    assert calls[0][0].startswith("滞在: 12.3s")
    assert calls[1][0].startswith("状態: STAYING")
    assert calls[2][0] == "警告"

    # 円描画により腰位置周辺に赤いピクセルが存在することを確認
    circle_region = result[52:68, 42:58]
    assert np.any(np.all(circle_region == (0, 0, 255), axis=-1))


def test_main_returns_when_input_missing(monkeypatch, capsys, tmp_path):
    missing_path = tmp_path / "not_found.mp4"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--input",
            str(missing_path),
            "--no_display",
            "--no_video_output",
            "--no_csv_output",
        ],
    )

    result = run_hand_raise.main()

    captured = capsys.readouterr()
    assert "エラー: 動画ファイルが見つかりません" in captured.out
    assert result is None


def test_main_processes_frames_with_stubs(monkeypatch, capsys, tmp_path):
    input_path = tmp_path / "sample.mp4"
    input_path.write_bytes(b"dummy")

    frames = [np.zeros((72, 128, 3), dtype=np.uint8) for _ in range(2)]

    capture_instances = []

    class DummyCapture:
        def __init__(self, path):
            self.path = Path(path)
            self.index = 0
            self.released = False
            capture_instances.append(self)

        def isOpened(self):  # noqa: N802 - mimic OpenCV (cv2) API
            return True

        def read(self):
            if self.index < len(frames):
                frame = frames[self.index]
                self.index += 1
                return True, frame.copy()
            return False, None

        def get(self, prop):
            if prop == run_hand_raise.cv2.CAP_PROP_FPS:
                return 25.0
            if prop == run_hand_raise.cv2.CAP_PROP_FRAME_WIDTH:
                return frames[0].shape[1]
            if prop == run_hand_raise.cv2.CAP_PROP_FRAME_HEIGHT:
                return frames[0].shape[0]
            if prop == run_hand_raise.cv2.CAP_PROP_FRAME_COUNT:
                return len(frames)
            return 0

        def release(self):
            self.released = True

    video_writer_instances = []

    class DummyVideoWriter:
        def __init__(self, path, fourcc, fps, size):
            self.path = path
            self.fourcc = fourcc
            self.fps = fps
            self.size = size
            self.frames = []
            self.released = False
            video_writer_instances.append(self)

        def isOpened(self):  # noqa: N802 - mimic OpenCV (cv2) API
            return True

        def write(self, frame):
            self.frames.append(frame.copy())

        def release(self):
            self.released = True

    class DummyAppConfig:
        values = {
            "hand_raise.visibility_threshold": 0.8,
            "hand_raise.min_consecutive_frames": 7,
            "dwell_time_detector.stay_threshold_sec": 20.0,
            "dwell_time_detector.confidence_threshold": 0.6,
            "dwell_time_detector.advanced_detection.spike_threshold": 2.5,
            "dwell_time_detector.advanced_detection.stability_threshold_px": 120.0,
            "dwell_time_detector.advanced_detection.grace_period_sec": 2.0,
            "dwell_time_detector.use_normalization": True,
            "dwell_time_detector.normalization_base": "hips",
        }
        calls = []
        instances = []

        def __init__(self, path):
            self.path = Path(path)
            DummyAppConfig.instances.append(self)

        def getfloat(self, key, fallback):
            DummyAppConfig.calls.append(("getfloat", key, fallback))
            return float(self.values.get(key, fallback))

        def getint(self, key, fallback):
            DummyAppConfig.calls.append(("getint", key, fallback))
            return int(self.values.get(key, fallback))

        def getboolean(self, key, fallback):
            DummyAppConfig.calls.append(("getboolean", key, fallback))
            return bool(self.values.get(key, fallback))

        def get(self, key, fallback):
            DummyAppConfig.calls.append(("get", key, fallback))
            return self.values.get(key, fallback)

    pose_estimator_instances = []

    class DummyPoseEstimator:
        def __init__(self):
            self.frames = []
            self.closed = False
            pose_estimator_instances.append(self)

        def estimate(self, frame):
            self.frames.append(frame)
            return "LANDMARKS"

        def close(self):
            self.closed = True

    hand_detector_instances = []

    class DummyHandRaiseDetector:
        def __init__(self, visibility_threshold, min_consecutive_frames):
            self.visibility_threshold = visibility_threshold
            self.min_consecutive_frames = min_consecutive_frames
            self.detect_calls = []
            hand_detector_instances.append(self)

        def detect(self, landmarks):
            self.detect_calls.append(landmarks)
            return {"left_hand_raised": True, "right_hand_raised": False}

    dwell_detector_instances = []

    class DummyDwellTimeDetector:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.updates = []
            dwell_detector_instances.append(self)

        def update(self, landmarks, frame_shape, timestamp):
            self.updates.append((landmarks, frame_shape, timestamp))
            return "LONG STAY" if len(self.updates) == 1 else None

        def get_current_status(self):
            return {
                "stay_duration": 4.2,
                "is_long_stay": True,
                "state": "ACTIVE",
                "confidence": 0.83,
                "hip_position": (10.0, 20.0),
            }

    draw_landmarks_calls = []
    monkeypatch.setattr(
        run_hand_raise,
        "draw_landmarks",
        lambda image, landmarks: draw_landmarks_calls.append((landmarks, image.shape)),
    )

    hand_status_calls = []

    def fake_draw_hand_raise_status(image, statuses):
        hand_status_calls.append(statuses)
        return image

    monkeypatch.setattr(run_hand_raise, "draw_hand_raise_status", fake_draw_hand_raise_status)

    dwell_draw_calls = []

    def fake_draw_dwell_status(image, status, alert):
        dwell_draw_calls.append((status, alert))
        return image

    monkeypatch.setattr(run_hand_raise, "draw_dwell_status", fake_draw_dwell_status)

    progress_updates = []

    class DummyProgress:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, value):
            progress_updates.append(value)

    monkeypatch.setattr(run_hand_raise, "tqdm", lambda *a, **kw: DummyProgress(*a, **kw))

    monkeypatch.setattr(run_hand_raise.cv2, "VideoCapture", DummyCapture)
    monkeypatch.setattr(run_hand_raise.cv2, "VideoWriter", DummyVideoWriter)
    monkeypatch.setattr(run_hand_raise.cv2, "VideoWriter_fourcc", lambda *args: 1234)
    monkeypatch.setattr(run_hand_raise.cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_hand_raise.cv2, "waitKey", lambda *args, **kwargs: 0)
    monkeypatch.setattr(run_hand_raise.cv2, "destroyAllWindows", lambda: None)

    monkeypatch.setattr(run_hand_raise, "AppConfig", DummyAppConfig)
    monkeypatch.setattr(run_hand_raise, "PoseEstimator", DummyPoseEstimator)
    monkeypatch.setattr(run_hand_raise, "HandRaiseDetector", DummyHandRaiseDetector)
    monkeypatch.setattr(run_hand_raise, "DwellTimeDetector", DummyDwellTimeDetector)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--input",
            str(input_path),
            "--no_display",
            "--draw_skeleton",
        ],
    )

    run_hand_raise.main()

    captured = capsys.readouterr()
    assert "処理が完了しました。" in captured.out
    assert str(input_path.with_name("sample_processed.mp4")) in captured.out
    assert str(input_path.with_name("sample_results.csv")) in captured.out

    assert DummyAppConfig.instances[0].path == Path("config.yaml")
    assert hand_detector_instances[0].visibility_threshold == pytest.approx(0.8)
    assert hand_detector_instances[0].min_consecutive_frames == 7

    dwell_kwargs = dwell_detector_instances[0].kwargs
    assert dwell_kwargs["stay_threshold_sec"] == pytest.approx(20.0)
    assert dwell_kwargs["confidence_threshold"] == pytest.approx(0.6)
    assert dwell_kwargs["spike_threshold"] == pytest.approx(2.5)
    assert dwell_kwargs["stability_threshold_px"] == pytest.approx(120.0)
    assert dwell_kwargs["grace_period_sec"] == pytest.approx(2.0)
    assert dwell_kwargs["use_normalization"] is True
    assert dwell_kwargs["normalization_base"] == "hips"

    pose_estimator = pose_estimator_instances[0]
    assert len(pose_estimator.frames) == len(frames)
    assert pose_estimator.closed is True

    hand_detector = hand_detector_instances[0]
    assert hand_detector.detect_calls == ["LANDMARKS", "LANDMARKS"]

    dwell_detector = dwell_detector_instances[0]
    assert len(dwell_detector.updates) == len(frames)
    assert progress_updates == [1, 1]

    video_writer = video_writer_instances[0]
    assert video_writer.path == str(input_path.with_name("sample_processed.mp4"))
    assert video_writer.frames
    assert video_writer.released is True

    capture = capture_instances[0]
    assert capture.released is True

    assert len(draw_landmarks_calls) == len(frames)
    assert len(hand_status_calls) == len(frames)
    assert len(dwell_draw_calls) == len(frames)

    csv_path = input_path.with_name("sample_results.csv")
    csv_content = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert csv_content[0].startswith("frame_number,timestamp")
    assert len(csv_content) == len(frames) + 1


def test_main_uses_defaults_when_config_missing(monkeypatch, capsys, tmp_path):
    input_path = tmp_path / "missing_config.mp4"
    input_path.write_bytes(b"data")

    class DummyCapture:
        def __init__(self, path):
            self.path = Path(path)

        def isOpened(self):  # noqa: N802 - mimic OpenCV API
            return True

        def read(self):
            return False, None

        def release(self):
            return None

        def get(self, _prop):
            return 0

    class DummyPoseEstimator:
        def estimate(self, _frame):
            return None

        def close(self):
            return None

    class DummyHandRaiseDetector:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def detect(self, _landmarks):
            return {"left_hand_raised": False, "right_hand_raised": False}

    class DummyDwellDetector:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def update(self, _landmarks, _frame_shape, _timestamp):
            return None

        def get_current_status(self):
            return {}

    def fake_app_config(_path):
        raise FileNotFoundError

    monkeypatch.setattr(run_hand_raise.cv2, "VideoCapture", DummyCapture)
    monkeypatch.setattr(run_hand_raise, "PoseEstimator", DummyPoseEstimator)
    monkeypatch.setattr(run_hand_raise, "HandRaiseDetector", DummyHandRaiseDetector)
    monkeypatch.setattr(run_hand_raise, "DwellTimeDetector", DummyDwellDetector)
    monkeypatch.setattr(run_hand_raise, "AppConfig", fake_app_config)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--input",
            str(input_path),
            "--no_display",
            "--no_video_output",
            "--no_csv_output",
        ],
    )

    run_hand_raise.main()

    captured = capsys.readouterr()
    assert "警告: config.yamlが見つかりません。デフォルト値を使用します。" in captured.out


def test_main_returns_when_capture_fails(monkeypatch, capsys, tmp_path):
    input_path = tmp_path / "fail_capture.mp4"
    input_path.write_bytes(b"data")

    class DummyCapture:
        def __init__(self, path):
            self.path = Path(path)

        def isOpened(self):  # noqa: N802 - mimic OpenCV API
            return False

        def release(self):
            return None

    class DummyConfig:
        def __init__(self, _path):
            return None

        def getfloat(self, key, fallback):
            return fallback

        def getint(self, key, fallback):
            return fallback

        def getboolean(self, key, fallback):
            return fallback

        def get(self, key, fallback):
            return fallback

    class DummyPoseEstimator:
        def estimate(self, _frame):
            return None

        def close(self):
            return None

    monkeypatch.setattr(run_hand_raise.cv2, "VideoCapture", DummyCapture)
    monkeypatch.setattr(run_hand_raise, "AppConfig", DummyConfig)
    monkeypatch.setattr(run_hand_raise, "PoseEstimator", DummyPoseEstimator)
    monkeypatch.setattr(run_hand_raise, "HandRaiseDetector", lambda **_: object())
    monkeypatch.setattr(run_hand_raise, "DwellTimeDetector", lambda **_: object())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--input",
            str(input_path),
            "--no_display",
            "--no_video_output",
            "--no_csv_output",
        ],
    )

    run_hand_raise.main()

    captured = capsys.readouterr()
    assert f"エラー: 動画ファイルが開けません: {input_path}" in captured.out


def test_main_sets_default_fps_when_zero(monkeypatch, capsys, tmp_path):
    input_path = tmp_path / "zero_fps.mp4"
    input_path.write_bytes(b"data")

    frames = [np.zeros((10, 10, 3), dtype=np.uint8)]

    class DummyCapture:
        def __init__(self, path):
            self.path = Path(path)
            self.index = 0

        def isOpened(self):  # noqa: N802 - mimic OpenCV API
            return True

        def read(self):
            if self.index < len(frames):
                frame = frames[self.index]
                self.index += 1
                return True, frame
            return False, None

        def get(self, prop):
            if prop == run_hand_raise.cv2.CAP_PROP_FPS:
                return 0.0
            if prop == run_hand_raise.cv2.CAP_PROP_FRAME_WIDTH:
                return frames[0].shape[1]
            if prop == run_hand_raise.cv2.CAP_PROP_FRAME_HEIGHT:
                return frames[0].shape[0]
            if prop == run_hand_raise.cv2.CAP_PROP_FRAME_COUNT:
                return len(frames)
            return 0

        def release(self):
            return None

    class DummyPoseEstimator:
        def __init__(self):
            self.closed = False

        def estimate(self, _frame):
            return None

        def close(self):
            self.closed = True

    class DummyHandRaiseDetector:
        def detect(self, _landmarks):
            return {"left_hand_raised": False, "right_hand_raised": False}

    class DummyDwellDetector:
        def __init__(self):
            self.timestamps: list[float] = []

        def update(self, _landmarks, _frame_shape, timestamp):
            self.timestamps.append(timestamp)
            return

        def get_current_status(self):
            return {}

    detectors = DummyDwellDetector()
    progress_instances: list[DummyProgress] = []

    def fake_tqdm(*args, **kwargs):
        instance = DummyProgress(*args, **kwargs)
        progress_instances.append(instance)
        return instance

    monkeypatch.setattr(run_hand_raise.cv2, "VideoCapture", DummyCapture)
    monkeypatch.setattr(run_hand_raise.cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_hand_raise.cv2, "waitKey", lambda *args, **kwargs: 0)
    monkeypatch.setattr(run_hand_raise.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(run_hand_raise, "PoseEstimator", DummyPoseEstimator)
    monkeypatch.setattr(run_hand_raise, "HandRaiseDetector", lambda **_: DummyHandRaiseDetector())
    monkeypatch.setattr(run_hand_raise, "DwellTimeDetector", lambda **_: detectors)
    monkeypatch.setattr(run_hand_raise, "draw_hand_raise_status", lambda image, statuses: image)
    monkeypatch.setattr(run_hand_raise, "draw_dwell_status", lambda image, status, alert: image)
    monkeypatch.setattr(run_hand_raise, "tqdm", fake_tqdm)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--input",
            str(input_path),
            "--no_display",
            "--no_video_output",
            "--no_csv_output",
        ],
    )

    run_hand_raise.main()

    captured = capsys.readouterr()
    assert "処理が完了しました。" in captured.out
    assert detectors.timestamps == [0.0]
    assert progress_instances[0].updates == [1]


def test_main_handles_video_writer_failure(monkeypatch, capsys, tmp_path):
    input_path = tmp_path / "writer_fail.mp4"
    input_path.write_bytes(b"data")

    class DummyCapture:
        def __init__(self, path):
            self.path = Path(path)

        def isOpened(self):  # noqa: N802 - mimic OpenCV API
            return True

        def read(self):
            return False, None

        def get(self, prop):
            if prop == run_hand_raise.cv2.CAP_PROP_FPS:
                return 30.0
            if prop == run_hand_raise.cv2.CAP_PROP_FRAME_WIDTH:
                return 160
            if prop == run_hand_raise.cv2.CAP_PROP_FRAME_HEIGHT:
                return 120
            if prop == run_hand_raise.cv2.CAP_PROP_FRAME_COUNT:
                return 0
            return 0

        def release(self):
            return None

    video_writers: list[object] = []

    class DummyVideoWriter:
        def __init__(self, path, *_args):
            self.path = path
            self.released = False
            video_writers.append(self)

        def isOpened(self):  # noqa: N802 - mimic OpenCV API
            return False

        def release(self):
            self.released = True

    class DummyPoseEstimator:
        def estimate(self, _frame):
            return None

        def close(self):
            return None

    monkeypatch.setattr(run_hand_raise.cv2, "VideoCapture", DummyCapture)
    monkeypatch.setattr(run_hand_raise.cv2, "VideoWriter", DummyVideoWriter)
    monkeypatch.setattr(run_hand_raise.cv2, "VideoWriter_fourcc", lambda *args: 0)
    monkeypatch.setattr(run_hand_raise, "PoseEstimator", DummyPoseEstimator)
    monkeypatch.setattr(run_hand_raise, "HandRaiseDetector", lambda **_: object())
    monkeypatch.setattr(run_hand_raise, "DwellTimeDetector", lambda **_: object())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--input",
            str(input_path),
            "--no_display",
            "--no_csv_output",
        ],
    )

    run_hand_raise.main()

    captured = capsys.readouterr()
    assert f"警告: 動画を書き出せませんでした: {input_path.with_name('writer_fail_processed.mp4')}" in captured.out
    assert video_writers and video_writers[0].released is True


def test_main_handles_csv_open_error(monkeypatch, capsys, tmp_path):
    input_path = tmp_path / "csv_error.mp4"
    input_path.write_bytes(b"data")
    expected_csv = input_path.with_name("csv_error_results.csv")

    class DummyCapture:
        def __init__(self, path):
            self.path = Path(path)

        def isOpened(self):  # noqa: N802 - mimic OpenCV API
            return True

        def read(self):
            return False, None

        def get(self, prop):
            if prop == run_hand_raise.cv2.CAP_PROP_FPS:
                return 30.0
            if prop == run_hand_raise.cv2.CAP_PROP_FRAME_WIDTH:
                return 100
            if prop == run_hand_raise.cv2.CAP_PROP_FRAME_HEIGHT:
                return 100
            if prop == run_hand_raise.cv2.CAP_PROP_FRAME_COUNT:
                return 0
            return 0

        def release(self):
            return None

    class DummyPoseEstimator:
        def estimate(self, _frame):
            return None

        def close(self):
            return None

    real_open = builtins.open

    def failing_open(path, *args, **kwargs):
        if Path(path) == expected_csv:
            raise OSError("permission denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(run_hand_raise.cv2, "VideoCapture", DummyCapture)
    monkeypatch.setattr(run_hand_raise, "PoseEstimator", DummyPoseEstimator)
    monkeypatch.setattr(run_hand_raise, "HandRaiseDetector", lambda **_: object())
    monkeypatch.setattr(run_hand_raise, "DwellTimeDetector", lambda **_: object())
    monkeypatch.setattr(run_hand_raise, "draw_hand_raise_status", lambda image, statuses: image)
    monkeypatch.setattr(run_hand_raise, "draw_dwell_status", lambda image, status, alert: image)
    monkeypatch.setattr(run_hand_raise, "tqdm", lambda *a, **kw: DummyProgress(*a, **kw))
    monkeypatch.setattr("builtins.open", failing_open)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--input",
            str(input_path),
            "--no_display",
            "--no_video_output",
        ],
    )

    run_hand_raise.main()

    captured = capsys.readouterr()
    assert "警告: CSVを書き出せませんでした (permission denied). CSV出力を無効化します。" in captured.out


def test_main_user_abort_via_display(monkeypatch, capsys, tmp_path):
    input_path = tmp_path / "user_abort.mp4"
    input_path.write_bytes(b"data")
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    class DummyCapture:
        def __init__(self, path):
            self.path = Path(path)
            self.index = 0

        def isOpened(self):  # noqa: N802 - mimic OpenCV API
            return True

        def read(self):
            if self.index == 0:
                self.index += 1
                return True, frame.copy()
            return False, None

        def get(self, prop):
            if prop == run_hand_raise.cv2.CAP_PROP_FPS:
                return 24.0
            if prop == run_hand_raise.cv2.CAP_PROP_FRAME_WIDTH:
                return frame.shape[1]
            if prop == run_hand_raise.cv2.CAP_PROP_FRAME_HEIGHT:
                return frame.shape[0]
            if prop == run_hand_raise.cv2.CAP_PROP_FRAME_COUNT:
                return 1
            return 0

        def release(self):
            return None

    class DummyPoseEstimator:
        def estimate(self, frame):
            return frame

        def close(self):
            return None

    class DummyHandRaiseDetector:
        def detect(self, _landmarks):
            return {"left_hand_raised": False, "right_hand_raised": False}

    class DummyDwellDetector:
        def update(self, _landmarks, _frame_shape, _timestamp):
            return None

        def get_current_status(self):
            return {}

    destroy_calls: list[bool] = []

    monkeypatch.setattr(run_hand_raise.cv2, "VideoCapture", DummyCapture)
    monkeypatch.setattr(run_hand_raise.cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(run_hand_raise.cv2, "waitKey", lambda *a, **k: ord("q"))
    monkeypatch.setattr(run_hand_raise.cv2, "destroyAllWindows", lambda: destroy_calls.append(True))
    monkeypatch.setattr(run_hand_raise, "PoseEstimator", DummyPoseEstimator)
    monkeypatch.setattr(run_hand_raise, "HandRaiseDetector", lambda **_: DummyHandRaiseDetector())
    monkeypatch.setattr(run_hand_raise, "DwellTimeDetector", lambda **_: DummyDwellDetector())
    monkeypatch.setattr(run_hand_raise, "draw_hand_raise_status", lambda image, statuses: image)
    monkeypatch.setattr(run_hand_raise, "draw_dwell_status", lambda image, status, alert: image)
    monkeypatch.setattr(run_hand_raise, "tqdm", lambda *a, **kw: DummyProgress(*a, **kw))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--input",
            str(input_path),
            "--no_video_output",
            "--no_csv_output",
        ],
    )

    run_hand_raise.main()

    captured = capsys.readouterr()
    assert "処理を途中で終了しました。" in captured.out
    assert destroy_calls == [True]


def test_main_handles_keyboard_interrupt(monkeypatch, capsys, tmp_path):
    input_path = tmp_path / "keyboard_interrupt.mp4"
    input_path.write_bytes(b"data")
    frame = np.zeros((30, 30, 3), dtype=np.uint8)

    class DummyCapture:
        def __init__(self, path):
            self.path = Path(path)

        def isOpened(self):  # noqa: N802 - mimic OpenCV API
            return True

        def read(self):
            return True, frame

        def get(self, prop):
            if prop == run_hand_raise.cv2.CAP_PROP_FPS:
                return 30.0
            if prop == run_hand_raise.cv2.CAP_PROP_FRAME_WIDTH:
                return frame.shape[1]
            if prop == run_hand_raise.cv2.CAP_PROP_FRAME_HEIGHT:
                return frame.shape[0]
            if prop == run_hand_raise.cv2.CAP_PROP_FRAME_COUNT:
                return 1
            return 0

        def release(self):
            return None

    class DummyPoseEstimator:
        def estimate(self, _frame):
            raise KeyboardInterrupt

        def close(self):
            return None

    class DummyDwellDetector:
        def update(self, _landmarks, _frame_shape, _timestamp):
            return None

        def get_current_status(self):
            return {}

    monkeypatch.setattr(run_hand_raise.cv2, "VideoCapture", DummyCapture)
    monkeypatch.setattr(run_hand_raise, "PoseEstimator", DummyPoseEstimator)
    monkeypatch.setattr(run_hand_raise, "HandRaiseDetector", lambda **_: object())
    monkeypatch.setattr(run_hand_raise, "DwellTimeDetector", lambda **_: DummyDwellDetector())
    monkeypatch.setattr(run_hand_raise, "draw_hand_raise_status", lambda image, statuses: image)
    monkeypatch.setattr(run_hand_raise, "draw_dwell_status", lambda image, status, alert: image)
    monkeypatch.setattr(run_hand_raise, "tqdm", lambda *a, **kw: DummyProgress(*a, **kw))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--input",
            str(input_path),
            "--no_display",
            "--no_video_output",
            "--no_csv_output",
        ],
    )

    run_hand_raise.main()

    captured = capsys.readouterr()
    assert "ユーザー操作により処理を中断しました。" in captured.out
    assert "処理を途中で終了しました。" in captured.out


def test_src_module_entrypoint_invokes_main(monkeypatch, tmp_path):
    input_path = tmp_path / "entry.mp4"
    input_path.write_bytes(b"data")

    class DummyCapture:
        def __init__(self, path):
            self.path = Path(path)
            self.released = False

        def isOpened(self):  # noqa: N802 - mimic OpenCV API
            return True

        def read(self):
            return False, None

        def get(self, prop):
            if prop == run_hand_raise.cv2.CAP_PROP_FPS:
                return 30.0
            if prop == run_hand_raise.cv2.CAP_PROP_FRAME_WIDTH:
                return 160
            if prop == run_hand_raise.cv2.CAP_PROP_FRAME_HEIGHT:
                return 120
            if prop == run_hand_raise.cv2.CAP_PROP_FRAME_COUNT:
                return 0
            return 0

        def release(self):
            self.released = True

    class DummyPoseEstimator:
        def __init__(self):
            self.closed = False

        def estimate(self, _frame):
            return None

        def close(self):
            self.closed = True

    class DummyHandRaiseDetector:
        def detect(self, _landmarks):
            return {"left_hand_raised": False, "right_hand_raised": False}

    class DummyDwellDetector:
        def update(self, _landmarks, _frame_shape, _timestamp):
            return None

        def get_current_status(self):
            return {}

    pose_instances: list[DummyPoseEstimator] = []
    capture_instances: list[DummyCapture] = []

    def pose_factory():
        instance = DummyPoseEstimator()
        pose_instances.append(instance)
        return instance

    def capture_factory(path):
        instance = DummyCapture(path)
        capture_instances.append(instance)
        return instance

    monkeypatch.setattr(run_hand_raise.cv2, "VideoCapture", capture_factory)
    monkeypatch.setattr("src.pose_estimator.PoseEstimator", pose_factory)
    monkeypatch.setattr("src.detectors.hand_raise_refactored.HandRaiseDetector", lambda **_: DummyHandRaiseDetector())
    monkeypatch.setattr("src.analysis.dwell_time_detector.DwellTimeDetector", lambda **_: DummyDwellDetector())
    monkeypatch.setattr("src.drawing_utils.draw_hand_raise_status", lambda image, statuses: image)
    monkeypatch.setattr("src.drawing_utils.draw_dwell_status", lambda image, status, alert: image)
    monkeypatch.setattr("src.drawing_utils.draw_landmarks", lambda image, landmarks: None)
    monkeypatch.setattr("tqdm.tqdm", lambda *a, **kw: DummyProgress(*a, **kw))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_hand_raise.py",
            "--input",
            str(input_path),
            "--no_display",
            "--no_video_output",
            "--no_csv_output",
        ],
    )

    runpy.run_path(Path("src/run_hand_raise.py"), run_name="__main__")

    assert pose_instances and pose_instances[0].closed is True
    assert capture_instances and capture_instances[0].released is True


def test_cli_entrypoint_invokes_main(monkeypatch):
    calls = []

    def fake_main():
        calls.append(True)

    monkeypatch.setattr("src.run_hand_raise.main", fake_main)

    runpy.run_path(Path("run_hand_raise.py"), run_name="__main__")

    assert calls == [True]
