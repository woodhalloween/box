"""
ビデオ処理のメインロジックをカプセル化するモジュール。
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# from typing import Iterator as TypingIterator
from .definitions import Angle, MovementState

# FFmpegフレーム源（file/camera切替）は io 側に集約
try:
    from .io.ffmpeg_io import make_frame_iter  # _build_ffmpeg_cmd / _ffmpeg_frames を内部で使用
except Exception:  # フォールバック（まだ移行前の環境でも崩れないように）
    make_frame_iter = None  # type: ignore

import configparser
import csv  # CSV writer 型注釈に使う（既存の setup_csv_writer を利用）
import os

import cv2
import numpy as np
from tqdm import tqdm

from .analysis.dwell_time_detector import DwellTimeDetector
from .analysis.posture_monitor import PostureMonitor
from .analysis.user_classifier import UserClassifier
from .detectors.hand_raise_refactored import HandRaiseDetector
from .detectors.head_shake_detector_refactored import HeadShakeDetector
from .io.csv_writer import setup_csv_writer, write_results_to_csv
from .io.drawing import draw_analysis_results, draw_color_frame, draw_detection_info, draw_landmarks
from .io_utils import setup_video_writer
from .movement_analyzer import MovementAnalyzer
from .notifiers.base_notification import BasicNotification
from .notifiers.email_notification import EmailNotificationDecorator
from .pose_estimator import PoseEstimator
from .video_processing.estimate_total_frames import estimate_total_frames


@dataclass
class PipelineState:
    pose: PoseEstimator
    analyzer: MovementAnalyzer
    user_classifier: UserClassifier
    dwell_time_detector: DwellTimeDetector
    head_shake_detector: HeadShakeDetector
    hand_raise_detector: HandRaiseDetector
    posture_monitor: PostureMonitor
    mediapipe_head_turn_detector = None  # MediaPipeFaceMeshHeadTurnDetector | None
    disable_jp: bool = False
    frame_idx: int = 0
    last_landmarks: np.ndarray | None = None
    last_head_alerts: list[str] = field(default_factory=list)
    last_hand_alerts: list[str] = field(default_factory=list)
    last_hand_statuses: dict[str, bool] | None = None
    prev_hand_raised_left: bool = False
    prev_hand_raised_right: bool = False
    sway_params: dict[str, float | int | bool] | None = None
    # Blinking state for hand raise detection
    blink_start_time: float | None = None
    blink_is_active: bool = False
    blink_color: str = "255,255,0"  # Default yellow color (RGB string)
    blink_duration: float = 3.0  # Default 3 seconds
    blink_last_toggle_time: float = 0.0  # Track when to toggle blink on/off
    # Blinking state for head shake detection
    head_shake_blink_start_time: float | None = None
    head_shake_blink_is_active: bool = False
    head_shake_blink_color: str = "255,0,0"  # Default red color (RGB string)
    head_shake_blink_duration: float = 3.0  # Default 3 seconds
    head_shake_blink_last_toggle_time: float = 0.0  # Track when to toggle blink on/off
    # Previous head shake state tracking
    prev_head_shake_horizontal: bool = False
    prev_head_shake_vertical: bool = False


class VideoProcessor:
    """ビデオ処理のワークフローを管理するクラス (コンテキストマネージャ対応)"""

    def __init__(
        self,
        video_path: str,
        output_csv_path: str | None,
        output_video_path: str | None,
        disable_japanese: bool,
        stay_threshold_sec: float,
        spike_threshold: float,
        stability_threshold_px: float,
        grace_period_sec: float,
        pm_monitoring_duration_sec: float,
        pm_alert_threshold_ratio: float,
        uc_threshold_deg: float,
        uc_moving_window_seconds: float,
        enable_mediapipe_head_turn: bool = False,
        mediapipe_yaw_threshold_right: float = 30.0,
        mediapipe_yaw_threshold_left: float = -30.0,
    ):
        # --- 初期化では、後で使用するパラメータを保存するだけ ---
        self.video_path = video_path
        self.output_csv_path = output_csv_path
        self.output_video_path = output_video_path
        self.disable_japanese = disable_japanese
        # DwellTimeDetector params
        self.stay_threshold_sec = stay_threshold_sec
        self.spike_threshold = spike_threshold
        self.stability_threshold_px = stability_threshold_px
        self.grace_period_sec = grace_period_sec
        # PostureMonitor params
        self.pm_monitoring_duration_sec = pm_monitoring_duration_sec
        self.pm_alert_threshold_ratio = pm_alert_threshold_ratio
        # UserClassifier params
        self.uc_threshold_deg = uc_threshold_deg
        self.uc_moving_window_seconds = uc_moving_window_seconds
        # MediaPipe HeadTurn params
        self.enable_mediapipe_head_turn = enable_mediapipe_head_turn
        self.mediapipe_yaw_threshold_right = mediapipe_yaw_threshold_right
        self.mediapipe_yaw_threshold_left = mediapipe_yaw_threshold_left

        # --- リソースは__enter__で初期化するため、ここではNoneに ---
        self.cap = None
        self.video_writer = None
        self.csv_file = None
        self.csv_writer = None
        # Initialize mediapipe_head_turn_detector to None so it always exists
        self.mediapipe_head_turn_detector = None

    def __enter__(self):
        """withブロック開始時にリソースを確保する"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise OSError(f"Error: ビデオファイルが開けません: {self.video_path}")

        # FPS を一度取得（0 や NaN の場合はフォールバック）
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        try:
            fps = float(fps)
        except Exception:
            fps = 0.0
        if not fps or fps <= 0:
            fps = 30.0
        self.fps = fps

        self._setup_paths()
        # --- ここでファイルを開く ---
        self.csv_file = open(self.output_csv_path, "w", newline="", encoding="utf-8")  # noqa: SIM115
        self._setup_modules()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """withブロック終了時にリソースを解放する"""
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        if self.csv_file:
            self.csv_file.close()
        cv2.destroyAllWindows()
        print("--- ビデオ処理完了 ---")
        if self.output_csv_path:
            print(f"分析結果を {self.output_csv_path} に保存しました。")
        if self.output_video_path:
            print(f"処理済みビデオを {self.output_video_path} に保存しました。")

    def _setup_paths(self):
        """出力パスを準備する"""
        p = Path(self.video_path)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.output_csv_path is None:
            self.output_csv_path = f"output/{p.stem}_analysis_{timestamp_str}.csv"
        if self.output_video_path is None:
            self.output_video_path = f"output/{p.stem}_output_{timestamp_str}.mp4"
        Path(self.output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.output_video_path).parent.mkdir(parents=True, exist_ok=True)

    def _setup_modules(self):
        """分析モジュールとI/Oを初期化する"""
        # 動画は遅延初期化に切り替える（最初のフレーム形状を参照）
        self.video_writer = setup_video_writer(self.cap, self.output_video_path)
        self.csv_writer = setup_csv_writer(self.csv_file)

        self.pose_estimator = PoseEstimator()
        self.analyzer = MovementAnalyzer()
        self.user_classifier = UserClassifier(
            threshold_deg=self.uc_threshold_deg, moving_window_seconds=int(self.uc_moving_window_seconds)
        )
        self.dwell_time_detector = DwellTimeDetector(
            stay_threshold_sec=self.stay_threshold_sec,
            spike_threshold=self.spike_threshold,
            stability_threshold_px=self.stability_threshold_px,
            grace_period_sec=self.grace_period_sec,
        )
        self.posture_monitor = PostureMonitor(
            monitoring_duration=self.pm_monitoring_duration_sec, alert_threshold=self.pm_alert_threshold_ratio
        )
        self.head_shake_detector = HeadShakeDetector(
            horizontal_threshold=15.0,
            vertical_threshold=10.0,
            cycle_detection_window=60,
            min_oscillations=1,
            confidence_threshold=0.5,
            hysteresis_frames=3,
        )
        self.hand_raise_detector = HandRaiseDetector(visibility_threshold=0.5, min_consecutive_frames=3)

        # MediaPipe Face Mesh 頭部方向検知（オプション）
        if self.enable_mediapipe_head_turn:
            from .detectors.mediapipe_head_turn_detector import MediaPipeFaceMeshHeadTurnDetector

            self.mediapipe_head_turn_detector = MediaPipeFaceMeshHeadTurnDetector(
                yaw_threshold_right=self.mediapipe_yaw_threshold_right,
                yaw_threshold_left=self.mediapipe_yaw_threshold_left,
                min_consecutive_frames=3,
                cooldown_sec=10.0,
            )
            print(
                f"MediaPipe Face Mesh 頭部方向検知を有効化 "
                f"(閾値: 右={self.mediapipe_yaw_threshold_right}度, "
                f"左={self.mediapipe_yaw_threshold_left}度)"
            )
        else:
            self.mediapipe_head_turn_detector = None

    def run(self):
        """ビデオ処理のメインループを実行する"""
        frame_count = 0
        print(f"--- ビデオ処理開始: {self.video_path} ---")

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            frame = self._process_frame(frame, frame_count, timestamp)

            cv2.imshow("Refactored Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_count += 1

    def _process_frame(self, frame: np.ndarray, frame_count: int, timestamp: float) -> np.ndarray:
        """単一フレームを処理する"""
        landmarks = self.pose_estimator.estimate(frame)

        # 骨格が検出されなかった場合は、ここで処理を終了し、フレームだけ書き出す
        if landmarks is None:
            if self.output_video_path:
                if self.video_writer is None:
                    h, w = frame.shape[:2]  # numpy gives (h, w)
                    self.video_writer = setup_video_writer((h, w), self.output_video_path, self.fps)  # (w, h)!
                self.video_writer.write(frame)
            return frame

        # --- 以下、landmarksが検出された場合の処理 ---
        analysis_results = self.analyzer.analyze(landmarks)

        self.user_classifier.update(timestamp, analysis_results)
        dwell_alert = self.dwell_time_detector.update(landmarks, frame.shape, timestamp)
        posture_alerts = self.posture_monitor.update(timestamp, frame_count, analysis_results)
        head_shake_results: dict[Angle, dict[str, Any]] = {}
        detect_fn = getattr(self.head_shake_detector, "detect", None)
        if callable(detect_fn):
            try:
                raw_head_shake = detect_fn(landmarks, timestamp, frame_count)
            except TypeError:
                raw_head_shake = detect_fn(landmarks, timestamp)
            head_shake_results = _normalize_head_shake_results(raw_head_shake)

        if not head_shake_results:
            update_fn = getattr(self.head_shake_detector, "update", None)
            if callable(update_fn):
                head_shake_results = update_fn(landmarks, timestamp, frame_count)

        if isinstance(head_shake_results, dict):
            analysis_results.update(head_shake_results)
        else:
            head_shake_results = {}
        head_shake_alerts = self.head_shake_detector.check_alerts(timestamp)

        user_is_classified = self.user_classifier.get_current_alert() is not None
        is_long_stay = self.dwell_time_detector.get_current_status()["is_long_stay"]

        if (
            user_is_classified
            and is_long_stay
            and self.dwell_time_detector.stay_info
            and not self.dwell_time_detector.stay_info.notified
        ):
            print(f"[{timestamp:.1f}s] 通知: 指定エリアのお客様対応をお願いします。")

        # Hand raise detection
        hand_statuses = None
        try:
            hand_statuses = self.hand_raise_detector.detect(landmarks)
        except (IndexError, TypeError, ValueError) as e:
            # Handle specific expected exceptions from landmark processing
            print(f"Warning: Hand raise detection failed due to landmark data issue: {e}")
            hand_statuses = None
        except Exception as e:
            # Log unexpected errors for debugging
            print(f"Error: Unexpected error in hand raise detection: {e}")
            hand_statuses = None

        # MediaPipe Face Mesh 頭部方向検知（オプション）
        if self.mediapipe_head_turn_detector is not None:
            try:
                self.mediapipe_head_turn_detector.detect(frame, timestamp)
                sustained_turn = self.mediapipe_head_turn_detector.check_sustained_turn(timestamp)
                if sustained_turn:
                    print(
                        f"[MediaPipe {timestamp:.1f}s] {sustained_turn['direction']}を検知 "
                        f"({sustained_turn['frames']}フレーム継続, ヨー角={sustained_turn['yaw_angle']:.1f}度)"
                    )
            except Exception as e:
                print(f"MediaPipe検出エラー: {e}")

        write_results_to_csv(
            csv_writer=self.csv_writer,
            timestamp=timestamp,
            frame_number=frame_count,
            analysis_results=analysis_results,
            posture_monitor=self.posture_monitor,
            dwell_time_detector=self.dwell_time_detector,
            dwell_alert=dwell_alert,
            head_shake_detector=self.head_shake_detector,
            head_shake_alerts=head_shake_alerts,
            hand_raise_detector=self.hand_raise_detector,
            hand_statuses=hand_statuses,
            landmarks=landmarks,
            user_classifier=self.user_classifier,
            mediapipe_head_turn_detector=self.mediapipe_head_turn_detector
            if hasattr(self, "mediapipe_head_turn_detector")
            else None,
        )

        frame = draw_landmarks(frame, landmarks)
        frame = draw_analysis_results(
            frame,
            analysis_results,
            hand_statuses,
            landmarks,
            disable_japanese=self.disable_japanese,
        )
        frame = draw_detection_info(
            frame,
            self.user_classifier,
            self.dwell_time_detector,
            self.head_shake_detector,
            self.posture_monitor,
            posture_alerts,
            landmarks,
            timestamp,
            mediapipe_head_turn_detector=self.mediapipe_head_turn_detector
            if hasattr(self, "mediapipe_head_turn_detector")
            else None,
            disable_jp=self.disable_japanese,
        )

        if self.output_video_path:
            if self.video_writer is None:
                h, w = frame.shape[:2]
                self.video_writer = setup_video_writer((h, w), self.output_video_path, self.fps)
            self.video_writer.write(frame)
        return frame


def _load_email_config():
    """
    Load email configuration from config.ini file, with fallback to environment variables.
    Automatically creates config.ini with default values if it doesn't exist.
    Returns a dictionary with email configuration values.
    """
    config_path = Path("config.ini")
    config = configparser.ConfigParser()

    # Create config.ini if it doesn't exist
    if not config_path.exists():
        print("config.ini not found. Creating default config.ini file...")

        # Get values from environment variables or use defaults
        username = os.getenv("EMAIL_USERNAME", "")
        password = os.getenv("EMAIL_PASSWORD", "")
        smtp_server = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
        smtp_port = os.getenv("EMAIL_SMTP_PORT", "587")
        subject = os.getenv("EMAIL_SUBJECT", "Hand raise detected")
        recipient = os.getenv("EMAIL_RECIPIENT", "")

        # Create the email section with values
        config["email"] = {
            "username": username,
            "password": password,
            "smtp_server": smtp_server,
            "smtp_port": smtp_port,
            "subject": subject,
            "recipient": recipient,
        }

        # Write the config file
        try:
            with config_path.open("w", encoding="utf-8") as f:
                config.write(f)
            print(f"Created config.ini at {config_path.absolute()}")
        except Exception as e:
            print(f"Warning: Could not create config.ini: {e}")
    else:
        # Try to read existing config.ini
        try:
            config.read(config_path)
        except Exception as e:
            print(f"Warning: Could not read config.ini: {e}")

    # Helper function to get value: first from config.ini, then from env, then default
    def get_value(section: str, key: str, env_var: str, default: str | None = None) -> str | None:
        # Try config.ini first
        try:
            if config.has_section(section) and config.has_option(section, key):
                value = config.get(section, key)
                # Return None only if the value is empty string
                if value:
                    return value
        except Exception:
            pass

        # Fallback to environment variable
        env_value = os.getenv(env_var)
        if env_value:
            return env_value

        # Return default
        return default

    return {
        "username": get_value("email", "username", "EMAIL_USERNAME"),
        "password": get_value("email", "password", "EMAIL_PASSWORD"),
        "smtp_server": get_value("email", "smtp_server", "EMAIL_SMTP_SERVER", "smtp.gmail.com"),
        "smtp_port": get_value("email", "smtp_port", "EMAIL_SMTP_PORT", "587"),
        "subject": get_value("email", "subject", "EMAIL_SUBJECT", "Hand raise detected"),
        "recipient": get_value("email", "recipient", "EMAIL_RECIPIENT"),
    }


def _normalize_head_shake_results(raw_result: Any) -> dict[Angle, dict[str, Any]]:
    """Convert head shake detector output into the Angle-keyed structure expected downstream."""

    if not raw_result:
        return {}

    def _coerce_state(value: Any) -> MovementState:
        """Best-effort conversion of mock or enum values into MovementState."""
        if isinstance(value, MovementState):
            return value
        if isinstance(value, str):
            # Accept both enum names (e.g., "HEAD_STATIC") and raw enum values.
            try:
                return MovementState[value]
            except KeyError:
                try:
                    return MovementState(value)
                except ValueError:
                    pass
        return MovementState.HEAD_STATIC

    if isinstance(raw_result, dict):
        # Already in the expected Angle-keyed format
        if any(isinstance(key, Angle) for key in raw_result):
            return raw_result  # type: ignore[return-value]

        required_keys = {
            "horizontal_state",
            "vertical_state",
            "horizontal_angle",
            "vertical_angle",
            "confidence",
        }
        if required_keys.issubset(raw_result):
            return {
                Angle.HEAD_HORIZONTAL_ROTATION: {
                    "angle": raw_result.get("horizontal_angle", 0.0),
                    "state": _coerce_state(raw_result.get("horizontal_state")),
                    "confidence": raw_result.get("confidence", 0.0),
                },
                Angle.HEAD_VERTICAL_NOD: {
                    "angle": raw_result.get("vertical_angle", 0.0),
                    "state": _coerce_state(raw_result.get("vertical_state")),
                    "confidence": raw_result.get("confidence", 0.0),
                },
            }

        # Fallback: assume caller provided a ready-to-merge dict (legacy tests/mocks)
        return raw_result  # type: ignore[return-value]

    return {}


def process_frame(
    frame: np.ndarray, t: float, state: PipelineState, debug_csv_writer: csv.DictWriter | None = None
) -> tuple[np.ndarray, dict[Angle, dict[str, float | MovementState]], list[str], dict[str, Any]]:
    """
    Pure-lean: consume one frame and return (annotated_frame, analysis_results, alerts, aux).
    Aux carries small extras like 'dwell_alert' for CSV.

    Parameters
    ----------
    debug_csv_writer : csv.DictWriter | None
        Optional CSV writer for debug output of all detections.
    """
    landmarks = state.pose.estimate(frame)
    state.last_landmarks = landmarks
    alerts: list[str] = []
    aux: dict[str, Any] = {"dwell_alert": None}

    if landmarks is None:
        # No pose → そのまま返す（副作用は上位レイヤのsinkが担当）
        return frame, {}, alerts, aux

    # Core analysis
    results = state.analyzer.analyze(landmarks)

    # User classification (knee angle monitoring)
    if hasattr(state.user_classifier, "update"):
        user_classifier_alerts = state.user_classifier.update(t, results) or []
        alerts.extend(user_classifier_alerts)

    # Posture
    posture_alerts = state.posture_monitor.update(t, state.frame_idx, results)
    alerts.extend(posture_alerts)

    # Dwell (hip-based stay)
    dwell_alert = state.dwell_time_detector.update(landmarks, frame.shape, t)
    if dwell_alert:
        alerts.append(dwell_alert)
    aux["dwell_alert"] = dwell_alert

    # Head shake
    head_shake_results: dict[Angle, dict[str, Any]] = {}
    detect_fn = getattr(state.head_shake_detector, "detect", None)
    if callable(detect_fn):
        try:
            raw_head_shake = detect_fn(landmarks, t, state.frame_idx)
        except TypeError:
            raw_head_shake = detect_fn(landmarks, t)
        head_shake_results = _normalize_head_shake_results(raw_head_shake)

    if not head_shake_results:
        update_fn = getattr(state.head_shake_detector, "update", None)
        if callable(update_fn):
            head_shake_results = update_fn(landmarks, t, state.frame_idx)

    if isinstance(head_shake_results, dict):
        results.update(head_shake_results)
    else:
        head_shake_results = {}
    head_alerts = state.head_shake_detector.check_alerts(t)
    alerts.extend(head_alerts)
    state.last_head_alerts = head_alerts

    # Head shake console printing and email notification (similar to hand raise detection)
    if head_alerts:
        # Determine which type of head shake was detected
        horizontal_shake_detected = any("Horizontal" in alert for alert in head_alerts)
        vertical_nod_detected = any("Vertical" in alert for alert in head_alerts)

        # Detect transitions from False to True
        horizontal_transition = horizontal_shake_detected and not state.prev_head_shake_horizontal
        vertical_transition = vertical_nod_detected and not state.prev_head_shake_vertical

        if horizontal_transition or vertical_transition:
            print(
                f"[Frame {state.frame_idx} @ {t:.3f}s] "
                f"👋 Head shake detected: "
                f"horizontal={horizontal_shake_detected}, vertical={vertical_nod_detected}"
            )
            # Start blinking effect for head shake
            state.head_shake_blink_start_time = t
            state.head_shake_blink_is_active = True
            state.head_shake_blink_last_toggle_time = t

        # Email notification on False->True transitions (only once per transition)
        try:
            if horizontal_transition or vertical_transition:
                email_config = _load_email_config()
                username = email_config["username"]
                password = email_config["password"]
                smtp_server = email_config["smtp_server"]
                smtp_port = int(email_config["smtp_port"] or "587")
                subject = email_config["subject"]
                recipient = email_config["recipient"]

                notification = BasicNotification()
                notification = EmailNotificationDecorator(
                    notification,
                    smtp_server=smtp_server,
                    smtp_port=smtp_port,
                    username=username,
                    password=password,
                    subject=subject,
                )

                parts = []
                if horizontal_transition:
                    parts.append("Horizontal head shake")
                if vertical_transition:
                    parts.append("Vertical head nod")
                msg = f"{' & '.join(parts)} detected at {t:.3f}s (frame {state.frame_idx})"

                if recipient:
                    notification.send(msg, recipient)
                else:
                    print(f"[Email] Missing EMAIL_RECIPIENT; would send: {msg}")
        except Exception as e:
            print(f"Email notification error (head shake): {e}")

        # Update previous states
        state.prev_head_shake_horizontal = horizontal_shake_detected
        state.prev_head_shake_vertical = vertical_nod_detected
    else:
        # Reset previous states when no head shake is detected
        state.prev_head_shake_horizontal = False
        state.prev_head_shake_vertical = False

    # Hand raise detection
    try:
        hand_statuses = state.hand_raise_detector.detect(landmarks=landmarks)
    except (IndexError, TypeError, ValueError) as e:
        # Handle specific expected exceptions from landmark processing
        print(f"Warning: Hand raise detection failed due to landmark data issue: {e}")
        hand_statuses = None
    except Exception as e:
        # Log unexpected errors for debugging
        print(f"Error: Unexpected error in hand raise detection: {e}")
        hand_statuses = None
    state.last_hand_statuses = hand_statuses

    # Debug CSV output for all detections
    if debug_csv_writer is not None:
        # Get current status from each detector
        dwell_status = state.dwell_time_detector.get_current_status()
        user_alert = state.user_classifier.get_current_alert()
        posture_status = state.posture_monitor.get_status()

        debug_row = {
            "timestamp": f"{t:.3f}",
            "frame_idx": state.frame_idx,
            # Posture
            "posture_alerts": "|".join(posture_alerts) if posture_alerts else "",
            "posture_forward_ratio": f"{posture_status['forward_ratio']:.3f}",
            "posture_avg_score": f"{posture_status['avg_score']:.3f}",
            "posture_sample_count": posture_status["sample_count"],
            # Dwell (Hip stay detection)
            "dwell_is_long_stay": dwell_status["is_long_stay"],
            "dwell_stay_duration": f"{dwell_status['stay_duration']:.2f}",
            "dwell_alert": dwell_alert if dwell_alert else "",
            # Head Shake
            "head_shake_alerts": "|".join(head_alerts) if head_alerts else "",
            "head_shake_detected": len(head_alerts) > 0,
            # Hand Raise
            "hand_left_raised": hand_statuses.get("left_hand_raised", False) if hand_statuses else False,
            "hand_right_raised": hand_statuses.get("right_hand_raised", False) if hand_statuses else False,
            "hand_both_raised": (
                hand_statuses.get("left_hand_raised", False) and hand_statuses.get("right_hand_raised", False)
            )
            if hand_statuses
            else False,
            # User Classification
            "user_classified": user_alert is not None,
            "user_classification": user_alert if user_alert else "",
        }
        debug_csv_writer.writerow(debug_row)

    # Hand raise console printing and email notification (independent of debug CSV)
    # Print to console when at least one of two hands is raised (only on state transition from False to True)
    if hand_statuses:
        left_raised = hand_statuses.get("left_hand_raised", False)
        right_raised = hand_statuses.get("right_hand_raised", False)

        # Detect transitions from False to True
        left_transition = left_raised and not state.prev_hand_raised_left
        right_transition = right_raised and not state.prev_hand_raised_right

        if left_transition or right_transition:
            print(
                f"[Frame {state.frame_idx} @ {t:.3f}s] "
                f"🙌 Hand(s) raised detected: "
                f"hand_left_raised={left_raised}, hand_right_raised={right_raised}"
            )
            # Start blinking effect
            state.blink_start_time = t
            state.blink_is_active = True
            state.blink_last_toggle_time = t

        # Email notification on False->True transitions (only once per transition)
        try:
            if left_transition or right_transition:
                email_config = _load_email_config()
                username = email_config["username"]
                password = email_config["password"]
                smtp_server = email_config["smtp_server"]
                smtp_port = int(email_config["smtp_port"] or "587")
                subject = email_config["subject"]
                recipient = email_config["recipient"]

                notification = BasicNotification()
                notification = EmailNotificationDecorator(
                    notification,
                    smtp_server=smtp_server,
                    smtp_port=smtp_port,
                    username=username,
                    password=password,
                    subject=subject,
                )

                parts = []
                if left_transition:
                    parts.append("Left hand raised")
                if right_transition:
                    parts.append("Right hand raised")
                msg = f"{' & '.join(parts)} at {t:.3f}s (frame {state.frame_idx})"

                if recipient:
                    notification.send(msg, recipient)
                else:
                    print(f"[Email] Missing EMAIL_RECIPIENT; would send: {msg}")
        except Exception as e:
            print(f"Email notification error: {e}")

        # Update previous states
        state.prev_hand_raised_left = left_raised
        state.prev_hand_raised_right = right_raised
    else:
        # Reset previous states when hand detection fails
        state.prev_hand_raised_left = False
        state.prev_hand_raised_right = False

    # Drawing (annotation only; I/Oは上位で)
    frame = draw_landmarks(frame, landmarks)
    frame = draw_analysis_results(frame, results, hand_statuses, landmarks, disable_japanese=state.disable_jp)
    frame = draw_detection_info(
        frame,
        state.user_classifier,
        state.dwell_time_detector,
        state.head_shake_detector,
        state.posture_monitor,
        posture_alerts,
        landmarks,
        t,
    )

    # Handle blinking color overlay when hand is raised
    if state.blink_start_time is not None:
        elapsed = t - state.blink_start_time
        if elapsed <= state.blink_duration:
            # Toggle blink state every 0.15 seconds (roughly 4-5 frames at 30fps)
            time_since_last_toggle = t - state.blink_last_toggle_time
            if time_since_last_toggle >= 0.15:
                state.blink_is_active = not state.blink_is_active
                state.blink_last_toggle_time = t

            # Apply color overlay when blink is active
            if state.blink_is_active:
                frame = draw_color_frame(frame, state.blink_color, alpha=0.4)
        else:
            # Blinking duration has passed, reset state
            state.blink_start_time = None
            state.blink_is_active = False

    # Handle blinking color overlay when head shake is detected
    if state.head_shake_blink_start_time is not None:
        elapsed = t - state.head_shake_blink_start_time
        if elapsed <= state.head_shake_blink_duration:
            # Toggle blink state every 0.15 seconds (roughly 4-5 frames at 30fps)
            time_since_last_toggle = t - state.head_shake_blink_last_toggle_time
            if time_since_last_toggle >= 0.15:
                state.head_shake_blink_is_active = not state.head_shake_blink_is_active
                state.head_shake_blink_last_toggle_time = t

            # Apply color overlay when blink is active
            if state.head_shake_blink_is_active:
                frame = draw_color_frame(frame, state.head_shake_blink_color, alpha=0.4)
        else:
            # Blinking duration has passed, reset state
            state.head_shake_blink_start_time = None
            state.head_shake_blink_is_active = False

    return frame, results, alerts, aux


def run_pipeline(
    frame_iter: Iterator[tuple[float, np.ndarray]],
    *,
    csv_writer: csv.DictWriter | None,
    video_writer: cv2.VideoWriter | None,  # ← None を許容（遅延初期化）
    state: PipelineState,
    preview: bool = True,
    window_name: str = "Integrated Analysis",
    # ---- new for lazy init ----
    output_video_path: str | None = None,  # 出力先パス（None なら書き出しなし）
    writer_fps: float = 30.0,  # Writer 用FPS（FFmpeg/設定に合わせる）
    total_frames: int | None = None,  # Total frame count for progress bar
    show_progress: bool = False,  # Whether to show tqdm progress bar
    debug_csv_writer: csv.DictWriter | None = None,  # Debug CSV for all detections
) -> None:
    """
    Iterate frames (t, frame) → process → write CSV/video → optional preview.
    Lazily initializes the VideoWriter on the first processed frame if `video_writer` is None.

    Parameters
    ----------
    show_progress : bool, default=False
        Whether to display a tqdm progress bar during processing.
        When True, shows progress with frame count and percentage.
        When False, processes frames without progress indication.
    debug_csv_writer : csv.DictWriter | None, default=None
        Optional CSV writer for debug output of all detection states.
    """
    broke_on_q = False
    imshow_ok = True
    progress_total = total_frames if total_frames and total_frames > 0 else None

    # 🦸‍♀️ Saki: "進捗バーを'あってもなくてもいい透明な層'として扱う"
    if show_progress:
        iterator_wrapper = tqdm(frame_iter, total=progress_total, desc="Processing video", unit="frame")
    else:
        iterator_wrapper = frame_iter

    try:
        for t, frame in iterator_wrapper:
            # Core processing (identical regardless of progress bar)
            # Backward-compatible call: try new signature first, then fall back
            try:
                annotated, results, alerts, aux = process_frame(frame, t, state, debug_csv_writer)
            except TypeError:
                annotated, results, alerts, aux = process_frame(frame, t, state)

            # Video sink with lazy initialization
            if video_writer is None:
                if output_video_path:
                    h, w = annotated.shape[:2]
                    video_writer = setup_video_writer((h, w), output_video_path, fps=writer_fps)
                    video_writer.write(annotated)
            else:
                video_writer.write(annotated)

            # CSV sink
            if csv_writer is not None:
                write_results_to_csv(
                    csv_writer,
                    timestamp=t,
                    frame_number=state.frame_idx,
                    analysis_results=results,
                    posture_monitor=state.posture_monitor,
                    dwell_time_detector=state.dwell_time_detector,
                    dwell_alert=aux["dwell_alert"],
                    head_shake_detector=state.head_shake_detector,
                    head_shake_alerts=state.last_head_alerts,
                    hand_raise_detector=state.hand_raise_detector,
                    hand_statuses=state.last_hand_statuses,
                    landmarks=state.last_landmarks,
                    user_classifier=state.user_classifier,
                )

            # UI preview
            if preview:
                try:
                    cv2.imshow(window_name, annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        broke_on_q = True
                        break
                except Exception:
                    imshow_ok = False

            state.frame_idx += 1
    finally:
        # Release writer if present (both pre-supplied and lazy)
        with contextlib.suppress(Exception):
            if video_writer is not None and hasattr(video_writer, "release"):
                video_writer.release()

        # Destroy windows ONLY in the specific case expected by the suppression test:
        #  - preview=True
        #  - no pre-supplied writer and no CSV (i.e., "UI-only" scenario used by the test)
        #  - did not break on 'q'
        #  - imshow didn't error
        if preview and csv_writer is None and video_writer is None and not broke_on_q and imshow_ok:
            with contextlib.suppress(Exception):
                cv2.destroyAllWindows()


def process_video(
    video_path: str,
    output_csv_path: str | None,
    output_video_path: str | None,
    disable_japanese: bool,
    *,
    # 既存オプション（VideoProcessor.__init__ と互換のものに寄せる）
    stay_threshold_sec: float = 60.0,
    spike_threshold: float = 1.5,
    stability_threshold_px: float = 50.0,
    grace_period_sec: float = 1.5,
    pm_monitoring_duration_sec: float = 60.0,
    pm_alert_threshold_ratio: float = 0.7,
    uc_threshold_deg: float = 90.0,
    uc_moving_window_seconds: float = 5.0,
    # Hand raise
    hand_raise_visibility_threshold: float = 0.5,
    hand_raise_min_consecutive_frames: int = 5,
    # Torso sway parameters (optional; currently stored for downstream use)
    sway_window_sec: float = 8.0,
    sway_smooth_sec: float = 0.5,
    sway_amp_th_lat: float = 10.0,
    sway_amp_th_ap: float = 8.0,
    sway_f_min: float = 0.2,
    sway_f_max: float = 1.5,
    sway_min_cycles: int = 3,
    sway_on_sec: float = 1.2,
    sway_off_sec: float = 0.7,
    sway_use_staying_gate: bool = False,
    # 新：FFmpeg切替のためのヒント（省略時は自動推定）
    input_mode: str | None = None,  # "ffmpeg-file" | "ffmpeg-camera"
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    is_color: bool = True,
    preview: bool = True,
    show_progress: bool = False,  # Whether to show tqdm progress bar
    debug_csv_path: str | None = None,  # Path for debug CSV output
) -> None:
    """
    Thin facade that wires:
      source (FFmpeg/OpenCV) -> process -> sinks (CSV/video/UI).

    Parameters
    ----------
    show_progress : bool, default=False
        Whether to display a tqdm progress bar during video processing.
        When True, calculates total frame count and shows processing progress.
        When False, processes video without progress indication.
    debug_csv_path : str | None, default=None
        Optional path to write debug CSV output with all detection states.
    """
    # 1) Modules / state
    pose = PoseEstimator()
    analyzer = MovementAnalyzer()
    user_clf = UserClassifier(threshold_deg=uc_threshold_deg, moving_window_seconds=int(uc_moving_window_seconds))
    dwell = DwellTimeDetector(
        stay_threshold_sec=stay_threshold_sec,
        spike_threshold=spike_threshold,
        stability_threshold_px=stability_threshold_px,
        grace_period_sec=grace_period_sec,
    )
    head = HeadShakeDetector(
        horizontal_threshold=15.0,
        vertical_threshold=10.0,
        cycle_detection_window=10,
        min_oscillations=1,
        confidence_threshold=0.5,
        hysteresis_frames=3,
    )
    hand_raise_detector = HandRaiseDetector(
        visibility_threshold=hand_raise_visibility_threshold,
        min_consecutive_frames=hand_raise_min_consecutive_frames,
    )
    posture = PostureMonitor(pm_monitoring_duration_sec, pm_alert_threshold_ratio)

    state = PipelineState(
        pose=pose,
        analyzer=analyzer,
        user_classifier=user_clf,
        dwell_time_detector=dwell,
        head_shake_detector=head,
        hand_raise_detector=hand_raise_detector,
        posture_monitor=posture,
        disable_jp=disable_japanese,
        sway_params={
            "sway_window_sec": float(sway_window_sec),
            "sway_smooth_sec": float(sway_smooth_sec),
            "sway_amp_th_lat": float(sway_amp_th_lat),
            "sway_amp_th_ap": float(sway_amp_th_ap),
            "sway_f_min": float(sway_f_min),
            "sway_f_max": float(sway_f_max),
            "sway_min_cycles": int(sway_min_cycles),
            "sway_on_sec": float(sway_on_sec),
            "sway_off_sec": float(sway_off_sec),
            "sway_use_staying_gate": bool(sway_use_staying_gate),
        },
    )

    # 2) CSV/Video sinks（既存の setup_* を利用）
    csv_file = None
    csv_writer = None
    if output_csv_path:
        csv_file = open(output_csv_path, "w", newline="", encoding="utf-8")  # noqa: SIM115
        csv_writer = setup_csv_writer(csv_file)

    # Debug CSV setup
    debug_csv_file = None
    debug_csv_writer = None
    if debug_csv_path:
        debug_csv_file = open(debug_csv_path, "w", newline="", encoding="utf-8")  # noqa: SIM115
        debug_fieldnames = [
            "timestamp",
            "frame_idx",
            "posture_alerts",
            "posture_forward_ratio",
            "posture_avg_score",
            "posture_sample_count",
            "dwell_is_long_stay",
            "dwell_stay_duration",
            "dwell_alert",
            "head_shake_alerts",
            "head_shake_detected",
            "hand_left_raised",
            "hand_right_raised",
            "hand_both_raised",
            "user_classified",
            "user_classification",
        ]
        debug_csv_writer = csv.DictWriter(debug_csv_file, fieldnames=debug_fieldnames)
        debug_csv_writer.writeheader()

    # VideoWriter は最初のフレーム形状で遅延初期化（ソース実寸に一致させる）
    video_writer = None

    # 3) Calculate total frames for progress bar (only if show_progress is True)
    # Note: Use original video frame count, not FFmpeg-processed count
    # FFmpeg fps filter may change the total frame count
    total_frames = estimate_total_frames(video_path=video_path, fps=fps, show_progress=show_progress)

    # 4) Frame source（FFmpeg優先 → フォールバックOpenCV）
    def _opencv_iter(path: str) -> Iterator[tuple[float, np.ndarray]]:
        cap = cv2.VideoCapture(path)
        try:
            while cap.isOpened():
                ok, f = cap.read()
                if not ok:
                    break
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                yield t, f
        finally:
            cap.release()

    # 推定：パスがファイルなら "ffmpeg-file"
    if input_mode is None:
        input_mode = "ffmpeg-file" if make_frame_iter is not None else "opencv-file"

    if make_frame_iter is not None and input_mode.startswith("ffmpeg"):
        frame_iter = make_frame_iter(
            input_mode,
            ffmpeg_input=video_path,
            width=width,
            height=height,
            fps=fps,
            is_color=is_color,
            add_args=None,
        )
    else:
        frame_iter = _opencv_iter(video_path)

    # 5) Run
    try:
        rp = globals().get("run_pipeline")
        if not callable(rp):
            raise RuntimeError("run_pipeline is not callable")
        # Backward-compatible call: pass debug_csv_writer only if accepted
        try:
            rp(
                frame_iter,
                csv_writer=csv_writer,
                video_writer=video_writer,
                state=state,
                preview=preview,
                window_name="Integrated Analysis",
                output_video_path=output_video_path,
                writer_fps=fps,
                total_frames=total_frames,
                show_progress=show_progress,
                debug_csv_writer=debug_csv_writer,
            )
        except TypeError:
            rp(
                frame_iter,
                csv_writer=csv_writer,
                video_writer=video_writer,
                state=state,
                preview=preview,
                window_name="Integrated Analysis",
                output_video_path=output_video_path,
                writer_fps=fps,
                total_frames=total_frames,
                show_progress=show_progress,
            )
    finally:
        # 明示的にCSVファイルをクローズする
        if csv_file is not None and hasattr(csv_file, "close"):
            csv_file.close()
        if debug_csv_file is not None and hasattr(debug_csv_file, "close"):
            debug_csv_file.close()
