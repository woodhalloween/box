"""
ビデオ処理のメインロジックをカプセル化するモジュール。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .analysis.dwell_time_detector import DwellTimeDetector
from .analysis.posture_monitor import PostureMonitor
from .analysis.user_classifier import UserClassifier
from .head_shake_detector import HeadShakeDetector
from .io.csv_writer import setup_csv_writer, write_results_to_csv
from .io.drawing import draw_analysis_results, draw_detection_info, draw_landmarks
from .io_utils import setup_video_writer
from .movement_analyzer import MovementAnalyzer
from .pose_estimator import PoseEstimator


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

        # --- リソースは__enter__で初期化するため、ここではNoneに ---
        self.cap = None
        self.video_writer = None
        self.csv_file = None
        self.csv_writer = None

    def __enter__(self):
        """withブロック開始時にリソースを確保する"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise OSError(f"Error: ビデオファイルが開けません: {self.video_path}")

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
            self.output_video_path = f"output/{p.stem}_output_{timestamp_str}.csv"
        Path(self.output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.output_video_path).parent.mkdir(parents=True, exist_ok=True)

    def _setup_modules(self):
        """分析モジュールとI/Oを初期化する"""
        self.video_writer = setup_video_writer(self.cap, self.output_video_path)
        self.csv_writer = setup_csv_writer(self.csv_file)

        self.pose_estimator = PoseEstimator()
        self.analyzer = MovementAnalyzer()
        self.user_classifier = UserClassifier(
            threshold_deg=self.uc_threshold_deg, moving_window_seconds=self.uc_moving_window_seconds
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
        self.head_shake_detector = HeadShakeDetector()

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
            self.video_writer.write(frame)
            return frame

        # --- 以下、landmarksが検出された場合の処理 ---
        analysis_results = self.analyzer.analyze(landmarks)

        self.user_classifier.update(timestamp, analysis_results)
        dwell_alert = self.dwell_time_detector.update(landmarks, frame.shape, timestamp)
        posture_alerts = self.posture_monitor.update(timestamp, frame_count, analysis_results)
        head_shake_results = self.head_shake_detector.update(landmarks, timestamp, frame_count)
        analysis_results.update(head_shake_results)
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
            landmarks=landmarks,
        )

        frame = draw_landmarks(frame, landmarks)
        frame = draw_analysis_results(frame, analysis_results, landmarks, disable_japanese=self.disable_japanese)
        frame = draw_detection_info(
            frame,
            self.user_classifier,
            self.dwell_time_detector,
            self.head_shake_detector,
            self.posture_monitor,
            posture_alerts,
            landmarks,
            timestamp,
        )

        self.video_writer.write(frame)
        return frame
