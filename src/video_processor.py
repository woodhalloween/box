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
from .io.debug_csv_writer import (
    setup_debug_csv_writer,
    write_debug_results_to_csv,
)
from .io.drawing import draw_analysis_results, draw_detection_info, draw_landmarks
from .io.video_writer import setup_video_writer
from .movement_analyzer import MovementAnalyzer
from .pose_estimator import PoseEstimator


class VideoProcessor:
    """ビデオ処理のワークフローを管理するクラス"""

    def __init__(
        self,
        video_path: str,
        output_csv_path: str | None,
        output_video_path: str | None,
        disable_japanese: bool,
        stay_threshold_sec: float,
    ):
        self.video_path = video_path
        self.output_csv_path = output_csv_path
        self.output_video_path = output_video_path
        self.disable_japanese = disable_japanese
        self.stay_threshold_sec = stay_threshold_sec

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise OSError(f"Error: ビデオファイルが開けません: {video_path}")

        self._setup_paths()
        self._setup_modules()

    def _setup_paths(self):
        """出力パスを準備する"""
        p = Path(self.video_path)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.output_csv_path is None:
            self.output_csv_path = f"output/{p.stem}_analysis_{timestamp_str}.csv"
        if self.output_video_path is None:
            self.output_video_path = f"output/{p.stem}_output_{timestamp_str}.mp4"
        self.debug_csv_path = f"output/debug_analysis_{timestamp_str}.csv"  # デバッグ用CSVパス
        Path(self.output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.output_video_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.debug_csv_path).parent.mkdir(parents=True, exist_ok=True)

    def _setup_modules(self):
        """分析モジュールとI/Oを初期化する"""
        self.video_writer = setup_video_writer(self.cap, self.output_video_path)
        self.csv_file = open(self.output_csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = setup_csv_writer(self.csv_file)
        self.debug_csv_file = open(self.debug_csv_path, "w", newline="", encoding="utf-8")
        self.debug_csv_writer = setup_debug_csv_writer(self.debug_csv_file)

        self.pose_estimator = PoseEstimator()
        self.analyzer = MovementAnalyzer()
        self.user_classifier = UserClassifier(threshold_deg=100.0, moving_window_seconds=2)
        self.dwell_time_detector = DwellTimeDetector(stay_threshold_sec=self.stay_threshold_sec)
        self.posture_monitor = PostureMonitor()
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
            if timestamp > 20.0:  # 20秒以上処理したらループを抜ける
                print("--- DEBUG: Reached 20 second limit for comparison. Exiting loop. ---")
                break

            self._process_frame(frame, frame_count, timestamp)

            cv2.imshow("Refactored Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_count += 1

        self._cleanup()

    def _process_frame(self, frame: np.ndarray, frame_count: int, timestamp: float):
        """単一フレームを処理する"""
        landmarks = self.pose_estimator.estimate(frame)

        # 骨格が検出されなかった場合は、ここで処理を終了し、フレームだけ書き出す
        if landmarks is None:
            # デバッグCSVには記録を残す
            write_debug_results_to_csv(
                self.debug_csv_writer,
                timestamp=timestamp,
                frame_number=frame_count,
                analysis_results={},
                landmarks=None,
            )
            self.video_writer.write(frame)
            return

        # --- 以下、landmarksが検出された場合の処理 ---
        analysis_results = self.analyzer.analyze(landmarks)

        # デバッグCSVに常に書き込む
        write_debug_results_to_csv(
            self.debug_csv_writer,
            timestamp=timestamp,
            frame_number=frame_count,
            analysis_results=analysis_results,
            landmarks=landmarks,
        )

        user_alerts = self.user_classifier.update(timestamp, analysis_results)
        dwell_alert = self.dwell_time_detector.update(landmarks, frame.shape, timestamp)
        posture_alerts = self.posture_monitor.update(timestamp, frame_count, analysis_results)
        head_shake_results = self.head_shake_detector.update(landmarks, timestamp, frame_count)
        analysis_results.update(head_shake_results)
        head_shake_alerts = self.head_shake_detector.check_alerts(timestamp)

        user_is_classified = self.user_classifier.get_current_alert() is not None
        is_long_stay = self.dwell_time_detector.get_current_status()["is_long_stay"]

        if user_is_classified and is_long_stay:
            if self.dwell_time_detector.stay_info and not self.dwell_time_detector.stay_info.notified:
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

        draw_landmarks(frame, landmarks)
        frame = draw_analysis_results(frame, analysis_results, landmarks, disable_japanese=self.disable_japanese)
        draw_detection_info(
            frame,
            user_classifier=self.user_classifier,
            dwell_time_detector=self.dwell_time_detector,
            head_shake_detector=self.head_shake_detector,
            timestamp=timestamp,
        )

        self.video_writer.write(frame)

    def _cleanup(self):
        """リソースを解放する"""
        self.cap.release()
        self.video_writer.release()
        self.csv_file.close()
        self.debug_csv_file.close()
        cv2.destroyAllWindows()
        print("--- ビデオ処理完了 ---")
        print(f"分析結果を {self.output_csv_path} に保存しました。")
        print(f"処理済みビデオを {self.output_video_path} に保存しました。")
