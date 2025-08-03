"""src/detect_joint_movement.py

This script is the main entry point for the joint movement detection application.
It captures video from a file, processes each frame to detect pose landmarks,
analyzes the landmarks to calculate joint angles, and displays the results in
real-time.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import cv2

from src.definitions import Angle, MovementState
from src.drawing_utils import draw_analysis_results, draw_landmarks
from src.movement_analyzer import MovementAnalyzer
from src.pose_estimator import PoseEstimator


@dataclass
class PostureSnapshot:
    """姿勢スナップショット - 特定時刻の姿勢データ"""

    timestamp: float
    frame_number: int
    analysis_results: dict[Angle, dict[str, Any]]
    is_forward_leaning: bool
    forward_lean_score: float


class PostureMonitor:
    """前傾姿勢の長期滞在を監視するクラス"""

    def __init__(self, monitoring_duration: float = 60.0, alert_threshold: float = 0.7):
        """
        Args:
            monitoring_duration: 監視期間（秒）
            alert_threshold: アラート閾値（0.0-1.0, 前傾姿勢の割合）
        """
        self.monitoring_duration = monitoring_duration
        self.alert_threshold = alert_threshold
        self.posture_history: deque[PostureSnapshot] = deque()
        self.last_alert_time: float = 0
        self.alert_cooldown: float = 30.0  # アラート間隔（秒）

    def is_forward_leaning_posture(self, analysis_results: dict[Angle, dict[str, Any]]) -> tuple[bool, float]:
        """
        前傾姿勢かどうかを判定

        Args:
            analysis_results: 関節分析結果

        Returns:
            (is_forward_leaning, confidence_score)
        """
        forward_indicators = []

        # 体の傾き角度チェック
        body_tilt = analysis_results.get(Angle.BODY_TILT)
        if body_tilt and "angle" in body_tilt:
            tilt_angle = body_tilt["angle"]
            # 体の傾きが150度以下の場合は前傾の可能性
            if tilt_angle <= 150:
                forward_indicators.append(1.0 - (tilt_angle / 150.0))
            else:
                forward_indicators.append(0.0)

        # 首・胴体角度チェック
        neck_trunk = analysis_results.get(Angle.NECK_TRUNK_ANGLE)
        if neck_trunk and "angle" in neck_trunk:
            neck_angle = neck_trunk["angle"]
            # 首が前に出ている状態（150度以下）
            if neck_angle <= 150:
                forward_indicators.append(1.0 - (neck_angle / 150.0))
            else:
                forward_indicators.append(0.0)

        # 肩の前方傾斜チェック
        right_shoulder = analysis_results.get(Angle.RIGHT_SHOULDER)
        left_shoulder = analysis_results.get(Angle.LEFT_SHOULDER)

        shoulder_flexion_count = 0
        shoulder_total = 0

        for shoulder in [right_shoulder, left_shoulder]:
            if shoulder and "state" in shoulder:
                shoulder_total += 1
                if shoulder["state"] == MovementState.FLEXION:
                    shoulder_flexion_count += 1

        if shoulder_total > 0:
            shoulder_flexion_ratio = shoulder_flexion_count / shoulder_total
            forward_indicators.append(shoulder_flexion_ratio)

        # 前傾スコア計算
        if forward_indicators:
            forward_score = sum(forward_indicators) / len(forward_indicators)
            is_leaning = forward_score > 0.5  # 50%以上で前傾と判定
            return is_leaning, forward_score

        return False, 0.0

    def update(self, timestamp: float, frame_number: int, analysis_results: dict[Angle, dict[str, Any]]) -> list[str]:
        """
        姿勢データを更新し、必要に応じてアラートを生成

        Args:
            timestamp: タイムスタンプ
            frame_number: フレーム番号
            analysis_results: 関節分析結果

        Returns:
            アラートメッセージのリスト
        """
        # 前傾姿勢判定
        is_forward, score = self.is_forward_leaning_posture(analysis_results)

        # スナップショット作成
        snapshot = PostureSnapshot(
            timestamp=timestamp,
            frame_number=frame_number,
            analysis_results=analysis_results.copy(),
            is_forward_leaning=is_forward,
            forward_lean_score=score,
        )

        # 履歴に追加
        self.posture_history.append(snapshot)

        # 古いデータを削除（監視期間外）
        while self.posture_history and timestamp - self.posture_history[0].timestamp > self.monitoring_duration:
            self.posture_history.popleft()

        # アラートチェック
        alerts = self._check_for_alerts(timestamp)

        return alerts

    def _check_for_alerts(self, current_time: float) -> list[str]:
        """アラート条件をチェック"""
        alerts = []

        # クールダウン期間中はアラートしない
        if current_time - self.last_alert_time < self.alert_cooldown:
            return alerts

        # 監視期間に達していない場合はチェックしない
        if not self.posture_history:
            return alerts

        oldest_timestamp = self.posture_history[0].timestamp
        if current_time - oldest_timestamp < self.monitoring_duration:
            return alerts

        # 前傾姿勢の割合を計算
        forward_leaning_count = sum(1 for snapshot in self.posture_history if snapshot.is_forward_leaning)
        total_count = len(self.posture_history)

        if total_count > 0:
            forward_ratio = forward_leaning_count / total_count

            if forward_ratio >= self.alert_threshold:
                # 平均前傾スコア計算
                avg_score = sum(snapshot.forward_lean_score for snapshot in self.posture_history) / total_count

                alert_msg = (
                    f"⚠️ 長時間前傾姿勢検知: {self.monitoring_duration:.0f}秒間の"
                    f"{forward_ratio:.1%}が前傾姿勢 (平均スコア: {avg_score:.2f})"
                )
                alerts.append(alert_msg)

                self.last_alert_time = current_time

        return alerts

    def get_status(self) -> dict[str, Any]:
        """現在の監視状態を取得"""
        if not self.posture_history:
            return {"monitoring_duration": 0, "forward_ratio": 0, "avg_score": 0, "sample_count": 0}

        oldest_timestamp = self.posture_history[0].timestamp
        latest_timestamp = self.posture_history[-1].timestamp
        monitoring_duration = latest_timestamp - oldest_timestamp

        forward_count = sum(1 for s in self.posture_history if s.is_forward_leaning)
        total_count = len(self.posture_history)
        forward_ratio = forward_count / total_count if total_count > 0 else 0

        avg_score = sum(s.forward_lean_score for s in self.posture_history) / total_count if total_count > 0 else 0

        return {
            "monitoring_duration": monitoring_duration,
            "forward_ratio": forward_ratio,
            "avg_score": avg_score,
            "sample_count": total_count,
        }


def open_output_file(path: str) -> IO[Any]:
    """出力用のCSVファイルを開き、ヘッダーを書き込む"""
    try:
        output_file = open(path, "w", newline="")
        # ヘッダーを準備
        header = ["timestamp", "frame_number"]
        all_angles = list(Angle)  # Enumの全メンバーを取得
        for angle in all_angles:
            header.extend([f"{angle.value}_ANGLE", f"{angle.value}_STATE"])

        # 前傾姿勢監視データを追加
        header.extend(["FORWARD_LEANING", "FORWARD_SCORE", "MONITORING_DURATION", "FORWARD_RATIO"])

        csv_writer = csv.writer(output_file)
        csv_writer.writerow(header)
        return output_file
    except OSError as e:
        sys.exit(f"Error: Cannot open output file {path} - {e}")


def write_results_to_csv(
    writer: Any, timestamp: float, frame_number: int, results: dict, posture_monitor: PostureMonitor
) -> None:
    """分析結果をCSVファイルに書き込む"""
    row = [timestamp, frame_number]

    # 関節角度データ
    for joint_name in MovementAnalyzer.ANGLE_DEFINITIONS:
        angle_data = results.get(joint_name)
        if angle_data:
            row.append(f"{angle_data['angle']:.2f}")
            row.append(angle_data["state"].value)
        else:
            # データがない場合は空欄を追加
            row.extend(["", ""])

    # 前傾姿勢監視データ
    is_forward, score = posture_monitor.is_forward_leaning_posture(results)
    status = posture_monitor.get_status()

    row.extend([is_forward, f"{score:.3f}", f"{status['monitoring_duration']:.1f}", f"{status['forward_ratio']:.3f}"])

    writer.writerow(row)


def setup_csv_writer(csv_file: IO[Any]):
    """CSVファイルのヘッダーを書き込む"""
    header = ["timestamp", "frame_number"]
    all_angles = list(Angle)  # Enumの全メンバーを取得
    for angle in all_angles:
        header.extend([f"{angle.value}_ANGLE", f"{angle.value}_STATE"])

    # 前傾姿勢監視データを追加
    header.extend(["FORWARD_LEANING", "FORWARD_SCORE", "MONITORING_DURATION", "FORWARD_RATIO"])

    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)
    return csv_writer


def setup_video_writer(cap: cv2.VideoCapture, output_video_path: str) -> cv2.VideoWriter:
    """ビデオライターを初期化し、ビデオを書き込む"""
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    return video_writer


def draw_posture_alerts(frame, alerts: list[str], status: dict[str, Any]):
    """フレームに前傾姿勢アラートと状態を描画"""
    y_offset = 30

    # アラート表示
    for alert in alerts:
        cv2.putText(frame, alert, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30

    # 監視状態表示
    if status["sample_count"] > 0:
        status_text = (
            f"Monitor: {status['monitoring_duration']:.1f}s | "
            f"Forward: {status['forward_ratio']:.1%} | "
            f"Score: {status['avg_score']:.2f}"
        )
        cv2.putText(frame, status_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


def process_video(
    video_path: str,
    output_csv_path: str | None,
    output_video_path: str | None,
    disable_japanese: bool,
    monitoring_duration: float = 60.0,
    alert_threshold: float = 0.7,
):
    """
    ビデオを処理して、関節の動きを分析し、結果をCSVとビデオに出力する。
    前傾姿勢の長期滞在も監視する。

    :param video_path: 入力ビデオのパス
    :param output_csv_path: 出力CSVファイルのパス
    :param output_video_path: 出力ビデオファイルのパス
    :param disable_japanese: 日本語テキストの描画を無効にするかどうか
    :param monitoring_duration: 前傾姿勢監視期間（秒）
    :param alert_threshold: アラート閾値（前傾姿勢の割合）
    """
    # 入力ビデオを開く
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # 出力ファイルパスが指定されていない場合、デフォルトパスを生成
    p = Path(video_path)
    if output_csv_path is None:
        output_csv_path = f"output/{p.stem}_analysis.csv"
    if output_video_path is None:
        output_video_path = f"output/{p.stem}_output.mp4"

    # 出力ディレクトリを作成
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)

    # CSVライターとビデオライターをセットアップ
    csv_file = open(output_csv_path, "w", newline="")
    csv_writer = setup_csv_writer(csv_file)
    video_writer = setup_video_writer(cap, output_video_path)

    pose_estimator = PoseEstimator()
    analyzer = MovementAnalyzer()
    posture_monitor = PostureMonitor(monitoring_duration, alert_threshold)

    # パフォーマンス計測用の変数を初期化
    frame_count = 0
    total_time_spent = 0.0
    time_reading = 0.0
    time_posing = 0.0
    time_analyzing = 0.0
    time_drawing = 0.0
    time_writing = 0.0
    time_monitoring = 0.0

    print(f"前傾姿勢監視開始: {monitoring_duration}秒間, 閾値: {alert_threshold:.1%}")

    while cap.isOpened():
        loop_start_time = time.perf_counter()

        start_time = time.perf_counter()
        success, frame = cap.read()
        if not success:
            break
        time_reading += time.perf_counter() - start_time

        start_time = time.perf_counter()
        landmarks = pose_estimator.estimate(frame)
        time_posing += time.perf_counter() - start_time

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        analysis_results = {}
        alerts = []

        if landmarks is not None:
            # Movement Analysis
            start_time = time.perf_counter()
            analysis_results = analyzer.analyze(landmarks)
            time_analyzing += time.perf_counter() - start_time

            # Posture Monitoring
            start_time = time.perf_counter()
            alerts = posture_monitor.update(timestamp, frame_count, analysis_results)
            time_monitoring += time.perf_counter() - start_time

            # Write to CSV
            start_time = time.perf_counter()
            write_results_to_csv(csv_writer, timestamp, frame_count, analysis_results, posture_monitor)
            time_writing += time.perf_counter() - start_time

        # 4. Drawing
        start_time = time.perf_counter()
        if landmarks is not None:
            # draw_analysis_resultsが描画済みの画像を返すように変更されたため、
            # 戻り値で変数を更新する

            # --- FPS計算と描画関数の呼び出し ---
            loop_time = time.perf_counter() - loop_start_time
            current_fps = 1.0 / loop_time if loop_time > 0 else 0

            frame = draw_analysis_results(
                image=frame,
                results=analysis_results,
                landmarks=landmarks,
                fps=current_fps,
                disable_japanese=disable_japanese,
            )
            # --- ここまで ---
            draw_landmarks(frame, landmarks)

        # アラートと監視状態の描画
        status = posture_monitor.get_status()
        draw_posture_alerts(frame, alerts, status)
        time_drawing += time.perf_counter() - start_time

        # アラート表示
        for alert in alerts:
            print(f"フレーム {frame_count}: {alert}")

        # 描画されたフレームをビデオに書き込む
        start_time = time.perf_counter()
        if video_writer:
            video_writer.write(frame)
        time_writing += time.perf_counter() - start_time

        # 5. Display
        cv2.imshow("Movement Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

        total_time_spent += time.perf_counter() - loop_start_time

    # クリーンアップ
    cap.release()
    video_writer.release()
    csv_file.close()
    cv2.destroyAllWindows()

    # --- パフォーマンス分析レポートを出力 ---
    if frame_count > 0 and total_time_spent > 0:
        print("\n--- Performance Analysis Report ---")
        print(f"Total frames processed: {frame_count}")
        print(f"Total processing time: {total_time_spent:.2f} seconds")
        print(f"Average FPS (including waitKey): {frame_count / total_time_spent:.2f}")
        print("-" * 33)

        total_tracked_time = time_reading + time_posing + time_analyzing + time_drawing + time_writing + time_monitoring
        if total_tracked_time == 0:
            total_tracked_time = 1

        print(f"Bottleneck Analysis (based on {total_tracked_time:.2f}s of tracked processing time):")
        print(f"  - AI Pose Estimation:   {time_posing:7.2f}s ({time_posing / total_tracked_time * 100:5.1f}%)")
        print(
            f"  - OpenCV Operations:    {time_reading + time_drawing + time_writing:7.2f}s ({(time_reading + time_drawing + time_writing) / total_tracked_time * 100:5.1f}%)"
        )
        print(f"  - Joint Analyzing:      {time_analyzing:7.2f}s ({time_analyzing / total_tracked_time * 100:5.1f}%)")
        print(f"  - Posture Monitoring:   {time_monitoring:7.2f}s ({time_monitoring / total_tracked_time * 100:5.1f}%)")
        print("-" * 33)

        # 最終監視結果表示
        final_status = posture_monitor.get_status()
        print("\n--- 前傾姿勢監視結果 ---")
        print(f"監視期間: {final_status['monitoring_duration']:.1f}秒")
        print(f"前傾姿勢割合: {final_status['forward_ratio']:.1%}")
        print(f"平均前傾スコア: {final_status['avg_score']:.3f}")
        print(f"分析サンプル数: {final_status['sample_count']}")
        print("--- End of Report ---")


def main() -> None:
    """
    Main function to parse arguments and start the video processing.
    """
    parser = argparse.ArgumentParser(description="Analyze joint movements from a video with forward leaning detection.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output-csv", type=str, help="Path to the output CSV file to save results.")
    parser.add_argument("--output-video", type=str, help="Path to the output video file to save the processed video.")
    parser.add_argument("--disable-japanese", action="store_true", help="Disable Japanese text in the output video.")
    parser.add_argument(
        "--monitoring-duration",
        type=float,
        default=60.0,
        help="Forward leaning monitoring duration in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--alert-threshold", type=float, default=0.7, help="Alert threshold for forward leaning ratio (default: 0.7)"
    )

    args = parser.parse_args()

    process_video(
        args.video,
        args.output_csv,
        args.output_video,
        args.disable_japanese,
        args.monitoring_duration,
        args.alert_threshold,
    )


if __name__ == "__main__":
    main()
