"""
新しいビジネスロジックに基づいた統合版スクリプト
1. 膝の角度からユーザーを分類 (車椅子ユーザーなど)
2. 腰の座標から滞在時間を検知
3. 店員へ通知 (ここではprint文で代替)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2

from .analysis.dwell_time_detector import DwellTimeDetector
from .analysis.posture_monitor import PostureMonitor
from .analysis.user_classifier import UserClassifier
from .drawing_utils import draw_analysis_results, draw_landmarks
from .head_shake_detector import HeadShakeDetector
from .io_utils import setup_csv_writer, setup_video_writer, write_results_to_csv
from .movement_analyzer import MovementAnalyzer
from .pose_estimator import PoseEstimator


def draw_detection_info(frame, user_classifier, dwell_time_detector, head_shake_detector):
    """検知情報をフレームに描画する"""
    y_offset = 30
    # ユーザー分類のアラートを描画
    user_alert = user_classifier.get_current_alert()
    if user_alert:
        cv2.putText(frame, user_alert, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += 30

    # 首振りアラートの表示
    if head_shake_detector:
        head_shake_alerts = head_shake_detector.check_alerts(cv2.getTickCount() / cv2.getTickFrequency())
        for alert in head_shake_alerts:
            cv2.putText(frame, alert, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            y_offset += 30

    # 滞在検知の情報を描画
    dwell_status = dwell_time_detector.get_current_status()
    if dwell_status["hip_position"]:
        pos = (int(dwell_status["hip_position"][0]), int(dwell_status["hip_position"][1]))
        duration = dwell_status["stay_duration"]
        color = (0, 0, 255) if dwell_status["is_long_stay"] else (0, 255, 0)
        cv2.circle(frame, pos, 8, color, -1)
        cv2.putText(
            frame,
            f"Stay: {duration:.1f}s",
            (pos[0] + 15, pos[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def process_video(
    video_path: str,
    output_csv_path: str | None,
    output_video_path: str | None,
    disable_japanese: bool,
    stay_threshold_sec: float,
):
    """
    ビデオを処理し、新しいビジネスロジックに基づいて分析を行う
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: ビデオファイルが開けません: {video_path}")
        return

    # --- 出力ファイルの準備 ---
    p = Path(video_path)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_csv_path is None:
        output_csv_path = f"output/{p.stem}_analysis_{timestamp_str}.csv"
    if output_video_path is None:
        output_video_path = f"output/{p.stem}_output_{timestamp_str}.mp4"
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)

    video_writer = setup_video_writer(cap, output_video_path)
    csv_file = open(output_csv_path, "w", newline="", encoding="utf-8")
    csv_writer = setup_csv_writer(csv_file)

    # --- 各分析モジュールの初期化 ---
    pose_estimator = PoseEstimator()
    analyzer = MovementAnalyzer()
    user_classifier = UserClassifier(threshold_deg=100.0, moving_window_seconds=2)
    dwell_time_detector = DwellTimeDetector(stay_threshold_sec=stay_threshold_sec)
    posture_monitor = PostureMonitor()
    head_shake_detector = HeadShakeDetector()

    frame_count = 0
    print(f"--- ビデオ処理開始: {video_path} ---")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        landmarks = pose_estimator.estimate(frame)

        if landmarks is not None:
            analysis_results = analyzer.analyze(landmarks)
            # print(f"Frame {frame_count}: Analysis results: {analysis_results}")  # デバッグ出力

            # 1. UserClassifierでユーザーを分類
            user_alerts = user_classifier.update(timestamp, analysis_results)

            # 2. DwellTimeDetectorで滞在時間を更新
            dwell_alert = dwell_time_detector.update(landmarks, frame.shape, timestamp)

            # 3. PostureMonitorで姿勢を更新
            posture_alerts = posture_monitor.update(timestamp, frame_count, analysis_results)

            # 4. HeadShakeDetectorで首振りを検知
            head_shake_results = head_shake_detector.update(landmarks, timestamp, frame_count)
            analysis_results.update(head_shake_results)
            head_shake_alerts = head_shake_detector.check_alerts(timestamp)

            # 5. 条件に応じて通知
            user_is_classified = user_classifier.get_current_alert() is not None
            is_long_stay = dwell_time_detector.get_current_status()["is_long_stay"]

            if user_is_classified and is_long_stay:
                # 一度通知したら、移動が検知されるまで再通知しないようにする
                if dwell_time_detector.stay_info and not dwell_time_detector.stay_info.notified:
                    print(f"[{timestamp:.1f}s] 通知: 指定エリアのお客様対応をお願いします。")
                    # 通知済みフラグはDwellTimeDetector側で自動的に管理される

            # --- CSV書き込み ---
            write_results_to_csv(
                csv_writer=csv_writer,
                timestamp=timestamp,
                frame_number=frame_count,
                analysis_results=analysis_results,
                posture_monitor=posture_monitor,
                dwell_time_detector=dwell_time_detector,
                dwell_alert=dwell_alert,
                head_shake_detector=head_shake_detector,
                head_shake_alerts=head_shake_alerts,
                landmarks=landmarks,
            )

            # --- 描画処理 ---
            draw_landmarks(frame, landmarks)
            draw_analysis_results(frame, analysis_results, landmarks, disable_japanese=disable_japanese)
            draw_detection_info(frame, user_classifier, dwell_time_detector, head_shake_detector)

        video_writer.write(frame)
        cv2.imshow("Refactored Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_count += 1

    cap.release()
    video_writer.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print("--- ビデオ処理完了 ---")
    print(f"分析結果を {output_csv_path} に保存しました。")
    print(f"処理済みビデオを {output_video_path} に保存しました。")


def main():
    parser = argparse.ArgumentParser(description="リファクタリング版 滞在検知システム")
    parser.add_argument("--video", required=True, help="入力ビデオファイルのパス")
    parser.add_argument("--output-csv", help="出力CSVファイルのパス")
    parser.add_argument("--output-video", help="出力ビデオファイルのパス")
    parser.add_argument("--disable-japanese", action="store_true", help="描画テキストを英語にする")
    parser.add_argument(
        "--stay-threshold",
        type=float,
        default=10.0,
        help="「長期滞在」と判定する時間の閾値（秒）",
    )
    args = parser.parse_args()

    process_video(
        video_path=args.video,
        output_csv_path=args.output_csv,
        output_video_path=args.output_video,
        disable_japanese=args.disable_japanese,
        stay_threshold_sec=args.stay_threshold,
    )


if __name__ == "__main__":
    main()
