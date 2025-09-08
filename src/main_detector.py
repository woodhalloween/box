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
from .pose.definitions import BodyPart
from .pose_estimator import PoseEstimator


def draw_detection_info(
    frame,
    user_classifier,
    dwell_time_detector,
    head_shake_detector,
    posture_monitor,
    posture_alerts,
    landmarks,
):
    """検知情報をフレームに描画する"""
    y_offset = 30
    # --- 上部にアラートを描画 ---
    all_alerts = []
    user_alert = user_classifier.get_current_alert()
    if user_alert:
        all_alerts.append((user_alert, (0, 255, 255)))  # Yellow for user/knee

    all_alerts.extend([(alert, (0, 0, 255)) for alert in posture_alerts])  # Red for posture

    if head_shake_detector:
        head_shake_alerts = head_shake_detector.check_alerts(cv2.getTickCount() / cv2.getTickFrequency())
        all_alerts.extend([(alert, (255, 0, 255)) for alert in head_shake_alerts])  # Magenta for head shake

    for alert_text, color in all_alerts:
        cv2.putText(frame, alert_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 30

    # --- 滞在検知の情報を描画 (元の詳細版に) ---
    dwell_status = dwell_time_detector.get_current_status()
    if dwell_status["hip_position"]:
        pos = (int(dwell_status["hip_position"][0]), int(dwell_status["hip_position"][1]))
        duration = dwell_status.get("stay_duration", 0.0)
        is_long_stay = dwell_status.get("is_long_stay", False)
        state = dwell_status.get("state", "N/A")
        confidence = dwell_status.get("confidence", 0.0)

        color = (0, 0, 255) if is_long_stay else (0, 255, 0)
        cv2.circle(frame, pos, 8, color, -1)

        # 元の'detect_joint_movement_with_hip_stay.py'のテキスト形式を復元
        text = f"{state}: {duration:.1f}s (Conf:{confidence:.2f})"
        cv2.putText(
            frame,
            text,
            (pos[0] + 15, pos[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    # --- 下部に姿勢監視のステータスを描画 ---
    status = posture_monitor.get_status()
    if status.get("sample_count", 0) > 0:
        status_text = (
            f"Monitor: {status.get('monitoring_duration', 0):.1f}s | "
            f"Forward: {status.get('forward_ratio', 0):.1%} | "
            f"Score: {status.get('avg_score', 0):.2f}"
        )
        cv2.putText(frame, status_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # --- 首振り情報を描画 ---
    if head_shake_detector is not None and landmarks is not None:
        try:
            nose = landmarks[BodyPart.NOSE.value]
            if nose[3] > 0.5:  # 信頼度
                height, width = frame.shape[:2]
                nose_pos = (int(nose[0] * width), int(nose[1] * height))
                status = head_shake_detector.get_status()

                h_state = status.get("horizontal_state", "HEAD_STATIC")
                color = (0, 255, 0)  # Green for static
                if h_state == "HORIZONTAL_SHAKE":
                    color = (0, 255, 255)  # Yellow
                elif h_state == "HEAD_LEFT_TURN":
                    color = (255, 0, 0)  # Blue
                elif h_state == "HEAD_RIGHT_TURN":
                    color = (0, 0, 255)  # Red

                cv2.circle(frame, nose_pos, 8, color, 2)
        except (IndexError, TypeError):
            pass  # ランドマークがない場合は何もしない


def process_video(
    video_path: str,
    output_csv_path: str | None,
    output_video_path: str | None,
    disable_japanese: bool,
    stay_threshold_sec: float,
    spike_threshold: float,
    stability_threshold_px: float,
    grace_period_sec: float,
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
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = setup_csv_writer(csv_file)

        # --- 各分析モジュールの初期化 ---
        pose_estimator = PoseEstimator()
        analyzer = MovementAnalyzer()
        user_classifier = UserClassifier(threshold_deg=100.0, moving_window_seconds=2)
        dwell_time_detector = DwellTimeDetector(
            stay_threshold_sec=stay_threshold_sec,
            spike_threshold=spike_threshold,
            stability_threshold_px=stability_threshold_px,
            grace_period_sec=grace_period_sec,
        )
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
                # 1. UserClassifierでユーザーを分類
                user_classifier.update(timestamp, analysis_results)

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

                if (
                    user_is_classified
                    and is_long_stay
                    and dwell_time_detector.stay_info
                    and not dwell_time_detector.stay_info.notified
                ):
                    # 一度通知したら、移動が検知されるまで再通知しないようにする
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
                frame = draw_analysis_results(frame, analysis_results, landmarks, disable_japanese=disable_japanese)
                draw_detection_info(
                    frame,
                    user_classifier,
                    dwell_time_detector,
                    head_shake_detector,
                    posture_monitor,
                    posture_alerts,
                    landmarks,
                )

            video_writer.write(frame)
            cv2.imshow("Refactored Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_count += 1

    cap.release()
    video_writer.release()
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
    # --- 滞在検知の高度な引数を追加 ---
    parser.add_argument(
        "--spike-threshold",
        type=float,
        default=1.5,
        help="移動スパイク検知の閾値（体幹長比）",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=50.0,
        help="検出安定性（体幹長ブレ）の閾値（px）",
    )
    parser.add_argument("--grace-period", type=float, default=1.5, help="移動検知の猶予期間（秒）")
    args = parser.parse_args()

    process_video(
        video_path=args.video,
        output_csv_path=args.output_csv,
        output_video_path=args.output_video,
        disable_japanese=args.disable_japanese,
        stay_threshold_sec=args.stay_threshold,
        spike_threshold=args.spike_threshold,
        stability_threshold_px=args.stability_threshold,
        grace_period_sec=args.grace_period,
    )


if __name__ == "__main__":
    main()
