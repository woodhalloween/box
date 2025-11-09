"""
新しいビジネスロジックに基づいた統合版スクリプト
1. 膝の角度からユーザーを分類 (車椅子ユーザーなど)
2. 腰の座標から滞在時間を検知
3. 店員へ通知 (ここではprint文で代替)
"""

from __future__ import annotations

import argparse

from .config import config
from .video_processor import VideoProcessor


def process_video(
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
    """
    ビデオを処理し、新しいビジネスロジックに基づいて分析を行う
    """
    try:
        with VideoProcessor(
            video_path=video_path,
            output_csv_path=output_csv_path,
            output_video_path=output_video_path,
            disable_japanese=disable_japanese,
            stay_threshold_sec=stay_threshold_sec,
            spike_threshold=spike_threshold,
            stability_threshold_px=stability_threshold_px,
            grace_period_sec=grace_period_sec,
            pm_monitoring_duration_sec=pm_monitoring_duration_sec,
            pm_alert_threshold_ratio=pm_alert_threshold_ratio,
            uc_threshold_deg=uc_threshold_deg,
            uc_moving_window_seconds=uc_moving_window_seconds,
            enable_mediapipe_head_turn=True,
            mediapipe_yaw_threshold_right=30.0,
            mediapipe_yaw_threshold_left=-30.0,
        ) as processor:
            processor.run()
    except OSError as e:
        print(e)


def main():
    parser = argparse.ArgumentParser(
        description="リファクタリング版 滞在検知システム",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- General arguments ---
    parser.add_argument("--video", required=True, help="入力ビデオファイルのパス")
    parser.add_argument("--output-csv", help="出力CSVファイルのパス")
    parser.add_argument("--output-video", help="出力ビデオファイルのパス")
    parser.add_argument("--disable-japanese", action="store_true", help="描画テキストを英語にする")

    # --- DwellTimeDetector arguments ---
    group_dwell = parser.add_argument_group("DwellTimeDetector Settings")
    group_dwell.add_argument(
        "--stay-threshold",
        type=float,
        default=config.getfloat("dwell_time_detector.stay_threshold_sec", fallback=10.0),
        help="「長期滞在」と判定する時間の閾値（秒）",
    )
    group_dwell.add_argument(
        "--spike-threshold",
        type=float,
        default=config.getfloat("dwell_time_detector.advanced_detection.spike_threshold", fallback=1.5),
        help="移動スパイク検知の閾値（体幹長比）",
    )
    group_dwell.add_argument(
        "--stability-threshold",
        type=float,
        default=config.getfloat("dwell_time_detector.advanced_detection.stability_threshold_px", fallback=50.0),
        help="検出安定性（体幹長ブレ）の閾値（px）",
    )
    group_dwell.add_argument(
        "--grace-period",
        type=float,
        default=config.getfloat("dwell_time_detector.advanced_detection.grace_period_sec", fallback=1.5),
        help="移動検知の猶予期間（秒）",
    )

    # --- PostureMonitor arguments ---
    group_posture = parser.add_argument_group("PostureMonitor Settings")
    group_posture.add_argument(
        "--pm-duration",
        type=float,
        default=config.getfloat("posture_monitor.monitoring_duration_sec", fallback=60.0),
        help="姿勢監視を行う期間（秒）",
    )
    group_posture.add_argument(
        "--pm-threshold",
        type=float,
        default=config.getfloat("posture_monitor.alert_threshold_ratio", fallback=0.7),
        help="アラートを発する前傾姿勢の割合の閾値",
    )

    # --- UserClassifier arguments ---
    group_user = parser.add_argument_group("UserClassifier Settings")
    group_user.add_argument(
        "--uc-threshold",
        type=float,
        default=config.getfloat("user_classifier.threshold_deg", fallback=90.0),
        help="ユーザー分類（膝角度）のアラートを発する閾値（度）",
    )
    group_user.add_argument(
        "--uc-window",
        type=int,
        default=config.getint("user_classifier.moving_window_seconds", fallback=5),
        help="ユーザー分類（膝角度）の移動平均を計算する期間（秒）",
    )

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
        pm_monitoring_duration_sec=args.pm_duration,
        pm_alert_threshold_ratio=args.pm_threshold,
        uc_threshold_deg=args.uc_threshold,
        uc_moving_window_seconds=args.uc_window,
    )


if __name__ == "__main__":
    main()
