"""
scripts/test_mediapipe_head_turn.py

MediaPipe Face Mesh頭部方向検知のテストスクリプト。
国宝さん首振り動画で検知をテストする。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.video_processor import VideoProcessor


def main():
    parser = argparse.ArgumentParser(description="MediaPipe Face Mesh頭部方向検知のテスト")
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/raw/国宝さん首振り1.mp4"),
        help="入力動画ファイルのパス（デフォルト: data/raw/国宝さん首振り1.mp4）",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="出力CSVファイルのパス（デフォルト: 自動生成）",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="出力動画ファイルのパス（デフォルト: 自動生成）",
    )
    parser.add_argument(
        "--yaw-right",
        type=float,
        default=30.0,
        help="右向き判定の閾値（度）（デフォルト: 30.0）",
    )
    parser.add_argument(
        "--yaw-left",
        type=float,
        default=-30.0,
        help="左向き判定の閾値（度）（デフォルト: -30.0）",
    )

    args = parser.parse_args()

    if not args.video.exists():
        print(f"エラー: 動画ファイルが見つかりません - {args.video}")
        return

    print("=" * 60)
    print("MediaPipe Face Mesh 頭部方向検知テスト")
    print("=" * 60)
    print(f"入力動画: {args.video}")
    print(f"右向き閾値: {args.yaw_right}度")
    print(f"左向き閾値: {args.yaw_left}度")
    print("=" * 60)

    # VideoProcessorを使用して処理
    with VideoProcessor(
        video_path=str(args.video),
        output_csv_path=str(args.output_csv) if args.output_csv else None,
        output_video_path=str(args.output_video) if args.output_video else None,
        disable_japanese=False,
        stay_threshold_sec=10.0,
        spike_threshold=1.5,
        stability_threshold_px=50.0,
        grace_period_sec=1.5,
        pm_monitoring_duration_sec=60.0,
        pm_alert_threshold_ratio=0.7,
        uc_threshold_deg=90.0,
        uc_moving_window_seconds=5.0,
        enable_mediapipe_head_turn=True,  # MediaPipe検知を有効化
        mediapipe_yaw_threshold_right=args.yaw_right,
        mediapipe_yaw_threshold_left=args.yaw_left,
    ) as processor:
        processor.run()

    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)


if __name__ == "__main__":
    main()

