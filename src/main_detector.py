"""
新しいビジネスロジックに基づいた統合版スクリプト
1. 膝の角度からユーザーを分類 (車椅子ユーザーなど)
2. 腰の座標から滞在時間を検知
3. 店員へ通知 (ここではprint文で代替)
"""

from __future__ import annotations

import argparse

from .video_processor import VideoProcessor


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
    try:
        with VideoProcessor(
            video_path=video_path,
            output_csv_path=output_csv_path,
            output_video_path=output_video_path,
            disable_japanese=disable_japanese,
            stay_threshold_sec=stay_threshold_sec,
        ) as processor:
            processor.run()
    except OSError as e:
        print(e)


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
    )


if __name__ == "__main__":
    main()
