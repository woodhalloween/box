#!/usr/bin/env python3
"""
頭部姿勢検出システム実行スクリプト
MediaPipeを使用して顔のランドマークから頭部姿勢（特に左右の動き）を検出し、「首振り」を検知する
"""

import os
import sys
import argparse
import time
from head_pose.detector import process_video, HeadPoseDetector


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="頭部姿勢検出システム")
    parser.add_argument(
        "--video",
        type=str,
        required=False,
        default="WIN_20250319_10_03_53_Pro.mp4",
        help="処理する動画ファイルのパス",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default="",
        help="出力動画のパス（指定しない場合は保存しない）",
    )
    parser.add_argument("--no-preview", action="store_true", help="プレビューを非表示にする")
    parser.add_argument("--log-dir", type=str, default="logs", help="ログを保存するディレクトリ")

    # 検出パラメータのチューニング用引数
    parser.add_argument(
        "--yaw-threshold",
        type=float,
        default=10.0,
        help="首振りと判定する角度変化の閾値（度、デフォルト: 10.0）",
    )
    parser.add_argument(
        "--consecutive-frames",
        type=int,
        default=2,
        help="首振りと判定する連続フレーム数（デフォルト: 2）",
    )
    parser.add_argument(
        "--max-history", type=int, default=40, help="履歴に保存するフレーム数（デフォルト: 40）"
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=1.0,
        help="検出後のクールダウン時間（秒、デフォルト: 1.0）",
    )

    args = parser.parse_args()

    # 動画ファイルの存在チェック
    if not os.path.exists(args.video):
        print(f"エラー: 指定された動画ファイル '{args.video}' が見つかりません。")
        return 1

    # 出力ディレクトリが指定されている場合、存在チェックと作成
    output_path = None
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError as e:
                print(f"エラー: 出力ディレクトリの作成に失敗しました: {e}")
                return 1
        output_path = args.output

    # 処理開始メッセージ
    print(f"頭部姿勢検出を開始します...")
    print(f"入力動画: {args.video}")
    if output_path:
        print(f"出力動画: {output_path}")
    print(f"プレビュー: {'無効' if args.no_preview else '有効'}")
    print(f"ログディレクトリ: {args.log_dir}")
    print(f"検出パラメータ:")
    print(f"  - 角度変化閾値: {args.yaw_threshold}度")
    print(f"  - 連続フレーム数: {args.consecutive_frames}フレーム")
    print(f"  - 履歴サイズ: {args.max_history}フレーム")
    print(f"  - クールダウン時間: {args.cooldown}秒")
    print("-" * 50)

    start_time = time.time()

    try:
        # カスタム検出器を作成
        detector = HeadPoseDetector(
            max_history=args.max_history,
            yaw_threshold=args.yaw_threshold,
            consecutive_frames=args.consecutive_frames,
            detection_cooldown=args.cooldown,
        )

        # 動画処理の実行（カスタム検出器を直接渡す）
        log_data = process_video(
            args.video,
            output_path,
            show_preview=not args.no_preview,
            log_dir=args.log_dir,
            detector=detector,  # この引数はprocess_video関数で受け取れるように修正が必要
        )

        # 処理時間の計算と表示
        total_time = time.time() - start_time
        print(f"\n処理が完了しました！")
        print(f"総処理時間: {total_time:.2f}秒")

        # 検出統計がすでにprocess_video内で表示されるため、ここでは省略

        return 0

    except KeyboardInterrupt:
        print("\n処理が中断されました。")
        return 130
    except Exception as e:
        print(f"\nエラー: 処理中に例外が発生しました: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
