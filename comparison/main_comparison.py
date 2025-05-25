"""
YOLO単体 vs ByteTrack ID付与比較実行メインファイル
"""

import argparse
import os
import sys
from pathlib import Path

# パス設定
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

from evaluators.comparison_evaluator import ComparisonEvaluator


def main():
    """
    メイン実行関数
    """
    parser = argparse.ArgumentParser(description="YOLO vs ByteTrack ID付与比較")

    # 必須引数
    parser.add_argument("video_path", type=str, help="入力動画ファイルのパス")

    # オプション引数
    parser.add_argument(
        "--model", type=str, default="yolo11n.pt", help="YOLOモデルのパス (default: yolo11n.pt)"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.3, help="検出信頼度閾値 (default: 0.3)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help='実行デバイス ("", "cpu", "mps", "0"等) (default: auto)',
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="処理する最大フレーム数 (default: 全フレーム)"
    )
    parser.add_argument("--display", action="store_true", help="リアルタイム比較表示を有効にする")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison/output",
        help="結果出力ディレクトリ (default: comparison/output)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="comparison",
        help="出力ファイル名のプレフィックス (default: comparison)",
    )

    args = parser.parse_args()

    # 引数の検証
    if not os.path.exists(args.video_path):
        print(f"エラー: 動画ファイルが見つかりません: {args.video_path}")
        return 1

    # モデルファイルの確認
    if not os.path.exists(args.model):
        print(f"警告: モデルファイルが見つかりません: {args.model}")
        print("YOLOの自動ダウンロード機能を利用します")

    print("=" * 80)
    print("YOLO単体 vs ByteTrack ID付与比較実行")
    print("=" * 80)
    print(f"入力動画: {args.video_path}")
    print(f"モデル: {args.model}")
    print(f"信頼度閾値: {args.confidence}")
    print(f"実行デバイス: {args.device if args.device else 'auto'}")
    print(f"最大フレーム数: {args.max_frames if args.max_frames else '全フレーム'}")
    print(f"リアルタイム表示: {'有効' if args.display else '無効'}")
    print(f"出力ディレクトリ: {args.output_dir}")
    print("=" * 80)

    try:
        # 比較評価器の初期化
        evaluator = ComparisonEvaluator(
            model_path=args.model,
            confidence=args.confidence,
            device=args.device,
            output_dir=args.output_dir,
        )

        # 動画処理実行
        print("\n動画処理を開始します...")
        results = evaluator.process_video(
            video_path=args.video_path, max_frames=args.max_frames, display=args.display
        )

        # メトリクス計算
        print("\nメトリクス計算中...")
        metrics = evaluator.calculate_final_metrics()

        # 結果表示
        evaluator.print_summary()

        # 結果保存
        print("\n結果を保存中...")
        evaluator.save_results(prefix=args.prefix)

        print("\n比較評価が完了しました！")
        return 0

    except KeyboardInterrupt:
        print("\n\nユーザーによって中断されました")
        return 1
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


def run_demo():
    """
    デモ実行用の関数
    """
    print("=" * 80)
    print("YOLO vs ByteTrack 比較デモ")
    print("=" * 80)

    # テスト動画ファイルの確認
    test_videos = [
        "data/test_video.mp4",
        "tracked_2025-05-03_152528フロア10minites_30sec_bytetrack.mp4",
        "2025-05-03_152528フロア10minites_30sec.mp4",
    ]

    available_video = None
    for video in test_videos:
        if os.path.exists(video):
            available_video = video
            break

    if not available_video:
        print("エラー: テスト用動画ファイルが見つかりません")
        print("以下のいずれかのファイルを配置してください:")
        for video in test_videos:
            print(f"  - {video}")
        return 1

    print(f"テスト動画: {available_video}")
    print("短時間のデモを実行します（最大300フレーム）")
    print("=" * 80)

    try:
        # デモ用の評価器初期化
        evaluator = ComparisonEvaluator(
            model_path="yolo11n.pt", confidence=0.3, device="", output_dir="comparison/output"
        )

        # デモ実行
        results = evaluator.process_video(
            video_path=available_video,
            max_frames=300,  # 10秒程度のデモ
            display=False,
        )

        # 結果表示
        evaluator.calculate_final_metrics()
        evaluator.print_summary()
        evaluator.save_results(prefix="demo")

        print("\nデモ実行が完了しました！")
        return 0

    except Exception as e:
        print(f"デモ実行中にエラーが発生しました: {str(e)}")
        return 1


if __name__ == "__main__":
    # コマンドライン引数をチェック
    if len(sys.argv) == 1:
        # 引数なしの場合はデモモードで実行
        print("引数が指定されていません。デモモードで実行します。")
        print("使用方法: python main_comparison.py <video_path> [options]")
        print("デモモードを開始します...")
        sys.exit(run_demo())
    else:
        # 通常モードで実行
        sys.exit(main())
