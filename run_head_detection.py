#!/usr/bin/env python3
"""
頭部検出システム実行スクリプト
YOLOを使用して人物の頭部を検出し、「首振り」（左右の動き）を検知する
"""

import os
import sys
import argparse
import time
from head_detection.detector import HeadDetector, process_video

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="頭部検出と首振り検知システム")
    parser.add_argument("--video", type=str, required=False, default="WIN_20250319_10_03_53_Pro.mp4",
                        help="処理する動画ファイルのパス")
    parser.add_argument("--output", type=str, required=False, 
                        default="data/output/head_detection_result.mp4",
                        help="出力動画のパス（指定しない場合はデフォルトパスに保存）")
    parser.add_argument("--no-save", action="store_true", 
                        help="動画を保存しない場合はこのオプションを指定")
    parser.add_argument("--no-preview", action="store_true", help="プレビューを非表示にする")
    parser.add_argument("--log-dir", type=str, default="logs", help="ログを保存するディレクトリ")
    
    # YOLOおよび検出パラメータのチューニング用引数
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                       help="YOLOモデルのパス（デフォルト: yolo11n.pt）")
    parser.add_argument("--confidence", type=float, default=0.35,
                       help="検出信頼度閾値（デフォルト: 0.35）")
    parser.add_argument("--movement-threshold", type=float, default=0.08,
                      help="首振りと判定する移動量の閾値（画像幅に対する割合、デフォルト: 0.08）")
    parser.add_argument("--consecutive-frames", type=int, default=3,
                      help="首振りと判定する連続フレーム数（デフォルト: 3）")
    parser.add_argument("--cooldown", type=float, default=2.0,
                       help="検出後のクールダウン時間（秒、デフォルト: 2.0）")
    parser.add_argument("--min-duration", type=float, default=0.5,
                       help="首振りイベントの最小持続時間（秒、デフォルト: 0.5）")
    
    args = parser.parse_args()
    
    # 動画ファイルの存在チェック
    if not os.path.exists(args.video):
        print(f"エラー: 指定された動画ファイル '{args.video}' が見つかりません。")
        return 1
    
    # 出力ディレクトリが指定されている場合、存在チェックと作成
    output_path = None
    if not args.no_save:
        output_path = args.output
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError as e:
                print(f"エラー: 出力ディレクトリの作成に失敗しました: {e}")
                return 1
    
    # 処理開始メッセージ
    print(f"頭部検出と首振り検知を開始します...")
    print(f"入力動画: {args.video}")
    if output_path:
        print(f"出力動画: {output_path} （保存します）")
    else:
        print(f"動画の保存: 無効")
    print(f"プレビュー: {'無効' if args.no_preview else '有効'}")
    print(f"ログディレクトリ: {args.log_dir}")
    print(f"検出パラメータ:")
    print(f"  - モデル: {args.model}")
    print(f"  - 信頼度閾値: {args.confidence}")
    print(f"  - 移動量閾値: {args.movement_threshold}")
    print(f"  - 連続フレーム数: {args.consecutive_frames}")
    print(f"  - クールダウン時間: {args.cooldown}秒")
    print(f"  - 最小持続時間: {args.min_duration}秒")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        # カスタム検出器を作成
        detector = HeadDetector(
            model_path=args.model,
            confidence=args.confidence,
            movement_threshold=args.movement_threshold,
            consecutive_frames=args.consecutive_frames,
            detection_cooldown=args.cooldown,
            min_event_duration=args.min_duration
        )
        
        # 動画処理の実行
        log_data = process_video(
            args.video, 
            output_path,
            show_preview=not args.no_preview,
            log_dir=args.log_dir,
            detector=detector
        )
        
        # 処理時間の計算と表示
        total_time = time.time() - start_time
        print(f"\n処理が完了しました！")
        print(f"総処理時間: {total_time:.2f}秒")
        
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