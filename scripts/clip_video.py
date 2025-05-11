import argparse
import os
from pathlib import Path
import cv2
import time

def clip_video(input_path, output_path, start_time=0, duration=30):
    """
    動画ファイルを指定された開始時間から一定時間切り取る
    
    Args:
        input_path (str): 入力動画のパス
        output_path (str): 出力動画のパス
        start_time (float): 切り取り開始時間（秒）
        duration (float): 切り取る秒数
    """
    # 入力動画を開く
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"エラー: ビデオファイルを開けませんでした: {input_path}")
        return False
    
    # 動画のプロパティを取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps if fps > 0 else 0
    
    print(f"入力動画情報: {width}x{height} @ {fps:.2f}fps, 長さ: {total_duration:.2f}秒")
    
    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 切り取りの終了時間を計算（元動画の長さを超えないようにする）
    end_time = min(start_time + duration, total_duration)
    actual_duration = end_time - start_time
    
    # 開始フレームに移動
    start_frame = int(start_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 出力動画の設定（MacOS互換のコーデック）
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264コーデック
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # フレームの切り取り
    current_time = start_time
    frame_count = 0
    print(f"処理開始: {start_time:.2f}秒から{end_time:.2f}秒まで（{actual_duration:.2f}秒間）")
    start_process_time = time.time()
    
    while cap.isOpened() and current_time < end_time:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 出力動画に書き込み
        out.write(frame)
        
        # 進捗表示（10フレームごと）
        frame_count += 1
        current_time = start_time + (frame_count / fps)
        if frame_count % 10 == 0:
            progress = (current_time - start_time) / actual_duration * 100
            elapsed = time.time() - start_process_time
            remaining = (actual_duration - (current_time - start_time)) / (current_time - start_time) * elapsed if current_time > start_time else 0
            print(f"進捗: {progress:.1f}% ({frame_count}フレーム, {current_time:.2f}秒) - 残り約{remaining:.1f}秒")
    
    # リソースを解放
    cap.release()
    out.release()
    
    elapsed_time = time.time() - start_process_time
    print(f"処理完了: {frame_count}フレーム処理, 所要時間: {elapsed_time:.2f}秒")
    print(f"出力動画: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="動画ファイルの一部を切り取ります")
    parser.add_argument("input_video", help="入力動画ファイルのパス")
    parser.add_argument("output_video", help="出力動画ファイルのパス")
    parser.add_argument("--start", type=float, default=0.0, help="切り取り開始時間（秒）、デフォルト: 0.0")
    parser.add_argument("--duration", type=float, default=30.0, help="切り取る長さ（秒）、デフォルト: 30.0")
    
    args = parser.parse_args()
    
    # ファイル名のみからフルパスを構築（相対パスが指定された場合）
    input_path = args.input_video
    if not os.path.isabs(input_path) and not input_path.startswith('./') and not input_path.startswith('../'):
        if os.path.exists(f"data/raw/{input_path}"):
            input_path = f"data/raw/{input_path}"
    
    output_path = args.output_video
    if not os.path.isabs(output_path) and not output_path.startswith('./') and not output_path.startswith('../'):
        # 出力先ディレクトリを明示的に指定されていない場合、clips/ディレクトリに配置
        if not os.path.dirname(output_path):
            output_path = f"data/clips/{output_path}"
    
    print(f"入力ファイル: {input_path}")
    print(f"出力ファイル: {output_path}")
    print(f"切り取り設定: 開始 {args.start}秒から {args.duration}秒間")
    
    clip_video(input_path, output_path, args.start, args.duration)

if __name__ == "__main__":
    main() 