import argparse
import os
import sys

import cv2


def clip_video_opencv(video_path, output_path, start_frame, end_frame):
    """
    OpenCVを使用して、指定されたフレーム番号に基づいて動画を切り取る。
    """
    try:
        # 入力ビデオを開く
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"エラー: 動画ファイルを開けません: {video_path}", file=sys.stderr)
            return

        # 動画のプロパティを取得
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # フレーム番号のバリデーション
        if start_frame >= end_frame or end_frame > total_frames:
            print(
                f"エラー: 無効なフレーム範囲です。開始: {start_frame}, 終了: {end_frame}, 総フレーム: {total_frames}",
                file=sys.stderr,
            )
            cap.release()
            return

        # 出力ビデオライターをセットアップ
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("動画を切り取っています (OpenCV)...")
        print(f"  - 入力ファイル: {os.path.basename(video_path)}")
        print(f"  - 期間: フレーム {start_frame} から {end_frame} まで")

        # 開始フレームまでシーク
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current_frame += 1

        # リソースを解放
        cap.release()
        out.release()

        print("\n成功！ 切り取った動画を以下に保存しました:")
        print(f"  -> {output_path}")

    except Exception as e:
        print(f"動画の切り取り中にエラーが発生しました: {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="指定されたフレーム範囲で動画を切り取ります。")
    parser.add_argument("video_path", type=str, help="入力動画ファイルのパス")
    parser.add_argument("output_path", type=str, help="出力動画ファイルのパス")
    parser.add_argument("start_frame", type=int, help="切り取り開始フレーム番号")
    parser.add_argument("end_frame", type=int, help="切り取り終了フレーム番号")

    args = parser.parse_args()

    clip_video_opencv(args.video_path, args.output_path, args.start_frame, args.end_frame)
