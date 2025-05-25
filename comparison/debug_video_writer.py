#!/usr/bin/env python3
"""
VideoWriterの問題を調査するためのデバッグスクリプト
"""

import os

import cv2
import numpy as np


def test_video_writer():
    """VideoWriterの基本動作をテスト"""

    # テスト用の設定
    output_path = "test_output.mp4"
    width, height = 640, 480
    fps = 30.0

    print("テスト設定:")
    print(f"  出力パス: {output_path}")
    print(f"  解像度: {width}x{height}")
    print(f"  FPS: {fps}")

    # 複数のコーデックをテスト
    codecs = [
        ("avc1", "H.264"),
        ("mp4v", "MPEG-4"),
        ("XVID", "XVID"),
    ]

    for codec, desc in codecs:
        try:
            print(f"\n=== {desc} ({codec}) テスト ===")

            # VideoWriterの作成
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # 初期化確認
            if not writer.isOpened():
                print("❌ VideoWriter初期化失敗")
                writer.release()
                continue

            print("✅ VideoWriter初期化成功")

            # テストフレームの作成と書き込み
            success_count = 0
            total_frames = 10

            for i in range(total_frames):
                # カラフルなテストフレームを作成
                frame = np.zeros((height, width, 3), dtype=np.uint8)

                # フレーム番号に応じて色を変更
                color = (
                    (i * 25) % 255,  # Blue
                    (i * 50) % 255,  # Green
                    (i * 75) % 255,  # Red
                )
                frame[:] = color

                # フレーム番号をテキストで描画
                cv2.putText(
                    frame,
                    f"Frame {i}",
                    (50, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 255, 255),
                    3,
                )

                # フレームサイズの確認
                if frame.shape != (height, width, 3):
                    print(f"❌ フレーム{i}: サイズ不一致 {frame.shape} != ({height}, {width}, 3)")
                    continue

                # 書き込み実行
                success = writer.write(frame)
                if success:
                    success_count += 1
                else:
                    print(f"❌ フレーム{i}: 書き込み失敗")

            # 結果表示
            writer.release()

            print(f"書き込み結果: {success_count}/{total_frames} フレーム成功")

            # ファイル確認
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / 1024  # KB
                print(f"✅ ファイル生成成功: {file_size:.2f}KB")

                # ファイルを再度開いて確認
                cap = cv2.VideoCapture(output_path)
                if cap.isOpened():
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    actual_fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"✅ 再生確認: {frame_count}フレーム, {actual_fps}fps")
                    cap.release()
                else:
                    print("❌ 再生テスト失敗")

                # テストファイル削除
                os.remove(output_path)
            else:
                print("❌ ファイル生成失敗")

        except Exception as e:
            print(f"❌ エラー: {e}")


if __name__ == "__main__":
    test_video_writer()
