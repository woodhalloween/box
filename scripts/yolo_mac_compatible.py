#!/usr/bin/env python3
"""
Mac互換のYOLOポーズ推定スクリプト
macOSで再生可能なMP4ファイルを出力します
"""

import cv2
import numpy as np
import time
import os
from ultralytics import YOLO

# YOLOポーズ推定の骨格ラインの定義
SKELETON = [
    (0, 1),
    (1, 3),
    (0, 2),
    (1, 2),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def process_video(input_video, output_video):
    """Mac互換形式でポーズ推定結果の動画を生成"""
    # 入力動画の読み込み
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"エラー: 動画 '{input_video}' を開けませんでした")
        return

    # 動画のメタデータ取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    # 出力用VideoWriterの作成（Mac互換のコーデックを使用）
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264コーデック
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # YOLOモデルのロード - 比較的小さいモデルを使用
    model = YOLO("yolov8n-pose.pt")

    frame_count = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # YOLOポーズ推定実行
            results = model.predict(frame, verbose=False)
            result = results[0]

            # 結果を描画
            if result.keypoints is not None:
                keypoints = result.keypoints.data

                # 検出された各人物に対する処理
                for kp_points in keypoints:
                    # 骨格ラインの描画
                    for p1_idx, p2_idx in SKELETON:
                        if p1_idx < len(kp_points) and p2_idx < len(kp_points):
                            p1 = kp_points[p1_idx]
                            p2 = kp_points[p2_idx]

                            # 信頼度が十分な場合のみ描画
                            if p1[2] > 0.5 and p2[2] > 0.5:
                                cv2.line(
                                    frame,
                                    (int(p1[0]), int(p1[1])),
                                    (int(p2[0]), int(p2[1])),
                                    (0, 0, 255),
                                    2,
                                )

                    # 関節点の描画
                    for point in kp_points:
                        x, y, conf = point
                        if conf > 0.5:  # 信頼度の閾値
                            cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)

            # 処理情報を表示
            elapsed_time = time.time() - start_time
            fps_text = f"FPS: {frame_count / elapsed_time:.1f}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 出力動画に書き込み
            out.write(frame)

            # 進捗表示
            if frame_count % 30 == 0:
                print(f"処理中: {frame_count}フレーム完了")

    except KeyboardInterrupt:
        print("処理が中断されました")
    finally:
        # リソース解放
        cap.release()
        out.release()

    total_time = time.time() - start_time
    print(f"処理完了: {frame_count}フレーム ({total_time:.2f}秒)")
    print(f"平均FPS: {frame_count / total_time:.2f}")
    print(f"出力動画: {output_video}")


if __name__ == "__main__":
    input_video = "data/videos/WIN_20250319_10_03_53_Pro.mp4"
    output_video = "data/output/yolo_pose_mac_compatible.mp4"

    process_video(input_video, output_video)
