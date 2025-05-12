#!/usr/bin/env python
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot.trackers.bytetrack.bytetrack import ByteTrack

# 動画ファイルをロード
video_path = "line_fortuna_demo_multipersons.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

# YOLOモデルをロード
model = YOLO("yolov8n.pt")

# ByteTrackトラッカーを初期化
tracker = ByteTrack(track_thresh=0.3, track_buffer=30, match_thresh=0.8)

# 検出実行
results = model.predict(frame, classes=0, conf=0.3)

# 検出結果をboxmot用の形式に変換
boxes = results[0].boxes
dets_for_tracker = []

if len(boxes) > 0:
    for i in range(len(boxes)):
        box = boxes[i].xyxy.cpu().numpy()[0]  # [x1, y1, x2, y2]
        conf = float(boxes[i].conf.cpu().numpy()[0])
        cls = int(boxes[i].cls.cpu().numpy()[0])

        # [x1, y1, x2, y2, conf, class]の形式
        dets_for_tracker.append([box[0], box[1], box[2], box[3], conf, cls])

    dets_for_tracker = np.array(dets_for_tracker)
else:
    dets_for_tracker = np.empty((0, 6))

# トラッキング実行
tracks = tracker.update(dets_for_tracker, frame)

# トラッキング結果の形式を確認
print(f"検出数: {len(dets_for_tracker)}")
print(f"トラック数: {len(tracks)}")

if len(tracks) > 0:
    print(f"トラック形式: {tracks.shape}")
    print(f"トラックの例: {tracks[0]}")
    print(
        f"トラックの各列の意味:\n"
        f"0-3: x1, y1, x2, y2 (バウンディングボックス座標)\n"
        f"4: track_id (トラッキングID)\n"
        f"残りの列: {tracks.shape[1] - 5}列"
    )

cap.release()
