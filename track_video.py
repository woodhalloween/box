#!/usr/bin/env python
import os
import cv2
import time
import argparse
import numpy as np
from pathlib import Path

from ultralytics import YOLO
from boxmot.trackers.botsort.botsort import BotSort
from boxmot.trackers.bytetrack.bytetrack import ByteTrack
from boxmot.trackers.ocsort.ocsort import OcSort
from boxmot.trackers.strongsort.strongsort import StrongSort
from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
from boxmot.utils import ROOT, WEIGHTS

def main(args):
    # 動画パス
    video_path = args.source
    
    # 検出モデルの設定
    model = YOLO(args.yolo_model)
    
    # トラッカーの設定
    if args.tracker == 'strongsort':
        tracker = StrongSort(
            reid_weights=Path(WEIGHTS / 'osnet_x0_25_msmt17.pt'),
            device=args.device,
            half=args.half
        )
    elif args.tracker == 'bytetrack':
        tracker = ByteTrack(
            track_thresh=args.conf,
            track_buffer=30,
            match_thresh=0.8
        )
    elif args.tracker == 'botsort':
        tracker = BotSort(
            reid_weights=Path(WEIGHTS / 'osnet_x0_25_msmt17.pt'),
            device=args.device,
            half=args.half
        )
    elif args.tracker == 'ocsort':
        tracker = OcSort(
            det_thresh=args.conf,
            iou_threshold=0.3,
            use_byte=False
        )
    elif args.tracker == 'deepocsort':
        tracker = DeepOcSort(
            model_weights=Path(WEIGHTS / 'osnet_x0_25_msmt17.pt'),
            device=args.device,
            fp16=args.half
        )
    else:
        raise ValueError(f'トラッカー {args.tracker} は対応していません。')
    
    # 動画の読み込み
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 出力ビデオライターの設定
    output_path = f'tracked_{Path(video_path).stem}_{args.tracker}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    # トラッキング処理
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # YOLOでの検出
        results = model.predict(frame, classes=args.classes, conf=args.conf)
        
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
        
        # トラッカーの更新
        tracks = tracker.update(dets_for_tracker, frame)
        
        # 検出結果とトラッキング結果の描画
        for d in tracks:
            # トラックの形式: [x1, y1, x2, y2, track_id, conf, cls_id, ...]
            x1, y1, x2, y2, track_id = d[:5]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # トラッキングIDとクラスラベルの表示
            # クラスIDは6列目（インデックス5）にある場合が多いが、トラッカーによって異なる可能性がある
            cls_id = 0  # デフォルトは人（クラス0）
            if d.shape[0] > 6:
                cls_id = int(d[6])
            
            label = f'ID: {int(track_id)}, {model.names[cls_id]}'
            
            # 枠の描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # ラベルの描画
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 処理中の情報表示
        elapsed_time = time.time() - start_time
        fps_text = f'FPS: {frame_count / elapsed_time:.1f}'
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 結果の書き込み
        writer.write(frame)
        
        # プレビュー（必要に応じて）
        if args.show:
            cv2.imshow('Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # リソースの解放
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    print(f'\nトラッキング完了: {output_path}')
    print(f'処理時間: {elapsed_time:.1f}秒')
    print(f'総フレーム数: {frame_count}')
    print(f'平均FPS: {frame_count / elapsed_time:.1f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='line_fortuna_demo_multipersons.mp4', help='動画ファイルパス')
    parser.add_argument('--yolo-model', type=str, default='yolov8n.pt', help='YOLOモデルパス')
    parser.add_argument('--tracker', type=str, default='bytetrack', choices=['strongsort', 'bytetrack', 'botsort', 'ocsort', 'deepocsort'], help='トラッカーの種類')
    parser.add_argument('--device', type=str, default='', help='使用するデバイス (例: cpu, 0)')
    parser.add_argument('--classes', type=int, nargs='+', default=0, help='検出するクラス（0:人）')
    parser.add_argument('--conf', type=float, default=0.3, help='検出信頼度閾値')
    parser.add_argument('--half', action='store_true', help='半精度浮動小数点数を使用')
    parser.add_argument('--show', action='store_true', help='プレビュー表示')
    
    args = parser.parse_args()
    
    main(args) 