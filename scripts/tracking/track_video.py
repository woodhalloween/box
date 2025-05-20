#!/usr/bin/env python
import argparse
import csv
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import psutil
from boxmot.trackers.boosttrack.boosttrack import BoostTrack
from boxmot.trackers.botsort.botsort import BotSort
from boxmot.trackers.bytetrack.bytetrack import ByteTrack
from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
from boxmot.trackers.ocsort.ocsort import OcSort
from boxmot.trackers.strongsort.strongsort import StrongSort
from boxmot.utils import WEIGHTS
from ultralytics import YOLO


def get_system_info():
    """システム情報を取得する関数"""
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu": platform.processor(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_total": round(psutil.virtual_memory().total / (1024**3), 2),  # GB単位
    }


def main(args):
    # 動画パス
    video_path = args.source

    # 検出モデルの設定
    model = YOLO(args.yolo_model)

    # モデル情報の取得
    model_info = {
        "name": Path(args.yolo_model).stem,
        "file_size_mb": round(os.path.getsize(args.yolo_model) / (1024**2), 2)
        if os.path.exists(args.yolo_model)
        else None,
        "task": model.task if hasattr(model, "task") else "unknown",
        "version": getattr(model, "version", "unknown") if hasattr(model, "version") else "unknown",
    }

    # トラッカーの設定
    if args.tracker == "strongsort":
        tracker = StrongSort(
            reid_weights=Path(WEIGHTS / "osnet_x0_25_msmt17.pt"), device=args.device, half=args.half
        )
    elif args.tracker == "bytetrack":
        tracker = ByteTrack(track_thresh=args.conf, track_buffer=30, match_thresh=0.8)
    elif args.tracker == "botsort":
        tracker = BotSort(
            reid_weights=Path(WEIGHTS / "osnet_x0_25_msmt17.pt"), device=args.device, half=args.half
        )
    elif args.tracker == "ocsort":
        tracker = OcSort(det_thresh=args.conf, iou_threshold=0.3, use_byte=False)
    elif args.tracker == "deepocsort":
        tracker = DeepOcSort(
            model_weights=Path(WEIGHTS / "osnet_x0_25_msmt17.pt"),
            device=args.device,
            fp16=args.half,
        )
    elif args.tracker == "boosttrack":
        tracker = BoostTrack(
            reid_weights=Path(WEIGHTS / "osnet_x0_25_msmt17.pt"),
            device=args.device,
            half=args.half,
            max_age=60,
            min_hits=3,
            det_thresh=args.conf,
            iou_threshold=0.3,
        )
    else:
        raise ValueError(f"トラッカー {args.tracker} は対応していません。")

    # 動画の読み込み
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 出力ビデオライターの設定
    output_path = f"tracked_{Path(video_path).stem}_{args.tracker}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # システム情報の取得
    system_info = get_system_info()

    # ログファイルの設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = (
        f"log_{Path(video_path).stem}_{args.tracker}_{Path(args.yolo_model).stem}_{timestamp}.csv"
    )
    with open(log_file, "w", newline="") as csvfile:
        log_writer = csv.writer(csvfile)

        # システム情報のヘッダー行
        log_writer.writerow(["# System Information"])
        log_writer.writerow(["# OS", system_info["os"], system_info["os_version"]])
        log_writer.writerow(["# Python", system_info["python_version"]])
        log_writer.writerow(
            [
                "# CPU",
                system_info["cpu"],
                f"{system_info['cpu_cores']} cores, {system_info['cpu_threads']} threads",
            ]
        )
        log_writer.writerow(["# RAM", f"{system_info['ram_total']} GB"])
        log_writer.writerow([])

        # モデル情報のヘッダー行
        log_writer.writerow(
            ["# YOLO Model", args.yolo_model, "Size", f"{model_info['file_size_mb']} MB"]
        )
        log_writer.writerow(["# Model Task", model_info["task"], "Version", model_info["version"]])
        log_writer.writerow(["# Tracker", args.tracker])
        log_writer.writerow(
            [
                "# Device",
                args.device if args.device else "auto",
                "Half Precision",
                str(args.half),
                "Confidence Threshold",
                str(args.conf),
            ]
        )
        log_writer.writerow([])

        # 動画情報
        log_writer.writerow(["# Video", video_path])
        log_writer.writerow(
            ["# Resolution", f"{width}x{height}", "FPS", fps, "Total Frames", total_frames]
        )
        log_writer.writerow([])

        # データ列のヘッダー行
        log_writer.writerow(
            [
                "Frame",
                "Time",
                "Detection_Time_ms",
                "Tracking_Time_ms",
                "Total_Time_ms",
                "Objects_Detected",
                "Objects_Tracked",
                "FPS",
                "YOLO_Preprocess_ms",
                "YOLO_Inference_ms",
                "YOLO_Postprocess_ms",
                "CUDA_Used",
                "Model",
                "Tracker",
                "Notes",
            ]
        )

        frame_count = 0
        start_time = time.time()

        # 処理時間計測用
        detection_times = []
        tracking_times = []

        # トラッキング処理
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # YOLOでの検出 - 時間計測
            detection_start = time.time()
            results = model.predict(frame, classes=args.classes, conf=args.conf)
            detection_end = time.time()
            detection_time = (detection_end - detection_start) * 1000  # ミリ秒に変換
            detection_times.append(detection_time)

            # YOLOの内部処理時間取得
            yolo_preprocess = 0
            yolo_inference = 0
            yolo_postprocess = 0

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

            # YOLOのSpeed行からpreprocess, inference, postprocessの時間を抽出
            if hasattr(results[0], "_speed"):
                speed_info = results[0]._speed
                if "preprocess" in speed_info:
                    yolo_preprocess = speed_info["preprocess"]
                if "inference" in speed_info:
                    yolo_inference = speed_info["inference"]
                if "postprocess" in speed_info:
                    yolo_postprocess = speed_info["postprocess"]

            # トラッカーの更新 - 時間計測
            tracking_start = time.time()
            tracks = tracker.update(dets_for_tracker, frame)
            tracking_end = time.time()
            tracking_time = (tracking_end - tracking_start) * 1000  # ミリ秒に変換
            tracking_times.append(tracking_time)

            # 合計処理時間を計算
            total_time = detection_time + tracking_time

            # ログ出力（10フレームごと）
            if frame_count % 10 == 0 or frame_count == 1:
                print(
                    f"Frame {frame_count}: Detection time {detection_time:.1f}ms, "
                    f"Tracking time {tracking_time:.1f}ms"
                )

            # ログをCSVに記録
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            # CSVに記録
            log_writer.writerow(
                [
                    frame_count,  # フレーム番号
                    f"{elapsed_time:.3f}",  # 経過時間（秒）
                    f"{detection_time:.1f}",  # 検出時間（ms）
                    f"{tracking_time:.1f}",  # 追跡時間（ms）
                    f"{total_time:.1f}",  # 合計処理時間（ms）
                    len(boxes),  # 検出されたオブジェクト数
                    len(tracks),  # 追跡されたオブジェクト数
                    f"{current_fps:.1f}",  # 現在のFPS
                    f"{yolo_preprocess:.1f}",  # YOLO前処理時間
                    f"{yolo_inference:.1f}",  # YOLO推論時間
                    f"{yolo_postprocess:.1f}",  # YOLO後処理時間
                    "No" if args.device == "cpu" else "Yes",  # CUDAの使用
                    Path(args.yolo_model).stem,  # モデル名
                    args.tracker,  # トラッカー名
                    "",  # メモ欄
                ]
            )

            # 検出結果とトラッキング結果の描画
            for d in tracks:
                # トラックの形式: [x1, y1, x2, y2, track_id, conf, cls_id, ...]
                x1, y1, x2, y2, track_id = d[:5]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # トラッキングIDとクラスラベルの表示
                # クラスIDは6列目（インデックス5）にある場合が多いが、
                # トラッカーによって異なる可能性がある
                cls_id = 0  # デフォルトは人（クラス0）
                if d.shape[0] > 6:
                    cls_id = int(d[6])

                label = f"ID: {int(track_id)}, {model.names[cls_id]}"

                # 枠の描画
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # ラベルの描画
                cv2.putText(
                    frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

            # 処理中の情報表示
            elapsed_time = time.time() - start_time
            fps_text = f"FPS: {frame_count / elapsed_time:.1f}"
            detect_text = f"Detection: {detection_time:.1f}ms"
            track_text = f"Tracking: {tracking_time:.1f}ms"

            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, detect_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, track_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 結果の書き込み
            writer.write(frame)

            # プレビュー（必要に応じて）
            if args.show:
                cv2.imshow("Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    # 平均処理時間の計算
    avg_detection_time = sum(detection_times) / len(detection_times) if detection_times else 0
    avg_tracking_time = sum(tracking_times) / len(tracking_times) if tracking_times else 0

    print(f"\nTracking completed: {output_path}")
    print(f"Total processing time: {elapsed_time:.1f} seconds")
    print(f"Total frames: {frame_count}")
    print(f"Average FPS: {frame_count / elapsed_time:.1f}")
    print(f"Average detection time: {avg_detection_time:.1f}ms")
    print(f"Average tracking time: {avg_tracking_time:.1f}ms")
    print(f"Average total processing time: {avg_detection_time + avg_tracking_time:.1f}ms")
    print(f"Log file saved to: {log_file}")

    # リソースの解放
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=str, default="line_fortuna_demo_multipersons.mp4", help="動画ファイルパス"
    )
    parser.add_argument("--yolo-model", type=str, default="yolo11n.pt", help="YOLOモデルパス")
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack",
        choices=["strongsort", "bytetrack", "botsort", "ocsort", "deepocsort", "boosttrack"],
        help="トラッカーの種類",
    )
    parser.add_argument("--device", type=str, default="", help="使用するデバイス (例: cpu, 0)")
    parser.add_argument("--classes", type=int, nargs="+", default=0, help="検出するクラス（0:人）")
    parser.add_argument("--conf", type=float, default=0.3, help="検出信頼度閾値")
    parser.add_argument("--half", action="store_true", help="半精度浮動小数点数を使用")
    parser.add_argument("--show", action="store_true", help="プレビュー表示")

    args = parser.parse_args()

    main(args)
