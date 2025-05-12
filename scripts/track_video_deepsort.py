#!/usr/bin/env python
import os
import cv2
import time
import argparse
import numpy as np
import csv
import platform
import psutil
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO

# boxmot トラッカーのインポート (必要に応じて)
# from boxmot.trackers.botsort.botsort import BotSort
# from boxmot.trackers.bytetrack.bytetrack import ByteTrack
# from boxmot.trackers.ocsort.ocsort import OcSort
# from boxmot.trackers.strongsort.strongsort import StrongSort
# from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
# from boxmot.trackers.boosttrack.boosttrack import BoostTrack
# from boxmot.trackers.hybridsort.hybridsort import HybridSORT
from deep_sort_realtime.deepsort_tracker import DeepSort
# from boxmot.utils import ROOT, WEIGHTS # BoxMOTを使わない場合は不要


def get_system_info():
    """システム情報を取得する関数"""
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu": platform.processor(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_total": round(psutil.virtual_memory().total / (1024**3), 2),  # GB単位
    }
    return info


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

    # トラッカーの設定 (DeepSORTのみ)
    if args.tracker == "deepsort":
        tracker = DeepSort(
            max_age=30,  # トラックが維持される最大フレーム数
            n_init=3,  # トラックが確定されるまでの最小ヒット数
            nms_max_overlap=1.0,  # NMSの重複閾値
            max_cosine_distance=0.2,  # 見た目の特徴量の最大コサイン距離
            nn_budget=None,  # 特徴量バジェット (Noneで無制限)
            # embedder='mobilenet', # 使用するReIDモデル (デフォルト)
            # half=args.half,       # 半精度 (DeepSORTでは設定不可の場合あり)
            bgr=True,  # 入力画像がBGR形式か
        )
    else:
        # このスクリプトはDeepSORT専用とするため、他のトラッカーはエラーとする
        raise ValueError(
            f"このスクリプトはDeepSORT専用です。指定されたトラッカー {args.tracker} はサポートされていません。"
        )

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
            if hasattr(results[0], "speed"):  # YOLOv8 8.0.196以降
                speed_info = results[0].speed
                yolo_preprocess = speed_info.get("preprocess", 0)
                yolo_inference = speed_info.get("inference", 0)
                yolo_postprocess = speed_info.get("postprocess", 0)
            elif hasattr(results[0], "_speed"):  # 古いバージョン
                speed_info = results[0]._speed
                yolo_preprocess = speed_info.get("preprocess", 0)
                yolo_inference = speed_info.get("inference", 0)
                yolo_postprocess = speed_info.get("postprocess", 0)

            # 検出結果をDeepSORT用の形式に変換
            boxes = results[0].boxes
            detections_for_deepsort = []
            objects_detected_count = 0

            if len(boxes) > 0:
                objects_detected_count = len(boxes)
                for i in range(len(boxes)):
                    box = boxes[i].xyxy.cpu().numpy()[0]  # [x1, y1, x2, y2]
                    conf = float(boxes[i].conf.cpu().numpy()[0])
                    cls = int(boxes[i].cls.cpu().numpy()[0])

                    # DeepSORT形式: ([left, top, width, height], confidence, class_id)
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    # クラスIDがargs.classesに含まれるものだけをトラッカーに渡す
                    if cls in args.classes:
                        detections_for_deepsort.append(
                            ([x1, y1, w, h], conf, str(cls))
                        )  # class_idは文字列にする必要あり

            # トラッカーの更新 - 時間計測
            tracking_start = time.time()
            tracks = tracker.update_tracks(detections_for_deepsort, frame=frame)
            tracking_end = time.time()
            tracking_time = (tracking_end - tracking_start) * 1000  # ミリ秒に変換
            tracking_times.append(tracking_time)

            # 合計処理時間を計算
            total_time = detection_time + tracking_time

            # ログ出力（10フレームごと）
            if frame_count % 10 == 0 or frame_count == 1:
                print(
                    f"Frame {frame_count}: Detection time {detection_time:.1f}ms, Tracking time {tracking_time:.1f}ms"
                )

            # ログをCSVに記録
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            # CSVに記録
            objects_tracked_count = len([t for t in tracks if t.is_confirmed()])
            # DeepSORTオブジェクトから use_cuda 属性を確認することはできないため、args.deviceに基づいて判断
            cuda_used_str = "Yes" if args.device and args.device != "cpu" else "No"
            log_writer.writerow(
                [
                    frame_count,  # フレーム番号
                    f"{elapsed_time:.3f}",  # 経過時間（秒）
                    f"{detection_time:.1f}",  # 検出時間（ms）
                    f"{tracking_time:.1f}",  # 追跡時間（ms）
                    f"{total_time:.1f}",  # 合計処理時間（ms）
                    objects_detected_count,  # 検出されたオブジェクト数
                    objects_tracked_count,  # 追跡されたオブジェクト数
                    f"{current_fps:.1f}",  # 現在のFPS
                    f"{yolo_preprocess:.1f}",  # YOLO前処理時間
                    f"{yolo_inference:.1f}",  # YOLO推論時間
                    f"{yolo_postprocess:.1f}",  # YOLO後処理時間
                    cuda_used_str,  # CUDAの使用 (引数に基づく)
                    Path(args.yolo_model).stem,  # モデル名
                    args.tracker,  # トラッカー名
                    "",  # メモ欄
                ]
            )

            # 検出結果とトラッキング結果の描画
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, ltrb)

                # 元の検出からクラスIDを取得（DeepSORTは直接クラスを返さないため）
                # この実装では、単純にargs.classesの最初のクラスを使う
                # より正確にするには、検出結果とのマッチングが必要
                cls_id = args.classes[0]
                label = f"ID: {track_id}, {model.names[cls_id]}"

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
                cv2.imshow("Tracking (DeepSORT)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    # 平均処理時間の計算
    avg_detection_time = sum(detection_times) / len(detection_times) if detection_times else 0
    avg_tracking_time = sum(tracking_times) / len(tracking_times) if tracking_times else 0
    total_elapsed_time = time.time() - start_time  # 最終的な経過時間

    print(f"\nTracking completed: {output_path}")
    print(f"Total processing time: {total_elapsed_time:.1f} seconds")
    print(f"Total frames: {frame_count}")
    print(f"Average FPS: {frame_count / total_elapsed_time:.1f}")
    print(f"Average detection time: {avg_detection_time:.1f}ms")
    print(f"Average tracking time: {avg_tracking_time:.1f}ms")
    print(f"Average total processing time: {avg_detection_time + avg_tracking_time:.1f}ms")
    print(f"Log file saved to: {log_file}")

    # リソースの解放
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track objects in a video using YOLO and DeepSORT."
    )
    parser.add_argument("--source", type=str, required=True, help="動画ファイルパス")
    parser.add_argument(
        "--yolo-model",
        type=str,
        required=True,
        help="YOLOモデルパス (例: yolov8n.pt, yolo11n-pose.pt)",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="deepsort",
        choices=["deepsort"],
        help="使用するトラッカー (このスクリプトではdeepsortのみ)",
    )
    parser.add_argument("--device", type=str, default="", help="使用するデバイス (例: cpu, 0)")
    parser.add_argument(
        "--classes", type=int, nargs="+", default=[0], help="検出・追跡するクラスID (例: 0 は人物)"
    )
    parser.add_argument("--conf", type=float, default=0.3, help="検出信頼度閾値")
    parser.add_argument(
        "--half",
        action="store_true",
        help="YOLOで半精度浮動小数点数を使用 (DeepSORT自体には影響しない場合あり)",
    )
    parser.add_argument("--show", action="store_true", help="プレビュー表示")

    args = parser.parse_args()

    # tracker引数がdeepsortであることを強制
    if args.tracker != "deepsort":
        print("エラー: このスクリプトは --tracker deepsort 専用です。")
        exit()

    main(args)
