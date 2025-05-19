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
from boxmot.trackers.bytetrack.bytetrack import ByteTrack
from ultralytics import YOLO


# データ保存用の構造体
class DetectionResults:
    def __init__(self):
        self.frame_count = 0
        self.detection_times = []  # 検出時間（ms）
        self.tracking_times = []  # 追跡時間（ms）
        self.fps_values = []  # FPS値
        self.objects_detected = []  # 検出されたオブジェクト数
        self.objects_tracked = []  # 追跡されたオブジェクト数
        self.memory_usages = []  # メモリ使用量（MB）
        self.detection_confidence = []  # 検出信頼度の平均

    def add_frame_result(
        self,
        detection_time,
        tracking_time,
        fps,
        objects_detected,
        objects_tracked,
        memory_usage,
        detection_conf=None,
    ):
        self.frame_count += 1
        self.detection_times.append(detection_time)
        self.tracking_times.append(tracking_time)
        self.fps_values.append(fps)
        self.objects_detected.append(objects_detected)
        self.objects_tracked.append(objects_tracked)
        self.memory_usages.append(memory_usage)
        if detection_conf is not None:
            self.detection_confidence.append(detection_conf)

    def get_summary(self):
        """結果のサマリを辞書で返す"""
        return {
            "frame_count": self.frame_count,
            "avg_detection_time": np.mean(self.detection_times) if self.detection_times else 0,
            "avg_tracking_time": np.mean(self.tracking_times) if self.tracking_times else 0,
            "avg_fps": np.mean(self.fps_values) if self.fps_values else 0,
            "avg_objects_detected": np.mean(self.objects_detected) if self.objects_detected else 0,
            "avg_objects_tracked": np.mean(self.objects_tracked) if self.objects_tracked else 0,
            "avg_memory_usage": np.mean(self.memory_usages) if self.memory_usages else 0,
            "avg_detection_conf": np.mean(self.detection_confidence)
            if self.detection_confidence
            else 0,
            "max_objects_detected": max(self.objects_detected) if self.objects_detected else 0,
            "max_objects_tracked": max(self.objects_tracked) if self.objects_tracked else 0,
        }


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


def initialize_log_file(log_file_path, input_file, model_path, resolutions):
    """結果ログファイルの初期化"""
    system_info = get_system_info()

    with open(log_file_path, "w", newline="") as log_f:
        log_writer = csv.writer(log_f)
        log_writer.writerow(["# 解像度別YOLOv11+ByteTrack精度比較"])
        log_writer.writerow(["# 実行日時", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        log_writer.writerow(["# システム情報"])
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
        log_writer.writerow(["# 入力動画", input_file])
        log_writer.writerow(["# YOLOモデル", model_path])
        log_writer.writerow(["# 解像度比較", ", ".join([f"{w}x{h}" for w, h in resolutions])])
        log_writer.writerow([])

        # ヘッダー行
        log_writer.writerow(
            [
                "解像度",
                "フレーム数",
                "平均検出時間(ms)",
                "平均追跡時間(ms)",
                "平均FPS",
                "平均検出オブジェクト数",
                "平均追跡オブジェクト数",
                "最大検出オブジェクト数",
                "最大追跡オブジェクト数",
                "平均検出信頼度",
                "平均メモリ使用量(MB)",
            ]
        )

    # ファイルハンドルとライターを返さないように修正
    return None, None


def process_frame_for_tracking(frame_rgb, model, tracker):
    """1フレームを処理して検出と追跡を行う"""
    detection_start_time = time.time()
    # 人物クラス(0)のみを検出
    results = model.predict(frame_rgb, verbose=False, classes=[0])
    detection_time_ms = (time.time() - detection_start_time) * 1000
    result = results[0]

    boxes = result.boxes
    # 検出結果をboxmot用の形式に変換
    dets_for_tracker = []
    avg_conf = 0.0

    if len(boxes) > 0:
        total_conf = 0.0
        for i in range(len(boxes)):
            box = boxes[i].xyxy.cpu().numpy()[0]  # [x1, y1, x2, y2]
            conf = float(boxes[i].conf.cpu().numpy()[0])
            cls = int(boxes[i].cls.cpu().numpy()[0])
            total_conf += conf

            # [x1, y1, x2, y2, conf, class]の形式
            dets_for_tracker.append([box[0], box[1], box[2], box[3], conf, cls])

        dets_for_tracker = np.array(dets_for_tracker)
        avg_conf = total_conf / len(boxes) if len(boxes) > 0 else 0
    else:
        dets_for_tracker = np.empty((0, 6))

    tracking_start_time = time.time()
    # ByteTrackのupdateメソッドにnumpy配列を渡す
    tracks = tracker.update(dets_for_tracker, frame_rgb)
    tracking_time_ms = (time.time() - tracking_start_time) * 1000

    return tracks, detection_time_ms, tracking_time_ms, len(boxes), len(tracks), avg_conf


def resize_frame(frame, target_width, target_height):
    """フレームを指定サイズにリサイズ"""
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def draw_tracking_info(frame, tracks):
    """トラッキング情報を描画"""
    for track in tracks:
        # トラックの形式: [x1, y1, x2, y2, track_id, conf, cls_id, ...]
        x1, y1, x2, y2, track_id = track[:5]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        track_id = int(track_id)

        # バウンディングボックスの描画
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # IDの表示
        label = f"ID: {track_id}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


def process_video(
    input_file,
    output_file,
    model,
    tracker,
    target_width,
    target_height,
    original_width,
    original_height,
    enable_preview=False,
):
    """動画を処理し結果を取得"""
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"エラー: 入力動画を開けません: {input_file}")
        return None

    # 入力動画の情報
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 出力ファイルの準備
    if output_file:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Mac互換コーデック
        out = cv2.VideoWriter(output_file, fourcc, fps, (target_width, target_height))
    else:
        out = None

    # 結果保存用のオブジェクト
    results = DetectionResults()
    frame_idx = 0
    loop_start_time = time.time()

    try:
        while True:
            frame_start_time = time.time()
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_idx += 1

            # フレームのリサイズ（必要な場合）
            if target_width != original_width or target_height != original_height:
                frame_bgr = resize_frame(frame_bgr, target_width, target_height)

            # 処理用にRGBに変換
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # 検出と追跡の実行
            (
                tracks,
                detection_time_ms,
                tracking_time_ms,
                objects_detected,
                objects_tracked,
                avg_conf,
            ) = process_frame_for_tracking(frame_rgb, model, tracker)

            # 追跡情報の描画
            if out or enable_preview:
                frame_bgr = draw_tracking_info(frame_bgr, tracks)

            # 出力動画に書き込み
            if out:
                out.write(frame_bgr)

            # プレビュー表示
            if enable_preview:
                window_name = f"Preview {target_width}x{target_height}"
                cv2.imshow(window_name, frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # 現在のメモリ使用量を測定（MB単位）
            current_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)

            # 経過時間とFPSの計算
            elapsed_time = time.time() - loop_start_time
            current_fps = frame_idx / elapsed_time if elapsed_time > 0 else 0

            # 結果の保存
            results.add_frame_result(
                detection_time_ms,
                tracking_time_ms,
                current_fps,
                objects_detected,
                objects_tracked,
                current_memory_mb,
                avg_conf,
            )

            # 進捗表示
            if frame_idx % 30 == 0 or frame_idx == 1:
                print(
                    f"処理中 ({target_width}x{target_height}): {frame_idx}フレーム "
                    f"(FPS: {current_fps:.1f}, メモリ: {current_memory_mb:.1f}MB)"
                )

    except KeyboardInterrupt:
        print(f"処理が中断されました ({target_width}x{target_height})")
    finally:
        cap.release()
        if out:
            out.release()

        # 結果のサマリを表示
        summary = results.get_summary()
        print(f"\n--- 処理結果サマリ ({target_width}x{target_height}) ---")
        print(f"合計フレーム数: {summary['frame_count']}")
        print(f"平均FPS: {summary['avg_fps']:.2f}")
        print(f"平均検出時間: {summary['avg_detection_time']:.1f} ms")
        print(f"平均追跡時間: {summary['avg_tracking_time']:.1f} ms")
        print(f"平均検出オブジェクト数: {summary['avg_objects_detected']:.1f}")
        print(f"平均追跡オブジェクト数: {summary['avg_objects_tracked']:.1f}")
        print(f"平均検出信頼度: {summary['avg_detection_conf']:.3f}")

    return results


def compare_resolutions(
    input_file, output_dir, model_path="yolov11n.pt", enable_preview=False, device=""
):
    """複数の解像度で比較実行"""
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    # ログファイルの準備
    log_file_path = os.path.join(
        output_dir,
        f"comparison_results_{Path(input_file).stem}_{Path(model_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )

    # 入力動画の情報取得
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"エラー: 入力動画を開けません: {input_file}")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # 使用する解像度のリスト (元のサイズと半分のサイズ)
    resolutions = [
        (original_width, original_height),  # 元のサイズ
        (original_width // 2, original_height // 2),  # 半分のサイズ
    ]

    # ログファイルの初期化
    initialize_log_file(log_file_path, input_file, model_path, resolutions)

    # モデルのロード
    print(f"YOLOモデルのロード中: {model_path}")
    try:
        if device.isdigit():
            yolo_device = int(device)
        elif device.lower() == "cpu":
            yolo_device = "cpu"
        else:
            yolo_device = None  # 自動選択
    except ValueError:
        print(f"警告: 無効なデバイス指定 '{device}'。自動選択を使用します。")
        yolo_device = None

    model = YOLO(model_path).to(yolo_device)

    # 各解像度で処理を実行
    all_results = {}
    for width, height in resolutions:
        resolution_str = f"{width}x{height}"
        print(f"\n処理開始: 解像度 {resolution_str} ({resolution_str})")

        # 出力ファイル名
        output_file = os.path.join(
            output_dir, f"{Path(input_file).stem}_{resolution_str}_{Path(model_path).stem}.mp4"
        )

        # ByteTrackトラッカーの初期化 (各解像度で新しいインスタンスを使用)
        tracker = ByteTrack(track_thresh=0.3, track_buffer=30, match_thresh=0.8)

        # 動画処理
        results = process_video(
            input_file,
            output_file,
            model,
            tracker,
            width,
            height,
            original_width,
            original_height,
            enable_preview,
        )

        if results:
            all_results[resolution_str] = results.get_summary()

            # 結果をログに記録 - 修正: 毎回ファイルを開いて追記する
            summary = results.get_summary()
            with open(log_file_path, "a", newline="") as log_f:
                log_writer = csv.writer(log_f)
                log_writer.writerow(
                    [
                        resolution_str,
                        summary["frame_count"],
                        f"{summary['avg_detection_time']:.1f}",
                        f"{summary['avg_tracking_time']:.1f}",
                        f"{summary['avg_fps']:.1f}",
                        f"{summary['avg_objects_detected']:.1f}",
                        f"{summary['avg_objects_tracked']:.1f}",
                        summary["max_objects_detected"],
                        summary["max_objects_tracked"],
                        f"{summary['avg_detection_conf']:.3f}",
                        f"{summary['avg_memory_usage']:.1f}",
                    ]
                )

    # ウィンドウのクリーンアップ
    if enable_preview:
        cv2.destroyAllWindows()

    # 比較結果の表示
    print("\n=== 解像度別比較結果 ===")
    print("解像度\t検出時間(ms)\t追跡時間(ms)\tFPS\t検出数\t追跡数\t信頼度")
    for res, summary in all_results.items():
        print(
            f"{res}\t{summary['avg_detection_time']:.1f}\t{summary['avg_tracking_time']:.1f}\t"
            f"{summary['avg_fps']:.1f}\t{summary['avg_objects_detected']:.1f}\t"
            f"{summary['avg_objects_tracked']:.1f}\t{summary['avg_detection_conf']:.3f}"
        )

    print(f"\n比較結果は '{log_file_path}' に保存されました。")


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv11とByteTrackによる異なる解像度での検出・追跡精度の比較"
    )
    parser.add_argument("--input", required=True, help="入力動画ファイルのパス")
    parser.add_argument("--output-dir", required=True, help="出力ディレクトリ")
    parser.add_argument(
        "--model", default="yolov11n.pt", help="YOLOモデルのパス (デフォルト: yolov11n.pt)"
    )
    parser.add_argument("--enable-preview", action="store_true", help="処理中のプレビューを表示")
    parser.add_argument("--device", type=str, default="", help="使用するデバイス (例: cpu, 0)")

    args = parser.parse_args()

    # 入力ファイルの確認
    if not os.path.exists(args.input):
        print(f"エラー: 入力ファイル '{args.input}' が見つかりません。")
        return

    # モデルファイルの確認
    if not os.path.exists(args.model):
        print(f"エラー: モデルファイル '{args.model}' が見つかりません。")
        return

    # 解像度比較の実行
    compare_resolutions(args.input, args.output_dir, args.model, args.enable_preview, args.device)


if __name__ == "__main__":
    main()
