import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import psutil

# 共通ユーティリティをインポート
from bytetrack_utils import (
    DetectionResults,
    draw_tracking_info,
    get_system_info,
    initialize_bytetrack,
    load_yolo_model,
    process_frame_for_tracking,
    resize_frame,
)


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
    """指定された解像度で動画を処理し、検出と追跡を行う"""
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"エラー: 入力動画を開けません: {input_file}")
        return None

    # 入力動画の情報
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 出力ファイルの準備
    if output_file:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Mac互換コーデック
        out = cv2.VideoWriter(output_file, fourcc, fps, (target_width, target_height))
    else:
        out = None

    print(f"処理開始: {target_width}x{target_height}, {fps}fps, 約{frame_count}フレーム")

    # 結果保存用のオブジェクト
    results = DetectionResults()
    frame_idx = 0
    start_time = time.time()
    last_fps_update = start_time
    fps_buffer = []

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_idx += 1
            frame_start_time = time.time()

            # フレームのリサイズ（必要な場合）
            if target_width != original_width or target_height != original_height:
                frame_bgr = resize_frame(frame_bgr, target_width, target_height)

            # 処理用にRGBに変換
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # 検出と追跡
            tracks, detection_time_ms, tracking_time_ms, num_detections, num_tracks, avg_conf = (
                process_frame_for_tracking(frame_rgb, model, tracker)
            )

            # 現在のメモリ使用量
            current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

            # 結果を記録
            frame_process_time = (time.time() - frame_start_time) * 1000
            current_fps = 1000 / frame_process_time if frame_process_time > 0 else 0
            results.add_frame_result(
                detection_time_ms,
                tracking_time_ms,
                current_fps,
                num_detections,
                num_tracks,
                current_memory,
                avg_conf,
            )

            # 追跡情報の描画と出力
            if output_file or enable_preview:
                fps_buffer.append(current_fps)
                if len(fps_buffer) > 10:
                    fps_buffer.pop(0)
                avg_fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0

                frame_bgr = draw_tracking_info(frame_bgr, tracks)

                # フレーム情報の表示
                cv2.putText(
                    frame_bgr,
                    f"Frame: {frame_idx}/{frame_count} FPS: {avg_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                if out:
                    out.write(frame_bgr)

                if enable_preview:
                    cv2.imshow(f"Video {target_width}x{target_height}", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            # 進捗表示
            if frame_idx % 30 == 0:
                elapsed = time.time() - start_time
                progress = frame_idx / frame_count * 100 if frame_count > 0 else 0
                print(
                    f"解像度 {target_width}x{target_height}: {frame_idx}/{frame_count} ({progress:.1f}%) "
                    f"FPS: {avg_fps:.1f} メモリ: {current_memory:.1f}MB"
                )

    except KeyboardInterrupt:
        print("\n処理が中断されました")
    finally:
        cap.release()
        if out:
            out.release()
        if enable_preview:
            cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print(
        f"\n処理完了 ({target_width}x{target_height}): {frame_idx}フレーム処理 ({total_time:.1f}秒)"
    )
    print(f"平均FPS: {frame_idx / total_time:.1f}")

    return results


def compare_resolutions(
    input_file, output_dir, model_path="yolov11n.pt", enable_preview=False, device=""
):
    """異なる解像度での性能を比較する"""
    print(f"モデル: {model_path}")

    # モデルの読み込み
    model = load_yolo_model(model_path, device)

    # 入力ファイルの情報を取得
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"エラー: 入力動画を開けません: {input_file}")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"入力動画: {original_width}x{original_height}, {orig_fps}fps, {frame_count}フレーム")

    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    # 比較する解像度のリスト（元の解像度と半分の解像度）
    half_width = original_width // 2
    half_height = original_height // 2
    resolutions = [
        (original_width, original_height),  # 元のサイズ
        (half_width, half_height),  # 半分のサイズ
        (640, 480),  # 標準的な小さいサイズ
    ]

    # ログファイルの初期化
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        output_dir, f"resolution_comparison_{Path(model_path).stem}_{timestamp}.csv"
    )
    initialize_log_file(log_file, input_file, model_path, resolutions)

    all_results = {}

    # 各解像度で処理を実行
    for width, height in resolutions:
        print(f"\n===== 解像度 {width}x{height} を処理中 =====")

        # トラッカーの初期化（各解像度で新しいインスタンスを使用）
        tracker = initialize_bytetrack()

        # 出力ファイル名の設定
        output_file = os.path.join(output_dir, f"output_{width}x{height}.mp4")

        # 動画処理の実行
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
            resolution_key = f"{width}x{height}"
            all_results[resolution_key] = results.get_summary()

    # 結果をログファイルに保存
    try:
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            for resolution, summary in all_results.items():
                writer.writerow(
                    [
                        resolution,
                        summary["frame_count"],
                        f"{summary['avg_detection_time']:.2f}",
                        f"{summary['avg_tracking_time']:.2f}",
                        f"{summary['avg_fps']:.2f}",
                        f"{summary['avg_objects_detected']:.2f}",
                        f"{summary['avg_objects_tracked']:.2f}",
                        summary["max_objects_detected"],
                        summary["max_objects_tracked"],
                        f"{summary['avg_detection_conf']:.4f}",
                        f"{summary['avg_memory_usage']:.2f}",
                    ]
                )
        print(f"\n結果を保存しました: {log_file}")
    except Exception as e:
        print(f"結果の保存中にエラーが発生しました: {e}")

    # 結果を表示
    print("\n===== 比較結果 =====")
    for resolution, summary in all_results.items():
        print(f"\n解像度 {resolution} の結果:")
        print(f"  処理フレーム数: {summary['frame_count']}")
        print(f"  平均検出時間: {summary['avg_detection_time']:.2f}ms")
        print(f"  平均追跡時間: {summary['avg_tracking_time']:.2f}ms")
        print(f"  平均FPS: {summary['avg_fps']:.2f}")
        print(f"  平均検出オブジェクト数: {summary['avg_objects_detected']:.2f}")
        print(f"  平均追跡オブジェクト数: {summary['avg_objects_tracked']:.2f}")
        print(f"  最大検出オブジェクト数: {summary['max_objects_detected']}")
        print(f"  最大追跡オブジェクト数: {summary['max_objects_tracked']}")
        print(f"  平均検出信頼度: {summary['avg_detection_conf']:.4f}")
        print(f"  平均メモリ使用量: {summary['avg_memory_usage']:.2f}MB")


def main():
    parser = argparse.ArgumentParser(description="異なる解像度でのYOLOv11+ByteTrack性能比較")
    parser.add_argument("input", help="入力動画ファイル")
    parser.add_argument(
        "-o", "--output-dir", default="output", help="出力ディレクトリ (デフォルト: output)"
    )
    parser.add_argument(
        "-m", "--model", default="yolov11n.pt", help="YOLOモデルパス (デフォルト: yolov11n.pt)"
    )
    parser.add_argument("--preview", action="store_true", help="処理中のプレビューを表示")
    parser.add_argument(
        "--device", default="", help="推論デバイス (例: cpu, 0, '', デフォルト: auto)"
    )

    args = parser.parse_args()

    compare_resolutions(
        args.input,
        args.output_dir,
        model_path=args.model,
        enable_preview=args.preview,
        device=args.device,
    )


if __name__ == "__main__":
    main()
