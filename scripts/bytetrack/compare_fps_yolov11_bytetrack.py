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
)


def initialize_log_file(log_file_path, input_file, model_path, fps_values):
    """結果ログファイルの初期化"""
    system_info = get_system_info()

    with open(log_file_path, "w", newline="") as log_f:
        log_writer = csv.writer(log_f)
        log_writer.writerow(["# FPS別YOLOv11+ByteTrack精度比較"])
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
        log_writer.writerow(["# 元の入力動画", input_file])
        log_writer.writerow(["# YOLOモデル", model_path])
        log_writer.writerow(["# FPS比較", ", ".join([f"{fps}fps" for fps in fps_values])])
        log_writer.writerow([])

        # ヘッダー行
        log_writer.writerow(
            [
                "FPS",
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


def create_lower_fps_video(input_file, output_file, target_fps=10):
    """入力動画からFPSを下げた動画を作成する"""
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"エラー: 入力動画を開けません: {input_file}")
        return False

    # 入力動画の情報
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 元のFPSが既に10以下の場合はそのままコピー
    if orig_fps <= target_fps:
        print(
            f"警告: 元の動画のFPS({orig_fps})が既に目標FPS({target_fps})以下です。そのままコピーします。"
        )
        cap.release()
        import shutil

        shutil.copy(input_file, output_file)
        return True

    # フレームの間引き率の計算
    frame_skip = int(orig_fps / target_fps)

    # 出力動画の設定
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4コーデック
    out = cv2.VideoWriter(output_file, fourcc, target_fps, (width, height))

    frame_idx = 0
    saved_frames = 0

    print(f"FPSを{orig_fps}から{target_fps}に変換中...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # フレームの間引き
            if frame_idx % frame_skip == 0:
                out.write(frame)
                saved_frames += 1

            frame_idx += 1

            # 進捗を表示
            if frame_idx % 100 == 0:
                progress = frame_idx / frame_count * 100
                print(f"処理中: {frame_idx}/{frame_count}フレーム ({progress:.1f}%)")

    finally:
        cap.release()
        out.release()

    print(f"FPS変換完了: {saved_frames}フレームを保存 ({target_fps}fps)")
    return True


def process_video(
    input_file,
    output_file,
    model,
    tracker,
    enable_preview=False,
):
    """指定された動画を処理し、検出と追跡を行う"""
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"エラー: 入力動画を開けません: {input_file}")
        return None

    # 入力動画の情報
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 出力ファイルの準備
    if output_file:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Mac互換コーデック
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    else:
        out = None

    print(f"処理開始: {width}x{height}, {fps}fps, 約{frame_count}フレーム")

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
                    cv2.imshow("Video", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            # 進捗表示
            if frame_idx % 30 == 0:
                elapsed = time.time() - start_time
                progress = frame_idx / frame_count * 100 if frame_count > 0 else 0
                print(
                    f"進捗: {frame_idx}/{frame_count} ({progress:.1f}%) "
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
    print(f"\n処理完了: {frame_idx}フレーム処理 ({total_time:.1f}秒)")
    print(f"平均FPS: {frame_idx / total_time:.1f}")

    return results


def compare_fps(
    input_file, output_dir, model_path="yolov11n.pt", enable_preview=False, device="", target_fps=10
):
    """異なるFPSでの性能を比較する"""
    print(f"モデル: {model_path}")

    # モデルの読み込み
    model = load_yolo_model(model_path, device)

    # 入力ファイルの情報を取得
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"エラー: 入力動画を開けません: {input_file}")
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"入力動画: {width}x{height}, {orig_fps}fps, {frame_count}フレーム")

    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    # 比較するFPS値のリスト
    fps_values = [10, int(orig_fps)]  # 10fpsと元の動画のFPSを比較

    # ログファイルの初期化
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"fps_comparison_{Path(model_path).stem}_{timestamp}.csv")
    initialize_log_file(log_file, input_file, model_path, fps_values)

    all_results = {}

    # 元のFPSの動画を処理
    original_fps_int = int(orig_fps)
    print(f"\n===== 元の動画 ({original_fps_int}fps) を処理中 =====")

    # トラッカーの初期化
    tracker = initialize_bytetrack()

    output_file = os.path.join(output_dir, f"output_{original_fps_int}fps.mp4")
    results_original = process_video(
        input_file, output_file, model, tracker, enable_preview=enable_preview
    )
    if results_original:
        all_results[original_fps_int] = results_original.get_summary()

    # 低FPSの動画を作成して処理
    if target_fps != original_fps_int:
        print(f"\n===== 低FPS動画 ({target_fps}fps) を作成中 =====")
        lower_fps_video = os.path.join(output_dir, f"input_{target_fps}fps.mp4")
        if create_lower_fps_video(input_file, lower_fps_video, target_fps):
            print(f"\n===== 低FPS動画 ({target_fps}fps) を処理中 =====")

            # 新しいトラッカーの初期化
            tracker = initialize_bytetrack()

            output_file = os.path.join(output_dir, f"output_{target_fps}fps.mp4")
            results_low_fps = process_video(
                lower_fps_video, output_file, model, tracker, enable_preview=enable_preview
            )
            if results_low_fps:
                all_results[target_fps] = results_low_fps.get_summary()

    # 結果をログファイルに保存
    try:
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            for fps, summary in all_results.items():
                writer.writerow(
                    [
                        fps,
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
    for fps, summary in all_results.items():
        print(f"\n{fps}fps の結果:")
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
    parser = argparse.ArgumentParser(description="異なるFPSでのYOLOv11+ByteTrack性能比較")
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
    parser.add_argument(
        "--target-fps", type=int, default=10, help="比較対象のFPS (デフォルト: 10fps)"
    )

    args = parser.parse_args()

    compare_fps(
        args.input,
        args.output_dir,
        model_path=args.model,
        enable_preview=args.preview,
        device=args.device,
        target_fps=args.target_fps,
    )


if __name__ == "__main__":
    main()
