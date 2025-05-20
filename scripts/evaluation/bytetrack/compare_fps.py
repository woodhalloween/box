import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import psutil

# 共通ユーティリティをインポート
from src.tracking.bytetrack_utils import (
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
    if not model:
        return

    # 入力ファイルの情報を取得
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"エラー: 元の入力動画を開けません: {input_file}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(
        f"元の入力動画: {original_width}x{original_height}, {original_fps:.2f}fps, {frame_count}フレーム"
    )

    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    # 比較するFPSの値 (元のFPSと指定されたターゲットFPS)
    fps_values_to_compare = [original_fps, target_fps]

    # ログファイルの初期化
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"fps_comparison_{Path(model_path).stem}_{timestamp}.csv")
    initialize_log_file(log_file, input_file, model_path, fps_values_to_compare)

    all_results = {}

    # 各FPSで処理を実行
    for current_proc_fps in fps_values_to_compare:
        print(f"\n===== FPS {current_proc_fps:.2f} を処理中 =====")

        video_to_process = input_file
        temp_video_file = None

        # 必要であればFPSを変換した一時動画を作成
        if abs(current_proc_fps - original_fps) > 0.1:  # わずかな違いは無視
            temp_video_file = os.path.join(output_dir, f"temp_video_{current_proc_fps:.0f}fps.mp4")
            if not create_lower_fps_video(input_file, temp_video_file, target_fps=current_proc_fps):
                print(
                    f"エラー: {current_proc_fps}fpsの動画作成に失敗しました。このFPSでの処理をスキップします。"
                )
                continue
            video_to_process = temp_video_file

        # トラッカーの初期化（各FPSで新しいインスタンスを使用）
        tracker = initialize_bytetrack(frame_rate=current_proc_fps)
        print(f"ByteTrack initialized with frame_rate: {current_proc_fps}")

        # 出力ファイル名の設定 (処理済み動画)
        output_video_file = os.path.join(
            output_dir, f"output_{Path(video_to_process).stem}_processed.mp4"
        )

        fps_results = process_video(
            video_to_process,
            output_video_file,
            model,
            tracker,
            enable_preview=enable_preview,
        )

        if fps_results:
            all_results[f"{current_proc_fps:.2f}fps"] = fps_results.get_summary()
            # ログファイルに追記
            with open(log_file, "a", newline="") as log_f:
                log_writer = csv.writer(log_f)
                summary = all_results[f"{current_proc_fps:.2f}fps"]
                log_writer.writerow(
                    [
                        f"{current_proc_fps:.2f}",
                        summary["frame_count"],
                        f"{summary['avg_detection_time']:.2f}",
                        f"{summary['avg_tracking_time']:.2f}",
                        f"{summary['avg_fps']:.2f}",
                        f"{summary['avg_objects_detected']:.2f}",
                        f"{summary['avg_objects_tracked']:.2f}",
                        summary["max_objects_detected"],
                        summary["max_objects_tracked"],
                        f"{summary['avg_detection_conf']:.3f}",
                        f"{summary['avg_memory_usage']:.2f}",
                    ]
                )
            print(f"結果 ({current_proc_fps:.2f}fps): {summary}")

        # 一時ファイルの削除
        if temp_video_file and os.path.exists(temp_video_file):
            try:
                os.remove(temp_video_file)
                print(f"一時ファイル {temp_video_file} を削除しました。")
            except OSError as e:
                print(f"エラー: 一時ファイル {temp_video_file} の削除に失敗しました: {e}")

    print("\n===== 全FPSの比較結果 =====")
    for fps_val, summary in all_results.items():
        print(f"FPS {fps_val}:")
        print(f"  平均FPS(処理能力): {summary['avg_fps']:.2f}")
        print(f"  平均検出時間: {summary['avg_detection_time']:.2f} ms")
        print(f"  平均追跡時間: {summary['avg_tracking_time']:.2f} ms")
        print(f"  平均検出オブジェクト数: {summary['avg_objects_detected']:.2f}")
        print(f"  平均追跡オブジェクト数: {summary['avg_objects_tracked']:.2f}")
        print(f"  平均検出信頼度: {summary['avg_detection_conf']:.3f}")

    print(f"\nログファイル: {log_file}")
    print("比較処理完了。")


def main():
    parser = argparse.ArgumentParser(
        description="異なるFPSでのYOLOv11とByteTrackの性能を比較します。"
    )
    parser.add_argument("--input", type=str, required=True, help="入力動画ファイルのパス")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/fps_comparison",
        help="出力ディレクトリのパス",
    )
    parser.add_argument("--model", type=str, default="yolov11n.pt", help="YOLOモデルファイルのパス")
    parser.add_argument(
        "--target_fps", type=int, default=10, help="比較対象とする低FPSの値 (例: 10)"
    )
    parser.add_argument("--preview", action="store_true", help="処理中のプレビューを表示する")
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="デバイス指定 (例: 'cpu', 'mps', '0' for GPU 0)",
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
