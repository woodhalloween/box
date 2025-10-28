"""
scripts/analyze_annotations_for_threshold.py

正解データ（annotations.csv）とMediaPipe検出結果を照合し、
最適な閾値を計算するスクリプト。

主な機能:
- annotations.csvを読み込み
- 各動画に対してMediaPipeでヨー角を計算
- ActionLabelごとにヨー角の統計を集計
- 最適閾値の推奨値を計算
- Markdownレポート出力
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detectors.mediapipe_head_turn_detector import MediaPipeFaceMeshHeadTurnDetector


def load_annotations(annotations_path: Path) -> dict[str, dict[int, str]]:
    """annotations.csvを読み込む。

    Args:
        annotations_path: annotations.csvのパス

    Returns:
        dict[VideoID, dict[Timestamp, ActionLabel]]
    """
    annotations = defaultdict(dict)

    with open(annotations_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["VideoID"]
            timestamp = int(row["Timestamp"])
            action_label = row["ActionLabel"]
            annotations[video_id][timestamp] = action_label

    return dict(annotations)


def analyze_video(
    video_path: Path, annotations: dict[int, str], detector: MediaPipeFaceMeshHeadTurnDetector
) -> list[dict]:
    """動画を分析してヨー角を計算する。

    Args:
        video_path: 動画ファイルのパス
        annotations: タイムスタンプごとのActionLabel
        detector: MediaPipe検出器

    Returns:
        list[dict]: フレームごとの分析結果
            - timestamp (int): タイムスタンプ（秒）
            - frame_number (int): フレーム番号
            - yaw_angle (float): ヨー角
            - action_label (str): 正解ラベル
            - face_detected (bool): 顔検出フラグ
    """
    if not video_path.exists():
        print(f"エラー: 動画ファイルが見つかりません - {video_path}")
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"エラー: 動画を開けません - {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # デフォルト

    results = []
    frame_number = 0

    print(f"動画を分析中: {video_path.name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = int(frame_number / fps)

        # annotations.csvにこのタイムスタンプのデータがあるか確認
        if timestamp_sec in annotations:
            action_label = annotations[timestamp_sec]

            # MediaPipeで検出
            detection_result = detector.detect(frame, timestamp_sec)

            # デバッグ: 最初の10個のデータポイントを出力
            if len(results) < 10:
                print(
                    f"  フレーム{frame_number}: ヨー角={detection_result['yaw_angle']:.2f}度, "
                    f"顔検出={detection_result['face_detected']}, ラベル={action_label}"
                )

            results.append(
                {
                    "timestamp": timestamp_sec,
                    "frame_number": frame_number,
                    "yaw_angle": detection_result["yaw_angle"],
                    "action_label": action_label,
                    "face_detected": detection_result["face_detected"],
                }
            )

        frame_number += 1

        if frame_number % 100 == 0:
            print(f"  処理中: {frame_number}フレーム")

    cap.release()
    print(f"完了: {len(results)}個のデータポイントを取得")

    return results


def calculate_statistics(results: list[dict]) -> dict[str, dict]:
    """ActionLabelごとにヨー角の統計を計算する。

    Args:
        results: 分析結果のリスト

    Returns:
        dict[ActionLabel, statistics]
    """
    # ActionLabelごとにデータを分類
    grouped_data = defaultdict(list)

    for result in results:
        if result["face_detected"]:
            label = result["action_label"]
            yaw_angle = result["yaw_angle"]
            grouped_data[label].append(yaw_angle)

    # 統計を計算
    statistics = {}

    for label, angles in grouped_data.items():
        if not angles:
            continue

        angles_array = np.array(angles)

        statistics[label] = {
            "count": len(angles),
            "mean": float(np.mean(angles_array)),
            "std": float(np.std(angles_array)),
            "min": float(np.min(angles_array)),
            "max": float(np.max(angles_array)),
            "median": float(np.median(angles_array)),
        }

    return statistics


def recommend_thresholds(statistics: dict[str, dict]) -> dict[str, float]:
    """統計データから最適な閾値を推奨する。

    Args:
        statistics: ActionLabelごとの統計

    Returns:
        dict: 推奨閾値
            - yaw_threshold_right: 右向き判定の閾値
            - yaw_threshold_left: 左向き判定の閾値
    """
    recommendations = {}

    # 右向きの閾値: 右向きの平均 - 1標準偏差
    if "右向き" in statistics:
        right_mean = statistics["右向き"]["mean"]
        right_std = statistics["右向き"]["std"]
        recommendations["yaw_threshold_right"] = right_mean - right_std
    else:
        recommendations["yaw_threshold_right"] = 20.0  # デフォルト

    # 左向きの閾値: 左向きの平均 + 1標準偏差
    if "左向き" in statistics:
        left_mean = statistics["左向き"]["mean"]
        left_std = statistics["左向き"]["std"]
        recommendations["yaw_threshold_left"] = left_mean + left_std
    else:
        recommendations["yaw_threshold_left"] = -20.0  # デフォルト

    return recommendations


def generate_report(
    statistics: dict[str, dict], recommendations: dict[str, float], output_path: Path
) -> None:
    """分析結果をMarkdownレポートとして出力する。

    Args:
        statistics: ActionLabelごとの統計
        recommendations: 推奨閾値
        output_path: 出力ファイルパス
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = [
        "# ヨー角分析レポート",
        "",
        f"**生成日時**: {timestamp}",
        "",
        "## 1. 概要",
        "",
        "正解データ（annotations.csv）に基づいて、MediaPipe Face Meshで検出したヨー角の統計を分析しました。",
        "",
        "## 2. ヨー角統計",
        "",
        "| ActionLabel | サンプル数 | 平均（度） | 標準偏差（度） | 最小値（度） | 最大値（度） | 中央値（度） |",
        "|------------|----------|----------|-------------|------------|------------|------------|",
    ]

    # 統計テーブルを生成
    for label in ["正面", "左向き", "右向き", "カメラ操作"]:
        if label in statistics:
            stats = statistics[label]
            report_lines.append(
                f"| {label} | {stats['count']} | {stats['mean']:.2f} | {stats['std']:.2f} | "
                f"{stats['min']:.2f} | {stats['max']:.2f} | {stats['median']:.2f} |"
            )
        else:
            report_lines.append(f"| {label} | 0 | - | - | - | - | - |")

    report_lines.extend(
        [
            "",
            "## 3. 推奨閾値",
            "",
            "統計分析に基づいた推奨閾値:",
            "",
            "```python",
            f"yaw_threshold_right = {recommendations['yaw_threshold_right']:.2f}  # 右向き判定",
            f"yaw_threshold_left = {recommendations['yaw_threshold_left']:.2f}   # 左向き判定",
            "```",
            "",
            "### 判定ルール",
            "",
            f"- **右向き**: ヨー角 >= {recommendations['yaw_threshold_right']:.2f}度",
            f"- **左向き**: ヨー角 <= {recommendations['yaw_threshold_left']:.2f}度",
            f"- **正面**: {recommendations['yaw_threshold_left']:.2f}度 < ヨー角 < {recommendations['yaw_threshold_right']:.2f}度",
            "",
            "## 4. 分析方法",
            "",
            "### 閾値の計算式",
            "",
            "```",
            "yaw_threshold_right = mean(右向きヨー角) - 1.0 × std(右向きヨー角)",
            "yaw_threshold_left = mean(左向きヨー角) + 1.0 × std(左向きヨー角)",
            "```",
            "",
            "この計算により、約68%のデータが正しく分類されることが期待されます（1標準偏差の範囲）。",
            "",
            "## 5. 使用方法",
            "",
            "推奨閾値を`MediaPipeFaceMeshHeadTurnDetector`に適用:",
            "",
            "```python",
            "detector = MediaPipeFaceMeshHeadTurnDetector(",
            f"    yaw_threshold_right={recommendations['yaw_threshold_right']:.2f},",
            f"    yaw_threshold_left={recommendations['yaw_threshold_left']:.2f},",
            "    min_consecutive_frames=3,",
            ")",
            "```",
            "",
            "## 6. 注意事項",
            "",
            "- サンプル数が少ない場合、閾値の精度が低くなる可能性があります",
            "- 動画の撮影条件（照明、角度、距離）によって閾値の調整が必要な場合があります",
            "- 実際の使用環境でテストし、必要に応じて微調整してください",
        ]
    )

    # ファイルに書き込み
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nレポートを保存しました: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="annotations.csvからヨー角の最適閾値を分析")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations.csv"),
        help="annotations.csvのパス（デフォルト: data/annotations.csv）",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("data/raw"),
        help="動画ファイルのディレクトリ（デフォルト: data/raw）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/analysis/yaw_angle_analysis.md"),
        help="出力レポートのパス（デフォルト: docs/analysis/yaw_angle_analysis.md）",
    )

    args = parser.parse_args()

    # annotations.csvを読み込み
    print(f"annotations.csvを読み込み中: {args.annotations}")
    annotations = load_annotations(args.annotations)
    print(f"読み込み完了: {len(annotations)}個の動画")

    # MediaPipe検出器を初期化
    print("\nMediaPipe Face Mesh検出器を初期化中...")
    detector = MediaPipeFaceMeshHeadTurnDetector()

    # 各動画を分析
    all_results = []

    for video_id, video_annotations in annotations.items():
        video_path = args.video_dir / video_id

        if not video_path.exists():
            print(f"警告: 動画ファイルが見つかりません - {video_path}")
            continue

        results = analyze_video(video_path, video_annotations, detector)
        all_results.extend(results)

    print(f"\n合計 {len(all_results)}個のデータポイントを取得")

    # 統計を計算
    print("\n統計を計算中...")
    statistics = calculate_statistics(all_results)

    # 閾値を推奨
    print("最適閾値を計算中...")
    recommendations = recommend_thresholds(statistics)

    print("\n=== 統計結果 ===")
    for label, stats in statistics.items():
        print(f"{label}: 平均={stats['mean']:.2f}度, 標準偏差={stats['std']:.2f}度, サンプル数={stats['count']}")

    print("\n=== 推奨閾値 ===")
    print(f"yaw_threshold_right = {recommendations['yaw_threshold_right']:.2f}度")
    print(f"yaw_threshold_left = {recommendations['yaw_threshold_left']:.2f}度")

    # レポートを生成
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_report(statistics, recommendations, args.output)


if __name__ == "__main__":
    main()

