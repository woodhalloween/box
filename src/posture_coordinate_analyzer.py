#!/usr/bin/env python3
"""
姿勢変化による座標変化を分析するツール

MediaPipeの関節データとYOLOのバウンディングボックスの中心座標を比較して、
姿勢変化時の安定性を検証する。
"""

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd

from pose_estimator import PoseEstimator


def calculate_midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """2点の中点を計算"""
    return (p1 + p2) / 2


class PostureCoordinateAnalyzer:
    """姿勢変化による座標変化を分析するクラス"""

    def __init__(self):
        self.pose_estimator = PoseEstimator()
        self.data = []

    def analyze_video(self, video_path: str, output_csv: str, sample_rate: int = 5):
        """
        ビデオを分析して座標データを収集

        Args:
            video_path: 入力ビデオのパス
            output_csv: 出力CSVファイルのパス
            sample_rate: フレームサンプリング率（N フレームごとに分析）
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frame_count = 0

        # CSVファイルの準備
        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "frame_number",
                "timestamp",
                "bbox_center_x",
                "bbox_center_y",
                "hip_center_x",
                "hip_center_y",
                "left_hip_x",
                "left_hip_y",
                "left_hip_visibility",
                "right_hip_x",
                "right_hip_y",
                "right_hip_visibility",
                "shoulder_center_x",
                "shoulder_center_y",
                "bbox_width",
                "bbox_height",
                "hip_available",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # サンプリング率に基づいてフレームをスキップ
                if frame_count % sample_rate != 0:
                    continue

                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                # MediaPipeで姿勢推定
                landmarks = self.pose_estimator.estimate(frame)

                # 仮想的なバウンディングボックス（実際のYOLOなしで概算）
                bbox_center_x, bbox_center_y = None, None
                bbox_width, bbox_height = None, None

                if landmarks is not None:
                    # 全ランドマークからバウンディングボックスを概算
                    visible_landmarks = landmarks[landmarks[:, 3] > 0.5]  # 可視性が0.5以上
                    if len(visible_landmarks) > 0:
                        x_coords = visible_landmarks[:, 0] * frame.shape[1]
                        y_coords = visible_landmarks[:, 1] * frame.shape[0]

                        x1, x2 = int(np.min(x_coords)), int(np.max(x_coords))
                        y1, y2 = int(np.min(y_coords)), int(np.max(y_coords))

                        # パディングを追加（実際のYOLOに近づける）
                        padding = 20
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(frame.shape[1], x2 + padding)
                        y2 = min(frame.shape[0], y2 + padding)

                        bbox_center_x = (x1 + x2) / 2
                        bbox_center_y = (y1 + y2) / 2
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1

                # 腰と肩の座標を取得
                hip_data = self._extract_hip_coordinates(landmarks, frame.shape)
                shoulder_data = self._extract_shoulder_coordinates(landmarks, frame.shape)

                # データを記録
                row = {
                    "frame_number": frame_count,
                    "timestamp": timestamp,
                    "bbox_center_x": bbox_center_x,
                    "bbox_center_y": bbox_center_y,
                    "hip_center_x": hip_data.get("center_x"),
                    "hip_center_y": hip_data.get("center_y"),
                    "left_hip_x": hip_data.get("left_x"),
                    "left_hip_y": hip_data.get("left_y"),
                    "left_hip_visibility": hip_data.get("left_visibility"),
                    "right_hip_x": hip_data.get("right_x"),
                    "right_hip_y": hip_data.get("right_y"),
                    "right_hip_visibility": hip_data.get("right_visibility"),
                    "shoulder_center_x": shoulder_data.get("center_x"),
                    "shoulder_center_y": shoulder_data.get("center_y"),
                    "bbox_width": bbox_width,
                    "bbox_height": bbox_height,
                    "hip_available": hip_data.get("available", False),
                }

                writer.writerow(row)
                self.data.append(row)

                # 進捗表示
                if frame_count % 100 == 0:
                    print(f"処理済み: {frame_count} フレーム (タイムスタンプ: {timestamp:.2f}秒)")

        cap.release()
        print(f"分析完了: {len(self.data)} サンプル収集")
        print(f"データ保存先: {output_csv}")

    def _extract_hip_coordinates(self, landmarks: np.ndarray | None, frame_shape: tuple) -> dict:
        """腰の座標データを抽出"""
        if landmarks is None:
            return {"available": False}

        try:
            left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]

            # 両方の腰が十分に見えているかチェック
            if left_hip[3] > 0.5 and right_hip[3] > 0.5:  # visibility > 0.5
                # 画像座標に変換
                left_x = left_hip[0] * frame_shape[1]
                left_y = left_hip[1] * frame_shape[0]
                right_x = right_hip[0] * frame_shape[1]
                right_y = right_hip[1] * frame_shape[0]

                # 中心座標を計算
                center_x = (left_x + right_x) / 2
                center_y = (left_y + right_y) / 2

                return {
                    "available": True,
                    "center_x": center_x,
                    "center_y": center_y,
                    "left_x": left_x,
                    "left_y": left_y,
                    "left_visibility": left_hip[3],
                    "right_x": right_x,
                    "right_y": right_y,
                    "right_visibility": right_hip[3],
                }
            return {"available": False}

        except (IndexError, KeyError):
            return {"available": False}

    def _extract_shoulder_coordinates(self, landmarks: np.ndarray | None, frame_shape: tuple) -> dict:
        """肩の座標データを抽出"""
        if landmarks is None:
            return {"available": False}

        try:
            left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]

            if left_shoulder[3] > 0.5 and right_shoulder[3] > 0.5:
                # 画像座標に変換
                left_x = left_shoulder[0] * frame_shape[1]
                left_y = left_shoulder[1] * frame_shape[0]
                right_x = right_shoulder[0] * frame_shape[1]
                right_y = right_shoulder[1] * frame_shape[0]

                # 中心座標を計算
                center_x = (left_x + right_x) / 2
                center_y = (left_y + right_y) / 2

                return {"available": True, "center_x": center_x, "center_y": center_y}
            return {"available": False}

        except (IndexError, KeyError):
            return {"available": False}

    def create_movement_analysis(self, csv_path: str, output_dir: str):
        """座標データから移動分析を実行してグラフを生成"""

        # データ読み込み
        df = pd.read_csv(csv_path)

        # 有効なデータのみを抽出
        valid_data = df[df["hip_available"] == True].copy()

        if len(valid_data) < 10:
            print("警告: 腰のデータが不十分です")
            return None

        # 移動距離の計算
        valid_data["bbox_movement"] = np.sqrt(
            valid_data["bbox_center_x"].diff() ** 2 + valid_data["bbox_center_y"].diff() ** 2
        )

        valid_data["hip_movement"] = np.sqrt(
            valid_data["hip_center_x"].diff() ** 2 + valid_data["hip_center_y"].diff() ** 2
        )

        # 外れ値の除去（上位5%を除外）
        bbox_movement_clean = valid_data["bbox_movement"].quantile(0.95)
        hip_movement_clean = valid_data["hip_movement"].quantile(0.95)

        # グラフ作成
        plt.figure(figsize=(15, 10))

        # 1. 座標の軌跡
        plt.subplot(2, 3, 1)
        plt.plot(
            valid_data["bbox_center_x"],
            valid_data["bbox_center_y"],
            "b-",
            alpha=0.7,
            label="バウンディングボックス中心",
        )
        plt.plot(valid_data["hip_center_x"], valid_data["hip_center_y"], "r-", alpha=0.7, label="腰の中心")
        plt.xlabel("X座標 (px)")
        plt.ylabel("Y座標 (px)")
        plt.title("座標の軌跡比較")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. X座標の時系列変化
        plt.subplot(2, 3, 2)
        plt.plot(
            valid_data["timestamp"], valid_data["bbox_center_x"], "b-", alpha=0.7, label="バウンディングボックス中心X"
        )
        plt.plot(valid_data["timestamp"], valid_data["hip_center_x"], "r-", alpha=0.7, label="腰の中心X")
        plt.xlabel("時間 (秒)")
        plt.ylabel("X座標 (px)")
        plt.title("X座標の時系列変化")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. Y座標の時系列変化
        plt.subplot(2, 3, 3)
        plt.plot(
            valid_data["timestamp"], valid_data["bbox_center_y"], "b-", alpha=0.7, label="バウンディングボックス中心Y"
        )
        plt.plot(valid_data["timestamp"], valid_data["hip_center_y"], "r-", alpha=0.7, label="腰の中心Y")
        plt.xlabel("時間 (秒)")
        plt.ylabel("Y座標 (px)")
        plt.title("Y座標の時系列変化")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4. 移動距離の比較
        plt.subplot(2, 3, 4)
        plt.plot(
            valid_data["timestamp"][1:],
            valid_data["bbox_movement"][1:],
            "b-",
            alpha=0.7,
            label="バウンディングボックス移動距離",
        )
        plt.plot(valid_data["timestamp"][1:], valid_data["hip_movement"][1:], "r-", alpha=0.7, label="腰の移動距離")
        plt.xlabel("時間 (秒)")
        plt.ylabel("移動距離 (px)")
        plt.title("フレーム間移動距離")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 5. 移動距離の分布
        plt.subplot(2, 3, 5)
        bbox_clean = valid_data["bbox_movement"][valid_data["bbox_movement"] <= bbox_movement_clean]
        hip_clean = valid_data["hip_movement"][valid_data["hip_movement"] <= hip_movement_clean]

        plt.hist(bbox_clean.dropna(), bins=30, alpha=0.7, label="バウンディングボックス", color="blue")
        plt.hist(hip_clean.dropna(), bins=30, alpha=0.7, label="腰の中心", color="red")
        plt.xlabel("移動距離 (px)")
        plt.ylabel("頻度")
        plt.title("移動距離の分布")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 6. 統計サマリー
        plt.subplot(2, 3, 6)
        plt.axis("off")

        # 統計計算
        bbox_stats = {
            "平均移動距離": f"{bbox_clean.mean():.2f} px",
            "移動距離標準偏差": f"{bbox_clean.std():.2f} px",
            "最大移動距離": f"{bbox_clean.max():.2f} px",
        }

        hip_stats = {
            "平均移動距離": f"{hip_clean.mean():.2f} px",
            "移動距離標準偏差": f"{hip_clean.std():.2f} px",
            "最大移動距離": f"{hip_clean.max():.2f} px",
        }

        stats_text = "=== 統計サマリー ===\n\n"
        stats_text += "【バウンディングボックス中心】\n"
        for key, value in bbox_stats.items():
            stats_text += f"  {key}: {value}\n"
        stats_text += "\n【腰の中心】\n"
        for key, value in hip_stats.items():
            stats_text += f"  {key}: {value}\n"

        stats_text += "\n【安定性評価】\n"
        stability_ratio = hip_clean.std() / bbox_clean.std() if bbox_clean.std() > 0 else 0
        if stability_ratio < 0.8:
            stats_text += f"  腰の中心の方が{1 / stability_ratio:.1f}倍安定"
        elif stability_ratio > 1.2:
            stats_text += f"  バウンディングボックス中心の方が{stability_ratio:.1f}倍安定"
        else:
            stats_text += "  両者の安定性は同程度"

        plt.text(
            0.1,
            0.9,
            stats_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.tight_layout()

        # グラフ保存
        output_path = Path(output_dir) / "coordinate_analysis.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print("分析グラフ保存:", output_path)

        # 詳細なCSVレポート生成
        report_path = Path(output_dir) / "movement_analysis_report.csv"
        summary_data = {
            "metric": [
                "bbox_mean_movement",
                "bbox_std_movement",
                "bbox_max_movement",
                "hip_mean_movement",
                "hip_std_movement",
                "hip_max_movement",
                "stability_ratio",
                "valid_frames",
                "total_frames",
            ],
            "value": [
                bbox_clean.mean(),
                bbox_clean.std(),
                bbox_clean.max(),
                hip_clean.mean(),
                hip_clean.std(),
                hip_clean.max(),
                stability_ratio,
                len(valid_data),
                len(df),
            ],
        }

        pd.DataFrame(summary_data).to_csv(report_path, index=False)
        print("数値レポート保存:", report_path)

        return {
            "bbox_stats": bbox_stats,
            "hip_stats": hip_stats,
            "stability_ratio": stability_ratio,
            "valid_frames": len(valid_data),
        }

    def close(self):
        """リソースを解放"""
        self.pose_estimator.close()


def main():
    parser = argparse.ArgumentParser(description="姿勢変化による座標変化を分析")
    parser.add_argument("--video", required=True, help="入力ビデオファイル")
    parser.add_argument("--output-csv", default="output/coordinate_analysis.csv", help="出力CSVファイル")
    parser.add_argument("--output-dir", default="output/analysis", help="分析結果の出力ディレクトリ")
    parser.add_argument("--sample-rate", type=int, default=5, help="フレームサンプリング率")
    parser.add_argument("--analyze-only", action="store_true", help="既存のCSVから分析のみ実行")

    args = parser.parse_args()

    analyzer = PostureCoordinateAnalyzer()

    try:
        if not args.analyze_only:
            # ビデオ分析
            print(f"ビデオ分析開始: {args.video}")
            analyzer.analyze_video(args.video, args.output_csv, args.sample_rate)

        # 移動分析とグラフ生成
        print("移動分析開始...")
        results = analyzer.create_movement_analysis(args.output_csv, args.output_dir)

        print("\n=== 分析結果 ===")
        print(f"安定性比率: {results['stability_ratio']:.3f}")
        print(f"有効フレーム数: {results['valid_frames']}")

    finally:
        analyzer.close()


if __name__ == "__main__":
    main()
