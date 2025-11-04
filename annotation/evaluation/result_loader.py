#!/usr/bin/env python3
"""
検出結果読み込みモジュール

MediaPipeとYOLO11の検出結果CSVを読み込み、統一フォーマットに変換します。
"""

import csv
from pathlib import Path


class DetectionResult:
    """検出結果を表すクラス"""

    def __init__(
        self,
        start_frame: int,
        end_frame: int,
        start_time: float,
        end_time: float,
        event_type: str,
        confidence: float | None = None,
    ):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.start_time = start_time
        self.end_time = end_time
        self.event_type = event_type
        self.confidence = confidence

    def __repr__(self):
        return (
            f"DetectionResult(frames={self.start_frame}-{self.end_frame}, "
            f"time={self.start_time:.2f}-{self.end_time:.2f}s, "
            f"type={self.event_type})"
        )

    def to_dict(self):
        """辞書形式に変換"""
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "event_type": self.event_type,
            "confidence": self.confidence,
        }


class ResultLoader:
    """検出結果を読み込むクラス"""

    @staticmethod
    def load_ground_truth(csv_path: str) -> list[DetectionResult]:
        """
        正解ラベルCSVを読み込み

        フォーマット: start_frame, end_frame, start_time, end_time, event_type
        """
        results = []

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                result = DetectionResult(
                    start_frame=int(row["start_frame"]),
                    end_frame=int(row["end_frame"]),
                    start_time=float(row["start_time"]),
                    end_time=float(row["end_time"]),
                    event_type=row["event_type"],
                )
                results.append(result)

        return results

    @staticmethod
    def load_mediapipe_results(csv_path: str, fps: float = 30.0) -> list[DetectionResult]:
        """
        MediaPipeの検出結果CSVを読み込み

        フォーマット: frame, type, details (または frame_number, detection_type, details)
        連続するフレームを1つのイベントにまとめます
        """
        if not Path(csv_path).exists():
            print(f"⚠️  MediaPipeの結果ファイルが見つかりません: {csv_path}")
            return []

        # CSVを読み込み
        detections = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # カラム名を確認（複数のフォーマットに対応）
            fieldnames = reader.fieldnames
            frame_col = "frame" if "frame" in fieldnames else "frame_number"
            type_col = "type" if "type" in fieldnames else "detection_type"

            for row in reader:
                frame_num = int(row[frame_col])
                detection_type = row[type_col]
                detections.append((frame_num, detection_type))

        # 連続するフレームをイベントにまとめる
        results = []
        if not detections:
            return results

        current_start = detections[0][0]
        current_type = detections[0][1]
        prev_frame = detections[0][0]

        for frame_num, detection_type in detections[1:]:
            # 同じタイプで連続している場合
            if detection_type == current_type and frame_num == prev_frame + 1:
                prev_frame = frame_num
            else:
                # イベントを保存
                result = DetectionResult(
                    start_frame=current_start,
                    end_frame=prev_frame,
                    start_time=current_start / fps,
                    end_time=prev_frame / fps,
                    event_type=current_type,
                )
                results.append(result)

                # 新しいイベント開始
                current_start = frame_num
                current_type = detection_type
                prev_frame = frame_num

        # 最後のイベントを保存
        result = DetectionResult(
            start_frame=current_start,
            end_frame=prev_frame,
            start_time=current_start / fps,
            end_time=prev_frame / fps,
            event_type=current_type,
        )
        results.append(result)

        return results

    @staticmethod
    def load_yolo11_results(csv_path: str, fps: float = 30.0, event_type_column: str = "idea") -> list[DetectionResult]:
        """
        YOLO11の検出結果CSVを読み込み

        メインプロジェクトの出力形式に対応:
        - movement_analysis.csv: frame, timestamp, idea, ...
        - results.csv: frame, timestamp, person_id, ...

        Args:
            csv_path: CSVファイルパス
            fps: フレームレート
            event_type_column: イベントタイプを示すカラム名（'idea'など）
        """
        if not Path(csv_path).exists():
            print(f"⚠️  YOLO11の結果ファイルが見つかりません: {csv_path}")
            return []

        # CSVを読み込み
        detections = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # カラム名を確認
            if event_type_column not in reader.fieldnames:
                print(f"⚠️  カラム '{event_type_column}' が見つかりません")
                print(f"   利用可能なカラム: {reader.fieldnames}")
                return []

            for row in reader:
                frame_num = int(row["frame"])
                event_type = row[event_type_column]

                # 空のイベントタイプはスキップ
                if not event_type or event_type.strip() == "":
                    continue

                detections.append((frame_num, event_type))

        # 連続するフレームをイベントにまとめる
        results = []
        if not detections:
            return results

        current_start = detections[0][0]
        current_type = detections[0][1]
        prev_frame = detections[0][0]

        for frame_num, event_type in detections[1:]:
            # 同じタイプで連続している場合
            if event_type == current_type and frame_num == prev_frame + 1:
                prev_frame = frame_num
            else:
                # イベントを保存
                result = DetectionResult(
                    start_frame=current_start,
                    end_frame=prev_frame,
                    start_time=current_start / fps,
                    end_time=prev_frame / fps,
                    event_type=current_type,
                )
                results.append(result)

                # 新しいイベント開始
                current_start = frame_num
                current_type = event_type
                prev_frame = frame_num

        # 最後のイベントを保存
        result = DetectionResult(
            start_frame=current_start,
            end_frame=prev_frame,
            start_time=current_start / fps,
            end_time=prev_frame / fps,
            event_type=current_type,
        )
        results.append(result)

        return results

    @staticmethod
    def filter_by_event_type(results: list[DetectionResult], event_type: str) -> list[DetectionResult]:
        """特定のイベントタイプでフィルタリング"""
        return [r for r in results if r.event_type == event_type]

    @staticmethod
    def print_summary(results: list[DetectionResult], label: str = "Results"):
        """検出結果のサマリーを表示"""
        print(f"\n=== {label} ===")
        print(f"総イベント数: {len(results)}")

        if not results:
            return

        # イベントタイプ別の集計
        event_types = {}
        total_duration = 0.0

        for result in results:
            event_type = result.event_type
            duration = result.end_time - result.start_time

            if event_type not in event_types:
                event_types[event_type] = {"count": 0, "duration": 0.0}

            event_types[event_type]["count"] += 1
            event_types[event_type]["duration"] += duration
            total_duration += duration

        print(f"総検出時間: {total_duration:.2f}秒")
        print()

        for event_type, stats in event_types.items():
            print(f"  {event_type}:")
            print(f"    - イベント数: {stats['count']}")
            print(f"    - 総時間: {stats['duration']:.2f}秒")
            print(f"    - 平均時間: {stats['duration']/stats['count']:.2f}秒")


def main():
    """テスト用のメイン関数"""
    import sys

    if len(sys.argv) < 2:
        print("使い方: python3 result_loader.py <csv_path> [type]")
        print("  type: ground_truth, mediapipe, yolo11")
        sys.exit(1)

    csv_path = sys.argv[1]
    result_type = sys.argv[2] if len(sys.argv) > 2 else "ground_truth"

    loader = ResultLoader()

    if result_type == "ground_truth":
        results = loader.load_ground_truth(csv_path)
    elif result_type == "mediapipe":
        results = loader.load_mediapipe_results(csv_path)
    elif result_type == "yolo11":
        results = loader.load_yolo11_results(csv_path)
    else:
        print(f"不明なタイプ: {result_type}")
        sys.exit(1)

    loader.print_summary(results, f"{result_type.upper()} Results")

    # 最初の5件を表示
    if results:
        print("\n最初の5件:")
        for i, result in enumerate(results[:5], 1):
            print(f"  {i}. {result}")


if __name__ == "__main__":
    main()
