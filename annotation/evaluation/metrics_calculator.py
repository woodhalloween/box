#!/usr/bin/env python3
"""
評価指標計算モジュール

フレーム単位とイベント単位の評価指標を計算します。
"""

import numpy as np
from result_loader import DetectionResult


class MetricsCalculator:
    """評価指標を計算するクラス"""

    @staticmethod
    def calculate_frame_level_metrics(
        ground_truth: list[DetectionResult],
        predictions: list[DetectionResult],
        total_frames: int,
        event_type: str = None,
    ) -> dict[str, float]:
        """
        フレーム単位の評価指標を計算

        各フレームを「イベント中」「非イベント中」に二値化し、
        Precision, Recall, F1-Score, Accuracy を計算します。

        Args:
            ground_truth: 正解ラベル
            predictions: 予測結果
            total_frames: 動画の総フレーム数
            event_type: 特定のイベントタイプでフィルタリング（Noneの場合は全て）

        Returns:
            評価指標の辞書
        """
        # イベントタイプでフィルタリング
        if event_type:
            ground_truth = [r for r in ground_truth if r.event_type == event_type]
            predictions = [r for r in predictions if r.event_type == event_type]

        # フレームごとの正解/予測を二値配列で表現
        gt_frames = np.zeros(total_frames, dtype=bool)
        pred_frames = np.zeros(total_frames, dtype=bool)

        # 正解ラベルをフレーム配列に変換
        for result in ground_truth:
            gt_frames[result.start_frame : result.end_frame + 1] = True

        # 予測結果をフレーム配列に変換
        for result in predictions:
            pred_frames[result.start_frame : result.end_frame + 1] = True

        # True Positive, False Positive, False Negative, True Negative を計算
        tp = np.sum(gt_frames & pred_frames)
        fp = np.sum(~gt_frames & pred_frames)
        fn = np.sum(gt_frames & ~pred_frames)
        tn = np.sum(~gt_frames & ~pred_frames)

        # 評価指標を計算
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total_frames if total_frames > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "true_positive": int(tp),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_negative": int(tn),
            "total_frames": total_frames,
            "positive_frames": int(np.sum(gt_frames)),
            "predicted_positive_frames": int(np.sum(pred_frames)),
        }

    @staticmethod
    def calculate_iou(gt_result: DetectionResult, pred_result: DetectionResult) -> float:
        """
        2つのイベント区間のIoU（Intersection over Union）を計算

        Args:
            gt_result: 正解イベント
            pred_result: 予測イベント

        Returns:
            IoU値（0.0〜1.0）
        """
        # 重なり区間を計算
        intersection_start = max(gt_result.start_frame, pred_result.start_frame)
        intersection_end = min(gt_result.end_frame, pred_result.end_frame)

        if intersection_start > intersection_end:
            return 0.0

        intersection = intersection_end - intersection_start + 1

        # 結合区間を計算
        union_start = min(gt_result.start_frame, pred_result.start_frame)
        union_end = max(gt_result.end_frame, pred_result.end_frame)
        union = union_end - union_start + 1

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def calculate_event_level_metrics(
        ground_truth: list[DetectionResult],
        predictions: list[DetectionResult],
        iou_threshold: float = 0.5,
        event_type: str = None,
    ) -> dict[str, float]:
        """
        イベント単位の評価指標を計算

        IoUベースで正解イベントと予測イベントをマッチングし、
        True Positive, False Positive, False Negative をカウントします。

        Args:
            ground_truth: 正解ラベル
            predictions: 予測結果
            iou_threshold: IoUの閾値（これ以上で一致とみなす）
            event_type: 特定のイベントタイプでフィルタリング

        Returns:
            評価指標の辞書
        """
        # イベントタイプでフィルタリング
        if event_type:
            ground_truth = [r for r in ground_truth if r.event_type == event_type]
            predictions = [r for r in predictions if r.event_type == event_type]

        # マッチング済みの予測を記録
        matched_predictions = set()

        # True Positive と False Negative をカウント
        tp = 0
        fn = 0
        iou_scores = []

        for gt_result in ground_truth:
            best_iou = 0.0
            best_pred_idx = -1

            # 最もIoUが高い予測を探す
            for i, pred_result in enumerate(predictions):
                if i in matched_predictions:
                    continue

                iou = MetricsCalculator.calculate_iou(gt_result, pred_result)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = i

            # IoUが閾値以上なら True Positive
            if best_iou >= iou_threshold:
                tp += 1
                matched_predictions.add(best_pred_idx)
                iou_scores.append(best_iou)
            else:
                fn += 1

        # False Positive をカウント（マッチしなかった予測）
        fp = len(predictions) - len(matched_predictions)

        # 評価指標を計算
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # 平均IoU
        mean_iou = np.mean(iou_scores) if iou_scores else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "mean_iou": mean_iou,
            "iou_threshold": iou_threshold,
            "total_ground_truth": len(ground_truth),
            "total_predictions": len(predictions),
        }

    @staticmethod
    def calculate_temporal_overlap(
        ground_truth: list[DetectionResult], predictions: list[DetectionResult], event_type: str = None
    ) -> dict[str, float]:
        """
        時間的重なりの統計を計算

        Args:
            ground_truth: 正解ラベル
            predictions: 予測結果
            event_type: 特定のイベントタイプでフィルタリング

        Returns:
            時間的重なりの統計
        """
        # イベントタイプでフィルタリング
        if event_type:
            ground_truth = [r for r in ground_truth if r.event_type == event_type]
            predictions = [r for r in predictions if r.event_type == event_type]

        # 正解イベントの総時間
        gt_total_duration = sum(r.end_time - r.start_time for r in ground_truth)

        # 予測イベントの総時間
        pred_total_duration = sum(r.end_time - r.start_time for r in predictions)

        # 重なり時間を計算
        overlap_duration = 0.0

        for gt_result in ground_truth:
            for pred_result in predictions:
                # 重なり区間を計算
                overlap_start = max(gt_result.start_time, pred_result.start_time)
                overlap_end = min(gt_result.end_time, pred_result.end_time)

                if overlap_start < overlap_end:
                    overlap_duration += overlap_end - overlap_start

        # 重なり率を計算
        overlap_ratio_gt = overlap_duration / gt_total_duration if gt_total_duration > 0 else 0.0
        overlap_ratio_pred = overlap_duration / pred_total_duration if pred_total_duration > 0 else 0.0

        return {
            "gt_total_duration": gt_total_duration,
            "pred_total_duration": pred_total_duration,
            "overlap_duration": overlap_duration,
            "overlap_ratio_gt": overlap_ratio_gt,
            "overlap_ratio_pred": overlap_ratio_pred,
        }

    @staticmethod
    def print_metrics(metrics: dict[str, float], title: str = "Metrics"):
        """評価指標を整形して表示"""
        print(f"\n{'=' * 60}")
        print(f"{title}")
        print(f"{'=' * 60}")

        # パーセンテージで表示する項目
        percentage_keys = [
            "precision",
            "recall",
            "f1_score",
            "accuracy",
            "mean_iou",
            "overlap_ratio_gt",
            "overlap_ratio_pred",
        ]

        for key, value in metrics.items():
            if key in percentage_keys:
                print(f"{key:30s}: {value:6.2%}")
            elif isinstance(value, float):
                print(f"{key:30s}: {value:10.2f}")
            else:
                print(f"{key:30s}: {value:10}")

        print(f"{'=' * 60}\n")


def main():
    """テスト用のメイン関数"""
    import sys

    from result_loader import ResultLoader

    if len(sys.argv) < 3:
        print("使い方: python3 metrics_calculator.py <ground_truth_csv> <prediction_csv> <total_frames>")
        sys.exit(1)

    gt_csv = sys.argv[1]
    pred_csv = sys.argv[2]
    total_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

    # データ読み込み
    loader = ResultLoader()
    ground_truth = loader.load_ground_truth(gt_csv)
    predictions = loader.load_mediapipe_results(pred_csv)  # または load_yolo11_results

    loader.print_summary(ground_truth, "Ground Truth")
    loader.print_summary(predictions, "Predictions")

    # 評価指標を計算
    calculator = MetricsCalculator()

    # フレーム単位の評価
    frame_metrics = calculator.calculate_frame_level_metrics(ground_truth, predictions, total_frames)
    calculator.print_metrics(frame_metrics, "Frame-Level Metrics")

    # イベント単位の評価
    event_metrics = calculator.calculate_event_level_metrics(ground_truth, predictions, iou_threshold=0.5)
    calculator.print_metrics(event_metrics, "Event-Level Metrics (IoU >= 0.5)")

    # 時間的重なりの統計
    overlap_stats = calculator.calculate_temporal_overlap(ground_truth, predictions)
    calculator.print_metrics(overlap_stats, "Temporal Overlap Statistics")


if __name__ == "__main__":
    main()
