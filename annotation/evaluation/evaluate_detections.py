#!/usr/bin/env python3
"""
検出結果評価スクリプト

正解ラベルと検出結果（MediaPipe、YOLO11）を比較し、評価指標を計算します。
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
from metrics_calculator import MetricsCalculator
from result_loader import ResultLoader


class DetectionEvaluator:
    """検出結果を評価するクラス"""

    def __init__(self, ground_truth_csv: str, video_path: str = None):
        """
        Args:
            ground_truth_csv: 正解ラベルのCSVパス
            video_path: 動画ファイルパス（総フレーム数とFPSを取得するため）
        """
        self.ground_truth_csv = ground_truth_csv
        self.video_path = video_path

        # 正解ラベルを読み込み
        self.loader = ResultLoader()
        self.ground_truth = self.loader.load_ground_truth(ground_truth_csv)

        # 動画情報を取得
        if video_path and Path(video_path).exists():
            cap = cv2.VideoCapture(video_path)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        else:
            # 動画がない場合は正解ラベルから推定
            max_frame = max(r.end_frame for r in self.ground_truth) if self.ground_truth else 1000
            self.total_frames = max_frame + 100  # 余裕を持たせる
            self.fps = 30.0  # デフォルト
            print("⚠️  動画ファイルが見つからないため、推定値を使用します")
            print(f"   総フレーム数: {self.total_frames}, FPS: {self.fps}")

        self.calculator = MetricsCalculator()

    def evaluate_method(self, predictions: list, method_name: str, event_type: str = None) -> dict:
        """
        1つの検出手法を評価

        Args:
            predictions: 予測結果のリスト
            method_name: 手法名（表示用）
            event_type: 評価対象のイベントタイプ

        Returns:
            評価結果の辞書
        """
        print(f"\n{'=' * 80}")
        print(f"評価対象: {method_name}")
        if event_type:
            print(f"イベントタイプ: {event_type}")
        print(f"{'=' * 80}")

        # データサマリー
        self.loader.print_summary(self.ground_truth, "Ground Truth")
        self.loader.print_summary(predictions, f"{method_name} Predictions")

        # フレーム単位の評価
        frame_metrics = self.calculator.calculate_frame_level_metrics(
            self.ground_truth, predictions, self.total_frames, event_type
        )
        self.calculator.print_metrics(frame_metrics, "📊 Frame-Level Metrics")

        # イベント単位の評価（複数のIoU閾値で）
        iou_thresholds = [0.3, 0.5, 0.7]
        event_metrics_list = []

        for iou_threshold in iou_thresholds:
            event_metrics = self.calculator.calculate_event_level_metrics(
                self.ground_truth, predictions, iou_threshold, event_type
            )
            event_metrics_list.append(event_metrics)
            self.calculator.print_metrics(event_metrics, f"🎯 Event-Level Metrics (IoU >= {iou_threshold})")

        # 時間的重なりの統計
        overlap_stats = self.calculator.calculate_temporal_overlap(self.ground_truth, predictions, event_type)
        self.calculator.print_metrics(overlap_stats, "⏱️  Temporal Overlap Statistics")

        return {
            "method_name": method_name,
            "frame_metrics": frame_metrics,
            "event_metrics_list": event_metrics_list,
            "overlap_stats": overlap_stats,
        }

    def compare_methods(self, mediapipe_csv: str = None, yolo11_csv: str = None, event_type: str = None) -> dict:
        """
        複数の検出手法を比較

        Args:
            mediapipe_csv: MediaPipeの結果CSVパス
            yolo11_csv: YOLO11の結果CSVパス
            event_type: 評価対象のイベントタイプ

        Returns:
            比較結果の辞書
        """
        results = {}

        # MediaPipeの評価
        if mediapipe_csv and Path(mediapipe_csv).exists():
            mediapipe_predictions = self.loader.load_mediapipe_results(mediapipe_csv, self.fps)
            results["mediapipe"] = self.evaluate_method(mediapipe_predictions, "MediaPipe", event_type)

        # YOLO11の評価
        if yolo11_csv and Path(yolo11_csv).exists():
            yolo11_predictions = self.loader.load_yolo11_results(yolo11_csv, self.fps)
            results["yolo11"] = self.evaluate_method(yolo11_predictions, "YOLO11", event_type)

        return results

    def generate_comparison_table(self, results: dict) -> str:
        """
        比較表をMarkdown形式で生成

        Args:
            results: compare_methodsの結果

        Returns:
            Markdown形式の比較表
        """
        lines = []
        lines.append("# 検出手法の比較結果")
        lines.append("")
        lines.append(f"**正解ラベル:** `{self.ground_truth_csv}`")
        lines.append(f"**評価日時:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**総フレーム数:** {self.total_frames}")
        lines.append(f"**FPS:** {self.fps:.2f}")
        lines.append("")

        # フレーム単位の比較表
        lines.append("## 📊 フレーム単位の評価")
        lines.append("")
        lines.append("| 手法 | Precision | Recall | F1-Score | Accuracy |")
        lines.append("|------|-----------|--------|----------|----------|")

        for _, result in results.items():
            metrics = result["frame_metrics"]
            lines.append(
                f"| {result['method_name']} | "
                f"{metrics['precision']:.2%} | "
                f"{metrics['recall']:.2%} | "
                f"{metrics['f1_score']:.2%} | "
                f"{metrics['accuracy']:.2%} |"
            )

        lines.append("")

        # イベント単位の比較表（IoU=0.5）
        lines.append("## 🎯 イベント単位の評価 (IoU >= 0.5)")
        lines.append("")
        lines.append("| 手法 | Precision | Recall | F1-Score | Mean IoU |")
        lines.append("|------|-----------|--------|----------|----------|")

        for _, result in results.items():
            # IoU=0.5の結果を取得
            event_metrics = result["event_metrics_list"][1]  # [0.3, 0.5, 0.7]の中間
            lines.append(
                f"| {result['method_name']} | "
                f"{event_metrics['precision']:.2%} | "
                f"{event_metrics['recall']:.2%} | "
                f"{event_metrics['f1_score']:.2%} | "
                f"{event_metrics['mean_iou']:.2%} |"
            )

        lines.append("")

        # 時間的重なりの比較表
        lines.append("## ⏱️  時間的重なり")
        lines.append("")
        lines.append("| 手法 | 正解総時間(秒) | 予測総時間(秒) | 重なり時間(秒) | 重なり率(正解基準) |")
        lines.append("|------|---------------|---------------|---------------|-------------------|")

        for _, result in results.items():
            stats = result["overlap_stats"]
            lines.append(
                f"| {result['method_name']} | "
                f"{stats['gt_total_duration']:.2f} | "
                f"{stats['pred_total_duration']:.2f} | "
                f"{stats['overlap_duration']:.2f} | "
                f"{stats['overlap_ratio_gt']:.2%} |"
            )

        lines.append("")

        # 詳細な評価指標
        lines.append("## 📈 詳細な評価指標")
        lines.append("")

        for _, result in results.items():
            lines.append(f"### {result['method_name']}")
            lines.append("")

            # フレーム単位
            lines.append("#### フレーム単位")
            metrics = result["frame_metrics"]
            lines.append(f"- True Positive: {metrics['true_positive']}")
            lines.append(f"- False Positive: {metrics['false_positive']}")
            lines.append(f"- False Negative: {metrics['false_negative']}")
            lines.append(f"- True Negative: {metrics['true_negative']}")
            lines.append(f"- Positive Frames (Ground Truth): {metrics['positive_frames']}")
            lines.append(f"- Predicted Positive Frames: {metrics['predicted_positive_frames']}")
            lines.append("")

            # イベント単位（各IoU閾値）
            lines.append("#### イベント単位")
            for event_metrics in result["event_metrics_list"]:
                iou_th = event_metrics["iou_threshold"]
                lines.append(f"**IoU >= {iou_th}:**")
                lines.append(f"- True Positive: {event_metrics['true_positive']}")
                lines.append(f"- False Positive: {event_metrics['false_positive']}")
                lines.append(f"- False Negative: {event_metrics['false_negative']}")
                lines.append(f"- Precision: {event_metrics['precision']:.2%}")
                lines.append(f"- Recall: {event_metrics['recall']:.2%}")
                lines.append(f"- F1-Score: {event_metrics['f1_score']:.2%}")
                lines.append("")

            lines.append("")

        return "\n".join(lines)

    def save_report(self, results: dict, output_path: str):
        """
        評価レポートを保存

        Args:
            results: compare_methodsの結果
            output_path: 出力ファイルパス
        """
        # 出力ディレクトリを作成
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Markdown形式で保存
        report = self.generate_comparison_table(results)
        output_path.write_text(report, encoding="utf-8")

        print(f"\n✅ 評価レポートを保存しました: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="検出結果を評価し、比較レポートを生成します")
    parser.add_argument("ground_truth", help="正解ラベルのCSVファイル")
    parser.add_argument("--video", help="動画ファイルパス（総フレーム数とFPSを取得）")
    parser.add_argument("--mediapipe", help="MediaPipeの検出結果CSVファイル")
    parser.add_argument("--yolo11", help="YOLO11の検出結果CSVファイル")
    parser.add_argument("--event-type", help="評価対象のイベントタイプ（例: HEAD_SHAKE）")
    parser.add_argument("--output", "-o", default="evaluation_report.md", help="出力レポートファイル（Markdown形式）")

    args = parser.parse_args()

    # 正解ラベルの存在確認
    if not Path(args.ground_truth).exists():
        print(f"エラー: 正解ラベルファイルが見つかりません: {args.ground_truth}")
        sys.exit(1)

    # 評価実行
    evaluator = DetectionEvaluator(args.ground_truth, args.video)

    results = evaluator.compare_methods(
        mediapipe_csv=args.mediapipe, yolo11_csv=args.yolo11, event_type=args.event_type
    )

    if not results:
        print("\n⚠️  評価対象の検出結果が見つかりませんでした")
        print("   --mediapipe または --yolo11 オプションで検出結果CSVを指定してください")
        sys.exit(1)

    # レポート保存
    evaluator.save_report(results, args.output)

    print("\n" + "=" * 80)
    print("評価完了！")
    print("=" * 80)


if __name__ == "__main__":
    main()
