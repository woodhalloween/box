"""
比較評価クラス
3つのID付与手法を並列で実行して比較評価する
"""

import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comparison.id_methods import BaseTracker, PerformanceMetrics, Track
from comparison.id_methods.bytetrack_wrapper import ByteTrackWrapper
from comparison.id_methods.yolo_simple_tracker import YoloSimpleTracker


@dataclass
class ComparisonResult:
    """比較結果を表すデータクラス"""

    frame_number: int
    timestamp: float
    method_results: dict[str, dict[str, Any]]


@dataclass
class TrackingMetrics:
    """追跡メトリクスを表すデータクラス"""

    method_name: str
    total_frames: int
    avg_processing_time_ms: float
    avg_fps: float
    avg_memory_usage_mb: float
    total_detections: int
    total_tracks: int
    id_switches: int
    track_consistency: float


class ComparisonEvaluator:
    """
    複数のトラッカーを並列実行して比較評価を行うクラス
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.3,
        device: str = "",
        output_dir: str = "comparison/output",
    ):
        """
        比較評価器の初期化

        Args:
            model_path: YOLOモデルのパス
            confidence: 検出信頼度閾値
            device: 実行デバイス
            output_dir: 出力ディレクトリ
        """
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.output_dir = output_dir

        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)

        # トラッカーの初期化
        self.trackers = self._initialize_trackers()

        # 結果保存用
        self.comparison_results: list[ComparisonResult] = []
        self.tracking_metrics: dict[str, TrackingMetrics] = {}

        # 前フレームのトラック情報（ID一貫性計算用）
        self.prev_tracks: dict[str, list[Track]] = {}
        self.id_switch_counts: dict[str, int] = dict.fromkeys(self.trackers.keys(), 0)

    def _initialize_trackers(self) -> dict[str, BaseTracker]:
        """
        各トラッカーを初期化

        Returns:
            トラッカー辞書
        """
        trackers = {}

        # YOLO簡易トラッカー
        trackers["YOLO_Simple"] = YoloSimpleTracker(
            model_path=self.model_path,
            confidence=self.confidence,
            device=self.device,
            distance_threshold=50.0,
            max_disappeared_frames=10,
        )

        # ByteTrackラッパー
        trackers["ByteTrack"] = ByteTrackWrapper(
            model_path=self.model_path,
            confidence=self.confidence,
            device=self.device,
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=30,
        )

        # 今回はYOLO高度版は省略（実装が複雑なため）

        return trackers

    def process_frame(self, frame: np.ndarray, frame_number: int) -> ComparisonResult:
        """
        1フレームを全トラッカーで処理

        Args:
            frame: 入力フレーム
            frame_number: フレーム番号

        Returns:
            比較結果
        """
        timestamp = time.time()
        method_results = {}

        for method_name, tracker in self.trackers.items():
            try:
                # トラッキング実行
                start_time = time.time()
                tracks, metrics = tracker.detect_and_track(frame)
                processing_time = (time.time() - start_time) * 1000

                # ID一貫性の計算
                id_switches = self._calculate_id_switches(method_name, tracks)

                # 結果の保存
                method_results[method_name] = {
                    "tracks": tracks,
                    "metrics": metrics,
                    "processing_time_ms": processing_time,
                    "id_switches": id_switches,
                    "track_count": len(tracks),
                    "detection_count": metrics.objects_detected,
                }

                # 前フレームのトラック情報を保存
                self.prev_tracks[method_name] = tracks.copy()

            except Exception as e:
                print(f"エラー - {method_name}: {str(e)}")
                method_results[method_name] = {
                    "tracks": [],
                    "metrics": PerformanceMetrics(),
                    "processing_time_ms": 0.0,
                    "id_switches": 0,
                    "track_count": 0,
                    "detection_count": 0,
                    "error": str(e),
                }

        result = ComparisonResult(
            frame_number=frame_number, timestamp=timestamp, method_results=method_results
        )

        self.comparison_results.append(result)
        return result

    def _calculate_id_switches(self, method_name: str, current_tracks: list[Track]) -> int:
        """
        ID切り替えの発生回数を計算

        Args:
            method_name: トラッカー名
            current_tracks: 現在フレームのトラック

        Returns:
            ID切り替え回数
        """
        if method_name not in self.prev_tracks:
            return 0

        prev_tracks = self.prev_tracks[method_name]
        switches = 0

        # 簡易的なID切り替え検出
        # 位置ベースで同一人物と思われるトラック間でIDが変わったかをチェック
        for curr_track in current_tracks:
            best_match_distance = float("inf")
            best_match_id = None

            for prev_track in prev_tracks:
                # 中心点間距離を計算
                curr_center = (
                    (curr_track.bbox[0] + curr_track.bbox[2]) / 2,
                    (curr_track.bbox[1] + curr_track.bbox[3]) / 2,
                )
                prev_center = (
                    (prev_track.bbox[0] + prev_track.bbox[2]) / 2,
                    (prev_track.bbox[1] + prev_track.bbox[3]) / 2,
                )

                distance = np.sqrt(
                    (curr_center[0] - prev_center[0]) ** 2 + (curr_center[1] - prev_center[1]) ** 2
                )

                if distance < best_match_distance:
                    best_match_distance = distance
                    best_match_id = prev_track.track_id

            # 近距離で最適マッチが見つかったが、IDが異なる場合はスイッチと判定
            if (
                best_match_distance < 100
                and best_match_id is not None
                and best_match_id != curr_track.track_id
            ):
                switches += 1

        self.id_switch_counts[method_name] += switches
        return switches

    def process_video(
        self, video_path: str, max_frames: int | None = None, display: bool = False
    ) -> list[ComparisonResult]:
        """
        動画全体を処理して比較評価を実行

        Args:
            video_path: 動画ファイルのパス
            max_frames: 処理する最大フレーム数
            display: リアルタイム表示するかどうか

        Returns:
            比較結果のリスト
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"動画ファイルを開けません: {video_path}")

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"動画処理開始: {video_path}")
        print(f"総フレーム数: {total_frames}")
        if max_frames:
            print(f"処理フレーム数: {max_frames}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames and frame_count >= max_frames:
                    break

                # フレーム処理
                result = self.process_frame(frame, frame_count)

                # プログレス表示
                if frame_count % 30 == 0:
                    print(
                        f"処理済みフレーム: {frame_count}/{total_frames if not max_frames else max_frames}"
                    )

                # リアルタイム表示（オプション）
                if display:
                    display_frame = self._create_comparison_display(frame, result)
                    cv2.imshow("Tracking Comparison", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1

        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

        print(f"動画処理完了: {frame_count}フレーム処理")
        return self.comparison_results

    def _create_comparison_display(self, frame: np.ndarray, result: ComparisonResult) -> np.ndarray:
        """
        比較表示用のフレームを作成

        Args:
            frame: 元フレーム
            result: 比較結果

        Returns:
            表示用フレーム
        """
        # 簡易的な横並び表示
        method_names = list(self.trackers.keys())
        num_methods = len(method_names)

        # フレームを縮小してコピー
        h, w = frame.shape[:2]
        display_w = w // num_methods
        display_h = h

        display_frame = np.zeros((display_h, display_w * num_methods, 3), dtype=np.uint8)

        for i, method_name in enumerate(method_names):
            # フレームのコピーとリサイズ
            method_frame = frame.copy()
            method_frame = cv2.resize(method_frame, (display_w, display_h))

            # トラック情報の描画
            if method_name in result.method_results:
                tracks = result.method_results[method_name]["tracks"]
                for track in tracks:
                    # バウンディングボックスをリサイズに合わせて調整
                    x1 = int(track.bbox[0] * display_w / w)
                    y1 = int(track.bbox[1] * display_h / h)
                    x2 = int(track.bbox[2] * display_w / w)
                    y2 = int(track.bbox[3] * display_h / h)

                    # 描画
                    cv2.rectangle(method_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        method_frame,
                        f"ID:{track.track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

            # 手法名とメトリクス表示
            cv2.putText(
                method_frame,
                method_name,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if method_name in result.method_results:
                metrics = result.method_results[method_name]["metrics"]
                cv2.putText(
                    method_frame,
                    f"FPS: {metrics.fps:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            # 表示フレームに配置
            display_frame[:, i * display_w : (i + 1) * display_w] = method_frame

        return display_frame

    def calculate_final_metrics(self) -> dict[str, TrackingMetrics]:
        """
        最終的な追跡メトリクスを計算

        Returns:
            各手法のメトリクス辞書
        """
        metrics = {}

        for method_name in self.trackers.keys():
            # 各フレームの結果から統計を計算
            processing_times = []
            fps_values = []
            memory_usages = []
            detection_counts = []
            track_counts = []

            for result in self.comparison_results:
                if method_name in result.method_results:
                    method_result = result.method_results[method_name]
                    processing_times.append(method_result["processing_time_ms"])

                    if "metrics" in method_result:
                        metrics_data = method_result["metrics"]
                        fps_values.append(metrics_data.fps)
                        memory_usages.append(metrics_data.memory_usage_mb)
                        detection_counts.append(metrics_data.objects_detected)
                        track_counts.append(method_result["track_count"])

            # トラック一貫性の計算（簡易版）
            total_switches = self.id_switch_counts[method_name]
            total_detections = sum(detection_counts) if detection_counts else 0
            track_consistency = max(0, 1.0 - (total_switches / max(total_detections, 1)))

            metrics[method_name] = TrackingMetrics(
                method_name=method_name,
                total_frames=len(self.comparison_results),
                avg_processing_time_ms=float(np.mean(processing_times))
                if processing_times
                else 0.0,
                avg_fps=float(np.mean(fps_values)) if fps_values else 0.0,
                avg_memory_usage_mb=float(np.mean(memory_usages)) if memory_usages else 0.0,
                total_detections=total_detections,
                total_tracks=sum(track_counts) if track_counts else 0,
                id_switches=total_switches,
                track_consistency=track_consistency,
            )

        self.tracking_metrics = metrics
        return metrics

    def save_results(self, prefix: str = "comparison"):
        """
        結果をファイルに保存

        Args:
            prefix: ファイル名のプレフィックス
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 詳細結果をCSVで保存
        csv_path = os.path.join(self.output_dir, f"{prefix}_results_{timestamp}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # ヘッダー
            header = ["frame_number", "timestamp"]
            for method_name in self.trackers.keys():
                header.extend(
                    [
                        f"{method_name}_processing_time_ms",
                        f"{method_name}_fps",
                        f"{method_name}_track_count",
                        f"{method_name}_detection_count",
                        f"{method_name}_id_switches",
                    ]
                )
            writer.writerow(header)

            # データ
            for result in self.comparison_results:
                row = [result.frame_number, result.timestamp]
                for method_name in self.trackers.keys():
                    if method_name in result.method_results:
                        method_result = result.method_results[method_name]
                        row.extend(
                            [
                                method_result["processing_time_ms"],
                                method_result["metrics"].fps,
                                method_result["track_count"],
                                method_result["detection_count"],
                                method_result["id_switches"],
                            ]
                        )
                    else:
                        row.extend([0, 0, 0, 0, 0])
                writer.writerow(row)

        # サマリーをJSONで保存
        json_path = os.path.join(self.output_dir, f"{prefix}_summary_{timestamp}.json")
        summary_data = {
            "tracking_metrics": {
                name: asdict(metrics) for name, metrics in self.tracking_metrics.items()
            },
            "comparison_info": {
                "total_frames": len(self.comparison_results),
                "model_path": self.model_path,
                "confidence": self.confidence,
                "methods_compared": list(self.trackers.keys()),
            },
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        print("結果保存完了:")
        print(f"  詳細結果: {csv_path}")
        print(f"  サマリー: {json_path}")

    def print_summary(self):
        """
        結果サマリーをコンソールに表示
        """
        if not self.tracking_metrics:
            self.calculate_final_metrics()

        print("\n" + "=" * 80)
        print("YOLO vs ByteTrack ID付与比較結果")
        print("=" * 80)

        for method_name, metrics in self.tracking_metrics.items():
            print(f"\n【{method_name}】")
            print(f"  処理フレーム数: {metrics.total_frames}")
            print(f"  平均処理時間: {metrics.avg_processing_time_ms:.2f} ms")
            print(f"  平均FPS: {metrics.avg_fps:.2f}")
            print(f"  平均メモリ使用量: {metrics.avg_memory_usage_mb:.2f} MB")
            print(f"  総検出数: {metrics.total_detections}")
            print(f"  総トラック数: {metrics.total_tracks}")
            print(f"  ID切り替え回数: {metrics.id_switches}")
            print(f"  トラック一貫性: {metrics.track_consistency:.3f}")

        print("\n" + "=" * 80)

    def reset(self):
        """
        評価器の状態をリセット
        """
        for tracker in self.trackers.values():
            tracker.reset()

        self.comparison_results = []
        self.tracking_metrics = {}
        self.prev_tracks = {}
        self.id_switch_counts = dict.fromkeys(self.trackers.keys(), 0)
