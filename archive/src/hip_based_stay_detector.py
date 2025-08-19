#!/usr/bin/env python3
"""
腰の位置を基準にした改良版長期滞在検知システム

MediaPipeの関節データ（腰の位置）を使用して、
姿勢変化に対してより耐性のある長期滞在検知を実現する。
"""

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark

from pose_estimator import PoseEstimator


@dataclass
class HipBasedStayInfo:
    """腰の位置ベースの滞在情報"""

    track_id: int
    hip_center: tuple[float, float]  # 腰の中心座標
    last_update_time: float
    stay_start_time: float
    stay_duration: float
    notified: bool
    hip_position_history: list[tuple[float, tuple[float, float]]]  # (time, position)
    confidence_score: float  # 腰の検出信頼度


class HipBasedStayDetector:
    """腰の位置を基準とした滞在検知器"""

    def __init__(
        self,
        move_threshold: float = 25.0,  # 分析結果より少し厳しく設定
        stay_threshold: float = 5.0,
        history_duration: float = 2.0,
        min_confidence: float = 0.7,
    ):
        self.move_threshold = move_threshold
        self.stay_threshold = stay_threshold
        self.history_duration = history_duration
        self.min_confidence = min_confidence

        self.pose_estimator = PoseEstimator()
        self.stay_info: dict[int, HipBasedStayInfo] = {}
        self.next_track_id = 1

    def extract_hip_position(self, landmarks: np.ndarray) -> tuple[float, float, float] | None:
        """
        MediaPipeランドマークから腰の中心位置を抽出

        Returns:
            (center_x, center_y, confidence) or None
        """
        if landmarks is None:
            return None

        try:
            left_hip = landmarks[PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[PoseLandmark.RIGHT_HIP.value]

            # 両方の腰が十分に見えているかチェック
            min_visibility = min(left_hip[3], right_hip[3])
            if min_visibility < self.min_confidence:
                return None

            # 中心座標を計算（正規化座標）
            center_x = (left_hip[0] + right_hip[0]) / 2
            center_y = (left_hip[1] + right_hip[1]) / 2

            return (center_x, center_y, min_visibility)

        except (IndexError, AttributeError):
            return None

    def update_stay_detection(self, frame: np.ndarray, current_time: float) -> list[dict]:
        """
        フレームを処理して滞在検知を更新

        Returns:
            通知リスト
        """
        height, width = frame.shape[:2]

        # MediaPipeで姿勢推定
        landmarks = self.pose_estimator.estimate(frame)

        notifications = []

        if landmarks is not None:
            hip_data = self.extract_hip_position(landmarks)

            if hip_data is not None:
                center_x, center_y, confidence = hip_data

                # 正規化座標を画像座標に変換
                pixel_x = center_x * width
                pixel_y = center_y * height
                hip_center = (pixel_x, pixel_y)

                # 既存の追跡との照合
                matched_track_id = self._match_hip_position(hip_center, current_time)

                if matched_track_id is None:
                    # 新しい人物として追跡開始
                    track_id = self.next_track_id
                    self.next_track_id += 1

                    self.stay_info[track_id] = HipBasedStayInfo(
                        track_id=track_id,
                        hip_center=hip_center,
                        last_update_time=current_time,
                        stay_start_time=current_time,
                        stay_duration=0.0,
                        notified=False,
                        hip_position_history=[(current_time, hip_center)],
                        confidence_score=confidence,
                    )
                else:
                    # 既存の追跡を更新
                    notifications.extend(
                        self._update_existing_track(matched_track_id, hip_center, current_time, confidence)
                    )

        # 古い追跡を削除
        self._cleanup_old_tracks(current_time)

        return notifications

    def _match_hip_position(self, hip_center: tuple[float, float], current_time: float) -> int | None:
        """
        現在の腰の位置を既存の追跡と照合
        """
        best_match_id = None
        min_distance = float("inf")

        for track_id, info in self.stay_info.items():
            # 最近更新された追跡のみ考慮
            if current_time - info.last_update_time > 2.0:
                continue

            # 距離を計算
            distance = np.sqrt((hip_center[0] - info.hip_center[0]) ** 2 + (hip_center[1] - info.hip_center[1]) ** 2)

            # 適切な距離内の最も近い追跡を選択
            if distance < 50.0 and distance < min_distance:  # 50px以内
                min_distance = distance
                best_match_id = track_id

        return best_match_id

    def _update_existing_track(
        self, track_id: int, hip_center: tuple[float, float], current_time: float, confidence: float
    ) -> list[dict]:
        """
        既存の追跡を更新
        """
        info = self.stay_info[track_id]
        notifications = []

        # 位置履歴を更新
        info.hip_position_history.append((current_time, hip_center))

        # 古い履歴を削除
        info.hip_position_history = [
            (t, pos) for t, pos in info.hip_position_history if current_time - t <= self.history_duration
        ]

        # 移動距離を計算（過去の平均位置との比較）
        if len(info.hip_position_history) >= 2:
            # 過去1秒間の平均位置を計算
            recent_positions = [pos for t, pos in info.hip_position_history if current_time - t <= 1.0]

            if recent_positions:
                avg_x = np.mean([pos[0] for pos in recent_positions])
                avg_y = np.mean([pos[1] for pos in recent_positions])
                avg_position = (avg_x, avg_y)

                # 現在位置との距離
                distance = np.sqrt((hip_center[0] - avg_position[0]) ** 2 + (hip_center[1] - avg_position[1]) ** 2)

                # 移動判定
                if distance < self.move_threshold:
                    # 滞在継続
                    time_diff = current_time - info.last_update_time
                    info.stay_duration += time_diff
                else:
                    # 移動と判定 - 滞在時間をリセット
                    info.stay_start_time = current_time
                    info.stay_duration = 0.0
                    info.notified = False

        # 位置と時刻を更新
        info.hip_center = hip_center
        info.last_update_time = current_time
        info.confidence_score = max(info.confidence_score, confidence)

        # 長期滞在の通知チェック
        if (
            info.stay_duration >= self.stay_threshold
            and not info.notified
            and info.confidence_score >= self.min_confidence
        ):
            notifications.append(
                {
                    "type": "long_stay_detected",
                    "track_id": track_id,
                    "duration": info.stay_duration,
                    "position": hip_center,
                    "confidence": info.confidence_score,
                    "timestamp": current_time,
                }
            )
            info.notified = True

        return notifications

    def _cleanup_old_tracks(self, current_time: float, timeout: float = 5.0):
        """古い追跡を削除"""
        to_remove = []
        for track_id, info in self.stay_info.items():
            if current_time - info.last_update_time > timeout:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.stay_info[track_id]

    def get_current_stays(self) -> list[dict]:
        """現在滞在中の人物情報を取得"""
        current_stays = []
        for track_id, info in self.stay_info.items():
            current_stays.append(
                {
                    "track_id": track_id,
                    "position": info.hip_center,
                    "duration": info.stay_duration,
                    "confidence": info.confidence_score,
                    "is_long_stay": info.stay_duration >= self.stay_threshold,
                }
            )
        return current_stays

    def draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """検知結果を描画"""

        for track_id, info in self.stay_info.items():
            x, y = info.hip_center
            x, y = int(x), int(y)

            # 滞在時間に応じて色を変更
            if info.stay_duration >= self.stay_threshold:
                color = (0, 0, 255)  # 赤: 長期滞在
                thickness = 3
            else:
                color = (0, 255, 255)  # 黄: 通常滞在
                thickness = 2

            # 腰の位置にマーカーを描画
            cv2.circle(frame, (x, y), 8, color, thickness)

            # ID と滞在時間を表示
            label = f"ID:{track_id} {info.stay_duration:.1f}s"
            cv2.putText(frame, label, (x - 30, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness - 1)

            # 信頼度を表示
            conf_label = f"conf:{info.confidence_score:.2f}"
            cv2.putText(frame, conf_label, (x - 30, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    def close(self):
        """リソースを解放"""
        self.pose_estimator.close()


def process_video_with_hip_detection(
    input_path: str,
    output_path: str,
    move_threshold: float = 25.0,
    stay_threshold: float = 5.0,
    enable_display: bool = True,
):
    """
    ビデオファイルを処理して腰ベースの滞在検知を実行
    """

    detector = HipBasedStayDetector(move_threshold=move_threshold, stay_threshold=stay_threshold)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    # ビデオ情報取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 出力ビデオ設定
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # ログファイル設定
    log_path = Path(output_path).with_suffix(".csv")

    frame_idx = 0
    start_time = time.time()

    try:
        with open(log_path, "w", newline="", encoding="utf-8") as logfile:
            log_writer = csv.writer(logfile)
            log_writer.writerow(
                [
                    "frame",
                    "timestamp",
                    "track_id",
                    "position_x",
                    "position_y",
                    "stay_duration",
                    "confidence",
                    "is_long_stay",
                    "notification",
                ]
            )

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                video_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                # 滞在検知実行
                notifications = detector.update_stay_detection(frame, video_timestamp)

                # 現在の滞在情報取得
                current_stays = detector.get_current_stays()

                # ログに記録
                for stay in current_stays:
                    notification_text = ""
                    for notif in notifications:
                        if notif["track_id"] == stay["track_id"]:
                            notification_text = f"Long stay detected: {notif['duration']:.1f}s"
                            break

                    log_writer.writerow(
                        [
                            frame_idx,
                            video_timestamp,
                            stay["track_id"],
                            stay["position"][0],
                            stay["position"][1],
                            stay["duration"],
                            stay["confidence"],
                            stay["is_long_stay"],
                            notification_text,
                        ]
                    )

                # 通知を表示
                for notif in notifications:
                    print(
                        f"Frame {frame_idx}: {notif['type']} - ID {notif['track_id']}, "
                        f"Duration: {notif['duration']:.1f}s, Pos: {notif['position']}"
                    )

                # 描画
                frame = detector.draw_detections(frame)

                # 情報テキスト追加
                info_text = f"Frame: {frame_idx}/{frame_count} | Hip-based Stay Detection"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                active_count = len([s for s in current_stays if s["duration"] > 0])
                long_stay_count = len([s for s in current_stays if s["is_long_stay"]])
                status_text = f"Active: {active_count} | Long stays: {long_stay_count}"
                cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # 出力書き込み
                out.write(frame)

                # 表示
                if enable_display:
                    cv2.imshow("Hip-based Stay Detection", frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        break

                # 進捗表示
                if frame_idx % 30 == 0:
                    elapsed = time.time() - start_time
                    processing_fps = frame_idx / elapsed if elapsed > 0 else 0
                    print(
                        f"Progress: {frame_idx}/{frame_count} ({frame_idx / frame_count * 100:.1f}%) "
                        f"| Processing FPS: {processing_fps:.1f}"
                    )

    finally:
        cap.release()
        out.release()
        if enable_display:
            cv2.destroyAllWindows()
        detector.close()

        print("\n処理完了!")
        print(f"出力ビデオ: {output_path}")
        print(f"ログファイル: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="腰の位置を基準にした長期滞在検知")
    parser.add_argument("--input", required=True, help="入力ビデオファイル")
    parser.add_argument("--output", default="output/hip_based_stay_detection.mp4", help="出力ビデオファイル")
    parser.add_argument("--move-threshold", type=float, default=25.0, help="移動判定の閾値 (ピクセル)")
    parser.add_argument("--stay-threshold", type=float, default=5.0, help="長期滞在判定の閾値 (秒)")
    parser.add_argument("--no-display", action="store_true", help="リアルタイム表示を無効化")

    args = parser.parse_args()

    # 出力ディレクトリ作成
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    process_video_with_hip_detection(
        input_path=args.input,
        output_path=args.output,
        move_threshold=args.move_threshold,
        stay_threshold=args.stay_threshold,
        enable_display=not args.no_display,
    )


if __name__ == "__main__":
    main()
