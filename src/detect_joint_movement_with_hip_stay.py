"""src/detect_joint_movement_with_hip_stay.py

統合版：骨格推定・前傾姿勢分析 + 腰の位置を基準とした長期滞在検知
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import IO, Any

import cv2
import mediapipe as mp
import numpy as np

from .definitions import Angle, MovementState
from .drawing_utils import draw_analysis_results, draw_landmarks
from .head_shake_detector import HeadShakeDetector
from .movement_analyzer import MovementAnalyzer
from .pose.definitions import BodyPart
from .pose_estimator import PoseEstimator

print("--- デバッグ: detect_joint_movement_with_hip_stay.py の実行開始 ---")

# OpenCVの型スタブがない環境での静的解析エラー回避用にAnyとして扱う
# cv2m: Any = cv2  <- この行を削除し、直接cv2を使用します


@dataclass
class PostureSnapshot:
    """姿勢スナップショット - 特定時刻の姿勢データ"""

    timestamp: float
    frame_number: int
    analysis_results: dict[Angle, dict[str, Any]]
    is_forward_leaning: bool
    forward_lean_score: float


@dataclass
class HipStayInfo:
    """腰の位置ベースの滞在情報"""

    last_hip_pos: tuple[float, float]  # 腰の中心座標
    last_update_time: float
    stay_start_time: float
    stay_duration: float
    notified: bool
    confidence_score: float  # 腰の検出信頼度


class PostureMonitor:
    """前傾姿勢の長期滞在を監視するクラス"""

    def __init__(self, monitoring_duration: float = 60.0, alert_threshold: float = 0.7):
        """
        Args:
            monitoring_duration: 監視期間（秒）
            alert_threshold: アラート閾値（0.0-1.0, 前傾姿勢の割合）
        """
        self.monitoring_duration = monitoring_duration
        self.alert_threshold = alert_threshold
        self.posture_history: deque[PostureSnapshot] = deque()
        self.last_alert_time: float = 0
        self.alert_cooldown: float = 30.0  # アラート間隔（秒）

    def update(self, timestamp: float, frame_number: int, analysis_results: dict) -> list[str]:
        """
        姿勢データを更新し、必要に応じてアラートを生成

        Args:
            timestamp: タイムスタンプ
            frame_number: フレーム番号
            analysis_results: 関節分析結果

        Returns:
            アラートメッセージのリスト
        """
        # 前傾姿勢判定
        is_forward, score = self.is_forward_leaning_posture(analysis_results)

        # スナップショット作成
        snapshot = PostureSnapshot(
            timestamp=timestamp,
            frame_number=frame_number,
            analysis_results=analysis_results.copy(),
            is_forward_leaning=is_forward,
            forward_lean_score=score,
        )

        # 履歴に追加
        self.posture_history.append(snapshot)

        # 古いデータを削除（監視期間外）
        while self.posture_history and timestamp - self.posture_history[0].timestamp > self.monitoring_duration:
            self.posture_history.popleft()

        # アラートチェック
        return self._check_for_alerts(timestamp)

    def is_forward_leaning_posture(self, analysis_results: dict) -> tuple[bool, float]:
        """
        前傾姿勢かどうかを判定

        Args:
            analysis_results: 関節分析結果

        Returns:
            (is_forward_leaning, confidence_score)
        """
        forward_indicators = []

        # 体の傾き角度チェック
        body_tilt = analysis_results.get(Angle.BODY_TILT)
        if body_tilt and "angle" in body_tilt:
            tilt_angle = body_tilt["angle"]
            # 体の傾きが150度以下の場合は前傾の可能性
            if tilt_angle <= 150:
                forward_indicators.append(1.0 - (tilt_angle / 150.0))
            else:
                forward_indicators.append(0.0)

        # 首・胴体角度チェック
        neck_trunk = analysis_results.get(Angle.NECK_TRUNK_ANGLE)
        if neck_trunk and "angle" in neck_trunk:
            neck_angle = neck_trunk["angle"]
            # 首が前に出ている状態（150度以下）
            if neck_angle <= 150:
                forward_indicators.append(1.0 - (neck_angle / 150.0))
            else:
                forward_indicators.append(0.0)

        # 肩の前方傾斜チェック
        right_shoulder = analysis_results.get(Angle.RIGHT_SHOULDER)
        left_shoulder = analysis_results.get(Angle.LEFT_SHOULDER)

        shoulder_flexion_count = 0
        shoulder_total = 0

        for shoulder in [right_shoulder, left_shoulder]:
            if shoulder and "state" in shoulder:
                shoulder_total += 1
                if shoulder["state"] == MovementState.FLEXION:
                    shoulder_flexion_count += 1

        if shoulder_total > 0:
            shoulder_flexion_ratio = shoulder_flexion_count / shoulder_total
            forward_indicators.append(shoulder_flexion_ratio)

        # 前傾スコア計算
        if forward_indicators:
            forward_score = sum(forward_indicators) / len(forward_indicators)
            is_leaning = forward_score > 0.5  # 50%以上で前傾と判定
            return is_leaning, forward_score

        return False, 0.0

    def _check_for_alerts(self, current_time: float) -> list[str]:
        """アラート条件をチェック"""
        alerts = []

        # クールダウン期間中はアラートしない（ただし初回アラートは除く）
        if self.last_alert_time > 0 and current_time - self.last_alert_time < self.alert_cooldown:
            return alerts

        # 監視期間に達していない場合はチェックしない
        if not self.posture_history:
            return alerts

        oldest_timestamp = self.posture_history[0].timestamp
        if current_time - oldest_timestamp < self.monitoring_duration:
            return alerts

        # 前傾姿勢の割合を計算
        forward_leaning_count = sum(1 for snapshot in self.posture_history if snapshot.is_forward_leaning)
        total_count = len(self.posture_history)

        if total_count > 0:
            forward_ratio = forward_leaning_count / total_count

            if forward_ratio >= self.alert_threshold:
                # 平均前傾スコア計算
                avg_score = sum(snapshot.forward_lean_score for snapshot in self.posture_history) / total_count

                alert_msg = (
                    f"[!] Forward Leaning: {self.monitoring_duration:.0f}s {forward_ratio:.1%} (Score: {avg_score:.2f})"
                )
                alerts.append(alert_msg)

                self.last_alert_time = current_time

        return alerts

    def get_status(self) -> dict[str, Any]:
        """現在の監視状態を取得"""
        if not self.posture_history:
            return {
                "monitoring_duration": 0,
                "forward_ratio": 0,
                "avg_score": 0,
                "sample_count": 0,
            }

        oldest_timestamp = self.posture_history[0].timestamp
        latest_timestamp = self.posture_history[-1].timestamp
        monitoring_duration = latest_timestamp - oldest_timestamp

        forward_count = sum(1 for s in self.posture_history if s.is_forward_leaning)
        total_count = len(self.posture_history)
        forward_ratio = forward_count / total_count if total_count > 0 else 0

        avg_score = sum(s.forward_lean_score for s in self.posture_history) / total_count if total_count > 0 else 0

        return {
            "monitoring_duration": monitoring_duration,
            "forward_ratio": forward_ratio,
            "avg_score": avg_score,
            "sample_count": total_count,
        }


class HipStayState(Enum):
    """腰ベース滞在検知の状態"""

    STAYING = auto()
    POTENTIAL_MOVE = auto()


class HipBasedStayDetector:
    """腰の位置を基準とし、複合条件と猶予期間を用いて滞在を検知する"""

    def __init__(
        self,
        stay_threshold_sec: float = 5.0,
        confidence_threshold: float = 0.5,
        # --- Idea 5 parameters ---
        spike_threshold: float = 1.5,
        spike_window_sec: float = 1.0,
        stability_threshold_px: float = 50.0,
        stability_window_sec: float = 2.0,
        # --- Grace period parameters ---
        grace_period_sec: float = 1.5,
        confirmation_ratio: float = 0.5,
        # --- Normalization ---
        use_normalization: bool = False,
        normalization_base: str = "torso",
    ):
        # --- Thresholds & parameters ---
        self.stay_threshold_sec = stay_threshold_sec
        self.confidence_threshold = confidence_threshold
        self.spike_threshold = spike_threshold
        self.spike_window_sec = spike_window_sec
        self.stability_threshold_px = stability_threshold_px
        self.stability_window_sec = stability_window_sec
        self.grace_period_sec = grace_period_sec
        self.confirmation_ratio = confirmation_ratio
        self.use_normalization = use_normalization
        self.normalization_base = normalization_base

        # --- State variables ---
        self.state = HipStayState.STAYING
        self.stay_info: HipStayInfo | None = None
        self.potential_move_start_time: float = 0.0

        # --- Data history for analysis ---
        # Assuming 30 FPS for deque maxlen calculation
        self.norm_dist_history = deque(maxlen=int(spike_window_sec * 30))
        self.scale_history = deque(maxlen=int(stability_window_sec * 30))
        self.grace_period_history: deque[bool] = deque(maxlen=int(grace_period_sec * 30))

    def _compute_person_scale(self, landmarks: np.ndarray, frame_shape: tuple) -> float | None:
        """人物スケールを計算して返す。"""
        height, width = frame_shape[:2]

        def to_px(pt: np.ndarray) -> tuple[float, float]:
            return pt[0] * width, pt[1] * height

        def visible(pt: np.ndarray) -> bool:
            try:
                return float(pt[3]) >= float(self.confidence_threshold)
            except (TypeError, ValueError):
                return False

        try:
            if self.normalization_base == "screen":
                return float(height)

            l_sh = landmarks[BodyPart.LEFT_SHOULDER]
            r_sh = landmarks[BodyPart.RIGHT_SHOULDER]
            l_hip = landmarks[BodyPart.LEFT_HIP]
            r_hip = landmarks[BodyPart.RIGHT_HIP]

            if self.normalization_base == "shoulder":
                if not (visible(l_sh) and visible(r_sh)):
                    return None
                l_sh_px = to_px(l_sh)
                r_sh_px = to_px(r_sh)
                shoulder_width = float(np.sqrt((l_sh_px[0] - r_sh_px[0]) ** 2 + (l_sh_px[1] - r_sh_px[1]) ** 2))
                return shoulder_width if shoulder_width > 0 else None

            # torso
            if not (visible(l_sh) and visible(r_sh) and visible(l_hip) and visible(r_hip)):
                return None

            l_sh_px, r_sh_px = to_px(l_sh), to_px(r_sh)
            l_hip_px, r_hip_px = to_px(l_hip), to_px(r_hip)

            shoulder_mid = ((l_sh_px[0] + r_sh_px[0]) / 2.0, (l_sh_px[1] + r_sh_px[1]) / 2.0)
            hip_mid = ((l_hip_px[0] + r_hip_px[0]) / 2.0, (l_hip_px[1] + r_hip_px[1]) / 2.0)
            torso_len = float(np.sqrt((shoulder_mid[0] - hip_mid[0]) ** 2 + (shoulder_mid[1] - hip_mid[1]) ** 2))
            return torso_len if torso_len > 0 else None

        except (IndexError, TypeError):
            return None

    def extract_hip_center(self, landmarks: np.ndarray, frame_shape: tuple) -> tuple[float, float, float] | None:
        """腰の中心座標と信頼度を抽出。片方の腰だけでも検出を試みる。"""
        if landmarks is None:
            return None
        try:
            height, width = frame_shape[:2]
            left_hip, right_hip = landmarks[BodyPart.LEFT_HIP], landmarks[BodyPart.RIGHT_HIP]
            left_hip_conf, right_hip_conf = left_hip[3], right_hip[3]
            left_visible = left_hip_conf >= self.confidence_threshold
            right_visible = right_hip_conf >= self.confidence_threshold
            left_hip_px = (left_hip[0] * width, left_hip[1] * height)
            right_hip_px = (right_hip[0] * width, right_hip[1] * height)

            if left_visible and right_visible:
                hip_center = ((left_hip_px[0] + right_hip_px[0]) / 2, (left_hip_px[1] + right_hip_px[1]) / 2)
                avg_conf = (left_hip_conf + right_hip_conf) / 2
                return hip_center[0], hip_center[1], avg_conf
            if left_visible:
                return left_hip_px[0], left_hip_px[1], left_hip_conf
            if right_visible:
                return right_hip_px[0], right_hip_px[1], right_hip_conf
            return None
        except (IndexError, TypeError):
            return None

    def _reset_stay_info(self, timestamp: float, current_pos: tuple[float, float], confidence: float):
        """滞在情報をリセットする"""
        self.stay_info = HipStayInfo(
            last_hip_pos=current_pos,
            last_update_time=timestamp,
            stay_start_time=timestamp,
            stay_duration=0.0,
            notified=False,
            confidence_score=confidence,
        )
        self.state = HipStayState.STAYING
        self.grace_period_history.clear()

    def update(self, landmarks: np.ndarray, frame_shape: tuple, timestamp: float) -> str | None:
        """新しいアルゴリズムに基づいて滞在検知を更新する"""
        hip_data = self.extract_hip_center(landmarks, frame_shape)
        if hip_data is None:
            if self.stay_info is not None:
                self._reset_stay_info(timestamp, self.stay_info.last_hip_pos, 0)
            return None

        hip_x, hip_y, confidence = hip_data
        current_pos = (hip_x, hip_y)
        person_scale = self._compute_person_scale(landmarks, frame_shape)

        if self.stay_info is None:
            self._reset_stay_info(timestamp, current_pos, confidence)
            return None

        # --- 移動メトリクスを計算 ---
        time_elapsed = timestamp - self.stay_info.last_update_time
        pixel_dist = np.linalg.norm(np.array(current_pos) - np.array(self.stay_info.last_hip_pos))

        norm_dist = 0.0
        if self.use_normalization and person_scale and person_scale > 0:
            norm_dist = pixel_dist / person_scale

        self.norm_dist_history.append(norm_dist)
        if person_scale:
            self.scale_history.append(person_scale)

        # --- 移動トリガー（アイデア5）をチェック ---
        is_spike = max(self.norm_dist_history, default=0) >= self.spike_threshold
        is_unstable = len(self.scale_history) > 1 and np.std(self.scale_history) > self.stability_threshold_px
        is_movement_detected = is_spike or is_unstable

        # --- 状態機械ロジック ---
        if self.state == HipStayState.STAYING:
            if is_movement_detected:
                self.state = HipStayState.POTENTIAL_MOVE
                self.potential_move_start_time = timestamp
                self.grace_period_history.clear()
                self.grace_period_history.append(bool(is_movement_detected))
            else:
                self.stay_info.stay_duration += time_elapsed

        elif self.state == HipStayState.POTENTIAL_MOVE:
            self.grace_period_history.append(bool(is_movement_detected))

            if timestamp - self.potential_move_start_time >= self.grace_period_sec:
                move_ratio = sum(self.grace_period_history) / len(self.grace_period_history)
                if move_ratio >= self.confirmation_ratio:
                    # 移動確定、リセット
                    self._reset_stay_info(timestamp, current_pos, confidence)
                    return "[!] Movement Confirmed"
                # 誤報と判断、滞在状態に復帰
                self.state = HipStayState.STAYING

        # --- 共通情報を更新 ---
        self.stay_info.last_hip_pos = current_pos
        self.stay_info.last_update_time = timestamp
        self.stay_info.confidence_score = confidence

        # --- 長期滞在アラートをチェック ---
        if self.stay_info.stay_duration >= self.stay_threshold_sec and not self.stay_info.notified:
            self.stay_info.notified = True
            return f"[!] Long Stay Detected: {self.stay_info.stay_duration:.1f}s"

        return None

    def get_current_status(self) -> dict[str, Any]:
        """現在の滞在状況を取得"""
        if self.stay_info is None:
            return {
                "hip_position": None,
                "stay_duration": 0.0,
                "confidence": 0.0,
                "is_long_stay": False,
                "state": self.state.name,
            }
        return {
            "hip_position": self.stay_info.last_hip_pos,
            "stay_duration": self.stay_info.stay_duration,
            "confidence": self.stay_info.confidence_score,
            "is_long_stay": self.stay_info.stay_duration >= self.stay_threshold_sec,
            "state": self.state.name,
        }


class KneeAngleMonitor:
    """膝角度の秒毎中央値と5秒移動平均を監視し、閾値を下回ると通知する."""

    def __init__(
        self,
        threshold_deg: float = 90.0,
        moving_window_seconds: int = 5,
        confidence_threshold: float = 0.7,
    ) -> None:
        """
        KneeAngleMonitorを初期化する.

        Args:
            threshold_deg: アラートを発する膝角度の閾値（度）.
            moving_window_seconds: 移動平均を計算するためのウィンドウサイズ（秒）.
            confidence_threshold: 角度計算の信頼度スコアの閾値.
        """
        self.threshold_deg = threshold_deg
        self.moving_window_seconds = moving_window_seconds
        self.confidence_threshold = confidence_threshold  # 信頼度の閾値を追加
        self.current_second: int | None = None
        self.current_left_values: list[float] = []
        self.current_right_values: list[float] = []
        self.medians_history: deque[tuple[int, float | None, float | None]] = deque(maxlen=moving_window_seconds)
        self.current_alert_message: str | None = None

    def _finalize_second(self, second: int) -> tuple[int, float | None, float | None] | None:
        if not self.current_left_values and not self.current_right_values:
            # データがない場合は履歴に追加しない
            return None

        left_med = float(np.median(self.current_left_values)) if self.current_left_values else None
        right_med = float(np.median(self.current_right_values)) if self.current_right_values else None

        self.medians_history.append((second, left_med, right_med))
        self.current_left_values.clear()
        self.current_right_values.clear()
        return (second, left_med, right_med)

    def _moving_average(self) -> tuple[float | None, float | None]:
        if not self.medians_history:
            return None, None
        # Noneをフィルタリングして平均を計算
        left_vals = [m[1] for m in self.medians_history if m[1] is not None]
        right_vals = [m[2] for m in self.medians_history if m[2] is not None]

        left_ma = float(np.mean(left_vals)) if left_vals else None
        right_ma = float(np.mean(right_vals)) if right_vals else None
        return left_ma, right_ma

    def update(self, timestamp_s: float, analysis_results: dict[Angle, dict[str, Any]]) -> list[str]:
        """各フレームで呼び出し.秒境界で集計を確定し、通知を返す."""
        alerts: list[str] = []
        sec = int(timestamp_s)

        # 最初のフレームで現在の秒を初期化
        if self.current_second is None:
            self.current_second = sec

        finalized_median_info = None
        # 秒が切り替わった時に、前の秒の結果を確定させる
        if sec != self.current_second:
            finalized_median_info = self._finalize_second(self.current_second)
            self.current_second = sec

        # 現在の秒のデータを蓄積
        left_knee = analysis_results.get(Angle.LEFT_KNEE)
        right_knee = analysis_results.get(Angle.RIGHT_KNEE)

        if left_knee and "angle" in left_knee:
            left_conf = min(
                left_knee.get("p1_confidence", 0.0),
                left_knee.get("p2_confidence", 0.0),
                left_knee.get("p3_confidence", 0.0),
            )
            if left_conf >= self.confidence_threshold:
                self.current_left_values.append(float(left_knee["angle"]))

        if right_knee and "angle" in right_knee:
            right_conf = min(
                right_knee.get("p1_confidence", 0.0),
                right_knee.get("p2_confidence", 0.0),
                right_knee.get("p3_confidence", 0.0),
            )
            if right_conf >= self.confidence_threshold:
                self.current_right_values.append(float(right_knee["angle"]))

        # 確定した前の秒の結果があれば、アラートをチェック
        if finalized_median_info:
            second, left_med, right_med = finalized_median_info

            left_ma, right_ma = self._moving_average()

            triggered = []
            # 中央値が閾値を下回るかチェック
            if (left_med is not None and left_med < self.threshold_deg) or (
                right_med is not None and right_med < self.threshold_deg
            ):
                lm = f"{left_med:.1f}" if left_med is not None else "-"
                rm = f"{right_med:.1f}" if right_med is not None else "-"
                triggered.append(f"Median L:{lm} R:{rm}")

            # 移動平均が閾値を下回るかチェック
            if (left_ma is not None and left_ma < self.threshold_deg) or (
                right_ma is not None and right_ma < self.threshold_deg
            ):
                lma = f"{left_ma:.1f}" if left_ma is not None else "-"
                rma = f"{right_ma:.1f}" if right_ma is not None else "-"
                triggered.append(f"MA5 L:{lma} R:{rma}")

            if triggered:
                alert_msg = f"[!] Knee Angle Low ({second}s): " + " / ".join(triggered)
                alerts.append(alert_msg)
                self.current_alert_message = alert_msg

        return alerts

    def get_current_alert(self) -> str | None:
        """現在表示中のアラートメッセージを取得."""
        return self.current_alert_message


def setup_csv_writer(csv_file: IO):
    """CSVライターをセットアップする"""
    fieldnames = [
        "timestamp",
        "frame_number",
        "right_elbow_angle",
        "right_elbow_state",
        "left_elbow_angle",
        "left_elbow_state",
        "right_shoulder_angle",
        "right_shoulder_state",
        "left_shoulder_angle",
        "left_shoulder_state",
        "right_hip_angle",
        "right_hip_state",
        "left_hip_angle",
        "left_hip_state",
        "right_knee_angle",
        "right_knee_state",
        "left_knee_angle",
        "left_knee_state",
        "right_knee_hip_confidence",
        "right_knee_knee_confidence",
        "right_knee_ankle_confidence",
        "left_knee_hip_confidence",
        "left_knee_knee_confidence",
        "left_knee_ankle_confidence",
        "body_tilt_angle",
        "neck_trunk_angle_angle",
        "neck_trunk_angle_state",
        "body_tilt_state",
        "lateral_tilt_angle",
        "lateral_tilt_state",
        # 首振り関連のフィールド
        "head_horizontal_rotation_angle",
        "head_horizontal_rotation_state",
        "head_vertical_nod_angle",
        "head_vertical_nod_state",
        "head_shake_horizontal_detected",
        "head_shake_vertical_detected",
        "head_shake_alerts",
        # 姿勢監視フィールド
        "is_forward_leaning",
        "forward_lean_score",
        "forward_lean_ratio",
        "avg_forward_lean_score",
        # 腰滞在検知フィールド
        "hip_center_x",
        "hip_center_y",
        "stay_duration",
        "hip_confidence",
        "is_long_stay",
        "long_stay_alert",
        "hip_detector_state",
        # 正規化パラメータフィールド
        "torso_length",
        "shoulder_width",
        "person_scale",
        "normalization_base",
        "normalization_applied",
    ]
    # ---ランドマーク座標のフィールドを追加---
    landmark_fieldnames = []
    for landmark in mp.solutions.pose.PoseLandmark:
        name = landmark.name
        landmark_fieldnames.extend([f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_visibility"])
    fieldnames.extend(landmark_fieldnames)
    # ---ここまで---

    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    return writer


def setup_video_writer(cap: Any, output_path: str):
    """ビデオライターをセットアップする"""
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def write_results_to_csv(
    csv_writer,
    timestamp: float,
    frame_number: int,
    analysis_results: dict,
    posture_monitor: PostureMonitor,
    hip_detector: HipBasedStayDetector,
    hip_alert: str | None,
    head_shake_detector: HeadShakeDetector | None = None,
    head_shake_alerts: list[str] | None = None,
    landmarks: np.ndarray | None = None,
    frame_shape: tuple | None = None,
):
    """結果をCSVに書き込む"""
    row = {"timestamp": timestamp, "frame_number": frame_number}

    for angle in Angle:
        angle_name = angle.name.lower()
        if angle in analysis_results:
            result = analysis_results[angle]
            row[f"{angle_name}_angle"] = result.get("angle", 0)
            row[f"{angle_name}_state"] = result.get("state", MovementState.STATIC).name
        else:
            row[f"{angle_name}_angle"] = 0
            row[f"{angle_name}_state"] = MovementState.STATIC.name

    # 膝関節の信頼度を追加
    rk_result = analysis_results.get(Angle.RIGHT_KNEE, {})
    row["right_knee_hip_confidence"] = rk_result.get("p1_confidence", 0.0)
    row["right_knee_knee_confidence"] = rk_result.get("p2_confidence", 0.0)
    row["right_knee_ankle_confidence"] = rk_result.get("p3_confidence", 0.0)

    lk_result = analysis_results.get(Angle.LEFT_KNEE, {})
    row["left_knee_hip_confidence"] = lk_result.get("p1_confidence", 0.0)
    row["left_knee_knee_confidence"] = lk_result.get("p2_confidence", 0.0)
    row["left_knee_ankle_confidence"] = lk_result.get("p3_confidence", 0.0)

    posture_stats = posture_monitor.get_status()
    is_forward_leaning, forward_lean_score = False, 0.0
    if posture_monitor.posture_history:
        latest = posture_monitor.posture_history[-1]
        is_forward_leaning = latest.is_forward_leaning
        forward_lean_score = latest.forward_lean_score
    row.update(
        {
            "is_forward_leaning": is_forward_leaning,
            "forward_lean_score": forward_lean_score,
            "forward_lean_ratio": posture_stats["forward_ratio"],
            "avg_forward_lean_score": posture_stats["avg_score"],
        }
    )

    hip_status = hip_detector.get_current_status()
    hip_pos = hip_status["hip_position"]
    row.update(
        {
            "hip_center_x": hip_pos[0] if hip_pos else 0,
            "hip_center_y": hip_pos[1] if hip_pos else 0,
            "stay_duration": hip_status["stay_duration"],
            "hip_confidence": hip_status["confidence"],
            "is_long_stay": hip_status["is_long_stay"],
            "long_stay_alert": hip_alert or "",
            "hip_detector_state": hip_status["state"],
        }
    )

    # 首振りデータ
    head_shake_status = head_shake_detector.get_status() if head_shake_detector else {}
    row.update(
        {
            "head_shake_horizontal_detected": head_shake_status.get("horizontal_state", "HEAD_STATIC") != "HEAD_STATIC",
            "head_shake_vertical_detected": head_shake_status.get("vertical_state", "HEAD_STATIC") != "HEAD_STATIC",
            "head_shake_alerts": "; ".join(head_shake_alerts) if head_shake_alerts else "",
        }
    )

    torso_length, shoulder_width, person_scale = 0.0, 0.0, 0.0
    if landmarks is not None and frame_shape is not None:
        # NOTE: Using protected method for simplicity in this script
        scale_val = hip_detector._compute_person_scale(landmarks, frame_shape)  # pylint: disable=protected-access
        if scale_val:
            person_scale = scale_val
            if hip_detector.normalization_base == "torso":
                torso_length = scale_val
            elif hip_detector.normalization_base == "shoulder":
                shoulder_width = scale_val
    row.update(
        {
            "torso_length": torso_length,
            "shoulder_width": shoulder_width,
            "person_scale": person_scale,
            "normalization_base": hip_detector.normalization_base,
            "normalization_applied": hip_detector.use_normalization,
        }
    )

    # --- ランドマーク座標を書き込む ---
    if landmarks is not None:
        for landmark in mp.solutions.pose.PoseLandmark:
            name = landmark.name
            idx = landmark.value
            row[f"{name}_x"] = landmarks[idx][0]
            row[f"{name}_y"] = landmarks[idx][1]
            row[f"{name}_z"] = landmarks[idx][2]
            row[f"{name}_visibility"] = landmarks[idx][3]
    else:
        # ランドマークがない場合は空欄（または0）で埋める
        for landmark in mp.solutions.pose.PoseLandmark:
            name = landmark.name
            row[f"{name}_x"] = 0.0
            row[f"{name}_y"] = 0.0
            row[f"{name}_z"] = 0.0
            row[f"{name}_visibility"] = 0.0
    # --- ここまで ---

    csv_writer.writerow(row)


def draw_posture_alerts(
    frame, alerts: list[str], status: dict[str, Any], knee_alert: str | None, head_shake_alerts: list[str] | None = None
):
    """フレームに前傾姿勢アラートと状態を描画"""
    y_offset = 30
    for alert in alerts:
        cv2.putText(frame, alert, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30
    if knee_alert:
        cv2.putText(frame, knee_alert, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30

    # 首振りアラートの表示
    if head_shake_alerts:
        for head_alert in head_shake_alerts:
            cv2.putText(frame, head_alert, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            y_offset += 30

    if status["sample_count"] > 0:
        status_text = (
            f"Monitor: {status['monitoring_duration']:.1f}s | "
            f"Forward: {status['forward_ratio']:.1%} | "
            f"Score: {status['avg_score']:.2f}"
        )
        cv2.putText(frame, status_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


def draw_hip_stay_info(frame: np.ndarray, hip_detector: HipBasedStayDetector) -> np.ndarray:
    """腰の滞在情報を描画"""
    hip_status = hip_detector.get_current_status()
    hip_pos = hip_status["hip_position"]
    if hip_pos is None:
        return frame

    center_x, center_y = int(hip_pos[0]), int(hip_pos[1])
    color = (0, 0, 255) if hip_status["is_long_stay"] else (0, 255, 0)
    thickness = 3 if hip_status["is_long_stay"] else 2
    cv2.circle(frame, (center_x, center_y), 8, color, thickness)

    state_text = hip_status["state"]
    stay_duration = hip_status["stay_duration"]
    confidence = hip_status["confidence"]
    text = f"{state_text}: {stay_duration:.1f}s (Conf:{confidence:.2f})"
    cv2.putText(frame, text, (center_x + 15, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def draw_head_shake_info(
    frame: np.ndarray, head_shake_detector: HeadShakeDetector | None, landmarks: np.ndarray | None
) -> np.ndarray:
    """首振り情報を描画"""
    if head_shake_detector is None or landmarks is None:
        return frame

    try:
        # 鼻の位置を取得
        nose = landmarks[BodyPart.NOSE]
        if nose[3] < 0.5:  # 信頼度が低い場合はスキップ
            return frame

        # 画面座標に変換
        height, width = frame.shape[:2]
        nose_x, nose_y = int(nose[0] * width), int(nose[1] * height)

        # 首振り状態を取得
        head_status = head_shake_detector.get_status()

        # 水平状態に応じた色設定
        horizontal_state = head_status.get("horizontal_state", "HEAD_STATIC")
        if horizontal_state == "HORIZONTAL_SHAKE":
            color = (0, 255, 255)  # 黄色（首振り検出）
        elif horizontal_state == "HEAD_LEFT_TURN":
            color = (255, 0, 0)  # 青色（左向き）
        elif horizontal_state == "HEAD_RIGHT_TURN":
            color = (0, 0, 255)  # 赤色（右向き）
        else:
            color = (0, 255, 0)  # 緑色（静止）

        # 鼻の位置にマーカーを描画
        cv2.circle(frame, (nose_x, nose_y), 6, color, 2)

        # 角度情報を表示
        h_angle = head_status.get("horizontal_angle", 0.0)
        v_angle = head_status.get("vertical_angle", 0.0)

        text = f"Head: H:{h_angle:.1f}° V:{v_angle:.1f}°"
        cv2.putText(frame, text, (nose_x + 10, nose_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 状態テキスト
        state_text = f"H:{horizontal_state.replace('HEAD_', '').replace('_', ' ')}"
        vertical_state = head_status.get("vertical_state", "HEAD_STATIC")
        state_text += f" V:{vertical_state.replace('HEAD_', '').replace('_', ' ')}"
        cv2.putText(frame, state_text, (nose_x + 10, nose_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    except (IndexError, TypeError):
        pass

    return frame


def process_video(
    video_path: str,
    output_csv_path: str | None,
    output_video_path: str | None,
    disable_japanese: bool,
    monitoring_duration: float = 60.0,
    alert_threshold: float = 0.7,
    hip_stay_threshold: float = 60.0,
    hip_normalize: bool = False,
    hip_norm_base: str = "torso",
    # --- New Algorithm Params ---
    spike_threshold: float = 1.5,
    stability_threshold_px: float = 50.0,
    grace_period_sec: float = 1.5,
):
    """ビデオを処理し、関節の動きと滞在を分析して結果を出力する"""
    print(f"--- デバッグ: process_video開始, 対象: {video_path} ---")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"--- デバッグ・エラー: ビデオファイルが開けません: {video_path} ---")
        return
    print("--- デバッグ: ビデオファイルを正常に開きました ---")

    p = Path(video_path)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_path = output_csv_path or f"output/{p.stem}_integrated_analysis_{timestamp_str}.csv"
    output_video_path = output_video_path or f"output/{p.stem}_integrated_output_{timestamp_str}.mp4"
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)

    video_writer = None
    try:
        with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
            csv_writer = setup_csv_writer(csv_file)
            video_writer = setup_video_writer(cap, output_video_path)

            pose_estimator = PoseEstimator()
            analyzer = MovementAnalyzer()
            posture_monitor = PostureMonitor(monitoring_duration, alert_threshold)
            hip_detector = HipBasedStayDetector(
                stay_threshold_sec=hip_stay_threshold,
                use_normalization=hip_normalize,
                normalization_base=hip_norm_base,
                confidence_threshold=0.1,  # Lowered for better detection
                spike_threshold=spike_threshold,
                stability_threshold_px=stability_threshold_px,
                grace_period_sec=grace_period_sec,
            )
            knee_monitor = KneeAngleMonitor(
                threshold_deg=90.0, moving_window_seconds=5, confidence_threshold=0.8
            )  # 信頼度閾値を追加
            head_shake_detector = HeadShakeDetector(
                horizontal_threshold=15.0,
                vertical_threshold=10.0,
                cycle_detection_window=60,  # 2秒@30fps
                min_oscillations=2,
            )

            frame_count = 0
            print(f"Processing video: {video_path}")
            print(f"Outputting to {output_csv_path} and {output_video_path}")
            if hip_normalize:
                print(
                    f"Hip-based Stay Detection: Spike > {spike_threshold:.2f} "
                    f"| Stability > {stability_threshold_px:.1f}px "
                    f"| Grace Period {grace_period_sec:.1f}s"
                )

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("--- DEBUG: Failed to read frame or end of video.")  # デバッグ出力
                    break

                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if frame_count % 100 == 0:  # 100フレーム毎に出力
                    print(f"--- DEBUG: Processing frame {frame_count}, timestamp: {timestamp:.2f}s")
                landmarks = pose_estimator.estimate(frame)
                hip_alert = None
                alerts = []
                head_shake_alerts = []
                analysis_results = {}

                if landmarks is not None:
                    analysis_results = analyzer.analyze(landmarks)
                    alerts = posture_monitor.update(timestamp, frame_count, analysis_results)
                    hip_alert = hip_detector.update(landmarks, frame.shape, timestamp)
                    knee_alerts = knee_monitor.update(timestamp, analysis_results)
                    alerts.extend(knee_alerts)

                    # 首振り検出
                    head_shake_results = head_shake_detector.update(landmarks, timestamp, frame_count)
                    analysis_results.update(head_shake_results)
                    head_shake_alerts = head_shake_detector.check_alerts(timestamp)
                    alerts.extend(head_shake_alerts)

                    if hip_alert:
                        print(f"Frame {frame_count}: {hip_alert}")

                    if head_shake_alerts:
                        for head_alert in head_shake_alerts:
                            print(f"Frame {frame_count}: {head_alert}")

                    write_results_to_csv(
                        csv_writer,
                        timestamp,
                        frame_count,
                        analysis_results,
                        posture_monitor,
                        hip_detector,
                        hip_alert,
                        head_shake_detector,
                        head_shake_alerts,
                        landmarks,
                        frame.shape,
                    )

                    frame = draw_analysis_results(
                        image=frame,
                        results=analysis_results,
                        landmarks=landmarks,
                        disable_japanese=disable_japanese,
                    )
                    draw_landmarks(frame, landmarks)
                    frame = draw_hip_stay_info(frame, hip_detector)
                    frame = draw_head_shake_info(frame, head_shake_detector, landmarks)

                status = posture_monitor.get_status()
                draw_posture_alerts(frame, alerts, status, knee_monitor.get_current_alert(), head_shake_alerts)
                if hip_alert:
                    cv2.putText(
                        frame,
                        f"HIP ALERT: {hip_alert}",
                        (10, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )

                video_writer.write(frame)
                cv2.imshow("Integrated Analysis", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                frame_count += 1
    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="統合版：関節動作分析 + 腰ベース長期滞在検知")
    parser.add_argument("--video", required=True, help="入力ビデオファイルのパス")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="検出信頼度の閾値")
    parser.add_argument("--no-display", action="store_true", help="結果表示ウィンドウを無効化")
    # --- Normalization parameters ---
    parser.add_argument(
        "--hip-normalize",
        action="store_true",
        help="腰の移動判定を人物スケールで正規化",
    )
    # --- New algorithm parameters ---
    parser.add_argument(
        "--spike-threshold",
        type=float,
        default=1.5,
        help="移動スパイク検知の閾値（体幹長比）",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=50.0,
        help="検出安定性（体幹長ブレ）の閾値（px）",
    )
    parser.add_argument("--grace-period", type=float, default=1.5, help="移動検知の猶予期間（秒）")
    args = parser.parse_args()

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    video_basename = f"{os.path.splitext(os.path.basename(args.video))[0]}_integrated"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = os.path.join(output_dir, f"{video_basename}_output_{timestamp}.mp4")
    output_csv_path = os.path.join(output_dir, f"{video_basename}_analysis_{timestamp}.csv")

    process_video(
        video_path=args.video,
        output_csv_path=output_csv_path,
        output_video_path=output_video_path,
        disable_japanese=False,
        monitoring_duration=60.0,
        alert_threshold=0.7,
        hip_stay_threshold=60.0,
        hip_normalize=args.hip_normalize,
        hip_norm_base="torso",
        spike_threshold=args.spike_threshold,
        stability_threshold_px=args.stability_threshold,
        grace_period_sec=args.grace_period,
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(f"--- デバッグ: argparseがSystemExitを発生させました (exit code: {e.code}) ---")
