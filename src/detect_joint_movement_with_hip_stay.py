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
from typing import IO, Any

import cv2
import mediapipe as mp
import numpy as np

from .analysis.dwell_time_detector import DwellTimeDetector  # noqa: F401
from .analysis.posture_monitor import PostureMonitor
from .analysis.user_classifier import UserClassifier  # noqa: F401
from .definitions import Angle, MovementState
from .head_shake_detector import HeadShakeDetector
from .movement_analyzer import MovementAnalyzer  # noqa: F401
from .pose.definitions import BodyPart
from .pose_estimator import PoseEstimator  # noqa: F401
from .video_processor import process_video

print("--- デバッグ: detect_joint_movement_with_hip_stay.py の実行開始 ---")

# OpenCVの型スタブがない環境での静的解析エラー回避用にAnyとして扱う
# cv2m: Any = cv2  <- この行を削除し、直接cv2を使用します


@dataclass
class HipStayInfo:
    """腰の位置ベースの滞在情報"""

    last_hip_pos: tuple[float, float]  # 腰の中心座標
    last_update_time: float
    stay_start_time: float
    stay_duration: float
    notified: bool
    confidence_score: float  # 腰の検出信頼度


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


# --- main & CLI (revised for the new process_video signature) ---
def main():
    parser = argparse.ArgumentParser(description="Integrated analysis: joint motion + hip-based dwell detection")

    # ---- Input source / switching ----
    parser.add_argument(
        "--video",
        required=True,
        help=(
            "FFmpeg/OpenCV input. File path for 'ffmpeg-file' / 'opencv-file', "
            "or device for 'ffmpeg-camera' (e.g., '/dev/video0', '0', or "
            "'video=Integrated Camera' on Windows)."
        ),
    )
    parser.add_argument(
        "--input-mode",
        choices=["ffmpeg-file", "ffmpeg-camera", "opencv-file"],
        default=None,
        help="Frame source backend. If omitted, defaults to 'ffmpeg-file' when available, else 'opencv-file'.",
    )

    # ---- Output / preview ----
    parser.add_argument("--no-display", action="store_true", help="Disable preview window (headless/CI).")
    parser.add_argument("--disable-japanese", action="store_true", help="Disable Japanese labels in overlays.")
    parser.add_argument(
        "--output-dir", type=str, default="./data/output/", help="Output directory for CSV and video files."
    )
    parser.add_argument("--output-csv", type=str, default=None, help="Output CSV path (overrides auto-generated path).")
    parser.add_argument(
        "--output-video", type=str, default=None, help="Output video path (overrides auto-generated path)."
    )

    # ---- FFmpeg/OpenCV capture geometry ----
    parser.add_argument("--width", type=int, default=1280, help="Output width for writer/pipeline.")
    parser.add_argument("--height", type=int, default=720, help="Output height for writer/pipeline.")
    parser.add_argument("--fps", type=float, default=30.0, help="Nominal FPS for writer/FFmpeg pacing.")
    parser.add_argument("--gray", action="store_true", help="Use mono pipeline hint (FFmpeg pix_fmt=gray).")

    # ---- Posture monitor (PM) ----
    parser.add_argument(
        "--pm-monitoring-duration-sec",
        type=float,
        default=60.0,
        help="PostureMonitor window length in seconds.",
    )
    parser.add_argument(
        "--pm-alert-threshold-ratio",
        type=float,
        default=0.7,
        help="PostureMonitor alert threshold ratio.",
    )

    # ---- Dwell (hip-based stay) ----
    parser.add_argument(
        "--stay-threshold-sec",
        type=float,
        default=60.0,
        help="Seconds required to declare long stay.",
    )
    parser.add_argument(
        "--spike-threshold",
        type=float,
        default=1.5,
        help="Spike detector threshold (torso-length-normalized).",
    )
    parser.add_argument(
        "--stability-threshold-px",
        type=float,
        default=50.0,
        help="Detector stability threshold (px jitter tolerance).",
    )
    parser.add_argument(
        "--grace-period-sec",
        type=float,
        default=1.5,
        help="Grace period for movement detection (seconds).",
    )

    # ---- User classifier (knee angle etc. if used downstream) ----
    parser.add_argument(
        "--uc-threshold-deg",
        type=float,
        default=90.0,
        help="Knee angle threshold (degrees).",
    )
    parser.add_argument(
        "--uc-moving-window-seconds",
        type=float,
        default=5.0,
        help="Rolling window for knee angle logic (seconds).",
    )

    # ---- Progress bar control ----
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show tqdm progress bar during video processing.",
    )

    args = parser.parse_args()

    # ---- Output paths ----
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Use provided paths or generate auto paths
    if args.output_csv:
        output_csv_path = args.output_csv
    else:
        video_basename = f"{os.path.splitext(os.path.basename(args.video))[0]}_integrated"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv_path = os.path.join(output_dir, f"{video_basename}_analysis_{timestamp}.csv")

    if args.output_video:
        output_video_path = args.output_video
    else:
        video_basename = f"{os.path.splitext(os.path.basename(args.video))[0]}_integrated"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = os.path.join(output_dir, f"{video_basename}_output_{timestamp}.mp4")

    # ---- Call revised process_video ----
    process_video(
        video_path=args.video,
        output_csv_path=output_csv_path,
        output_video_path=output_video_path,
        disable_japanese=args.disable_japanese,
        # dwell / detectors
        stay_threshold_sec=args.stay_threshold_sec,
        spike_threshold=args.spike_threshold,
        stability_threshold_px=args.stability_threshold_px,
        grace_period_sec=args.grace_period_sec,
        # posture monitor
        pm_monitoring_duration_sec=args.pm_monitoring_duration_sec,
        pm_alert_threshold_ratio=args.pm_alert_threshold_ratio,
        # user classifier (knee angle)
        uc_threshold_deg=args.uc_threshold_deg,
        uc_moving_window_seconds=args.uc_moving_window_seconds,
        # source switching / geometry
        input_mode=args.input_mode,
        width=args.width,
        height=args.height,
        fps=args.fps,
        is_color=not args.gray,
        preview=not args.no_display,
        show_progress=args.show_progress,
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(f"--- DEBUG: argparse raised SystemExit (exit code: {e.code}) ---")
