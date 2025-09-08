from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from ..pose.definitions import BodyPart


@dataclass
class DwellTimeInfo:
    """腰の位置ベースの滞在情報"""

    last_hip_pos: tuple[float, float]  # 腰の中心座標
    last_update_time: float
    stay_start_time: float
    stay_duration: float
    notified: bool
    confidence_score: float  # 腰の検出信頼度


class DwellTimeState(Enum):
    """腰ベース滞在検知の状態"""

    STAYING = auto()
    POTENTIAL_MOVE = auto()


class DwellTimeDetector:
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
        self.state = DwellTimeState.STAYING
        self.stay_info: DwellTimeInfo | None = None
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
        self.stay_info = DwellTimeInfo(
            last_hip_pos=current_pos,
            last_update_time=timestamp,
            stay_start_time=timestamp,
            stay_duration=0.0,
            notified=False,
            confidence_score=confidence,
        )
        self.state = DwellTimeState.STAYING
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
        if self.state == DwellTimeState.STAYING:
            if is_movement_detected:
                self.state = DwellTimeState.POTENTIAL_MOVE
                self.potential_move_start_time = timestamp
                self.grace_period_history.clear()
                self.grace_period_history.append(bool(is_movement_detected))
            else:
                self.stay_info.stay_duration += time_elapsed

        elif self.state == DwellTimeState.POTENTIAL_MOVE:
            self.grace_period_history.append(bool(is_movement_detected))

            if timestamp - self.potential_move_start_time >= self.grace_period_sec:
                move_ratio = sum(self.grace_period_history) / len(self.grace_period_history)
                if move_ratio >= self.confirmation_ratio:
                    # 移動確定、リセット
                    self._reset_stay_info(timestamp, current_pos, confidence)
                    return "[!] Movement Confirmed"
                # 誤報と判断、滞在状態に復帰
                self.state = DwellTimeState.STAYING

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
