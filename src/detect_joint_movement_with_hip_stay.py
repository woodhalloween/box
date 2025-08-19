"""src/detect_joint_movement_with_hip_stay.py

統合版：骨格推定・前傾姿勢分析 + 腰の位置を基準とした長期滞在検知
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Any

import cv2
import numpy as np

from .definitions import Angle, MovementState
from .drawing_utils import draw_analysis_results, draw_landmarks
from .movement_analyzer import MovementAnalyzer
from .pose.definitions import BodyPart
from .pose_estimator import PoseEstimator

# OpenCVの型スタブがない環境での静的解析エラー回避用にAnyとして扱う
cv2m: Any = cv2


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

        # クールダウン期間中はアラートしない
        if current_time - self.last_alert_time < self.alert_cooldown:
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
            return {"monitoring_duration": 0, "forward_ratio": 0, "avg_score": 0, "sample_count": 0}

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


class HipBasedStayDetector:
    """腰の位置を基準とした滞在検知器"""

    def __init__(
        self,
        move_threshold: float = 25.0,
        stay_threshold: float = 5.0,
        confidence_threshold: float = 0.5,
        use_normalization: bool = False,
        normalization_base: str = "torso",
    ):
        """
        Args:
            move_threshold: 移動判定の閾値。
                - 正規化なしの場合: ピクセル値（例: 25）
                - 正規化ありの場合: 基準スケールに対する比率（例: 0.10）
            stay_threshold: 長期滞在判定の閾値（秒）
            confidence_threshold: 検出可視性(visibility)の閾値
            use_normalization: 正規化して距離を評価するか
            normalization_base: 正規化の基準（"torso"|"shoulder"|"screen"）
        """
        self.move_threshold = move_threshold
        self.stay_threshold = stay_threshold
        self.confidence_threshold = confidence_threshold
        self.use_normalization = use_normalization
        self.normalization_base = (
            normalization_base if normalization_base in {"torso", "shoulder", "screen"} else "torso"
        )
        self.stay_info: HipStayInfo | None = None
        self.current_frame_idx = 0

    def _compute_person_scale(self, landmarks: np.ndarray, frame_shape: tuple) -> float | None:
        """人物スケールを計算して返す。

        normalization_base に従って以下のいずれかを返す:
          - torso: 肩中心-腰中心の距離（体幹長）
          - shoulder: 左右肩の距離（肩幅）
          - screen: 画面高さ（フォールバック、常に取得可能）
        """
        height, width = frame_shape[:2]

        # 画素座標に変換するヘルパ
        def to_px(pt: np.ndarray) -> tuple[float, float]:
            return pt[0] * width, pt[1] * height

        # 可視性チェック
        def visible(pt: np.ndarray) -> bool:
            try:
                return float(pt[3]) >= float(self.confidence_threshold)
            except (TypeError, ValueError):
                return False

        try:
            if self.normalization_base == "screen":
                return float(height)

            # 必要ランドマークを取得
            l_sh = landmarks[BodyPart.LEFT_SHOULDER]
            r_sh = landmarks[BodyPart.RIGHT_SHOULDER]
            l_hip = landmarks[BodyPart.LEFT_HIP]
            r_hip = landmarks[BodyPart.RIGHT_HIP]

            # 可視性チェック（必要な点）
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

            l_sh_px = to_px(l_sh)
            r_sh_px = to_px(r_sh)
            l_hip_px = to_px(l_hip)
            r_hip_px = to_px(r_hip)

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

            # 左右の腰の位置を取得
            left_hip = landmarks[BodyPart.LEFT_HIP]
            right_hip = landmarks[BodyPart.RIGHT_HIP]

            # 信頼度
            left_hip_confidence = left_hip[3]
            right_hip_confidence = right_hip[3]

            left_visible = left_hip_confidence >= self.confidence_threshold
            right_visible = right_hip_confidence >= self.confidence_threshold

            # 正規化座標をピクセル座標に変換
            left_hip_px = (left_hip[0] * width, left_hip[1] * height)
            right_hip_px = (right_hip[0] * width, right_hip[1] * height)

            if left_visible and right_visible:
                # 両方見える場合: 中心を計算
                hip_center = ((left_hip_px[0] + right_hip_px[0]) / 2, (left_hip_px[1] + right_hip_px[1]) / 2)
                avg_confidence = (left_hip_confidence + right_hip_confidence) / 2
                return hip_center[0], hip_center[1], avg_confidence
            elif left_visible:
                # 左のみ見える場合
                return left_hip_px[0], left_hip_px[1], left_hip_confidence
            elif right_visible:
                # 右のみ見える場合
                return right_hip_px[0], right_hip_px[1], right_hip_confidence
            else:
                # どちらも見えない
                return None

        except (IndexError, TypeError):
            return None

    def update(self, landmarks: np.ndarray, frame_shape: tuple, timestamp: float) -> str | None:
        """滞在検知を更新"""
        self.current_frame_idx += 1

        hip_data = self.extract_hip_center(landmarks, frame_shape)
        if hip_data is None:
            # 検出失敗時は滞在情報をリセット
            self.stay_info = None
            return None

        hip_x, hip_y, confidence = hip_data
        current_pos = (hip_x, hip_y)

        if self.stay_info is None:
            # 初回検出
            self.stay_info = HipStayInfo(
                last_hip_pos=current_pos,
                last_update_time=timestamp,
                stay_start_time=timestamp,
                stay_duration=0.0,
                notified=False,
                confidence_score=confidence,
            )
            return None

        # 前回位置からの移動距離を計算
        distance = np.sqrt(
            (current_pos[0] - self.stay_info.last_hip_pos[0]) ** 2
            + (current_pos[1] - self.stay_info.last_hip_pos[1]) ** 2
        )

        time_elapsed = timestamp - self.stay_info.last_update_time

        # 正規化の適用
        if self.use_normalization:
            scale = self._compute_person_scale(landmarks, frame_shape)
            if scale is None or scale <= 0:
                # フォールバック: 画面高さで正規化
                scale = float(frame_shape[0])
            distance_value = float(distance) / float(scale)
            threshold_value = float(self.move_threshold)
        else:
            distance_value = float(distance)
            threshold_value = float(self.move_threshold)

        if distance_value < threshold_value:
            # 滞在中: 滞在時間を更新
            self.stay_info.stay_duration += time_elapsed
        else:
            # 移動検出: 滞在情報をリセット
            self.stay_info = HipStayInfo(
                last_hip_pos=current_pos,
                last_update_time=timestamp,
                stay_start_time=timestamp,
                stay_duration=0.0,
                notified=False,
                confidence_score=confidence,
            )
            return None

        # 位置と時刻を更新
        self.stay_info.last_hip_pos = current_pos
        self.stay_info.last_update_time = timestamp
        self.stay_info.confidence_score = confidence

        # 長期滞在判定
        if self.stay_info.stay_duration >= self.stay_threshold and not self.stay_info.notified:
            self.stay_info.notified = True
            return f"[!] Long Stay Detected: {self.stay_info.stay_duration:.1f}s"

        return None

    def get_current_status(self) -> dict[str, Any]:
        """現在の滞在状況を取得"""
        if self.stay_info is None:
            return {"hip_position": None, "stay_duration": 0.0, "confidence": 0.0, "is_long_stay": False}

        return {
            "hip_position": self.stay_info.last_hip_pos,
            "stay_duration": self.stay_info.stay_duration,
            "confidence": self.stay_info.confidence_score,
            "is_long_stay": self.stay_info.stay_duration >= self.stay_threshold,
        }


class KneeAngleMonitor:
    """膝角度の秒毎中央値と5秒移動平均を監視し、閾値を下回ると通知する"""

    def __init__(self, threshold_deg: float = 90.0, moving_window_seconds: int = 5) -> None:
        self.threshold_deg = threshold_deg
        self.moving_window_seconds = moving_window_seconds

        # 現在集計中の秒と、その秒のフレーム内角度リスト
        self.current_second: int | None = None
        self.current_left_values: list[float] = []
        self.current_right_values: list[float] = []

        # 直近N秒の中央値履歴（左/右）
        self.medians_history: deque[tuple[int, float, float]] = deque(maxlen=moving_window_seconds)

        # 現在表示中のアラートメッセージ
        self.current_alert_message: str | None = None

    def _finalize_second(self, second: int) -> tuple[int, float, float] | None:
        if not self.current_left_values or not self.current_right_values:
            return None
        left_med = float(np.median(self.current_left_values))
        right_med = float(np.median(self.current_right_values))
        self.medians_history.append((second, left_med, right_med))
        # 次秒に備えてリセット
        self.current_left_values.clear()
        self.current_right_values.clear()
        return (second, left_med, right_med)

    def _moving_average(self) -> tuple[float | None, float | None]:
        if not self.medians_history:
            return None, None
        left_vals = [m[1] for m in self.medians_history]
        right_vals = [m[2] for m in self.medians_history]
        return float(np.mean(left_vals)), float(np.mean(right_vals))

    def update(self, timestamp_s: float, analysis_results: dict[Angle, dict[str, Any]]) -> list[str]:
        """各フレームで呼び出し。秒境界で集計を確定し、通知を返す。"""
        alerts: list[str] = []
        sec = int(timestamp_s)

        # 膝角度を取得
        left_knee = analysis_results.get(Angle.LEFT_KNEE)
        right_knee = analysis_results.get(Angle.RIGHT_KNEE)
        if left_knee and "angle" in left_knee:
            self.current_left_values.append(float(left_knee["angle"]))
        if right_knee and "angle" in right_knee:
            self.current_right_values.append(float(right_knee["angle"]))

        # 秒が切り替わったら前秒を確定
        if self.current_second is None:
            self.current_second = sec
            return alerts

        if sec != self.current_second:
            finalized = self._finalize_second(self.current_second)
            prev_second = self.current_second
            self.current_second = sec

            if finalized is not None:
                _, left_med, right_med = finalized
                left_ma, right_ma = self._moving_average()

                # 閾値判定（中央値、移動平均のどちらか）
                def below(x: float | None) -> bool:
                    return x is not None and x < self.threshold_deg

                triggered: list[str] = []
                if left_med < self.threshold_deg or right_med < self.threshold_deg:
                    triggered.append(f"Median L:{left_med:.1f} R:{right_med:.1f}")
                if below(left_ma) or below(right_ma):
                    lm = f"{left_ma:.1f}" if left_ma is not None else "-"
                    rm = f"{right_ma:.1f}" if right_ma is not None else "-"
                    triggered.append(f"MA5 L:{lm} R:{rm}")

                if triggered:
                    self.current_alert_message = f"[!] Knee Angle Low: {prev_second}s | " + " / ".join(triggered)
                    alerts.append(self.current_alert_message)

        return alerts

    def get_current_alert(self) -> str | None:
        """現在表示中のアラートメッセージを取得"""
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
        "body_tilt_angle",
        "neck_trunk_angle_angle",
        "neck_trunk_angle_state",
        "body_tilt_state",
        "lateral_tilt_angle",
        "lateral_tilt_state",
        "is_forward_leaning",
        "forward_lean_score",
        "forward_lean_ratio",
        "avg_forward_lean_score",
        # 腰ベース滞在検知のフィールド
        "hip_center_x",
        "hip_center_y",
        "stay_duration",
        "hip_confidence",
        "is_long_stay",
        "long_stay_alert",
        # 正規化パラメータのフィールド
        "torso_length",
        "shoulder_width",
        "person_scale",
        "normalization_base",
        "normalization_applied",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    return writer


def setup_video_writer(cap: Any, output_path: str):
    """ビデオライターをセットアップする"""
    fps = int(cap.get(getattr(cv2m, "CAP_PROP_FPS", 5)))
    width = int(cap.get(getattr(cv2m, "CAP_PROP_FRAME_WIDTH", 3)))
    height = int(cap.get(getattr(cv2m, "CAP_PROP_FRAME_HEIGHT", 4)))

    fourcc = cv2m.VideoWriter_fourcc(*"mp4v")
    return cv2m.VideoWriter(output_path, fourcc, fps, (width, height))


def write_results_to_csv(
    csv_writer,
    timestamp: float,
    frame_number: int,
    analysis_results: dict,
    posture_monitor: PostureMonitor,
    hip_detector: HipBasedStayDetector,
    hip_alert: str | None,
    landmarks: np.ndarray | None = None,
    frame_shape: tuple | None = None,
):
    """結果をCSVに書き込む"""
    # 関節角度データ
    row = {
        "timestamp": timestamp,
        "frame_number": frame_number,
    }

    # 各関節の角度と状態を記録
    for angle in Angle:
        angle_name = angle.name.lower()
        if angle in analysis_results:
            result = analysis_results[angle]
            row[f"{angle_name}_angle"] = result.get("angle", 0)
            row[f"{angle_name}_state"] = result.get("state", MovementState.STATIC).name
        else:
            row[f"{angle_name}_angle"] = 0
            row[f"{angle_name}_state"] = MovementState.STATIC.name

    # 前傾姿勢監視データ
    posture_stats = posture_monitor.get_status()

    # 最新の前傾姿勢判定
    is_forward_leaning = False
    forward_lean_score = 0.0
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

    # 腰ベース滞在検知データ
    hip_status = hip_detector.get_current_status()
    hip_pos = hip_status["hip_position"]

    row.update(
        {
            "hip_center_x": hip_pos[0] if hip_pos else 0,
            "hip_center_y": hip_pos[1] if hip_pos else 0,
            "stay_duration": hip_status["stay_duration"],
            "hip_confidence": hip_status["confidence"],
            "is_long_stay": hip_status["is_long_stay"],
            "long_stay_alert": hip_alert if hip_alert else "",
        }
    )

    # 正規化パラメータの計算と記録
    torso_length = 0.0
    shoulder_width = 0.0
    person_scale = 0.0

    if landmarks is not None and frame_shape is not None:
        # 胴体長の計算
        # NOTE: 公開メソッドがないため、一時的にprotectedメソッドを使用
        scale_torso = hip_detector._compute_person_scale(landmarks, frame_shape)  # pylint: disable=protected-access
        if scale_torso is not None:
            if hip_detector.normalization_base == "torso":
                torso_length = scale_torso
                person_scale = scale_torso
            elif hip_detector.normalization_base == "shoulder":
                shoulder_width = scale_torso  # この場合は肩幅が返される
                person_scale = scale_torso
            else:  # screen
                person_scale = scale_torso

        # 個別に胴体長と肩幅を計算（記録用）
        if hip_detector.normalization_base != "torso":
            # 胴体長を個別計算
            try:
                height, width = frame_shape[:2]
                l_sh = landmarks[BodyPart.LEFT_SHOULDER]
                r_sh = landmarks[BodyPart.RIGHT_SHOULDER]
                l_hip = landmarks[BodyPart.LEFT_HIP]
                r_hip = landmarks[BodyPart.RIGHT_HIP]

                def visible(pt: np.ndarray) -> bool:
                    try:
                        return float(pt[3]) >= float(hip_detector.confidence_threshold)
                    except (TypeError, ValueError):
                        return False

                if visible(l_sh) and visible(r_sh) and visible(l_hip) and visible(r_hip):
                    l_sh_px = (l_sh[0] * width, l_sh[1] * height)
                    r_sh_px = (r_sh[0] * width, r_sh[1] * height)
                    l_hip_px = (l_hip[0] * width, l_hip[1] * height)
                    r_hip_px = (r_hip[0] * width, r_hip[1] * height)

                    shoulder_mid = ((l_sh_px[0] + r_sh_px[0]) / 2.0, (l_sh_px[1] + r_sh_px[1]) / 2.0)
                    hip_mid = ((l_hip_px[0] + r_hip_px[0]) / 2.0, (l_hip_px[1] + r_hip_px[1]) / 2.0)
                    torso_length = float(
                        np.sqrt((shoulder_mid[0] - hip_mid[0]) ** 2 + (shoulder_mid[1] - hip_mid[1]) ** 2)
                    )

                    if hip_detector.normalization_base != "shoulder":
                        # 肩幅も個別計算
                        shoulder_width = float(np.sqrt((l_sh_px[0] - r_sh_px[0]) ** 2 + (l_sh_px[1] - r_sh_px[1]) ** 2))

            except (IndexError, TypeError):
                pass

    row.update(
        {
            "torso_length": torso_length,
            "shoulder_width": shoulder_width,
            "person_scale": person_scale,
            "normalization_base": hip_detector.normalization_base,
            "normalization_applied": hip_detector.use_normalization,
        }
    )

    csv_writer.writerow(row)


def draw_posture_alerts(frame, alerts: list[str], status: dict[str, Any], knee_alert: str | None):
    """フレームに前傾姿勢アラートと状態を描画"""
    y_offset = 30

    # アラート表示
    for alert in alerts:
        cv2m.putText(frame, alert, (10, y_offset), cv2m.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30

    # 膝角度アラートの持続表示
    if knee_alert:
        cv2m.putText(frame, knee_alert, (10, y_offset), cv2m.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30

    # 監視状態表示
    if status["sample_count"] > 0:
        status_text = (
            f"Monitor: {status['monitoring_duration']:.1f}s | "
            f"Forward: {status['forward_ratio']:.1%} | "
            f"Score: {status['avg_score']:.2f}"
        )
        cv2m.putText(frame, status_text, (10, frame.shape[0] - 20), cv2m.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


def draw_hip_stay_info(frame: np.ndarray, hip_detector: HipBasedStayDetector) -> np.ndarray:
    """腰の滞在情報を描画"""
    hip_status = hip_detector.get_current_status()
    hip_pos = hip_status["hip_position"]

    if hip_pos is None:
        return frame

    # 腰の位置にマーカーを描画
    center_x, center_y = int(hip_pos[0]), int(hip_pos[1])

    # 長期滞在かどうかで色を変更
    if hip_status["is_long_stay"]:
        color = (0, 0, 255)  # 赤色（長期滞在）
        thickness = 3
    else:
        color = (0, 255, 0)  # 緑色（通常）
        thickness = 2

    # 腰の中心に円を描画
    cv2m.circle(frame, (center_x, center_y), 8, color, thickness)

    # 滞在時間を表示
    stay_duration = hip_status["stay_duration"]
    confidence = hip_status["confidence"]

    text = f"Stay: {stay_duration:.1f}s (Conf:{confidence:.2f})"
    cv2m.putText(frame, text, (center_x + 15, center_y - 10), cv2m.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


def process_video(
    video_path: str,
    output_csv_path: str | None,
    output_video_path: str | None,
    disable_japanese: bool,
    monitoring_duration: float = 60.0,
    alert_threshold: float = 0.7,
    hip_move_threshold: float = 25.0,
    hip_stay_threshold: float = 60.0,
    hip_normalize: bool = False,
    hip_norm_base: str = "torso",
):
    """
    ビデオを処理して、関節の動きを分析し、結果をCSVとビデオに出力する。
    前傾姿勢の長期滞在も監視する。腰の位置を基準とした滞在検知も実行する。
    """
    cap = cv2m.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    p = Path(video_path)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_csv_path is None:
        output_csv_path = f"output/{p.stem}_integrated_analysis_{timestamp_str}.csv"
    if output_video_path is None:
        output_video_path = f"output/{p.stem}_integrated_output_{timestamp_str}.mp4"

    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)

    video_writer = None  # finallyブロックで参照できるよう初期化
    try:
        with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
            csv_writer = setup_csv_writer(csv_file)
            video_writer = setup_video_writer(cap, output_video_path)

            pose_estimator = PoseEstimator()
            analyzer = MovementAnalyzer()
            posture_monitor = PostureMonitor(monitoring_duration, alert_threshold)
            hip_detector = HipBasedStayDetector(
                move_threshold=hip_move_threshold,
                stay_threshold=hip_stay_threshold,
                use_normalization=hip_normalize,
                normalization_base=hip_norm_base,
                confidence_threshold=0.1,  # 閾値を低く設定
            )
            knee_monitor = KneeAngleMonitor(threshold_deg=90.0, moving_window_seconds=5)

            # パフォーマンス計測用の変数を初期化
            frame_count = 0
            total_time_spent = 0.0
            time_reading, time_posing, time_analyzing = 0.0, 0.0, 0.0
            time_drawing, time_writing, time_monitoring = 0.0, 0.0, 0.0

            print(f"Processing video: {video_path}")
            print(f"Output CSV: {output_csv_path}")
            print(f"Output Video: {output_video_path}")
            print(f"Forward Leaning Monitor: {monitoring_duration}s, Threshold: {alert_threshold:.1%}")
            if hip_normalize:
                print(
                    f"Hip-based Stay Detection: Move {hip_move_threshold} "
                    f"(normalized, base={hip_norm_base}), Stay {hip_stay_threshold}s"
                )
            else:
                print(f"Hip-based Stay Detection: Move {hip_move_threshold}px, Stay {hip_stay_threshold}s")

            while cap.isOpened():
                loop_start_time = time.perf_counter()

                start_time = time.perf_counter()
                success, frame = cap.read()
                if not success:
                    break
                time_reading += time.perf_counter() - start_time

                start_time = time.perf_counter()
                landmarks = pose_estimator.estimate(frame)
                time_posing += time.perf_counter() - start_time

                timestamp = cap.get(getattr(cv2m, "CAP_PROP_POS_MSEC", 0)) / 1000.0
                analysis_results = {}
                alerts = []
                hip_alert = None

                if landmarks is not None:
                    start_time = time.perf_counter()
                    analysis_results = analyzer.analyze(landmarks)
                    time_analyzing += time.perf_counter() - start_time

                    start_time = time.perf_counter()
                    alerts = posture_monitor.update(timestamp, frame_count, analysis_results)
                    hip_alert = hip_detector.update(landmarks, frame.shape, timestamp)
                    knee_alerts = knee_monitor.update(timestamp, analysis_results)
                    alerts.extend(knee_alerts)
                    time_monitoring += time.perf_counter() - start_time

                    start_time = time.perf_counter()
                    write_results_to_csv(
                        csv_writer,
                        timestamp,
                        frame_count,
                        analysis_results,
                        posture_monitor,
                        hip_detector,
                        hip_alert,
                        landmarks,
                        frame.shape,
                    )
                    time_writing += time.perf_counter() - start_time

                start_time = time.perf_counter()
                if landmarks is not None:
                    loop_time = time.perf_counter() - loop_start_time
                    current_fps = 1.0 / loop_time if loop_time > 0 else 0
                    frame = draw_analysis_results(
                        image=frame,
                        results=analysis_results,
                        landmarks=landmarks,
                        fps=current_fps,
                        disable_japanese=disable_japanese,
                    )
                    draw_landmarks(frame, landmarks)
                    frame = draw_hip_stay_info(frame, hip_detector)

                status = posture_monitor.get_status()
                draw_posture_alerts(frame, alerts, status, knee_monitor.get_current_alert())

                if hip_alert:
                    cv2m.putText(
                        frame,
                        f"HIP ALERT: {hip_alert}",
                        (10, frame.shape[0] - 40),
                        cv2m.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )
                time_drawing += time.perf_counter() - start_time

                for alert in alerts:
                    print(f"Frame {frame_count}: {alert}")
                if hip_alert:
                    print(f"Frame {frame_count}: {hip_alert}")

                video_writer.write(frame)
                cv2m.imshow("Integrated Analysis - Joint Movement & Hip Stay Detection", frame)
                if cv2m.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_count += 1
                total_time_spent = time.perf_counter() - loop_start_time

                if frame_count % 30 == 0:
                    total_frames = int(cap.get(getattr(cv2m, "CAP_PROP_FRAME_COUNT", 7)))
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    processing_fps = 1.0 / total_time_spent if total_time_spent > 0 else 0
                    print(
                        f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) | "
                        f"Processing FPS: {processing_fps:.1f}"
                    )

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2m.destroyAllWindows()

    # パフォーマンス分析レポート
    print("\n--- Performance Analysis Report ---")
    print(f"Total frames processed: {frame_count}")
    total_tracked_time = time_reading + time_posing + time_analyzing + time_drawing + time_writing + time_monitoring
    print(f"Total processing time: {total_tracked_time:.2f} seconds")
    if total_tracked_time > 0:
        print(f"Average FPS (including waitKey): {frame_count / total_tracked_time:.2f}")
    print("---------------------------------")
    print(f"Bottleneck Analysis (based on {total_tracked_time:.2f}s of tracked processing time):")
    if total_tracked_time > 0:
        print(f"  - AI Pose Estimation:     {time_posing:.2f}s ({100 * time_posing / total_tracked_time:5.1f}%)")
        print(
            f"  - OpenCV Operations:      {time_reading + time_drawing + time_writing:.2f}s "
            f"({100 * (time_reading + time_drawing + time_writing) / total_tracked_time:5.1f}%)"
        )
        print(f"  - Joint Analyzing:        {time_analyzing:.2f}s ({100 * time_analyzing / total_tracked_time:5.1f}%)")
        print(
            f"  - Posture & Hip Monitoring: {time_monitoring:.2f}s ({100 * time_monitoring / total_tracked_time:5.1f}%)"
        )
    print("---------------------------------")

    # 前傾姿勢監視結果
    final_status = posture_monitor.get_status()
    print("--- Forward Leaning Results ---")
    print(f"Monitor Duration: {final_status['monitoring_duration']:.1f}s")
    print(f"Forward Ratio: {final_status['forward_ratio']:.1%}")
    print(f"Average Score: {final_status['avg_score']:.3f}")
    print(f"Sample Count: {final_status['sample_count']}")

    # 腰ベース滞在検知結果
    hip_status = hip_detector.get_current_status()
    print("--- Hip-based Stay Results ---")
    print(f"Final Stay Duration: {hip_status['stay_duration']:.1f}s")
    print(f"Final Confidence: {hip_status['confidence']:.3f}")
    print(f"Final Long Stay: {hip_status['is_long_stay']}")

    print("--- End of Report ---")


def main():
    parser = argparse.ArgumentParser(description="統合版：関節動作分析 + 腰ベース長期滞在検知")
    parser.add_argument("--video", required=True, help="入力ビデオファイルのパス")
    parser.add_argument("--output-csv", help="出力CSVファイルのパス（オプション）")
    parser.add_argument("--output-video", help="出力ビデオファイルのパス（オプション）")
    parser.add_argument("--disable-japanese", action="store_true", help="日本語テキストの描画を無効にする")
    parser.add_argument(
        "--monitoring-duration", type=float, default=60.0, help="前傾姿勢監視期間（秒、デフォルト：60）"
    )
    parser.add_argument(
        "--alert-threshold", type=float, default=0.7, help="前傾姿勢アラート閾値（0.0-1.0、デフォルト：0.7）"
    )
    parser.add_argument(
        "--hip-move-threshold", type=float, default=25.0, help="腰の移動判定閾値（ピクセル、デフォルト：25）"
    )
    parser.add_argument(
        "--hip-stay-threshold", type=float, default=60.0, help="腰の長期滞在判定閾値（秒、デフォルト：60）"
    )
    parser.add_argument(
        "--hip-normalize",
        action="store_true",
        help="腰の移動判定を人物スケールで正規化（--hip-move-threshold は比率として解釈）",
    )
    parser.add_argument(
        "--hip-norm-base",
        choices=["torso", "shoulder", "screen"],
        default="torso",
        help="正規化の基準を選択（torso=体幹長, shoulder=肩幅, screen=画面高さ）",
    )

    args = parser.parse_args()

    process_video(
        video_path=args.video,
        output_csv_path=args.output_csv,
        output_video_path=args.output_video,
        disable_japanese=args.disable_japanese,
        monitoring_duration=args.monitoring_duration,
        alert_threshold=args.alert_threshold,
        hip_move_threshold=args.hip_move_threshold,
        hip_stay_threshold=args.hip_stay_threshold,
        hip_normalize=args.hip_normalize,
        hip_norm_base=args.hip_norm_base,
    )


if __name__ == "__main__":
    main()
