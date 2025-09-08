from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from ..definitions import Angle, MovementState


@dataclass
class PostureSnapshot:
    """姿勢スナップショット - 特定時刻の姿勢データ"""

    timestamp: float
    frame_number: int
    analysis_results: dict[Angle, dict[str, Any]]
    is_forward_leaning: bool
    forward_lean_score: float


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
