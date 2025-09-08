from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ..definitions import Angle


class UserClassifier:
    """膝角度の秒毎中央値と5秒移動平均を監視し、閾値を下回ると通知する."""

    def __init__(
        self,
        threshold_deg: float = 90.0,
        moving_window_seconds: int = 5,
        confidence_threshold: float = 0.7,
    ) -> None:
        """
        UserClassifierを初期化する.

        Args:
            threshold_deg: アラートを発する膝角度の閾値（度）.
            moving_window_seconds: 移動平均を計算するためのウィンドウサイズ（秒）.
            confidence_threshold: 角度計算の信頼度スコアの閾値.
        """
        self.threshold_deg = threshold_deg
        self.moving_window_seconds = moving_window_seconds
        self.confidence_threshold = confidence_threshold
        self.current_second: int | None = None
        self.current_left_values: list[float] = []
        self.current_right_values: list[float] = []
        self.medians_history: deque[tuple[int, float | None, float | None]] = deque(maxlen=moving_window_seconds)
        self.current_alert_message: str | None = None

    def _finalize_second(self, second: int) -> tuple[int, float | None, float | None] | None:
        if not self.current_left_values and not self.current_right_values:
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
        left_vals = [m[1] for m in self.medians_history if m[1] is not None]
        right_vals = [m[2] for m in self.medians_history if m[2] is not None]

        left_ma = float(np.mean(left_vals)) if left_vals else None
        right_ma = float(np.mean(right_vals)) if right_vals else None
        return left_ma, right_ma

    def update(self, timestamp_s: float, analysis_results: dict[Angle, dict[str, Any]]) -> list[str]:
        alerts: list[str] = []
        sec = int(timestamp_s)

        if self.current_second is None:
            self.current_second = sec

        finalized_median_info = None
        if sec != self.current_second:
            finalized_median_info = self._finalize_second(self.current_second)
            self.current_second = sec

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

        if finalized_median_info:
            second, left_med, right_med = finalized_median_info
            left_ma, right_ma = self._moving_average()
            triggered = []
            if (left_med is not None and left_med < self.threshold_deg) or (
                right_med is not None and right_med < self.threshold_deg
            ):
                lm = f"{left_med:.1f}" if left_med is not None else "-"
                rm = f"{right_med:.1f}" if right_med is not None else "-"
                triggered.append(f"Median L:{lm} R:{rm}")

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
