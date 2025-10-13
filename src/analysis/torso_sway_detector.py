from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Literal

import numpy as np


SwayLevel = Literal["none", "small", "medium", "large"]


@dataclass
class SwayMetrics:
    amp: float
    freq: float
    cycles: int
    sway: bool
    level: SwayLevel


class TorsoSwayDetector:
    """Detect torso sway (lateral and anterior-posterior) from angle time series.

    The detector operates continuously (not gated by dwell/STAYING by default) and
    suppresses flicker using hysteresis windows.
    """

    def __init__(
        self,
        *,
        fps: float,
        window_sec: float = 8.0,
        smooth_sec: float = 0.5,
        amp_th_lat: float = 10.0,
        amp_th_ap: float = 8.0,
        f_min: float = 0.2,
        f_max: float = 1.5,
        min_cycles: int = 3,
        on_sec: float = 1.2,
        off_sec: float = 0.7,
        use_staying_gate: bool = False,
    ) -> None:
        self.fps = float(max(1.0, fps))
        self.window_len = int(max(1.0, window_sec) * self.fps)
        self.smooth_len = max(1, int(max(0.0, smooth_sec) * self.fps))

        self.amp_th_lat = float(amp_th_lat)
        self.amp_th_ap = float(amp_th_ap)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.min_cycles = int(max(0, min_cycles))

        self.on_sec = float(max(0.0, on_sec))
        self.off_sec = float(max(0.0, off_sec))
        self.use_staying_gate = bool(use_staying_gate)

        self.lat_hist: Deque[float] = deque(maxlen=self.window_len)
        self.ap_hist: Deque[float] = deque(maxlen=self.window_len)
        self.t_hist: Deque[float] = deque(maxlen=self.window_len)

        # Hysteresis accumulators (seconds)
        self._lat_on_acc: float = 0.0
        self._lat_off_acc: float = 0.0
        self._ap_on_acc: float = 0.0
        self._ap_off_acc: float = 0.0
        self._lat_state: bool = False
        self._ap_state: bool = False

    # ---------------- Internal helpers ---------------- #
    def _smooth(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        if self.smooth_len <= 1 or x.size < self.smooth_len:
            return x.copy()
        k = self.smooth_len
        kernel = np.ones(k, dtype=float) / float(k)
        y = np.convolve(x, kernel, mode="valid")
        # left pad to keep length consistent
        pad = np.full(x.size - y.size, y[0], dtype=float)
        return np.concatenate([pad, y])

    @staticmethod
    def _percentile_amp(x: np.ndarray) -> float:
        # Robust amplitude: half of (P95 - P5)
        if x.size == 0:
            return 0.0
        p95 = np.percentile(x, 95)
        p5 = np.percentile(x, 5)
        return 0.5 * float(p95 - p5)

    @staticmethod
    def _zero_crossings(series: np.ndarray) -> int:
        if series.size < 2:
            return 0
        signs = np.signbit(series)
        return int(np.count_nonzero(signs[:-1] != signs[1:]))

    @staticmethod
    def _level_from_amp(amp: float, small: float, medium: float, large: float) -> SwayLevel:
        if amp >= large:
            return "large"
        if amp >= medium:
            return "medium"
        if amp >= small:
            return "small"
        return "none"

    def _analyze_window(self, series: Deque[float]) -> tuple[float, float, int]:
        if len(self.t_hist) < 2 or len(series) < 2:
            return 0.0, 0.0, 0
        s = np.asarray(series, dtype=float)
        s = self._smooth(s)
        d = s - float(np.median(s))

        amp = self._percentile_amp(d)
        zc = self._zero_crossings(d)
        cycles = max(0, zc // 2)
        duration = max(1e-6, float(self.t_hist[-1] - self.t_hist[0]))
        freq = cycles / duration
        return float(amp), float(freq), int(cycles)

    # ---------------- Public API ---------------- #
    def update(
        self,
        *,
        t: float,
        lateral_deg: float | None,
        body_tilt_deg: float | None,
        hip_state: str | None = None,
    ) -> Dict[str, SwayMetrics]:
        """Update detector with new angles.

        Args:
            t: timestamp in seconds
            lateral_deg: lateral tilt angle (deg)
            body_tilt_deg: body tilt angle (deg) (use 180 - angle as AP sway series)
            hip_state: optional gating state (e.g., "STAYING"); used only if use_staying_gate=True
        Returns:
            dict with keys "lateral" and "ap" mapping to SwayMetrics
        """
        # Optional gating by STAYING
        gated = self.use_staying_gate and (hip_state is not None) and (hip_state != "STAYING")
        if gated:
            self.lat_hist.clear()
            self.ap_hist.clear()
            self.t_hist.clear()
            self._lat_on_acc = self._lat_off_acc = 0.0
            self._ap_on_acc = self._ap_off_acc = 0.0
            self._lat_state = self._ap_state = False
            return {
                "lateral": SwayMetrics(amp=0.0, freq=0.0, cycles=0, sway=False, level="none"),
                "ap": SwayMetrics(amp=0.0, freq=0.0, cycles=0, sway=False, level="none"),
            }

        # Append samples
        self.t_hist.append(float(t))
        self.lat_hist.append(float(lateral_deg) if lateral_deg is not None else 0.0)
        ap_delta = 0.0
        if body_tilt_deg is not None:
            ap_delta = 180.0 - float(body_tilt_deg)
        self.ap_hist.append(ap_delta)

        # Analyze window for both axes
        lat_amp, lat_f, lat_cycles = self._analyze_window(self.lat_hist)
        ap_amp, ap_f, ap_cycles = self._analyze_window(self.ap_hist)

        lat_ok = (lat_amp >= self.amp_th_lat) and (self.f_min <= lat_f <= self.f_max) and (lat_cycles >= self.min_cycles)
        ap_ok = (ap_amp >= self.amp_th_ap) and (self.f_min <= ap_f <= self.f_max) and (ap_cycles >= self.min_cycles)

        # Hysteresis accumulators
        dt = 0.0
        if len(self.t_hist) >= 2:
            dt = float(self.t_hist[-1] - self.t_hist[-2])

        # Lateral
        if lat_ok:
            self._lat_on_acc += dt
            self._lat_off_acc = 0.0
        else:
            self._lat_off_acc += dt
            self._lat_on_acc = 0.0
        if not self._lat_state and self._lat_on_acc >= self.on_sec:
            self._lat_state = True
        if self._lat_state and self._lat_off_acc >= self.off_sec:
            self._lat_state = False

        # AP
        if ap_ok:
            self._ap_on_acc += dt
            self._ap_off_acc = 0.0
        else:
            self._ap_off_acc += dt
            self._ap_on_acc = 0.0
        if not self._ap_state and self._ap_on_acc >= self.on_sec:
            self._ap_state = True
        if self._ap_state and self._ap_off_acc >= self.off_sec:
            self._ap_state = False

        lat_level = self._level_from_amp(lat_amp, small=self.amp_th_lat, medium=max(self.amp_th_lat + 2.0, self.amp_th_lat * 1.2), large=max(self.amp_th_lat + 8.0, self.amp_th_lat * 1.8))
        ap_level = self._level_from_amp(ap_amp, small=self.amp_th_ap, medium=max(self.amp_th_ap + 2.0, self.amp_th_ap * 1.25), large=max(self.amp_th_ap + 7.0, self.amp_th_ap * 1.8))

        return {
            "lateral": SwayMetrics(amp=float(lat_amp), freq=float(lat_f), cycles=int(lat_cycles), sway=bool(self._lat_state), level=lat_level),
            "ap": SwayMetrics(amp=float(ap_amp), freq=float(ap_f), cycles=int(ap_cycles), sway=bool(self._ap_state), level=ap_level),
        }


