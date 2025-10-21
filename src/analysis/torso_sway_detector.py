from __future__ import annotations

from typing import Dict


class TorsoSwayDetector:
    """Simplified torso sway detector.

    判定は「絶対角度がしきい値を連続フレーム数以上でON、下回りが連続でOFF」。
    周波数・周期・平滑・窓解析およびSTAYINGゲートは使用しません。
    互換性のため、従来の引数は受け取りますが未使用のものがあります。
    """

    def __init__(
        self,
        *,
        fps: float,
        amp_th_lat: float = 10.0,
        amp_th_ap: float = 8.0,
        on_sec: float = 1.2,
        off_sec: float = 0.7,
        **_compat: object,
    ) -> None:
        self.fps = float(max(1.0, fps))
        self.amp_th_lat = float(amp_th_lat)
        self.amp_th_ap = float(amp_th_ap)
        # on/off をフレーム数に変換（少なくとも1フレーム）
        self._on_frames = max(1, int(round(float(max(0.0, on_sec)) * self.fps)))
        self._off_frames = max(1, int(round(float(max(0.0, off_sec)) * self.fps)))

        # Lateral state machine
        self._lat_state: bool = False
        self._lat_over: int = 0
        self._lat_under: int = 0

        # AP state machine
        self._ap_state: bool = False
        self._ap_over: int = 0
        self._ap_under: int = 0

    @staticmethod
    def _step(
        val: float,
        threshold: float,
        state: bool,
        over: int,
        under: int,
        on_frames: int,
        off_frames: int,
    ) -> tuple[bool, int, int]:
        if val >= threshold:
            over += 1
            under = 0
        else:
            under += 1
            over = 0

        if (not state) and over >= on_frames:
            state = True
        if state and under >= off_frames:
            state = False
        return state, over, under

    # ---------------- Public API ---------------- #
    def update(
        self,
        *,
        t: float,  # kept for compatibility (unused)
        lateral_deg: float | None,
        body_tilt_deg: float | None,
        hip_state: str | None = None,  # kept for compatibility (unused)
    ) -> Dict[str, bool]:
        """Update detector with new angles (boolean flags only).

        Args:
            t: timestamp in seconds (unused)
            lateral_deg: lateral tilt angle (deg)
            body_tilt_deg: body tilt angle (deg)
            hip_state: ignored
        Returns:
            dict with boolean flags: {"lateral": bool, "ap": bool}
        """
        lat_val = abs(float(lateral_deg)) if lateral_deg is not None else 0.0
        ap_val = abs(180.0 - float(body_tilt_deg)) if body_tilt_deg is not None else 0.0

        # Lateral axis
        self._lat_state, self._lat_over, self._lat_under = self._step(
            lat_val,
            self.amp_th_lat,
            self._lat_state,
            self._lat_over,
            self._lat_under,
            self._on_frames,
            self._off_frames,
        )

        # AP axis
        self._ap_state, self._ap_over, self._ap_under = self._step(
            ap_val,
            self.amp_th_ap,
            self._ap_state,
            self._ap_over,
            self._ap_under,
            self._on_frames,
            self._off_frames,
        )

        # explicitly touch unused args to satisfy linters
        _ = (t, hip_state)
        return {"lateral": bool(self._lat_state), "ap": bool(self._ap_state)}


