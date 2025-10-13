import math

import numpy as np

from src.analysis.torso_sway_detector import TorsoSwayDetector


def test_torso_sway_detector_detects_lateral_sine():
    fps = 30.0
    det = TorsoSwayDetector(
        fps=fps,
        window_sec=8.0,
        smooth_sec=0.2,
        amp_th_lat=8.0,
        amp_th_ap=6.0,
        f_min=0.2,
        f_max=1.5,
        min_cycles=3,
        on_sec=0.5,
        off_sec=0.3,
        use_staying_gate=False,
    )

    # Generate 6 seconds of 0.6Hz sine with 12deg amplitude for lateral.
    dur = 6.0
    n = int(dur * fps)
    t = np.arange(n) / fps
    lateral = 12.0 * np.sin(2 * math.pi * 0.6 * t)

    last = None
    for i in range(n):
        last = det.update(t=t[i], lateral_deg=float(lateral[i]), body_tilt_deg=180.0, hip_state="STAYING")

    assert last is not None
    lat = last["lateral"]
    # Expect sway=True and frequency around 0.6Hz
    assert lat.sway is True
    assert 0.4 <= lat.freq <= 0.9
    assert lat.amp >= 8.0


def test_torso_sway_detector_ap_sine_detects():
    fps = 30.0
    det = TorsoSwayDetector(
        fps=fps,
        window_sec=8.0,
        smooth_sec=0.2,
        amp_th_lat=8.0,
        amp_th_ap=6.0,
        f_min=0.2,
        f_max=1.5,
        min_cycles=3,
        on_sec=0.5,
        off_sec=0.3,
        use_staying_gate=False,
    )

    # BODY_TILT around 170 with AP oscillation +/- 10deg → ap_delta ~ 10*sin()
    dur = 6.0
    n = int(dur * fps)
    t = np.arange(n) / fps
    ap = 10.0 * np.sin(2 * math.pi * 0.7 * t)

    last = None
    for i in range(n):
        body_tilt = 180.0 - float(ap[i])
        last = det.update(t=t[i], lateral_deg=0.0, body_tilt_deg=body_tilt, hip_state="STAYING")

    apm = last["ap"]
    assert apm.sway is True
    assert 0.5 <= apm.freq <= 1.0
    assert apm.amp >= 6.0
