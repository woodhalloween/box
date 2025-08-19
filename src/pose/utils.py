from __future__ import annotations

import numpy as np


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """3つの点の座標から角度を計算する (p2が中心)。"""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    return angle


def calculate_midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """2つの点の座標から中点を計算する。"""
    return (p1 + p2) / 2
