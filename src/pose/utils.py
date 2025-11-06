from __future__ import annotations

import numpy as np


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """3つの点の座標から角度を計算する (p2が中心)。"""
    v1 = p1 - p2
    v2 = p3 - p2
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    # ゼロベクトル（または非常に小さいベクトル）の場合は NaN を返す（角度が定義できない）
    epsilon = 1e-10
    if norm1 < epsilon or norm2 < epsilon:
        return np.nan

    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def calculate_midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """2つの点の座標から中点を計算する。"""
    return (p1 + p2) / 2
