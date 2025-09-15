"""src/pose/utils.pyのテスト"""

import numpy as np
import pytest

from src.pose.utils import calculate_angle, calculate_midpoint


@pytest.mark.parametrize(
    "p1, p2, p3, expected_angle",
    [
        # 直角のケース
        (np.array([1, 0]), np.array([0, 0]), np.array([0, 1]), 90.0),
        # 直線のケース
        (np.array([1, 0]), np.array([0, 0]), np.array([-1, 0]), 180.0),
        # 0度のケース
        (np.array([1, 0]), np.array([0, 0]), np.array([2, 0]), 0.0),
        # 45度のケース
        (np.array([1, 0]), np.array([0, 0]), np.array([1, 1]), 45.0),
        # 3D座標のケース
        (np.array([1, 0, 0]), np.array([0, 0, 0]), np.array([0, 1, 0]), 90.0),
    ],
)
def test_calculate_angle(p1, p2, p3, expected_angle):
    """calculate_angleが正しい角度を計算するかテストする"""
    angle = calculate_angle(p1, p2, p3)
    assert np.isclose(angle, expected_angle)


@pytest.mark.parametrize(
    "p1, p2, expected_midpoint",
    [
        # 2D座標
        (np.array([0, 0]), np.array([2, 2]), np.array([1, 1])),
        # 2D座標（負の値を含む）
        (np.array([-1, -1]), np.array([1, 1]), np.array([0, 0])),
        # 3D座標
        (np.array([0, 0, 0]), np.array([2, 4, 6]), np.array([1, 2, 3])),
    ],
)
def test_calculate_midpoint(p1, p2, expected_midpoint):
    """calculate_midpointが正しい中点を計算するかテストする"""
    midpoint = calculate_midpoint(p1, p2)
    np.testing.assert_array_equal(midpoint, expected_midpoint)
