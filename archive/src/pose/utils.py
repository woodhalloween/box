"""src/pose/utils.py

This module provides utility functions for pose analysis, primarily focusing on
mathematical calculations required to interpret pose landmarks. These functions are
designed to be pure, reusable, and independent of the application's state.
"""

import numpy as np


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculates the angle between three 3D points (in degrees).

    The angle is computed at point 'b', between the vectors ba and bc.

    Args:
        a: A numpy array representing the 3D coordinates of the first point.
        b: A numpy array representing the 3D coordinates of the vertex (middle point).
        c: A numpy array representing the 3D coordinates of the third point.

    Returns:
        The calculated angle in degrees, ranging from 0 to 180.
    """
    # Create vectors from the points
    # Vector ba = a - b
    # Vector bc = c - b
    ba = a - b
    bc = c - b

    # Calculate the dot product and the norms of the vectors
    dot_product = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # Avoid division by zero
    if norm_ba == 0 or norm_bc == 0:
        # Return a default value or raise an error, depending on desired behavior.
        # Here, we return 0.0 as a neutral value.
        return 0.0

    # Calculate the cosine of the angle
    cosine_angle = dot_product / (norm_ba * norm_bc)

    # Clip the value to the valid range [-1.0, 1.0] to prevent floating point errors
    # with np.arccos
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Calculate the angle in radians and convert to degrees
    angle = np.arccos(cosine_angle)
    return float(np.degrees(angle))
