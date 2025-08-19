"""src/pose/drawing_utils.py

This module contains utility functions for drawing pose-related information onto
image frames using OpenCV. These functions are designed to be used by the main
application loop to visualize the output of the pose analysis.
"""

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

from .definitions import Movement

# MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def draw_landmarks(image: np.ndarray, landmarks: landmark_pb2.NormalizedLandmarkList) -> None:
    """フレームに骨格ランドマークを描画する"""
    if landmarks:
        mp_drawing.draw_landmarks(
            image,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )


def draw_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_scale: float = 1.0,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """
    Draws text with a background on the frame.

    Args:
        frame: The image frame on which to draw.
        text: The text string to be drawn.
        position: A tuple (x, y) representing the top-left corner of the text.
        font_scale: The scale of the font.
        color: The color of the text in BGR format.
        thickness: The thickness of the font.
        bg_color: The color of the text's background.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    rect_start = position
    rect_end = (position[0] + text_width + 10, position[1] - text_height - 10)

    # Draw a filled rectangle as a background
    cv2.rectangle(frame, rect_start, rect_end, bg_color, -1)

    # Put the text on top of the rectangle
    cv2.putText(
        frame,
        text,
        (position[0] + 5, position[1] - 5),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_analysis_results(image: np.ndarray, results: dict, font_size: float = 0.7, thickness: int = 2) -> None:
    """フレームに分析結果（関節角度と運動状態）を描画する"""
    y_offset = 30
    for joint, (angle, movement) in results.items():
        # movement Enumメンバーを可読な文字列に変換
        movement_text = movement.name if movement != Movement.NONE else ""

        text = f"{joint.replace('_', ' ').title()}: {angle:.1f} deg {movement_text}"
        cv2.putText(
            image,
            text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y_offset += 30
