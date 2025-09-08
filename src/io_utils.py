"""
CSVやビデオの入出力に関するヘルパー関数をまとめたモジュール。
"""
from __future__ import annotations

import csv
from typing import IO, Any

import cv2
import mediapipe as mp
import numpy as np

from .analysis.dwell_time_detector import DwellTimeDetector
from .analysis.posture_monitor import PostureMonitor
from .definitions import Angle, MovementState
from .head_shake_detector import HeadShakeDetector


def setup_video_writer(cap: cv2.VideoCapture, output_path: str) -> cv2.VideoWriter:
    """ビデオライターをセットアップする"""
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def setup_csv_writer(csv_file: IO) -> csv.DictWriter:
    """CSVライターをセットアップする"""
    fieldnames = [
        "timestamp",
        "frame_number",
        "right_elbow_angle",
        "right_elbow_state",
        "left_elbow_angle",
        "left_elbow_state",
        "right_shoulder_angle",
        "right_shoulder_state",
        "left_shoulder_angle",
        "left_shoulder_state",
        "right_hip_angle",
        "right_hip_state",
        "left_hip_angle",
        "left_hip_state",
        "right_knee_angle",
        "right_knee_state",
        "left_knee_angle",
        "left_knee_state",
        "right_knee_hip_confidence",
        "right_knee_knee_confidence",
        "right_knee_ankle_confidence",
        "left_knee_hip_confidence",
        "left_knee_knee_confidence",
        "left_knee_ankle_confidence",
        "body_tilt_angle",
        "neck_trunk_angle_angle",
        "neck_trunk_angle_state",
        "body_tilt_state",
        "lateral_tilt_angle",
        "lateral_tilt_state",
        "head_horizontal_rotation_angle",
        "head_horizontal_rotation_state",
        "head_vertical_nod_angle",
        "head_vertical_nod_state",
        "head_shake_horizontal_detected",
        "head_shake_vertical_detected",
        "head_shake_alerts",
        "is_forward_leaning",
        "forward_lean_score",
        "forward_lean_ratio",
        "avg_forward_lean_score",
        "hip_center_x",
        "hip_center_y",
        "stay_duration",
        "hip_confidence",
        "is_long_stay",
        "long_stay_alert",
        "hip_detector_state",
    ]
    landmark_fieldnames = []
    for landmark in mp.solutions.pose.PoseLandmark:
        name = landmark.name
        landmark_fieldnames.extend([f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_visibility"])
    fieldnames.extend(landmark_fieldnames)

    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    return writer


def write_results_to_csv(
    csv_writer: csv.DictWriter,
    timestamp: float,
    frame_number: int,
    analysis_results: dict,
    posture_monitor: PostureMonitor,
    dwell_time_detector: DwellTimeDetector,
    dwell_alert: str | None,
    head_shake_detector: HeadShakeDetector | None,
    head_shake_alerts: list[str] | None,
    landmarks: np.ndarray | None,
):
    """結果をCSVに書き込む"""
    row: dict[str, Any] = {"timestamp": timestamp, "frame_number": frame_number}

    for angle in Angle:
        angle_name = angle.name.lower()
        if angle in analysis_results:
            result = analysis_results[angle]
            row[f"{angle_name}_angle"] = result.get("angle", 0)
            row[f"{angle_name}_state"] = result.get("state", MovementState.STATIC).name
        else:
            row[f"{angle_name}_angle"] = 0
            row[f"{angle_name}_state"] = MovementState.STATIC.name

    rk_result = analysis_results.get(Angle.RIGHT_KNEE, {})
    row["right_knee_hip_confidence"] = rk_result.get("p1_confidence", 0.0)
    row["right_knee_knee_confidence"] = rk_result.get("p2_confidence", 0.0)
    row["right_knee_ankle_confidence"] = rk_result.get("p3_confidence", 0.0)

    lk_result = analysis_results.get(Angle.LEFT_KNEE, {})
    row["left_knee_hip_confidence"] = lk_result.get("p1_confidence", 0.0)
    row["left_knee_knee_confidence"] = lk_result.get("p2_confidence", 0.0)
    row["left_knee_ankle_confidence"] = lk_result.get("p3_confidence", 0.0)

    posture_stats = posture_monitor.get_status()
    is_forward_leaning, forward_lean_score = False, 0.0
    if posture_monitor.posture_history:
        latest = posture_monitor.posture_history[-1]
        is_forward_leaning = latest.is_forward_leaning
        forward_lean_score = latest.forward_lean_score
    row.update(
        {
            "is_forward_leaning": is_forward_leaning,
            "forward_lean_score": forward_lean_score,
            "forward_lean_ratio": posture_stats["forward_ratio"],
            "avg_forward_lean_score": posture_stats["avg_score"],
        }
    )

    dwell_status = dwell_time_detector.get_current_status()
    hip_pos = dwell_status["hip_position"]
    row.update(
        {
            "hip_center_x": hip_pos[0] if hip_pos else 0,
            "hip_center_y": hip_pos[1] if hip_pos else 0,
            "stay_duration": dwell_status["stay_duration"],
            "hip_confidence": dwell_status["confidence"],
            "is_long_stay": dwell_status["is_long_stay"],
            "long_stay_alert": dwell_alert or "",
            "hip_detector_state": dwell_status["state"],
        }
    )

    head_shake_status = head_shake_detector.get_status() if head_shake_detector else {}
    row.update(
        {
            "head_shake_horizontal_detected": head_shake_status.get("horizontal_state", "HEAD_STATIC")
            != "HEAD_STATIC",
            "head_shake_vertical_detected": head_shake_status.get("vertical_state", "HEAD_STATIC") != "HEAD_STATIC",
            "head_shake_alerts": "; ".join(head_shake_alerts) if head_shake_alerts else "",
        }
    )

    if landmarks is not None:
        for landmark in mp.solutions.pose.PoseLandmark:
            name = landmark.name
            idx = landmark.value
            row[f"{name}_x"] = landmarks[idx][0]
            row[f"{name}_y"] = landmarks[idx][1]
            row[f"{name}_z"] = landmarks[idx][2]
            row[f"{name}_visibility"] = landmarks[idx][3]
    else:
        for landmark in mp.solutions.pose.PoseLandmark:
            name = landmark.name
            row.update({f"{name}_x": 0.0, f"{name}_y": 0.0, f"{name}_z": 0.0, f"{name}_visibility": 0.0})

    csv_writer.writerow(row)
