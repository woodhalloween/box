"""
CSVの書き出しに関するヘルパー関数をまとめたモジュール。
"""

from __future__ import annotations

import csv
from typing import IO, Any

import mediapipe as mp
import numpy as np

from ..analysis.dwell_time_detector import DwellTimeDetector
from ..analysis.posture_monitor import PostureMonitor
from ..analysis.user_classifier import UserClassifier
from ..definitions import Angle, MovementState
from ..detectors import HandRaiseDetector
from ..head_shake_detector import HeadShakeDetector


def setup_csv_writer(csv_file: IO) -> csv.DictWriter:
    """CSVライターをセットアップする"""
    fieldnames = [
        "timestamp",
        "frame_number",
        "debug_analysis_results_empty",  # デバッグ用の列を追加
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
        "left_hand_raised",
        "right_hand_raised",
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
        "user_classifier_alert",
        # --- Torso sway metrics (lateral/AP) ---
        "torso_sway_lat_amp",
        "torso_sway_lat_freq",
        "torso_sway_lat_cycles",
        "torso_sway_lat_flag",
        "torso_sway_lat_level",
        "torso_sway_ap_amp",
        "torso_sway_ap_freq",
        "torso_sway_ap_cycles",
        "torso_sway_ap_flag",
        "torso_sway_ap_level",
        # --- MediaPipe Face Mesh 頭部方向検知 ---
        "mediapipe_yaw_angle",
        "mediapipe_face_detected",
        "mediapipe_turn_direction",
        "mediapipe_sustained_turn_detected",
        "mediapipe_sustained_direction",
        "mediapipe_sustained_frames",
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
    hand_raise_detector: HandRaiseDetector | None,
    hand_statuses: dict[str, bool] | None,
    landmarks: np.ndarray | None,
    user_classifier: UserClassifier | None = None,
    torso_sway: dict | None = None,
    mediapipe_head_turn_detector=None,  # MediaPipeFaceMeshHeadTurnDetector | None
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

    # Head Shake Status
    head_shake_status = head_shake_detector.get_status() if head_shake_detector else {}
    row.update(
        {
            "head_shake_horizontal_detected": head_shake_status.get("horizontal_state", "HEAD_STATIC") != "HEAD_STATIC",
            "head_shake_vertical_detected": head_shake_status.get("vertical_state", "HEAD_STATIC") != "HEAD_STATIC",
            "head_shake_alerts": "; ".join(head_shake_alerts) if head_shake_alerts else "",
        }
    )

    # Hand Raise Status
    if hand_statuses is not None:
        row["left_hand_raised"] = bool(hand_statuses.get("left_hand_raised", False))
        row["right_hand_raised"] = bool(hand_statuses.get("right_hand_raised", False))
    else:
        row["left_hand_raised"] = False
        row["right_hand_raised"] = False

    # User classifier alert
    user_classifier_alert = user_classifier.get_current_alert() if user_classifier else None
    row["user_classifier_alert"] = user_classifier_alert or ""

    # Torso sway metrics (optional)
    lat = (torso_sway or {}).get("lateral", {})
    ap = (torso_sway or {}).get("ap", {})
    row.update(
        {
            "torso_sway_lat_amp": float(lat.get("amp", 0.0) or 0.0),
            "torso_sway_lat_freq": float(lat.get("freq", 0.0) or 0.0),
            "torso_sway_lat_cycles": int(lat.get("cycles", 0) or 0),
            "torso_sway_lat_flag": bool(lat.get("sway", False)),
            "torso_sway_lat_level": str(lat.get("level", "none")),
            "torso_sway_ap_amp": float(ap.get("amp", 0.0) or 0.0),
            "torso_sway_ap_freq": float(ap.get("freq", 0.0) or 0.0),
            "torso_sway_ap_cycles": int(ap.get("cycles", 0) or 0),
            "torso_sway_ap_flag": bool(ap.get("sway", False)),
            "torso_sway_ap_level": str(ap.get("level", "none")),
        }
    )

    # MediaPipe Face Mesh 頭部方向検知の結果
    if mediapipe_head_turn_detector:
        status = mediapipe_head_turn_detector.get_status()
        row.update(
            {
                "mediapipe_yaw_angle": status.get("yaw_angle", 0.0),
                "mediapipe_face_detected": status.get("face_detected", False),
                "mediapipe_turn_direction": status.get("direction", ""),
                "mediapipe_sustained_turn_detected": status.get("is_sustained", False),
                "mediapipe_sustained_direction": status.get("current_direction", ""),
                "mediapipe_sustained_frames": status.get("consecutive_frames", 0),
            }
        )
    else:
        row.update(
            {
                "mediapipe_yaw_angle": 0.0,
                "mediapipe_face_detected": False,
                "mediapipe_turn_direction": "",
                "mediapipe_sustained_turn_detected": False,
                "mediapipe_sustained_direction": "",
                "mediapipe_sustained_frames": 0,
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
