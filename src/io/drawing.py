from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark
from PIL import Image, ImageDraw, ImageFont

from ..analysis.dwell_time_detector import DwellTimeDetector
from ..analysis.user_classifier import UserClassifier
from ..definitions import Angle, MovementState
from ..head_shake_detector import HeadShakeDetector

CONNECTIONS = [
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
    (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
    (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
    (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
    (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
    (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
]

# --- 日本語表示用マッピング ---
ANGLE_JP = {
    Angle.RIGHT_ELBOW: "右肘",
    Angle.LEFT_ELBOW: "左肘",
    Angle.RIGHT_SHOULDER: "右肩",
    Angle.LEFT_SHOULDER: "左肩",
    Angle.RIGHT_HIP: "右股関節",
    Angle.LEFT_HIP: "左股関節",
    Angle.RIGHT_KNEE: "右膝",
    Angle.LEFT_KNEE: "左膝",
    Angle.BODY_TILT: "体幹の傾き",
    Angle.NECK_TRUNK_ANGLE: "頸部-体幹角度",
    Angle.LATERAL_TILT: "体幹の側屈",
}

MOVEMENT_STATE_JP = {
    MovementState.UNKNOWN: "不明",
    MovementState.FLEXION: "屈曲",
    MovementState.EXTENSION: "伸展",
    MovementState.STATIC: "静止",
    MovementState.FORWARD_TILT: "前傾",
    MovementState.UPRIGHT: "直立",
    MovementState.HUNCH: "猫背",
    MovementState.STRAIGHT: "直立（姿勢）",
    MovementState.LEFT_TILT: "左側屈",
    MovementState.RIGHT_TILT: "右側屈",
}
# --- ここまで ---


def draw_japanese_text(
    image: np.ndarray, text: str, position: tuple[int, int], font_size: int, color: tuple[int, int, int]
) -> np.ndarray:
    """Pillowを使用して画像に日本語テキストを描画する。"""
    # OpenCVの画像(BGR)をPillowの画像(RGB)に変換
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # macOSの標準的な日本語フォントを指定
    try:
        font = ImageFont.truetype("Hiragino Sans GB.ttc", font_size)
    except OSError:
        # フォントが見つからない場合は、デフォルトフォントを使用（日本語は表示されない可能性）
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)

    # Pillowの画像をOpenCVの画像に戻す
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_landmarks(image: np.ndarray, landmarks: np.ndarray) -> None:
    """骨格を描画する"""
    h, w, _ = image.shape
    for landmark in landmarks:
        x, y = int(landmark[0] * w), int(landmark[1] * h)
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    for connection in CONNECTIONS:
        start_idx = connection[0].value
        end_idx = connection[1].value
        start_point = (
            int(landmarks[start_idx][0] * w),
            int(landmarks[start_idx][1] * h),
        )
        end_point = int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h)
        cv2.line(image, start_point, end_point, (255, 255, 255), 2)


def draw_analysis_results(
    image: np.ndarray,
    results: dict[Angle, dict[str, Any]],
    landmarks: np.ndarray | None,
    fps: float = 0.0,
    disable_japanese: bool = False,
) -> np.ndarray:
    """分析結果を日本語で画像に描画する。"""
    h, w, _ = image.shape
    y_offset = 30

    img_with_text = image.copy()

    # --- FPSを描画 ---
    fps_text = f"FPS: {fps:.2f}"
    # 右上に白で描画
    img_with_text = draw_japanese_text(img_with_text, fps_text, (w - 150, 30), 20, (255, 255, 255))
    # --- ここまで ---

    for angle, data in results.items():
        angle_val = data["angle"]
        state = data["state"]

        if disable_japanese:
            # 英語表示
            angle_name = angle.value.replace("_", " ").title()
            state_name = state.value
            text = f"{angle_name}: {angle_val:.1f} deg, {state_name}"
        else:
            # 日本語表示
            angle_jp_name = ANGLE_JP.get(angle, angle.value)
            state_jp_name = MOVEMENT_STATE_JP.get(state, state.value)
            text = f"{angle_jp_name}: {angle_val:.1f} 度, {state_jp_name}"

        # 状態に応じてテキストの色を決定
        text_color = (139, 0, 0)  # デフォルトは濃い青
        if angle == Angle.BODY_TILT and state == MovementState.FORWARD_TILT:
            text_color = (0, 128, 0)  # 傾き検知（前傾）の場合は濃い緑
        elif angle == Angle.NECK_TRUNK_ANGLE and state == MovementState.HUNCH:
            text_color = (11, 134, 184)  # うつむき検知（猫背）の場合は濃い山吹色

        if disable_japanese:
            # 英語の場合は通常のOpenCV描画
            cv2.putText(img_with_text, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        else:
            # 日本語の場合はPIL描画
            img_with_text = draw_japanese_text(img_with_text, text, (10, y_offset), 20, text_color)
        y_offset += 30

    return img_with_text


def draw_detection_info(
    frame: np.ndarray,
    user_classifier: UserClassifier,
    dwell_time_detector: DwellTimeDetector,
    head_shake_detector: HeadShakeDetector,
    timestamp: float,
):
    """検知情報をフレームに描画する"""
    y_offset = 30
    user_alert = user_classifier.get_current_alert()
    if user_alert:
        cv2.putText(frame, user_alert, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += 30

    if head_shake_detector:
        head_shake_alerts = head_shake_detector.check_alerts(timestamp)
        for alert in head_shake_alerts:
            cv2.putText(frame, alert, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            y_offset += 30

    dwell_status = dwell_time_detector.get_current_status()
    if dwell_status["hip_position"]:
        pos = (int(dwell_status["hip_position"][0]), int(dwell_status["hip_position"][1]))
        duration = dwell_status["stay_duration"]
        color = (0, 0, 255) if dwell_status["is_long_stay"] else (0, 255, 0)
        cv2.circle(frame, pos, 8, color, -1)
        cv2.putText(
            frame, f"Stay: {duration:.1f}s", (pos[0] + 15, pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
