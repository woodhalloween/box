from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark
from PIL import Image, ImageDraw, ImageFont

from src.definitions import Angle, MovementState

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
    landmarks: np.ndarray,
) -> np.ndarray:
    """分析結果を日本語で画像に描画する。"""
    h, w, _ = image.shape
    y_offset = 30

    img_with_text = image.copy()

    for angle, data in results.items():
        angle_val = data["angle"]
        state = data["state"]

        # マッピングから日本語名を取得。見つからなければEnumの値をそのまま使う
        angle_jp_name = ANGLE_JP.get(angle, angle.value)
        state_jp_name = MOVEMENT_STATE_JP.get(state, state.value)

        text = f"{angle_jp_name}: {angle_val:.1f} 度, {state_jp_name}"
        img_with_text = draw_japanese_text(
            img_with_text,
            text,
            (10, y_offset),
            20,
            (255, 0, 0),  # 白色から赤色に変更
        )
        y_offset += 30

    # --- デバッグ用の描画: 体幹の中心線と垂直線 ---
    if Angle.BODY_TILT in results and landmarks is not None:
        p_left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER.value]
        p_right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER.value]
        p_left_hip = landmarks[PoseLandmark.LEFT_HIP.value]
        p_right_hip = landmarks[PoseLandmark.RIGHT_HIP.value]

        p_shoulder_mid = (
            int(((p_left_shoulder[0] + p_right_shoulder[0]) / 2) * w),
            int(((p_left_shoulder[1] + p_right_shoulder[1]) / 2) * h),
        )
        p_hip_mid = (
            int(((p_left_hip[0] + p_right_hip[0]) / 2) * w),
            int(((p_left_hip[1] + p_right_hip[1]) / 2) * h),
        )

        # 体幹の中心線 (緑)
        cv2.line(image, p_hip_mid, p_shoulder_mid, (0, 255, 0), 2)
        # 垂直線 (青)
        cv2.line(
            image,
            p_hip_mid,
            (p_hip_mid[0], p_hip_mid[1] + 100),
            (255, 0, 0),
            2,
        )

    return img_with_text
