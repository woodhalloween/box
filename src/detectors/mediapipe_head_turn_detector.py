"""
mediapipe_head_turn_detector.py

MediaPipe Face Meshを使用した頭部方向検出モジュール。
PostureAndMotionDetectorBaseを継承し、顔ランドマークから
ヨー角を計算して頭部の方向（左向き・右向き・正面）を検知する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from ..definitions import MovementState
from .base_detector import PostureAndMotionDetectorBase


@dataclass
class _HeadTurnState:
    """頭部方向の内部状態を保持するデータクラス。

    HandRaiseDetectorの_HandStateと同様の設計パターンを採用。

    Attributes:
        consecutive_frames (int): 同じ方向が連続して検知されたフレーム数
        current_direction (str): 現在の方向（"正面" / "左向き" / "右向き"）
        is_sustained (bool): 持続的な方向転換が検知されているか
        yaw_angle (float): 最新のヨー角（度）
        confidence (float): ランドマーク検出の信頼度
        face_detected (bool): 顔が検出されたか
    """

    consecutive_frames: int = 0
    current_direction: str = "正面"
    is_sustained: bool = False
    yaw_angle: float = 0.0
    confidence: float = 0.0
    face_detected: bool = False


class MediaPipeFaceMeshHeadTurnDetector(PostureAndMotionDetectorBase):
    """MediaPipe Face Meshを使用した頭部方向検出器。

    PostureAndMotionDetectorBaseを継承し、468個の顔ランドマークから
    ヨー角を計算して頭部の左右方向を高精度で検知する。

    主な機能:
    - MediaPipe Face Meshで顔ランドマークを検出
    - ヨー角（Yaw angle）を計算
    - 連続フレームでの方向判定
    - 持続的な方向転換の検知（クールダウン管理付き）

    Attributes:
        yaw_threshold_right (float): 右向き判定の閾値（度）
        yaw_threshold_left (float): 左向き判定の閾値（度）
        min_consecutive_frames (int): 持続的方向転換に必要な連続フレーム数
        cooldown_sec (float): 再検知抑制期間（秒）
    """

    # MediaPipe Face Meshで使用するランドマークインデックス
    # https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md
    NOSE_TIP = 1  # 鼻先
    LEFT_EYE_OUTER = 33  # 左目外側
    RIGHT_EYE_OUTER = 263  # 右目外側
    LEFT_MOUTH_CORNER = 61  # 左口角
    RIGHT_MOUTH_CORNER = 291  # 右口角

    def __init__(
        self,
        yaw_threshold_right: float = 30.0,  # 分析結果に基づき調整
        yaw_threshold_left: float = -30.0,  # 分析結果に基づき調整
        min_consecutive_frames: int = 3,
        cooldown_sec: float = 10.0,
        confidence_threshold: float = 0.5,
    ):
        """MediaPipeFaceMeshHeadTurnDetectorを初期化する。

        Args:
            yaw_threshold_right (float): 右向き判定の閾値（度）。
                この値以上のヨー角で「右向き」と判定される。
            yaw_threshold_left (float): 左向き判定の閾値（度）。
                この値以下のヨー角で「左向き」と判定される。
            min_consecutive_frames (int): 持続的方向転換に必要な連続フレーム数。
                HandRaiseDetectorと同様のパターン。
            cooldown_sec (float): 再検知抑制期間（秒）。
                一度検知した後、この期間内は再通知しない。
            confidence_threshold (float): ランドマーク検出の信頼度閾値。
                この値を下回る場合は検出を無効とする。
        """
        super().__init__(confidence_threshold=confidence_threshold)

        self.yaw_threshold_right = yaw_threshold_right
        self.yaw_threshold_left = yaw_threshold_left
        self.min_consecutive_frames = min_consecutive_frames
        self.cooldown_sec = cooldown_sec

        # MediaPipe Face Mesh初期化
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold,
        )

        # 内部状態
        self._head_state = _HeadTurnState()

        # 履歴管理（base_detectorの機能を使用）
        self._direction_history = self._create_history_deque("direction", maxlen=min_consecutive_frames * 2)

        # クールダウン管理
        self._last_sustained_turn_time: float = float("-inf")

    def calculate_yaw_angle(self, face_landmarks) -> tuple[float, float]:
        """顔ランドマークからヨー角と信頼度を計算する。

        MediaPipe Face Meshの鼻先・目・口のランドマークを使用して、
        頭部の左右方向の回転角度（ヨー角）を計算する。

        計算方法:
        1. 左右の目の外側点の中点を計算
        2. 鼻先から目中点までのオフセットを計算
        3. 目間距離で正規化し、角度に変換

        Args:
            face_landmarks: MediaPipe Face Meshのランドマークリスト

        Returns:
            tuple[float, float]: (ヨー角, 信頼度)
                - ヨー角: -90度（左向き）〜 +90度（右向き）
                - 信頼度: 使用したランドマークの最小信頼度
        """
        try:
            # 必要なランドマークを取得
            nose = face_landmarks.landmark[self.NOSE_TIP]
            left_eye = face_landmarks.landmark[self.LEFT_EYE_OUTER]
            right_eye = face_landmarks.landmark[self.RIGHT_EYE_OUTER]

            # MediaPipe Face Meshはpresenceを持つが、visibilityは常に1.0なので
            # 信頼度は1.0として扱う
            confidence = 1.0

            # 目の中点を計算
            eye_midpoint_x = (left_eye.x + right_eye.x) / 2

            # 目間距離を計算（ゼロ除算防止）
            eye_distance = np.sqrt((right_eye.x - left_eye.x) ** 2 + (right_eye.y - left_eye.y) ** 2)

            if eye_distance < 1e-6:
                return 0.0, confidence

            # 鼻先のオフセットを計算
            offset_x = nose.x - eye_midpoint_x

            # 正規化されたオフセット比率
            offset_ratio = offset_x / eye_distance

            # ヨー角に変換（-180度〜+180度の範囲に拡大）
            # 目間距離は顔の幅の約1/3なので、係数を調整
            yaw_angle = offset_ratio * 180.0

            return float(yaw_angle), float(confidence)

        except (AttributeError, IndexError, TypeError, ValueError) as e:
            # デバッグ用にエラー情報を出力
            print(f"ヨー角計算エラー: {e}")
            return 0.0, 0.0

    def _classify_direction(self, yaw_angle: float) -> str:
        """ヨー角から頭部の方向を分類する。

        Args:
            yaw_angle (float): ヨー角（度）

        Returns:
            str: 方向（"正面" / "左向き" / "右向き"）
        """
        if yaw_angle >= self.yaw_threshold_right:
            return "右向き"
        if yaw_angle <= self.yaw_threshold_left:
            return "左向き"
        return "正面"

    def _update_consecutive_frames(self, current_direction: str) -> int:
        """連続フレーム数を更新する。

        HandRaiseDetectorの連続フレーム判定ロジックと同様のパターン。

        Args:
            current_direction (str): 現在の方向

        Returns:
            int: 更新後の連続フレーム数
        """
        # 履歴に追加
        self._direction_history.append(current_direction)

        # 前回と同じ方向ならカウントアップ
        if current_direction == self._head_state.current_direction:
            self._head_state.consecutive_frames += 1
        else:
            # 方向が変わったらリセット
            self._head_state.consecutive_frames = 1
            self._head_state.current_direction = current_direction

        # 閾値以上なら持続的方向転換フラグを立てる
        if self._head_state.consecutive_frames >= self.min_consecutive_frames:
            self._head_state.is_sustained = True
        else:
            self._head_state.is_sustained = False

        return self._head_state.consecutive_frames

    def detect(self, frame: np.ndarray, timestamp: float) -> dict[str, Any]:
        """フレームごとに頭部方向を検知する。

        HandRaiseDetector.detect()と同様の構造を採用。

        Args:
            frame (np.ndarray): 入力フレーム（BGR形式）
            timestamp (float): タイムスタンプ（秒）

        Returns:
            dict[str, Any]: 検知結果
                - face_detected (bool): 顔が検出されたか
                - yaw_angle (float): ヨー角（度）
                - direction (str): 方向
                - consecutive_frames (int): 連続フレーム数
                - confidence (float): 信頼度
                - is_sustained (bool): 持続的方向転換フラグ
        """
        # フレームをRGBに変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe Face Meshで処理
        results = self.face_mesh.process(frame_rgb)

        # 顔が検出されなかった場合
        if not results.multi_face_landmarks:
            self._reset_state()
            return self._export_status()

        # 最初の顔のランドマークを使用
        face_landmarks = results.multi_face_landmarks[0]

        # ヨー角を計算
        yaw_angle, confidence = self.calculate_yaw_angle(face_landmarks)

        # 状態を更新
        self._head_state.face_detected = True
        self._head_state.yaw_angle = yaw_angle
        self._head_state.confidence = confidence

        # 方向を分類
        direction = self._classify_direction(yaw_angle)

        # 連続フレーム数を更新
        self._update_consecutive_frames(direction)

        return self._export_status()

    def check_sustained_turn(self, timestamp: float) -> dict[str, Any] | None:
        """持続的な方向転換を検知する。

        クールダウン管理付きで、連続フレーム数が閾値以上の場合に
        検知イベントを返す。

        Args:
            timestamp (float): 現在のタイムスタンプ（秒）

        Returns:
            dict[str, Any] | None: 検知イベント、または None
                - detected (bool): 検知フラグ
                - direction (str): 検知された方向
                - frames (int): 継続フレーム数
                - timestamp (float): タイムスタンプ
                - yaw_angle (float): ヨー角
        """
        # 持続的方向転換が検知されていない場合
        if not self._head_state.is_sustained:
            return None

        # 正面の場合は通知しない
        if self._head_state.current_direction == "正面":
            return None

        # クールダウン期間内なら検知しない
        if (timestamp - self._last_sustained_turn_time) < self.cooldown_sec:
            return None

        # 検知イベントを生成
        self._last_sustained_turn_time = timestamp

        return {
            "detected": True,
            "direction": self._head_state.current_direction,
            "frames": self._head_state.consecutive_frames,
            "timestamp": timestamp,
            "yaw_angle": self._head_state.yaw_angle,
        }

    def get_status(self) -> dict[str, Any]:
        """検出器の現在の状態を返す。

        PostureAndMotionDetectorBaseの抽象メソッド実装。

        Returns:
            dict[str, Any]: 検出器の状態
        """
        # 持続的方向転換の方向を決定
        sustained_direction = ""
        if self._head_state.is_sustained and self._head_state.current_direction != "正面":
            if self._head_state.current_direction == "左向き":
                sustained_direction = MovementState.MEDIAPIPE_SUSTAINED_LEFT.value
            elif self._head_state.current_direction == "右向き":
                sustained_direction = MovementState.MEDIAPIPE_SUSTAINED_RIGHT.value

        return {
            "face_detected": self._head_state.face_detected,
            "yaw_angle": self._head_state.yaw_angle,
            "direction": self._head_state.current_direction,
            "consecutive_frames": self._head_state.consecutive_frames,
            "confidence": self._head_state.confidence,
            "is_sustained": self._head_state.is_sustained,
            "sustained_direction": sustained_direction,  # 持続的方向転換の方向
            "min_consecutive_frames": self.min_consecutive_frames,
            "yaw_threshold_right": self.yaw_threshold_right,
            "yaw_threshold_left": self.yaw_threshold_left,
            "cooldown_sec": self.cooldown_sec,
            "history_length": len(self._direction_history),
        }

    def reset(self) -> None:
        """検出器の状態をリセットする。"""
        self._reset_state()

    def _reset_state(self) -> None:
        """内部状態をリセットする。"""
        self._head_state = _HeadTurnState()
        self._clear_all_history()

    def _export_status(self) -> dict[str, Any]:
        """現在の状態をエクスポートする（内部用）。"""
        return {
            "face_detected": self._head_state.face_detected,
            "yaw_angle": self._head_state.yaw_angle,
            "direction": self._head_state.current_direction,
            "consecutive_frames": self._head_state.consecutive_frames,
            "confidence": self._head_state.confidence,
            "is_sustained": self._head_state.is_sustained,
        }
