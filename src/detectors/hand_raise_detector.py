import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark

# MediaPipe landmark component indices for readability
LANDMARK_X = 0
LANDMARK_Y = 1
LANDMARK_Z = 2
LANDMARK_VISIBILITY = 3


class HandRaiseDetector:
    """
    姿勢ランドマークから手の挙上状態を検出するクラス。

    このクラスは、連続するフレームにわたって手の挙上状態を追跡し、
    指定されたフレーム数以上連続して条件を満たした場合にのみ「挙手」と判定する。
    これにより、一時的な誤検出（チャタリング）をフィルタリングする。
    """

    def __init__(self, visibility_threshold: float, min_consecutive_frames: int):
        """
        HandRaiseDetectorを初期化する。

        Args:
            visibility_threshold (float):
                ランドマークが有効と見なされるための信頼度(visibility)の最小閾値。
            min_consecutive_frames (int):
                手を挙げている状態と判定するために必要な最小連続フレーム数。
        """
        self.visibility_threshold = visibility_threshold
        self.min_consecutive_frames = min_consecutive_frames

        # 左手の連続検出フレーム数と現在の状態
        self._left_hand_consecutive_frames = 0
        self._is_left_hand_raised = False

        # 右手の連続検出フレーム数と現在の状態
        self._right_hand_consecutive_frames = 0
        self._is_right_hand_raised = False

    def detect(self, landmarks: np.ndarray | None) -> dict[str, bool]:
        """
        与えられたランドマークから左右の手が挙がっているかを判定する。

        Args:
            landmarks (np.ndarray | None):
                姿勢推定モデルから出力されたランドマーク。
                形状は (33, 4) で、各行は [x, y, z, visibility] を表す。
                人物が検出されなかった場合はNone。

        Returns:
            dict[str, bool]:
                左右の手の挙上状態を示す辞書。
                例: {'left_hand_raised': True, 'right_hand_raised': False}
        """
        if landmarks is None:
            # ランドマークが検出されなかった場合、両手のカウンターをリセット
            self._left_hand_consecutive_frames = 0
            self._right_hand_consecutive_frames = 0
        else:
            # 左手の状態をチェック
            is_left_criteria_met = self._check_hand_raise_criteria(
                landmarks, PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_WRIST
            )
            if is_left_criteria_met:
                self._left_hand_consecutive_frames += 1
            else:
                self._left_hand_consecutive_frames = 0

            # 右手の状態をチェック
            is_right_criteria_met = self._check_hand_raise_criteria(
                landmarks, PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_WRIST
            )
            if is_right_criteria_met:
                self._right_hand_consecutive_frames += 1
            else:
                self._right_hand_consecutive_frames = 0

        # 連続フレーム数が閾値を超えたら、正式に手を挙げたと判定
        self._is_left_hand_raised = self._left_hand_consecutive_frames >= self.min_consecutive_frames
        self._is_right_hand_raised = self._right_hand_consecutive_frames >= self.min_consecutive_frames

        return {
            "left_hand_raised": self._is_left_hand_raised,
            "right_hand_raised": self._is_right_hand_raised,
        }

    def _check_hand_raise_criteria(self, landmarks: np.ndarray, shoulder_idx: int, wrist_idx: int) -> bool:
        """
        単一の腕について、手が挙がっているかの基準を満たすか判定する。

        Args:
            landmarks (np.ndarray): 姿勢ランドマーク。
            shoulder_idx (int): 肩のランドマークのインデックス。
            wrist_idx (int): 手首のランドマークのインデックス。

        Returns:
            bool: 手挙げの基準を満たしていればTrue、そうでなければFalse。
        """
        shoulder_landmark = landmarks[shoulder_idx]
        wrist_landmark = landmarks[wrist_idx]

        # 信頼度が閾値未満の場合は判定しない
        if (
            shoulder_landmark[LANDMARK_VISIBILITY] < self.visibility_threshold
            or wrist_landmark[LANDMARK_VISIBILITY] < self.visibility_threshold
        ):
            return False

        # 手首のy座標が肩のy座標より上にあるか（画像座標系なので値が小さいか）
        return wrist_landmark[LANDMARK_Y] < shoulder_landmark[LANDMARK_Y]
