"""src/pose/definitions.py

This module defines the core enumerations and constants used for pose analysis.

It includes definitions for pose landmarks, which directly correspond to the keypoints
provided by the MediaPipe Pose model, as well as definitions for different types
of joint movements. Using these enums improves code readability and maintainability
by avoiding the use of "magic numbers" for landmark indices.
"""

from enum import Enum, IntEnum, auto


class BodyPart(IntEnum):
    """MediaPipe Poseのランドマークに対応する身体部位のインデックス"""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


class Joint(Enum):
    """関節の種類を定義するEnum"""

    NOSE = auto()
    LEFT_EYE_INNER = auto()
    LEFT_EYE = auto()
    LEFT_EYE_OUTER = auto()
    RIGHT_EYE_INNER = auto()
    RIGHT_EYE = auto()
    RIGHT_EYE_OUTER = auto()
    LEFT_EAR = auto()
    RIGHT_EAR = auto()
    MOUTH_LEFT = auto()
    MOUTH_RIGHT = auto()
    LEFT_SHOULDER = auto()
    RIGHT_SHOULDER = auto()
    LEFT_ELBOW = auto()
    RIGHT_ELBOW = auto()
    LEFT_WRIST = auto()
    RIGHT_WRIST = auto()
    LEFT_PINKY = auto()
    RIGHT_PINKY = auto()
    LEFT_INDEX = auto()
    RIGHT_INDEX = auto()
    LEFT_THUMB = auto()
    RIGHT_THUMB = auto()
    LEFT_HIP = auto()
    RIGHT_HIP = auto()
    LEFT_KNEE = auto()
    RIGHT_KNEE = auto()
    LEFT_ANKLE = auto()
    RIGHT_ANKLE = auto()
    LEFT_HEEL = auto()
    RIGHT_HEEL = auto()
    LEFT_FOOT_INDEX = auto()
    RIGHT_FOOT_INDEX = auto()


class Movement(IntEnum):
    """運動の種類を定義するEnum"""

    NONE = 0  # 動きなし
    FLEXION = 1  # 屈曲
    EXTENSION = 2  # 伸展
