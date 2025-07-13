from enum import Enum


class Angle(Enum):
    """関節角度の種類"""

    RIGHT_ELBOW = "RIGHT_ELBOW"
    LEFT_ELBOW = "LEFT_ELBOW"
    RIGHT_SHOULDER = "RIGHT_SHOULDER"
    LEFT_SHOULDER = "LEFT_SHOULDER"
    RIGHT_HIP = "RIGHT_HIP"
    LEFT_HIP = "LEFT_HIP"
    RIGHT_KNEE = "RIGHT_KNEE"
    LEFT_KNEE = "LEFT_KNEE"
    BODY_TILT = "BODY_TILT"
    NECK_TRUNK_ANGLE = "NECK_TRUNK_ANGLE"


class MovementState(Enum):
    """運動の状態"""

    UNKNOWN = "UNKNOWN"  # 不明
    FLEXION = "FLEXION"  # 屈曲
    EXTENSION = "EXTENSION"  # 伸展
    STATIC = "STATIC"  # 静止
    FORWARD_TILT = "FORWARD_TILT"  # 前傾
    UPRIGHT = "UPRIGHT"  # 直立
    HUNCH = "HUNCH"  # 猫背
    STRAIGHT = "STRAIGHT"  # 直立（姿勢）
 