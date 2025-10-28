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
    LATERAL_TILT = "LATERAL_TILT"
    # 首振り関連の角度
    HEAD_HORIZONTAL_ROTATION = "HEAD_HORIZONTAL_ROTATION"  # 水平首振り角度
    HEAD_VERTICAL_NOD = "HEAD_VERTICAL_NOD"  # 垂直うなずき角度


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
    LEFT_TILT = "LEFT_TILT"  # 左側屈（左傾斜）
    RIGHT_TILT = "RIGHT_TILT"  # 右側屈（右傾斜）
    # 首振り関連の状態
    HORIZONTAL_SHAKE = "HORIZONTAL_SHAKE"  # 水平首振り
    VERTICAL_NOD = "VERTICAL_NOD"  # 垂直うなずき
    HEAD_STATIC = "HEAD_STATIC"  # 首静止
    HEAD_LEFT_TURN = "HEAD_LEFT_TURN"  # 左向き
    HEAD_RIGHT_TURN = "HEAD_RIGHT_TURN"  # 右向き
    HEAD_UP_NOD = "HEAD_UP_NOD"  # 上向きうなずき
    HEAD_DOWN_NOD = "HEAD_DOWN_NOD"  # 下向きうなずき
    # MediaPipe Face Mesh 頭部方向検知用
    MEDIAPIPE_LEFT_TURN = "MEDIAPIPE_LEFT_TURN"  # MediaPipe: 左向き
    MEDIAPIPE_RIGHT_TURN = "MEDIAPIPE_RIGHT_TURN"  # MediaPipe: 右向き
    MEDIAPIPE_FRONT_FACE = "MEDIAPIPE_FRONT_FACE"  # MediaPipe: 正面
    MEDIAPIPE_SUSTAINED_LEFT = "MEDIAPIPE_SUSTAINED_LEFT"  # MediaPipe: 持続的左向き
    MEDIAPIPE_SUSTAINED_RIGHT = "MEDIAPIPE_SUSTAINED_RIGHT"  # MediaPipe: 持続的右向き
