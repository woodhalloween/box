"""src/definitions.pyのテスト"""

from src.definitions import Angle, MovementState


def test_angle_enum():
    """Angle Enumのメンバーをテストする"""
    assert Angle.RIGHT_ELBOW.value == "RIGHT_ELBOW"
    assert Angle.LEFT_ELBOW.value == "LEFT_ELBOW"
    assert Angle.RIGHT_SHOULDER.value == "RIGHT_SHOULDER"
    assert Angle.LEFT_SHOULDER.value == "LEFT_SHOULDER"
    assert Angle.RIGHT_HIP.value == "RIGHT_HIP"
    assert Angle.LEFT_HIP.value == "LEFT_HIP"
    assert Angle.RIGHT_KNEE.value == "RIGHT_KNEE"
    assert Angle.LEFT_KNEE.value == "LEFT_KNEE"
    assert Angle.BODY_TILT.value == "BODY_TILT"
    assert Angle.NECK_TRUNK_ANGLE.value == "NECK_TRUNK_ANGLE"
    assert Angle.LATERAL_TILT.value == "LATERAL_TILT"
    assert Angle.HEAD_HORIZONTAL_ROTATION.value == "HEAD_HORIZONTAL_ROTATION"
    assert Angle.HEAD_VERTICAL_NOD.value == "HEAD_VERTICAL_NOD"


def test_movement_state_enum():
    """MovementState Enumのメンバーをテストする"""
    assert MovementState.UNKNOWN.value == "UNKNOWN"
    assert MovementState.FLEXION.value == "FLEXION"
    assert MovementState.EXTENSION.value == "EXTENSION"
    assert MovementState.STATIC.value == "STATIC"
    assert MovementState.FORWARD_TILT.value == "FORWARD_TILT"
    assert MovementState.UPRIGHT.value == "UPRIGHT"
    assert MovementState.HUNCH.value == "HUNCH"
    assert MovementState.STRAIGHT.value == "STRAIGHT"
    assert MovementState.LEFT_TILT.value == "LEFT_TILT"
    assert MovementState.RIGHT_TILT.value == "RIGHT_TILT"
    assert MovementState.HORIZONTAL_SHAKE.value == "HORIZONTAL_SHAKE"
    assert MovementState.VERTICAL_NOD.value == "VERTICAL_NOD"
    assert MovementState.HEAD_STATIC.value == "HEAD_STATIC"
    assert MovementState.HEAD_LEFT_TURN.value == "HEAD_LEFT_TURN"
    assert MovementState.HEAD_RIGHT_TURN.value == "HEAD_RIGHT_TURN"
    assert MovementState.HEAD_UP_NOD.value == "HEAD_UP_NOD"
    assert MovementState.HEAD_DOWN_NOD.value == "HEAD_DOWN_NOD"
