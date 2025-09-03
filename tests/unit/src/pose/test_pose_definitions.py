"""src/pose/definitions.pyのテスト"""

import unittest

from src.pose.definitions import BodyPart, Joint, Movement


class TestPoseDefinitions(unittest.TestCase):
    def test_body_part_enum_values(self):
        """BodyPart Enumの主要なメンバーの値が正しいかテストする"""
        self.assertEqual(BodyPart.NOSE, 0)
        self.assertEqual(BodyPart.LEFT_SHOULDER, 11)
        self.assertEqual(BodyPart.RIGHT_SHOULDER, 12)
        self.assertEqual(BodyPart.LEFT_HIP, 23)
        self.assertEqual(BodyPart.RIGHT_HIP, 24)
        self.assertEqual(BodyPart.RIGHT_FOOT_INDEX, 32)

    def test_body_part_enum_completeness(self):
        """BodyPart Enumが33個のメンバーを持つことをテストする"""
        self.assertEqual(len(BodyPart), 33)

    def test_joint_enum_uniqueness(self):
        """Joint Enumの全てのメンバーがユニークな値を持つことをテストする"""
        values = [member.value for member in Joint]
        self.assertEqual(len(values), len(set(values)), "Joint enum values are not unique")

    def test_movement_enum_values(self):
        """Movement Enumのメンバーの値が正しいかテストする"""
        self.assertEqual(Movement.NONE, 0)
        self.assertEqual(Movement.FLEXION, 1)
        self.assertEqual(Movement.EXTENSION, 2)


if __name__ == "__main__":
    unittest.main()
