"""src/detect_joint_movement_with_hip_stay.pyのテスト"""
# pylint: disable=protected-access

from __future__ import annotations

import unittest
from dataclasses import fields
from enum import Enum

import numpy as np

from src.detect_joint_movement_with_hip_stay import (
    HipBasedStayDetector,
    HipStayInfo,
    HipStayState,
)
from src.pose.definitions import BodyPart


def test_hip_stay_info_dataclass():
    """HipStayInfoデータクラスが正しく定義されているかテストする"""
    info = HipStayInfo(
        last_hip_pos=(100.0, 200.0),
        last_update_time=10.5,
        stay_start_time=5.0,
        stay_duration=5.5,
        notified=False,
        confidence_score=0.9,
    )
    assert info.last_hip_pos == (100.0, 200.0)
    assert info.stay_duration == 5.5
    assert info.notified is False
    assert len(fields(info)) == 6  # 6つのフィールドを持つことを確認


def test_hip_stay_state_enum():
    """HipStayState Enumのメンバーをテストする"""
    assert isinstance(HipStayState.STAYING, Enum)
    # auto()で自動採番されるため、具体的な値ではなく型や存在をテスト
    assert HipStayState.STAYING.name == "STAYING"
    assert HipStayState.POTENTIAL_MOVE.name == "POTENTIAL_MOVE"


class TestHipBasedStayDetector(unittest.TestCase):
    """HipBasedStayDetectorクラスのテストスイート"""

    def setUp(self):
        """テスト用のHipBasedStayDetectorインスタンスを初期化"""
        self.detector = HipBasedStayDetector(
            stay_threshold_sec=5.0,
            spike_threshold=1.0,  # 小さな動きで検知
            grace_period_sec=1.0,  # 短い猶予期間
            use_normalization=True,  # 正規化を有効化
            normalization_base="torso",
        )
        self.frame_shape = (720, 1280, 3)  # ダミーのフレームサイズ

    def _create_dummy_landmarks(self, hip_y_norm: float, torso_len_norm: float = 0.2) -> np.ndarray:
        """指定された腰のy座標を持つダミーランドマークを作成する"""
        landmarks = np.zeros((33, 4), dtype=np.float32)
        landmarks[:, 3] = 1.0  # visibility

        # Torso (for scaling)
        landmarks[BodyPart.LEFT_SHOULDER] = [0.4, hip_y_norm - torso_len_norm, 0, 1]
        landmarks[BodyPart.RIGHT_SHOULDER] = [0.6, hip_y_norm - torso_len_norm, 0, 1]

        # Hips
        landmarks[BodyPart.LEFT_HIP] = [0.4, hip_y_norm, 0, 1]
        landmarks[BodyPart.RIGHT_HIP] = [0.6, hip_y_norm, 0, 1]
        return landmarks

    def test_initialization(self):
        """初期状態をテストする"""
        self.assertEqual(self.detector.state, HipStayState.STAYING)
        self.assertIsNone(self.detector.stay_info)

    def test_extract_hip_center(self):
        """腰の中心座標が正しく抽出されるかテストする"""
        landmarks = self._create_dummy_landmarks(hip_y_norm=0.5)
        hip_data = self.detector.extract_hip_center(landmarks, self.frame_shape)
        self.assertIsNotNone(hip_data)
        expected_x = 0.5 * self.frame_shape[1]
        expected_y = 0.5 * self.frame_shape[0]
        self.assertAlmostEqual(hip_data[0], expected_x, places=4)
        self.assertAlmostEqual(hip_data[1], expected_y, places=4)

    def test_compute_person_scale(self):
        """_compute_person_scaleメソッドをテストする"""
        landmarks = self._create_dummy_landmarks(hip_y_norm=0.7, torso_len_norm=0.2)
        height, width, _ = self.frame_shape
        expected_torso_len_px = 0.2 * height

        # 1. Torso-based scaling
        self.detector.normalization_base = "torso"
        scale = self.detector._compute_person_scale(landmarks, self.frame_shape)
        self.assertAlmostEqual(scale, expected_torso_len_px, places=4)

        # 2. Shoulder-based scaling
        self.detector.normalization_base = "shoulder"
        landmarks[BodyPart.LEFT_SHOULDER] = [0.4, 0.5, 0, 1]
        landmarks[BodyPart.RIGHT_SHOULDER] = [0.6, 0.5, 0, 1]
        expected_shoulder_width_px = 0.2 * width
        scale = self.detector._compute_person_scale(landmarks, self.frame_shape)
        self.assertAlmostEqual(scale, expected_shoulder_width_px, places=4)

        # 3. Screen-based scaling
        self.detector.normalization_base = "screen"
        scale = self.detector._compute_person_scale(landmarks, self.frame_shape)
        self.assertEqual(scale, height)

        # 4. Invisible landmarks
        landmarks[BodyPart.LEFT_SHOULDER][3] = 0.1  # visibility < threshold
        self.detector.normalization_base = "torso"
        scale = self.detector._compute_person_scale(landmarks, self.frame_shape)
        self.assertIsNone(scale)

    def test_compute_person_scale_shoulders_not_visible(self):
        """(No. 24) 肩が見えない場合にNoneを返すかテスト"""
        landmarks = self._create_dummy_landmarks(hip_y_norm=0.7, torso_len_norm=0.2)
        landmarks[BodyPart.LEFT_SHOULDER][3] = 0.1  # 左肩の信頼度を低くする

        # ケース1: normalization_base = "shoulder"
        self.detector.normalization_base = "shoulder"
        scale = self.detector._compute_person_scale(landmarks, self.frame_shape)
        self.assertIsNone(scale, "Should return None if one shoulder is not visible for 'shoulder' base")

        # ケース2: normalization_base = "torso"
        self.detector.normalization_base = "torso"
        scale = self.detector._compute_person_scale(landmarks, self.frame_shape)
        self.assertIsNone(scale, "Should return None if one shoulder is not visible for 'torso' base")

    def test_movement_detection_by_stability(self):
        """人物スケールの不安定性によって移動が検知されるかテストする"""
        # 安定性閾値を低く、スパイク閾値を高く設定
        self.detector.stability_threshold_px = 10.0
        self.detector.spike_threshold = 999.0

        # 1. 安定したスケールで初期化
        landmarks_stable = self._create_dummy_landmarks(hip_y_norm=0.5, torso_len_norm=0.2)
        self.detector.update(landmarks_stable, self.frame_shape, timestamp=0.0)

        # 2. スケールを変動させる (位置は同じ)
        for i in range(1, 5):
            torso_len = 0.2 + (i % 2) * 0.1  # 0.3, 0.2, 0.3, 0.2 ...
            landmarks_unstable = self._create_dummy_landmarks(hip_y_norm=0.5, torso_len_norm=torso_len)
            self.detector.update(landmarks_unstable, self.frame_shape, timestamp=float(i))

        # スケールの標準偏差が閾値を超え、POTENTIAL_MOVEに遷移するはず
        self.assertEqual(self.detector.state, HipStayState.POTENTIAL_MOVE)

    def test_get_current_status(self):
        """get_current_statusメソッドが正しい値を返すかテストする"""
        # 1. 初期状態
        initial_status = self.detector.get_current_status()
        self.assertIsNone(initial_status["hip_position"])
        self.assertEqual(initial_status["stay_duration"], 0.0)
        self.assertFalse(initial_status["is_long_stay"])
        self.assertEqual(initial_status["state"], "STAYING")

        # 2. 滞在開始後
        landmarks = self._create_dummy_landmarks(hip_y_norm=0.5)
        self.detector.update(landmarks, self.frame_shape, timestamp=1.0)
        status_after_start = self.detector.get_current_status()
        self.assertIsNotNone(status_after_start["hip_position"])
        self.assertEqual(status_after_start["stay_duration"], 0.0)

        # 3. 長時間滞在後
        self.detector.update(landmarks, self.frame_shape, timestamp=6.0)  # 5s threshold
        status_long_stay = self.detector.get_current_status()
        self.assertAlmostEqual(status_long_stay["stay_duration"], 5.0)
        self.assertTrue(status_long_stay["is_long_stay"])

    def test_state_transition_stay_to_move(self):
        """STAYINGからPOTENTIAL_MOVEへの遷移をテストする"""
        # 1. 初期位置
        landmarks_stay = self._create_dummy_landmarks(hip_y_norm=0.5)
        self.detector.update(landmarks_stay, self.frame_shape, timestamp=0.0)
        self.assertEqual(self.detector.state, HipStayState.STAYING)

        # 2. 大きく動かす (spike_thresholdを超える)
        landmarks_move = self._create_dummy_landmarks(hip_y_norm=0.8)
        self.detector.update(landmarks_move, self.frame_shape, timestamp=1.0)

        # 3. 状態がPOTENTIAL_MOVEに遷移する
        self.assertEqual(self.detector.state, HipStayState.POTENTIAL_MOVE)

    def test_state_transition_move_to_stay(self):
        """POTENTIAL_MOVEからSTAYINGへの復帰（誤報）をテストする"""
        # 1. POTENTIAL_MOVE状態にする
        self.test_state_transition_stay_to_move()
        self.assertEqual(self.detector.state, HipStayState.POTENTIAL_MOVE)

        # 2. 猶予期間中に静止する
        landmarks_stay = self._create_dummy_landmarks(hip_y_norm=0.8)  # 動いた先の位置で静止
        self.detector.update(landmarks_stay, self.frame_shape, timestamp=1.5)

        # 3. 猶予期間(1s)を過ぎるとSTAYINGに戻る
        self.detector.update(landmarks_stay, self.frame_shape, timestamp=2.1)
        self.assertEqual(self.detector.state, HipStayState.STAYING)

    def test_state_transition_move_confirmed(self):
        """POTENTIAL_MOVEから移動確定（リセット）をテストする"""
        # 1. POTENTIAL_MOVE状態にする
        self.test_state_transition_stay_to_move()
        self.assertEqual(self.detector.state, HipStayState.POTENTIAL_MOVE)
        initial_start_time = self.detector.stay_info.stay_start_time

        # 2. 猶予期間中も動き続ける
        landmarks_move2 = self._create_dummy_landmarks(hip_y_norm=0.5)
        self.detector.update(landmarks_move2, self.frame_shape, timestamp=1.5)

        # 3. 猶予期間(1s)を過ぎると移動が確定し、滞在情報がリセットされる
        alert = self.detector.update(landmarks_move2, self.frame_shape, timestamp=2.1)
        self.assertEqual(alert, "[!] Movement Confirmed")
        self.assertEqual(self.detector.state, HipStayState.STAYING)
        self.assertNotEqual(self.detector.stay_info.stay_start_time, initial_start_time)

    def test_long_stay_alert(self):
        """長期滞在アラートが正しく発火するかテストする"""
        landmarks = self._create_dummy_landmarks(hip_y_norm=0.5)
        alert = None
        # stay_threshold_secが5.0なので、timestamp=5.0のupdate呼び出しでアラートが出るはず
        for i in range(6):  # 0, 1, 2, 3, 4, 5
            alert = self.detector.update(landmarks, self.frame_shape, timestamp=float(i))
            if i == 5:
                break

        self.assertIsNotNone(alert)
        self.assertIn("[!] Long Stay Detected", alert)
        self.assertTrue(self.detector.stay_info.notified)

    def test_compute_person_scale_exceptions(self):
        """(No. 23, 25) _compute_person_scaleの例外処理をテストする"""
        # (No. 23) visible() 内の TypeError/ValueError
        landmarks_bad_visibility = self._create_dummy_landmarks(hip_y_norm=0.5).astype(object)
        landmarks_bad_visibility[BodyPart.LEFT_SHOULDER][3] = "invalid"
        scale = self.detector._compute_person_scale(landmarks_bad_visibility, self.frame_shape)
        self.assertIsNone(scale, "Should return None with invalid visibility")

        # (No. 25) IndexError
        landmarks_index_error = np.zeros((10, 4))  # Not enough landmarks
        scale = self.detector._compute_person_scale(landmarks_index_error, self.frame_shape)
        self.assertIsNone(scale, "Should handle IndexError")

        # (No. 25) TypeError from None landmarks
        landmarks_type_error_none = None
        scale = self.detector._compute_person_scale(landmarks_type_error_none, self.frame_shape)
        self.assertIsNone(scale, "Should handle TypeError when landmarks is None")


class TestStandaloneFunctions(unittest.TestCase):
    """モジュールレベルの独立した関数のテスト"""

    def setUp(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_draw_posture_alerts(self):
        """(No. 26) draw_posture_alertsがクラッシュしないことをテスト"""
        from src.detect_joint_movement_with_hip_stay import draw_posture_alerts

        draw_posture_alerts(
            self.frame,
            alerts=["Test Alert"],
            status={"sample_count": 1, "monitoring_duration": 1.0, "forward_ratio": 0.5, "avg_score": 0.6},
            knee_alert="Knee Alert",
            head_shake_alerts=["Head Shake Alert"],
        )

    def test_draw_hip_stay_info(self):
        """(No. 27) draw_hip_stay_infoがクラッシュしないことをテスト"""
        from src.detect_joint_movement_with_hip_stay import draw_hip_stay_info

        detector = HipBasedStayDetector()
        detector.stay_info = HipStayInfo(
            last_hip_pos=(320, 240),
            last_update_time=1.0,
            stay_start_time=0.0,
            stay_duration=1.0,
            notified=False,
            confidence_score=0.9,
        )
        draw_hip_stay_info(self.frame, detector)

    def test_draw_head_shake_info(self):
        """(No. 28) draw_head_shake_infoがクラッシュしないことをテスト"""
        from src.detect_joint_movement_with_hip_stay import draw_head_shake_info
        from src.head_shake_detector import HeadShakeDetector

        detector = HeadShakeDetector()
        landmarks = np.zeros((33, 4))
        landmarks[BodyPart.NOSE] = [0.5, 0.5, 0, 0.9]  # visible nose
        draw_head_shake_info(self.frame, detector, landmarks)
        # detector or landmarksがNoneの場合もテスト
        draw_head_shake_info(self.frame, None, landmarks)
        draw_head_shake_info(self.frame, detector, None)
