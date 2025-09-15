from __future__ import annotations

import unittest

import numpy as np

from src.analysis.dwell_time_detector import DwellTimeDetector, DwellTimeState
from src.pose.definitions import BodyPart


class TestDwellTimeDetector(unittest.TestCase):
    """DwellTimeDetectorクラスのテストスイート"""

    def setUp(self):
        """テスト用のDwellTimeDetectorインスタンスを初期化"""
        self.detector = DwellTimeDetector(
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
        self.assertEqual(self.detector.state, DwellTimeState.STAYING)
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
        self.assertEqual(self.detector.state, DwellTimeState.POTENTIAL_MOVE)

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
        self.assertEqual(self.detector.state, DwellTimeState.STAYING)

        # 2. 大きく動かす (spike_thresholdを超える)
        landmarks_move = self._create_dummy_landmarks(hip_y_norm=0.8)
        self.detector.update(landmarks_move, self.frame_shape, timestamp=1.0)

        # 3. 状態がPOTENTIAL_MOVEに遷移する
        self.assertEqual(self.detector.state, DwellTimeState.POTENTIAL_MOVE)

    def test_state_transition_move_to_stay(self):
        """POTENTIAL_MOVEからSTAYINGへの復帰（誤報）をテストする"""
        # 1. POTENTIAL_MOVE状態にする
        self.test_state_transition_stay_to_move()
        self.assertEqual(self.detector.state, DwellTimeState.POTENTIAL_MOVE)

        # 2. 猶予期間中に静止する
        landmarks_stay = self._create_dummy_landmarks(hip_y_norm=0.8)  # 動いた先の位置で静止
        self.detector.update(landmarks_stay, self.frame_shape, timestamp=1.5)

        # 3. 猶予期間(1s)を過ぎるとSTAYINGに戻る
        self.detector.update(landmarks_stay, self.frame_shape, timestamp=2.1)
        self.assertEqual(self.detector.state, DwellTimeState.STAYING)

    def test_state_transition_move_confirmed(self):
        """POTENTIAL_MOVEから移動確定（リセット）をテストする"""
        # 1. POTENTIAL_MOVE状態にする
        self.test_state_transition_stay_to_move()
        self.assertEqual(self.detector.state, DwellTimeState.POTENTIAL_MOVE)
        initial_start_time = self.detector.stay_info.stay_start_time

        # 2. 猶予期間中も動き続ける
        landmarks_move2 = self._create_dummy_landmarks(hip_y_norm=0.5)
        self.detector.update(landmarks_move2, self.frame_shape, timestamp=1.5)

        # 3. 猶予期間(1s)を過ぎると移動が確定し、滞在情報がリセットされる
        alert = self.detector.update(landmarks_move2, self.frame_shape, timestamp=2.1)
        self.assertEqual(alert, "[!] Movement Confirmed")
        self.assertEqual(self.detector.state, DwellTimeState.STAYING)
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

    def test_compute_person_scale_visible_exception_returns_none(self):
        """visible() should catch TypeError/ValueError and treat the point as not visible -> returns None for scale."""
        # Use object dtype to inject an invalid confidence value that raises when cast to float
        landmarks = np.zeros((33, 4), dtype=object)
        height, width = self.frame_shape[:2]
        # Set shoulders and hips coordinates (normalized)
        landmarks[BodyPart.LEFT_SHOULDER] = [0.4, 0.4, 0.0, "INVALID"]
        landmarks[BodyPart.RIGHT_SHOULDER] = [0.6, 0.4, 0.0, 1.0]
        landmarks[BodyPart.LEFT_HIP] = [0.4, 0.6, 0.0, 1.0]
        landmarks[BodyPart.RIGHT_HIP] = [0.6, 0.6, 0.0, 1.0]
        # Shoulder-based normalization requires both shoulders visible;
        # LEFT_SHOULDER confidence casting will raise -> visible() returns False -> returns None
        self.detector.normalization_base = "shoulder"
        scale = self.detector._compute_person_scale(landmarks, self.frame_shape)
        self.assertIsNone(scale)

    def test_compute_person_scale_shoulder_requires_both_visible(self):
        """When normalization_base='shoulder', if either shoulder is not visible, return None."""
        landmarks = self._create_dummy_landmarks(hip_y_norm=0.5, torso_len_norm=0.2).copy()
        # Hide right shoulder by lowering its visibility below threshold
        landmarks[BodyPart.RIGHT_SHOULDER][3] = 0.0
        self.detector.normalization_base = "shoulder"
        scale = self.detector._compute_person_scale(landmarks, self.frame_shape)
        self.assertIsNone(scale)

    def test_compute_person_scale_handles_index_error(self):
        """_compute_person_scale should return None when landmarks indexing fails (IndexError/TypeError)."""
        # Too few landmarks to index shoulders/hips -> IndexError inside function
        malformed = np.zeros((1, 4), dtype=np.float32)
        self.detector.normalization_base = "torso"
        scale = self.detector._compute_person_scale(malformed, self.frame_shape)
        self.assertIsNone(scale)

    def test_extract_hip_center_returns_none_when_landmarks_none(self):
        """extract_hip_center should return None when landmarks is None."""
        center = self.detector.extract_hip_center(None, self.frame_shape)
        self.assertIsNone(center)

    def test_extract_hip_center_prefers_left_then_right_and_none(self):
        """extract_hip_center should return left hip if visible, else right hip if visible, else None."""
        # Start with both visible
        landmarks = self._create_dummy_landmarks(hip_y_norm=0.5)
        # Case 1: Both visible -> function may choose center or specific hip; we force left-only visible
        landmarks_left_only = landmarks.copy()
        landmarks_left_only[BodyPart.RIGHT_HIP][3] = 0.0  # hide right
        res_left = self.detector.extract_hip_center(landmarks_left_only, self.frame_shape)
        self.assertIsNotNone(res_left)
        # Expect pixel coordinates of left hip
        left_px_x = int(self.frame_shape[1] * landmarks_left_only[BodyPart.LEFT_HIP][0])
        left_px_y = int(self.frame_shape[0] * landmarks_left_only[BodyPart.LEFT_HIP][1])
        self.assertAlmostEqual(res_left[0], left_px_x, delta=1.0)
        self.assertAlmostEqual(res_left[1], left_px_y, delta=1.0)
        self.assertGreaterEqual(res_left[2], self.detector.confidence_threshold)

        # Case 2: Only right visible
        landmarks_right_only = landmarks.copy()
        landmarks_right_only[BodyPart.LEFT_HIP][3] = 0.0  # hide left
        res_right = self.detector.extract_hip_center(landmarks_right_only, self.frame_shape)
        self.assertIsNotNone(res_right)
        right_px_x = int(self.frame_shape[1] * landmarks_right_only[BodyPart.RIGHT_HIP][0])
        right_px_y = int(self.frame_shape[0] * landmarks_right_only[BodyPart.RIGHT_HIP][1])
        self.assertAlmostEqual(res_right[0], right_px_x, delta=1.0)
        self.assertAlmostEqual(res_right[1], right_px_y, delta=1.0)
        self.assertGreaterEqual(res_right[2], self.detector.confidence_threshold)

        # Case 3: Neither visible -> None
        landmarks_none = landmarks.copy()
        landmarks_none[BodyPart.LEFT_HIP][3] = 0.0
        landmarks_none[BodyPart.RIGHT_HIP][3] = 0.0
        res_none = self.detector.extract_hip_center(landmarks_none, self.frame_shape)
        self.assertIsNone(res_none)

    def test_extract_hip_center_handles_index_error(self):
        """extract_hip_center should return None when indexing fails (malformed array)."""
        malformed = np.zeros((1, 4), dtype=np.float32)
        res = self.detector.extract_hip_center(malformed, self.frame_shape)
        self.assertIsNone(res)

    def test_update_resets_on_missing_hip_data(self):
        """When hip_data becomes None while staying, detector should reset stay info and return None."""
        # Initialize by providing valid hip data
        landmarks = self._create_dummy_landmarks(hip_y_norm=0.5)
        self.detector.update(landmarks, self.frame_shape, timestamp=0.0)
        self.assertIsNotNone(self.detector.stay_info)

        # Now provide no landmarks so hip_data is None
        alert = self.detector.update(None, self.frame_shape, timestamp=0.5)
        self.assertIsNone(alert)
        # After reset, stay_info should exist and stay_duration reset to 0
        self.assertIsNotNone(self.detector.stay_info)
        self.assertEqual(self.detector.stay_info.stay_duration, 0.0)

    def test_potential_move_grace_fallback_to_staying_explicit(self):
        """Explicitly verify POTENTIAL_MOVE reverts to STAYING after grace period without confirming movement."""
        # 1) Establish initial stay
        start = self._create_dummy_landmarks(hip_y_norm=0.5)
        self.detector.update(start, self.frame_shape, timestamp=0.0)
        self.assertEqual(self.detector.state, DwellTimeState.STAYING)

        # 2) Large movement to trigger POTENTIAL_MOVE
        moved = self._create_dummy_landmarks(hip_y_norm=0.8)
        self.detector.update(moved, self.frame_shape, timestamp=0.2)
        self.assertEqual(self.detector.state, DwellTimeState.POTENTIAL_MOVE)

        # 3) Hold position (no further large movement) past grace period
        self.detector.update(moved, self.frame_shape, timestamp=1.3)  # grace_period is 1.0
        # Should revert to STAYING (false alarm resolved)
        self.assertEqual(self.detector.state, DwellTimeState.STAYING)

    def test_potential_move_reverts_to_staying_after_grace_period(self):
        """When in POTENTIAL_MOVE and no further significant movement occurs beyond the grace period, the state should revert to STAYING (false alarm)."""
        # 1) Initialize with a stable position
        start = self._create_dummy_landmarks(hip_y_norm=0.5)
        self.detector.update(start, self.frame_shape, timestamp=0.0)
        self.assertEqual(self.detector.state, DwellTimeState.STAYING)

        # 2) Make a large movement to trigger POTENTIAL_MOVE
        moved = self._create_dummy_landmarks(hip_y_norm=0.8)
        self.detector.update(moved, self.frame_shape, timestamp=0.2)
        self.assertEqual(self.detector.state, DwellTimeState.POTENTIAL_MOVE)

        # 3) Avoid spike-based confirmation during grace
        self.detector.spike_threshold = 9999.0

        # 4) Within grace, keep position (no extra movement)
        self.detector.update(moved, self.frame_shape, timestamp=0.7)

        # 5) After grace expires, detector should revert to STAYING (false alarm)
        self.detector.update(moved, self.frame_shape, timestamp=1.3)
        self.assertEqual(self.detector.state, DwellTimeState.STAYING)



    def test_potential_move_false_alarm_updates_fields_and_returns_none(self):
        """After POTENTIAL_MOVE times out (false alarm), state must be STAYING and common fields updated with no alert returned."""
        # 1) Begin with a stable stay
        start = self._create_dummy_landmarks(hip_y_norm=0.5)
        self.detector.update(start, self.frame_shape, timestamp=0.0)
        self.assertEqual(self.detector.state, DwellTimeState.STAYING)

        # 2) Trigger POTENTIAL_MOVE via a large displacement
        moved = self._create_dummy_landmarks(hip_y_norm=0.8)
        self.detector.update(moved, self.frame_shape, timestamp=0.2)
        self.assertEqual(self.detector.state, DwellTimeState.POTENTIAL_MOVE)

        # 3) To avoid movement confirmation from spike history, raise the spike threshold during grace
        self.detector.spike_threshold = 9999.0  # ensure is_spike=False for subsequent frames

        # 4) Provide an update within grace period (no additional movement)
        alert_mid = self.detector.update(moved, self.frame_shape, timestamp=0.7)  # still within grace
        self.assertIsNone(alert_mid)

        # 5) Exceed grace period with continued stability -> should revert to STAYING (false alarm)
        alert = self.detector.update(moved, self.frame_shape, timestamp=1.3)  # > 0.2 + grace(1.0)
        self.assertIsNone(alert)
        self.assertEqual(self.detector.state, DwellTimeState.STAYING)

        # 6) Common fields should be updated to the latest frame's values
        self.assertAlmostEqual(self.detector.stay_info.last_update_time, 1.3, places=6)
        self.assertGreaterEqual(self.detector.stay_info.confidence_score, self.detector.confidence_threshold)
