"""src/detect_joint_movement_with_hip_stay.pyのテスト"""
# pylint: disable=protected-access

from __future__ import annotations

import unittest
from dataclasses import fields
from enum import Enum

import numpy as np

from src.definitions import Angle, MovementState
from src.detect_joint_movement_with_hip_stay import (
    HipBasedStayDetector,
    HipStayInfo,
    HipStayState,
    PostureMonitor,
    PostureSnapshot,
)
from src.pose.definitions import BodyPart


def test_posture_snapshot_dataclass():
    """PostureSnapshotデータクラスが正しく定義されているかテストする"""
    analysis_results = {Angle.BODY_TILT: {"angle": 140.0, "state": MovementState.FORWARD_TILT}}
    snapshot = PostureSnapshot(
        timestamp=1.0,
        frame_number=1,
        analysis_results=analysis_results,
        is_forward_leaning=True,
        forward_lean_score=0.8,
    )
    assert snapshot.timestamp == 1.0
    assert snapshot.analysis_results[Angle.BODY_TILT]["angle"] == 140.0
    assert snapshot.is_forward_leaning is True
    assert len(fields(snapshot)) == 5  # 5つのフィールドを持つことを確認


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


class TestPostureMonitor(unittest.TestCase):
    """PostureMonitorクラスのテストスイート"""

    def setUp(self):
        """各テストの前にPostureMonitorインスタンスを初期化"""
        self.monitor = PostureMonitor(monitoring_duration=5.0, alert_threshold=0.5)

    def _create_dummy_results(self, is_leaning: bool) -> dict:
        """前傾または直立のダミー分析結果を作成する"""
        if is_leaning:
            # 前傾姿勢を示す値（体の傾き、首の角度、肩の屈曲）
            # スコアが0.5を超えるように、より明確な前傾の値に設定
            return {
                Angle.BODY_TILT: {"angle": 100.0},  # より深く前傾
                Angle.NECK_TRUNK_ANGLE: {"angle": 100.0},  # より深く前傾
                Angle.RIGHT_SHOULDER: {"state": MovementState.FLEXION},
                Angle.LEFT_SHOULDER: {"state": MovementState.FLEXION},
            }
        # 直立姿勢を示す値
        return {
            Angle.BODY_TILT: {"angle": 170.0},
            Angle.NECK_TRUNK_ANGLE: {"angle": 170.0},
            Angle.RIGHT_SHOULDER: {"state": MovementState.EXTENSION},
            Angle.LEFT_SHOULDER: {"state": MovementState.EXTENSION},
        }

    def test_initialization(self):
        """初期状態をテストする"""
        self.assertEqual(self.monitor.monitoring_duration, 5.0)
        self.assertEqual(self.monitor.alert_threshold, 0.5)
        self.assertEqual(len(self.monitor.posture_history), 0)

    def test_update_and_history_management(self):
        """updateメソッドで履歴が正しく管理されるかテストする"""
        # データを5秒分追加
        for i in range(5):
            results = self._create_dummy_results(is_leaning=True)
            self.monitor.update(timestamp=float(i), frame_number=i, analysis_results=results)
        self.assertEqual(len(self.monitor.posture_history), 5)

        # 6.0秒のデータを追加すると、0.0秒のデータが削除される (6.0 - 0.0 > 5.0)
        self.monitor.update(timestamp=6.0, frame_number=5, analysis_results=self._create_dummy_results(False))
        self.assertEqual(len(self.monitor.posture_history), 5)
        self.assertEqual(self.monitor.posture_history[0].timestamp, 1.0)

    def test_is_forward_leaning_posture(self):
        """前傾姿勢の判定ロジックをテストする"""
        leaning_results = self._create_dummy_results(is_leaning=True)
        is_leaning, score = self.monitor.is_forward_leaning_posture(leaning_results)
        self.assertTrue(is_leaning)
        self.assertGreater(score, 0.5)

        upright_results = self._create_dummy_results(is_leaning=False)
        is_leaning, score = self.monitor.is_forward_leaning_posture(upright_results)
        self.assertFalse(is_leaning)
        self.assertLess(score, 0.5)

    def test_is_forward_leaning_posture_boundary(self):
        """(No. 19) is_leaningのreturn部分の境界値テスト"""
        # ケース1: スコアがちょうど0.5の場合 -> False
        # body_tilt: 1 - 75/150 = 0.5
        # neck_trunk: 1 - 75/150 = 0.5
        # shoulder: 1/2 = 0.5
        # avg_score = (0.5 + 0.5 + 0.5) / 3 = 0.5
        results_equal = {
            Angle.BODY_TILT: {"angle": 75.0},
            Angle.NECK_TRUNK_ANGLE: {"angle": 75.0},
            Angle.RIGHT_SHOULDER: {"state": MovementState.FLEXION},
            Angle.LEFT_SHOULDER: {"state": MovementState.EXTENSION},
        }
        is_leaning, score = self.monitor.is_forward_leaning_posture(results_equal)
        self.assertFalse(is_leaning, "Score of 0.5 should not be considered leaning")
        self.assertAlmostEqual(score, 0.5)

        # ケース2: スコアが0.5をわずかに超える場合 -> True
        # body_tilt: 1 - 74/150 = 0.5066...
        # neck_trunk: 1 - 75/150 = 0.5
        # shoulder: 1/2 = 0.5
        # avg_score = (0.5066 + 0.5 + 0.5) / 3 = 0.5022... > 0.5
        results_greater = {
            Angle.BODY_TILT: {"angle": 74.0},
            Angle.NECK_TRUNK_ANGLE: {"angle": 75.0},
            Angle.RIGHT_SHOULDER: {"state": MovementState.FLEXION},
            Angle.LEFT_SHOULDER: {"state": MovementState.EXTENSION},
        }
        is_leaning, score = self.monitor.is_forward_leaning_posture(results_greater)
        self.assertTrue(is_leaning, "Score > 0.5 should be considered leaning")
        self.assertGreater(score, 0.5)

    def test_alert_triggering(self):
        """アラートが正しく発火するかテストする"""
        alerts = []
        # 監視期間中、ほとんどのフレームで前傾姿勢
        for i in range(6):
            is_leaning = i < 4  # 最初の4フレームは前傾
            results = self._create_dummy_results(is_leaning)
            alerts = self.monitor.update(timestamp=float(i), frame_number=i, analysis_results=results)

        # 最後のフレーム(5s)で監視期間(5s)に達し, 前傾率(4/5=0.8)が閾値(0.5)を超える
        self.assertTrue(alerts)
        self.assertIn("[!] Forward Leaning:", alerts[0])

    def test_alert_cooldown(self):
        """アラートのクールダウン機能をテストする"""
        # 1. 一度アラートを発火させる
        alerts = []
        for i in range(6):
            alerts = self.monitor.update(float(i), i, self._create_dummy_results(is_leaning=True))
        self.assertTrue(alerts)  # まずアラートが出たことを確認

        # 2. クールダウン期間中に再度アラート条件を満たしてもアラートは出ない
        self.monitor.last_alert_time = 5.0
        alerts_in_cooldown = self.monitor.update(6.0, 6, self._create_dummy_results(is_leaning=True))
        self.assertEqual(len(alerts_in_cooldown), 0)

        # 3. クールダウン期間が過ぎれば再度アラートが出る
        self.monitor.alert_cooldown = 2.0
        # 履歴をクリアして再テスト
        self.monitor.posture_history.clear()
        final_alerts = []
        for i in range(8, 14):  # 8sから13sまで
            final_alerts = self.monitor.update(float(i), i, self._create_dummy_results(is_leaning=True))

        self.assertTrue(final_alerts)
        self.assertIn("[!] Forward Leaning:", final_alerts[0])

    def test_get_status(self):
        """get_statusメソッドが正しい値を返すかテストする"""
        # 初期状態
        initial_status = self.monitor.get_status()
        self.assertEqual(initial_status["sample_count"], 0)
        self.assertEqual(initial_status["forward_ratio"], 0)

        # データを3秒分追加 (2フレームが前傾)
        self.monitor.update(0.0, 0, self._create_dummy_results(is_leaning=True))
        self.monitor.update(1.0, 1, self._create_dummy_results(is_leaning=True))
        self.monitor.update(2.0, 2, self._create_dummy_results(is_leaning=False))

        status = self.monitor.get_status()
        self.assertEqual(status["sample_count"], 3)
        self.assertAlmostEqual(status["monitoring_duration"], 2.0)
        self.assertAlmostEqual(status["forward_ratio"], 2 / 3)
        # スコアを計算して検証 ( leaning_score > 0.5, upright_score < 0.5 )
        _, leaning_score = self.monitor.is_forward_leaning_posture(self._create_dummy_results(is_leaning=True))
        _, upright_score = self.monitor.is_forward_leaning_posture(self._create_dummy_results(is_leaning=False))
        expected_avg_score = (leaning_score * 2 + upright_score) / 3
        self.assertAlmostEqual(status["avg_score"], expected_avg_score)

    def test_check_for_alerts_with_empty_history(self):
        """(No. 20) posture_historyが空の場合に_check_for_alertsが早期リターンすることをテスト"""
        self.monitor.posture_history.clear()
        alerts = self.monitor._check_for_alerts(current_time=10.0)
        self.assertEqual(len(alerts), 0)


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
