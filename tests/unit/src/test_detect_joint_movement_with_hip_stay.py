"""src/detect_joint_movement_with_hip_stay.pyのテスト"""

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
    KneeAngleMonitor,
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


class TestKneeAngleMonitor(unittest.TestCase):
    """KneeAngleMonitorクラスのテストスイート"""

    def setUp(self):
        """テスト用のKneeAngleMonitorインスタンスを初期化"""
        self.monitor = KneeAngleMonitor(threshold_deg=90.0, moving_window_seconds=3, confidence_threshold=0.5)

    def _create_dummy_results(self, angle: float, confidence: float = 1.0) -> dict:
        """指定された角度と信頼度を持つダミーの分析結果を作成する"""
        return {
            Angle.LEFT_KNEE: {
                "angle": angle,
                "p1_confidence": confidence,
                "p2_confidence": confidence,
                "p3_confidence": confidence,
            },
            Angle.RIGHT_KNEE: {
                "angle": angle,
                "p1_confidence": confidence,
                "p2_confidence": confidence,
                "p3_confidence": confidence,
            },
        }

    def test_angle_accumulation_and_finalization(self):
        """角度が秒単位で正しく集計・確定されるかテストする"""
        # 0.1秒, 0.2秒 -> 角度80.0
        self.monitor.update(0.1, self._create_dummy_results(80.0))
        self.monitor.update(0.2, self._create_dummy_results(80.0))
        # 0.3秒 -> 角度120.0 (中央値に影響を与える)
        self.monitor.update(0.3, self._create_dummy_results(120.0))

        # 1.1秒のフレームで、0秒目の集計が確定される
        self.monitor.update(1.1, self._create_dummy_results(100.0))

        # 0秒目の中央値は80.0になるはず
        self.assertEqual(len(self.monitor.medians_history), 1)
        second, left_med, right_med = self.monitor.medians_history[0]
        self.assertEqual(second, 0)
        self.assertAlmostEqual(left_med, 80.0)

    def test_confidence_threshold(self):
        """信頼度が閾値未満のデータが無視されるかテストする"""
        # 0秒台に信頼度の低いデータを追加
        self.monitor.update(0.1, self._create_dummy_results(80.0, confidence=0.4))
        # 1秒台のフレームで0秒台の集計が確定される
        self.monitor.update(1.1, self._create_dummy_results(100.0))

        # 0秒台のデータは信頼度不足で無視されたため、履歴には追加されない
        # (1秒台のデータは履歴に残っている)
        self.assertEqual(len(self.monitor.medians_history), 0)

    def test_alert_triggering_by_median(self):
        """中央値によってアラートが正しく発火するかテストする"""
        # 0秒台は危険な角度(80度)
        self.monitor.update(0.1, self._create_dummy_results(80.0))
        # 1秒台のフレームで0秒台の集計が確定し、アラートが生成される
        alerts = self.monitor.update(1.1, self._create_dummy_results(100.0))

        self.assertTrue(alerts, "Alert should be triggered by median")
        self.assertIn("[!] Knee Angle Low", alerts[0])
        self.assertIn("Median", alerts[0])

    def test_alert_triggering_by_moving_average(self):
        """移動平均によってアラートが正しく発火するかテストする"""
        # 0, 1, 2秒台で危険な角度を記録し、移動平均が閾値を下回るようにする
        self.monitor.update(0.1, self._create_dummy_results(80.0))
        self.monitor.update(1.1, self._create_dummy_results(80.0))
        self.monitor.update(2.1, self._create_dummy_results(80.0))

        # 3秒台のフレームで2秒台の集計が確定し、移動平均によるアラートが生成される
        alerts = self.monitor.update(3.1, self._create_dummy_results(120.0))

        self.assertTrue(alerts, "Alert should be triggered by moving average")
        # alertsリストのいずれかのメッセージに"MA"が含まれているかチェック
        self.assertTrue(any("MA" in alert for alert in alerts), "MA alert should be in the alerts list")

    def test_no_alert(self):
        """安全な角度ではアラートが発火しないことをテストする"""
        self.monitor.update(0.1, self._create_dummy_results(120.0))
        alerts = self.monitor.update(1.1, self._create_dummy_results(120.0))
        self.assertFalse(alerts)
