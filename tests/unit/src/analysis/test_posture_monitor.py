from __future__ import annotations

import unittest

from src.analysis.posture_monitor import PostureMonitor
from src.definitions import Angle, MovementState


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

    def test_is_forward_leaning_posture_returns_default_when_no_data(self):
        """Ensure is_forward_leaning_posture returns (False, 0.0) when no relevant data is provided"""
        # Empty analysis_results should lead to default (False, 0.0)
        is_leaning, score = self.monitor.is_forward_leaning_posture({})
        self.assertFalse(is_leaning)
        self.assertEqual(score, 0.0)

    def test_check_for_alerts_returns_empty_when_no_history(self):
        """Ensure _check_for_alerts returns empty alerts when posture_history is empty"""
        # Confirm posture_history is empty
        self.assertEqual(len(self.monitor.posture_history), 0)
        # Directly call _check_for_alerts with arbitrary timestamp
        alerts = self.monitor._check_for_alerts(current_time=10.0)
        self.assertEqual(alerts, [])
