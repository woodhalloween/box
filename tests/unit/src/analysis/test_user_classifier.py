"""src/analysis/user_classifier.pyのテスト"""

from __future__ import annotations

import unittest

from src.analysis.user_classifier import UserClassifier
from src.definitions import Angle


class TestUserClassifier(unittest.TestCase):
    """UserClassifierクラスのテストスイート"""

    def setUp(self):
        """テスト用のUserClassifierインスタンスを初期化"""
        self.monitor = UserClassifier(threshold_deg=90.0, moving_window_seconds=3, confidence_threshold=0.5)

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
