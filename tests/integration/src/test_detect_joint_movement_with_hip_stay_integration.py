"""src/detect_joint_movement_with_hip_stay.py の統合テスト"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np

from src.detect_joint_movement_with_hip_stay import main, process_video


class TestProcessVideo(unittest.TestCase):
    """(No. 29) process_video 関数の統合テスト"""

    # 多くの依存関係をモック化
    @patch("src.detect_joint_movement_with_hip_stay.cv2.VideoCapture")
    @patch("src.detect_joint_movement_with_hip_stay.setup_csv_writer")
    @patch("src.detect_joint_movement_with_hip_stay.setup_video_writer")
    @patch("src.detect_joint_movement_with_hip_stay.PoseEstimator")
    @patch("src.detect_joint_movement_with_hip_stay.MovementAnalyzer")
    @patch("src.detect_joint_movement_with_hip_stay.PostureMonitor")
    @patch("src.detect_joint_movement_with_hip_stay.HipBasedStayDetector")
    @patch("src.detect_joint_movement_with_hip_stay.KneeAngleMonitor")
    @patch("src.detect_joint_movement_with_hip_stay.HeadShakeDetector")
    @patch("src.detect_joint_movement_with_hip_stay.write_results_to_csv")
    @patch("src.detect_joint_movement_with_hip_stay.draw_analysis_results")
    @patch("src.detect_joint_movement_with_hip_stay.draw_landmarks")
    @patch("src.detect_joint_movement_with_hip_stay.draw_hip_stay_info")
    @patch("src.detect_joint_movement_with_hip_stay.draw_head_shake_info")
    @patch("src.detect_joint_movement_with_hip_stay.draw_posture_alerts")
    @patch("src.detect_joint_movement_with_hip_stay.cv2.imshow")
    @patch("src.detect_joint_movement_with_hip_stay.cv2.waitKey", return_value=1)  # ループが即時終了しないように変更
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.detect_joint_movement_with_hip_stay.Path")
    def test_process_video_runs_without_error(
        self,
        mock_path: MagicMock,
        mock_open_func: MagicMock,
        mock_wait_key: MagicMock,
        mock_imshow: MagicMock,
        _mock_draw_posture: MagicMock,
        mock_draw_head: MagicMock,
        mock_draw_hip: MagicMock,
        _mock_draw_landmarks: MagicMock,
        mock_draw_analysis: MagicMock,
        mock_write_csv: MagicMock,
        mock_head_detector: MagicMock,
        _mock_knee_monitor: MagicMock,
        _mock_hip_detector: MagicMock,
        _mock_posture_monitor: MagicMock,
        _mock_analyzer: MagicMock,
        mock_pose_estimator: MagicMock,
        mock_setup_video_writer: MagicMock,
        mock_setup_csv_writer: MagicMock,
        mock_video_capture: MagicMock,
    ):
        """関連関数をモック化し、一連の処理がエラーなく呼び出されることを確認"""
        # --- モックの設定 ---
        # VideoCaptureのモック
        mock_cap_instance = mock_video_capture.return_value
        mock_cap_instance.isOpened.return_value = True
        # 2フレーム処理 -> 終了 をシミュレート
        mock_cap_instance.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None),
        ]
        mock_cap_instance.get.return_value = 1000.0  # timestamp

        # PoseEstimatorのモック (ランドマークを返す)
        mock_pose_instance = mock_pose_estimator.return_value
        mock_pose_instance.estimate.return_value = np.zeros((33, 4))

        # 描画関数がフレーム(numpy配列)を返すように設定
        mock_draw_analysis.side_effect = lambda image, **_kwargs: image
        mock_draw_hip.side_effect = lambda frame, _detector: frame
        mock_draw_head.side_effect = lambda frame, _detector, _landmarks: frame

        # --- テスト対象の関数を実行 ---
        process_video(
            video_path="dummy.mp4",
            output_csv_path="dummy.csv",
            output_video_path="dummy_out.mp4",
            disable_japanese=False,
            hip_normalize=True,
        )

        # --- 呼び出しアサーション ---
        # ファイルパス関連
        self.assertTrue(mock_path.called)
        mock_open_func.assert_called_with("dummy.csv", "w", newline="", encoding="utf-8")

        # ビデオ・CSVライターのセットアップ
        mock_video_capture.assert_called_with("dummy.mp4")
        mock_setup_csv_writer.assert_called()
        mock_setup_video_writer.assert_called()

        # 各処理クラスのインスタンス化
        self.assertTrue(mock_pose_estimator.called)
        self.assertTrue(_mock_analyzer.called)
        self.assertTrue(_mock_posture_monitor.called)
        self.assertTrue(_mock_hip_detector.called)
        self.assertTrue(_mock_knee_monitor.called)
        self.assertTrue(mock_head_detector.called)

        # ループ内の主要な処理 (2フレーム分呼ばれる)
        self.assertEqual(mock_pose_instance.estimate.call_count, 2)
        self.assertEqual(mock_write_csv.call_count, 2)
        self.assertEqual(mock_draw_analysis.call_count, 2)
        self.assertEqual(mock_setup_video_writer.return_value.write.call_count, 2)

        # 表示関連
        self.assertEqual(mock_imshow.call_count, 2)
        self.assertEqual(mock_wait_key.call_count, 2)

        # リソース解放処理
        mock_cap_instance.release.assert_called_once()
        mock_setup_video_writer.return_value.release.assert_called_once()


class TestMainFunction(unittest.TestCase):
    """(No. 30) main 関数の統合テスト"""

    @patch("src.detect_joint_movement_with_hip_stay.process_video")
    @patch("argparse.ArgumentParser.parse_args")
    def test_main_calls_process_video_with_args(self, mock_parse_args: MagicMock, mock_process_video: MagicMock):
        """コマンドライン引数をシミュレートし、process_videoが適切な引数で呼び出されることを確認"""
        # --- モックの設定 ---
        # parse_argsが返すオブジェクトをモック化
        mock_args = MagicMock(
            video="test.mp4",
            hip_normalize=True,
            hip_stay_threshold=30.0,
            spike_threshold=1.0,
            stability_threshold=40.0,
            grace_period=2.0,
        )
        mock_parse_args.return_value = mock_args

        # --- テスト対象の関数を実行 ---
        with (
            patch("src.detect_joint_movement_with_hip_stay.os.path.basename", return_value="test.mp4"),
            patch("src.detect_joint_movement_with_hip_stay.os.path.splitext", return_value=("test", ".mp4")),
        ):
            main()

        # --- 呼び出しアサーション ---
        mock_process_video.assert_called_once()
        _, kwargs = mock_process_video.call_args
        self.assertEqual(kwargs["video_path"], "test.mp4")
        self.assertTrue(kwargs["hip_normalize"])
        self.assertEqual(kwargs["hip_stay_threshold"], 30.0)
        self.assertEqual(kwargs["spike_threshold"], 1.0)
        self.assertEqual(kwargs["stability_threshold_px"], 40.0)
        self.assertEqual(kwargs["grace_period_sec"], 2.0)
