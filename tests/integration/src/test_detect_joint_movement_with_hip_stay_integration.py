"""src/detect_joint_movement_with_hip_stay.py の統合テスト"""

from __future__ import annotations

import contextlib
import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np

from src.detect_joint_movement_with_hip_stay import main, process_video


class TestProcessVideo(unittest.TestCase):
    """(No. 29) process_video 関数の統合テスト（新シグネチャ対応）"""

    # NOTE: make_frame_iter, run_pipeline は存在しない場合もあるため create=True
    @patch("src.video_processor.run_pipeline", create=True)
    @patch("src.video_processor.make_frame_iter", create=True)
    @patch("src.video_processor.setup_csv_writer")  # <-- add this
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.detect_joint_movement_with_hip_stay.PoseEstimator")
    @patch("src.detect_joint_movement_with_hip_stay.MovementAnalyzer")
    @patch("src.detect_joint_movement_with_hip_stay.UserClassifier")
    @patch("src.detect_joint_movement_with_hip_stay.DwellTimeDetector")
    @patch("src.detect_joint_movement_with_hip_stay.HeadShakeDetector")
    @patch("src.detect_joint_movement_with_hip_stay.PostureMonitor")
    def test_ffmpeg_path_wires_correctly(
        self,
        _mock_posture: MagicMock,
        _mock_head: MagicMock,
        _mock_dwell: MagicMock,
        _mock_userclf: MagicMock,
        _mock_analyzer: MagicMock,
        _mock_pose: MagicMock,
        mock_open_fn: MagicMock,
        mock_setup_csv_writer,  # <-- receive it
        mock_make_frame_iter: MagicMock,
        mock_run_pipeline: MagicMock,
    ):
        """FFmpeg 経路（make_frame_iter 利用）の配線と引数を確認"""
        dummy_frames = iter(
            [
                (0.00, np.zeros((720, 1280, 3), dtype=np.uint8)),
                (0.04, np.zeros((720, 1280, 3), dtype=np.uint8)),
            ]
        )
        mock_make_frame_iter.return_value = dummy_frames

        mock_csv_writer = MagicMock()
        mock_setup_csv_writer.return_value = mock_csv_writer  # <-- return your mock

        process_video(
            video_path="video.mp4",
            output_csv_path="out.csv",
            output_video_path=None,
            disable_japanese=True,
            stay_threshold_sec=30.0,
            spike_threshold=1.0,
            stability_threshold_px=40.0,
            grace_period_sec=2.0,
            pm_monitoring_duration_sec=90.0,
            pm_alert_threshold_ratio=0.6,
            uc_threshold_deg=85.0,
            uc_moving_window_seconds=7.0,
            input_mode="ffmpeg-file",
            width=960,
            height=540,
            fps=24.0,
            is_color=True,
            preview=False,
        )

        # CSV open & writer
        mock_open_fn.assert_called_with("out.csv", "w", newline="", encoding="utf-8")

        # make_frame_iter 引数
        mock_make_frame_iter.assert_called_once()
        _args, kwargs = mock_make_frame_iter.call_args
        self.assertEqual(_args[0], "ffmpeg-file")
        self.assertEqual(kwargs["ffmpeg_input"], "video.mp4")
        self.assertEqual(kwargs["width"], 960)
        self.assertEqual(kwargs["height"], 540)
        self.assertEqual(kwargs["fps"], 24.0)
        self.assertTrue(kwargs["is_color"])

        # run_pipeline 配線
        mock_run_pipeline.assert_called_once()
        _pa, _pk = mock_run_pipeline.call_args
        self.assertIs(_pa[0], dummy_frames)
        self.assertIs(_pk["csv_writer"], mock_csv_writer)
        self.assertIsNone(_pk["video_writer"])
        self.assertFalse(_pk["preview"])
        self.assertEqual(_pk["window_name"], "Integrated Analysis")

        state = _pk["state"]
        self.assertTrue(state.disable_jp)
        for attr in (
            "pose",
            "analyzer",
            "user_classifier",
            "dwell_time_detector",
            "head_shake_detector",
            "posture_monitor",
        ):
            self.assertTrue(hasattr(state, attr))

    @patch("src.video_processor.run_pipeline", create=True)
    @patch("src.video_processor.cv2.VideoCapture")  # <-- use this one
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.detect_joint_movement_with_hip_stay.PoseEstimator")
    @patch("src.detect_joint_movement_with_hip_stay.MovementAnalyzer")
    @patch("src.detect_joint_movement_with_hip_stay.UserClassifier")
    @patch("src.detect_joint_movement_with_hip_stay.DwellTimeDetector")
    @patch("src.detect_joint_movement_with_hip_stay.HeadShakeDetector")
    @patch("src.detect_joint_movement_with_hip_stay.PostureMonitor")
    def test_opencv_fallback_when_no_ffmpeg(
        self,
        _mock_posture,
        _mock_head,
        _mock_dwell,
        _mock_userclf,
        _mock_analyzer,
        _mock_pose,
        mock_open_fn,
        mock_vcap,  # <-- use this
        mock_run_pipeline,
    ):
        # Make run_pipeline pull at least one frame so _opencv_iter executes
        def _consume_once(frame_iter, **kwargs):
            with contextlib.suppress(StopIteration):
                next(frame_iter)

        mock_run_pipeline.side_effect = _consume_once

        # (Optional) If you want to ensure the OpenCV branch by dependency state:
        # BUT not needed when input_mode="opencv-file"
        # with patch("src.video_processor.make_frame_iter", None, create=True):
        cap = mock_vcap.return_value
        cap.isOpened.return_value = True
        cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None),
        ]
        cap.get.return_value = 1000.0

        process_video(
            video_path="video.mp4",
            output_csv_path=None,
            output_video_path=None,
            disable_japanese=True,
            input_mode="opencv-file",  # force OpenCV path
        )

        mock_vcap.assert_called_once_with("video.mp4")
        mock_run_pipeline.assert_called_once()


class TestMainFunction(unittest.TestCase):
    """(No. 30) main 関数の統合テスト（process_video 新シグネチャ対応）"""

    @patch("src.detect_joint_movement_with_hip_stay.process_video")
    @patch("argparse.ArgumentParser.parse_args")
    @patch("src.detect_joint_movement_with_hip_stay.datetime")  # タイムスタンプ固定
    def test_main_calls_process_video_with_args(
        self,
        mock_dt: MagicMock,
        mock_parse_args: MagicMock,
        mock_process_video: MagicMock,
    ) -> None:
        """CLI 引数をシミュレートし、生成された出力パスと新引数で呼ばれることを確認"""
        mock_dt.now.return_value.strftime.return_value = "20250920_202147"

        mock_args = MagicMock(
            video="test.mp4",
            output_csv=None,
            output_video=None,
            disable_japanese=True,
            stay_threshold_sec=30.0,
            spike_threshold=1.0,
            stability_threshold_px=40.0,
            grace_period_sec=2.0,
            pm_monitoring_duration_sec=90.0,
            pm_alert_threshold_ratio=0.6,
            uc_threshold_deg=85.0,
            uc_moving_window_seconds=7.0,
            input_mode="ffmpeg-file",
            width=960,
            height=540,
            fps=24.0,
            no_display=True,  # => preview=False
            gray=False,  # => is_color=True
        )
        mock_parse_args.return_value = mock_args

        with (
            patch("src.detect_joint_movement_with_hip_stay.os.path.basename", return_value="test.mp4"),
            patch("src.detect_joint_movement_with_hip_stay.os.path.splitext", return_value=("test", ".mp4")),
            patch("src.detect_joint_movement_with_hip_stay.os.makedirs"),
        ):
            main()

        mock_process_video.assert_called_once()
        args, kwargs = mock_process_video.call_args

        def get_arg(name: str, pos: int | None = None):
            if name in kwargs:
                return kwargs[name]
            if pos is not None and len(args) > pos:
                return args[pos]
            raise AssertionError(f"{name} not found in process_video call")

        self.assertEqual(get_arg("video_path", 0), "test.mp4")
        expected_csv = "output/test_integrated_analysis_20250920_202147.csv"
        expected_mp4 = "output/test_integrated_output_20250920_202147.mp4"
        self.assertEqual(get_arg("output_csv_path", 1), expected_csv)
        self.assertEqual(get_arg("output_video_path", 2), expected_mp4)
        self.assertTrue(get_arg("disable_japanese", 3))

        self.assertEqual(get_arg("stay_threshold_sec"), 30.0)
        self.assertEqual(get_arg("spike_threshold"), 1.0)
        self.assertEqual(get_arg("stability_threshold_px"), 40.0)
        self.assertEqual(get_arg("grace_period_sec"), 2.0)
        self.assertEqual(get_arg("pm_monitoring_duration_sec"), 90.0)
        self.assertEqual(get_arg("pm_alert_threshold_ratio"), 0.6)
        self.assertEqual(get_arg("uc_threshold_deg"), 85.0)
        self.assertEqual(get_arg("uc_moving_window_seconds"), 7.0)
        self.assertEqual(get_arg("input_mode"), "ffmpeg-file")
        self.assertEqual(get_arg("width"), 960)
        self.assertEqual(get_arg("height"), 540)
        self.assertEqual(get_arg("fps"), 24.0)
        self.assertTrue(get_arg("is_color"))
        self.assertFalse(get_arg("preview"))
