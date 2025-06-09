from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.tracking.bytetrack_utils import (
    process_frame_for_tracking,
)


@pytest.fixture
def mock_yolo_result():
    """YOLOモデルの結果をモック"""
    mock_box = MagicMock()
    mock_box.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 20, 50, 60]])
    mock_box.conf.cpu.return_value.numpy.return_value = np.array([0.85])
    mock_box.cls.cpu.return_value.numpy.return_value = np.array([0])

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]

    mock_results = MagicMock()
    mock_results.__getitem__.return_value = mock_result

    return mock_results


class TestTracking:
    """検出と追跡関連の関数のテスト"""

    @patch("time.time", side_effect=[0, 0.05, 0.051, 0.052])  # 検出0.05秒、追跡0.001秒かかるとする
    def test_process_frame_for_tracking(self, mock_time, mock_yolo_result):
        """process_frame_for_tracking関数のテスト"""
        # テスト用の簡単な画像とモック
        test_frame = np.zeros((100, 200, 3), dtype=np.uint8)
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_yolo_result

        mock_tracker = MagicMock()
        mock_tracker.update.return_value = [[10, 20, 50, 60, 1, 0.85, 0]]  # トラック結果

        # 関数実行
        tracks, det_time, track_time, num_det, num_track, avg_conf = process_frame_for_tracking(
            test_frame, mock_model, mock_tracker
        )

        # 結果の検証
        assert det_time == pytest.approx(50.0)  # 0.05秒 = 50ms
        assert track_time == pytest.approx(1.0)  # 0.001秒 = 1ms
        assert num_det == 1  # 検出数
        assert num_track == 1  # 追跡数
        assert avg_conf == pytest.approx(0.85)  # 平均信頼度

    @patch("time.time", side_effect=[0, 0.01, 0.02, 0.03])  # Simulated timings
    def test_process_frame_for_tracking_no_detections(self, mock_time):
        """Should handle case with no detections and return empty dets_for_tracker (0,6)"""

        test_frame = np.zeros((100, 200, 3), dtype=np.uint8)

        # Mock YOLO model with no detections
        mock_result = MagicMock()
        mock_result.boxes = []  # No boxes

        mock_results = MagicMock()
        mock_results.__getitem__.return_value = mock_result

        mock_model = MagicMock()
        mock_model.predict.return_value = mock_results

        mock_tracker = MagicMock()
        mock_tracker.update.return_value = []  # No tracks

        # Run function
        tracks, det_time, track_time, num_det, num_track, avg_conf = process_frame_for_tracking(
            test_frame, mock_model, mock_tracker
        )

        # Assert correct behavior
        assert det_time == pytest.approx(10.0)
        assert track_time == pytest.approx(10.0)
        assert num_det == 0
        assert num_track == 0
        assert avg_conf == 0.0

        # Check that tracker.update received an empty (0, 6) array
        dets_passed = mock_tracker.update.call_args[0][0]
        assert isinstance(dets_passed, np.ndarray)
        assert dets_passed.shape == (0, 6)
