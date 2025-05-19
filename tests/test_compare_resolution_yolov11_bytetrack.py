# compare_resolution_yolov11_bytetrack.py の単体テスト
#
# このファイルはpytestフレームワークを使用して、YOLOv11とByteTrackによる
# 解像度別精度比較スクリプトの単体テストを行います。
#
# 実行方法:
#   - pytest -v tests/test_compare_resolution_yolov11_bytetrack.py
#   - または ./run_tests.sh
#
# 注意: pytestとモック関連のライブラリ（pytest-mock）が必要です。
# pip install -r requirements-dev.txt でインストールできます。

import os
import sys
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

# テスト対象のモジュールをインポートできるようにパスを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.compare_resolution_yolov11_bytetrack import (
    DetectionResults,
    draw_tracking_info,
    get_system_info,
    initialize_log_file,
    process_frame_for_tracking,
    resize_frame,
)


class TestDetectionResults:
    """DetectionResultsクラスのテスト"""

    def test_init(self):
        """初期化メソッドのテスト"""
        results = DetectionResults()
        assert results.frame_count == 0
        assert results.detection_times == []
        assert results.tracking_times == []
        assert results.fps_values == []
        assert results.objects_detected == []
        assert results.objects_tracked == []
        assert results.memory_usages == []
        assert results.detection_confidence == []

    def test_add_frame_result(self):
        """add_frame_resultメソッドのテスト"""
        results = DetectionResults()
        results.add_frame_result(10.0, 5.0, 30.0, 3, 2, 500.0, 0.85)

        assert results.frame_count == 1
        assert results.detection_times == [10.0]
        assert results.tracking_times == [5.0]
        assert results.fps_values == [30.0]
        assert results.objects_detected == [3]
        assert results.objects_tracked == [2]
        assert results.memory_usages == [500.0]
        assert results.detection_confidence == [0.85]

    def test_get_summary_empty(self):
        """空のDetectionResultsからのサマリ取得テスト"""
        results = DetectionResults()
        summary = results.get_summary()

        assert summary["frame_count"] == 0
        assert summary["avg_detection_time"] == 0
        assert summary["avg_tracking_time"] == 0
        assert summary["avg_fps"] == 0
        assert summary["avg_objects_detected"] == 0
        assert summary["avg_objects_tracked"] == 0
        assert summary["avg_memory_usage"] == 0
        assert summary["avg_detection_conf"] == 0
        assert summary["max_objects_detected"] == 0
        assert summary["max_objects_tracked"] == 0

    def test_get_summary_with_data(self):
        """データがあるDetectionResultsからのサマリ取得テスト"""
        results = DetectionResults()
        results.add_frame_result(10.0, 5.0, 30.0, 3, 2, 500.0, 0.85)
        results.add_frame_result(12.0, 6.0, 28.0, 4, 3, 550.0, 0.75)

        summary = results.get_summary()

        assert summary["frame_count"] == 2
        assert summary["avg_detection_time"] == pytest.approx(11.0)
        assert summary["avg_tracking_time"] == pytest.approx(5.5)
        assert summary["avg_fps"] == pytest.approx(29.0)
        assert summary["avg_objects_detected"] == pytest.approx(3.5)
        assert summary["avg_objects_tracked"] == pytest.approx(2.5)
        assert summary["avg_memory_usage"] == pytest.approx(525.0)
        assert summary["avg_detection_conf"] == pytest.approx(0.8)
        assert summary["max_objects_detected"] == 4
        assert summary["max_objects_tracked"] == 3


@pytest.fixture
def mock_system_info():
    """システム情報のモック"""
    return {
        "os": "TestOS",
        "os_version": "1.0",
        "python_version": "3.11.0",
        "cpu": "TestCPU",
        "cpu_cores": 4,
        "cpu_threads": 8,
        "ram_total": 16.0,
    }


class TestSystemInfo:
    """システム情報関連の関数のテスト"""

    @patch("platform.system", return_value="TestOS")
    @patch("platform.version", return_value="1.0")
    @patch("platform.python_version", return_value="3.11.0")
    @patch("platform.processor", return_value="TestCPU")
    @patch("psutil.cpu_count", side_effect=[4, 8])  # logical=False, logical=True
    @patch("psutil.virtual_memory")
    def test_get_system_info(
        self, mock_vm, mock_cpu_count, mock_processor, mock_py_ver, mock_version, mock_system
    ):
        """get_system_info関数のテスト"""
        # psutilのvirtual_memory()の戻り値をモック
        mock_vm.return_value.total = 17179869184  # 16GB in bytes

        info = get_system_info()

        assert info["os"] == "TestOS"
        assert info["os_version"] == "1.0"
        assert info["python_version"] == "3.11.0"
        assert info["cpu"] == "TestCPU"
        assert info["cpu_cores"] == 4
        assert info["cpu_threads"] == 8
        assert info["ram_total"] == pytest.approx(16.0)


class TestLogFile:
    """ログファイル関連の関数のテスト"""

    @patch("builtins.open", new_callable=mock_open)
    @patch("scripts.compare_resolution_yolov11_bytetrack.get_system_info")
    def test_initialize_log_file(self, mock_get_info, mock_file, mock_system_info):
        """initialize_log_file関数のテスト"""
        mock_get_info.return_value = mock_system_info

        log_file_path = "test_log.csv"
        input_file = "test_video.mp4"
        model_path = "test_model.pt"
        resolutions = [(1920, 1080), (960, 540)]

        log_f, log_writer = initialize_log_file(log_file_path, input_file, model_path, resolutions)

        # ファイルが開かれたことを確認
        mock_file.assert_called_once_with(log_file_path, "w", newline="")

        # ログヘッダーの書き込みが行われたことを確認
        handle = mock_file()
        write_calls = handle.write.call_args_list

        # write_callsを確認するか、csvを使った検証を行う
        assert len(write_calls) > 0


class TestImageProcessing:
    """画像処理関連の関数のテスト"""

    def test_resize_frame(self):
        """resize_frame関数のテスト"""
        # テスト用の簡単な画像配列を作成
        test_frame = np.zeros((100, 200, 3), dtype=np.uint8)
        test_frame[40:60, 80:120] = 255  # 中央に白い四角形を描画

        # リサイズ
        resized_frame = resize_frame(test_frame, 100, 50)

        # 期待されるサイズを確認
        assert resized_frame.shape == (50, 100, 3)

    @patch("cv2.rectangle")
    @patch("cv2.putText")
    def test_draw_tracking_info(self, mock_put_text, mock_rectangle):
        """draw_tracking_info関数のテスト"""
        # テスト用の簡単な画像配列とトラック情報
        test_frame = np.zeros((100, 200, 3), dtype=np.uint8)
        test_tracks = [
            [10, 20, 50, 60, 1, 0.9, 0],  # x1, y1, x2, y2, track_id, conf, cls_id
            [70, 30, 120, 80, 2, 0.8, 0],
        ]

        result_frame = draw_tracking_info(test_frame, test_tracks)

        # cv2.rectangleとcv2.putTextが各トラックに対して呼ばれたことを確認
        assert mock_rectangle.call_count == 2
        assert mock_put_text.call_count == 2

        # フレームが変更されずに返されたことを確認
        assert result_frame is test_frame


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


class TestComplexFunctions:
    """より複雑な関数のモックテスト"""

    @patch("scripts.compare_resolution_yolov11_bytetrack.cv2.VideoCapture")
    @patch("scripts.compare_resolution_yolov11_bytetrack.cv2.VideoWriter")
    @patch("scripts.compare_resolution_yolov11_bytetrack.cv2.cvtColor")
    @patch("scripts.compare_resolution_yolov11_bytetrack.process_frame_for_tracking")
    @patch("scripts.compare_resolution_yolov11_bytetrack.draw_tracking_info")
    @patch("scripts.compare_resolution_yolov11_bytetrack.time.time")
    @patch("scripts.compare_resolution_yolov11_bytetrack.psutil.Process")
    def test_process_video_basic(
        self,
        mock_process,
        mock_time,
        mock_draw,
        mock_process_frame,
        mock_cvt,
        mock_vid_writer,
        mock_vid_cap,
    ):
        """process_video関数の基本的なテスト"""
        from scripts.compare_resolution_yolov11_bytetrack import process_video

        # VideoCapture設定
        mock_vid_cap.return_value.isOpened.return_value = True
        mock_vid_cap.return_value.get.side_effect = [30.0, 1920, 1080]  # FPS, width, height
        # 2フレーム分の読み込みを模擬
        mock_vid_cap.return_value.read.side_effect = [
            (True, np.zeros((1080, 1920, 3))),
            (True, np.zeros((1080, 1920, 3))),
            (False, None),
        ]

        # 変換後の画像
        mock_cvt.return_value = np.zeros((1080, 1920, 3))

        # 検出と追跡の結果
        tracks = [[10, 20, 50, 60, 1, 0.85, 0]]
        mock_process_frame.return_value = (tracks, 10.0, 2.0, 1, 1, 0.85)

        # 時間計測（より多くの呼び出しに対応するために十分な値を設定）
        mock_time.side_effect = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

        # メモリ使用量
        mock_process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB

        # 関数を実行
        results = process_video(
            "test.mp4", "out.mp4", MagicMock(), MagicMock(), 1920, 1080, 1920, 1080, False
        )

        # 結果の検証
        assert results is not None
        assert results.frame_count == 2
        assert len(results.detection_times) == 2
        assert len(results.tracking_times) == 2

        # VideoCapture, VideoWriterが適切に呼ばれたか確認
        mock_vid_cap.assert_called_once_with("test.mp4")
        mock_vid_cap.return_value.release.assert_called_once()
        mock_vid_writer.return_value.release.assert_called_once()

    @patch("scripts.compare_resolution_yolov11_bytetrack.os.makedirs")
    @patch("scripts.compare_resolution_yolov11_bytetrack.os.path.join")
    @patch("scripts.compare_resolution_yolov11_bytetrack.cv2.VideoCapture")
    @patch("scripts.compare_resolution_yolov11_bytetrack.initialize_log_file")
    @patch("scripts.compare_resolution_yolov11_bytetrack.YOLO")
    @patch("scripts.compare_resolution_yolov11_bytetrack.ByteTrack")
    @patch("scripts.compare_resolution_yolov11_bytetrack.process_video")
    def test_compare_resolutions_basic(
        self,
        mock_process_video,
        mock_bytetrack,
        mock_yolo,
        mock_init_log,
        mock_vid_cap,
        mock_path_join,
        mock_makedirs,
    ):
        """compare_resolutions関数の基本的なテスト"""
        from scripts.compare_resolution_yolov11_bytetrack import compare_resolutions

        # VideoCapture設定
        mock_vid_cap.return_value.isOpened.return_value = True
        mock_vid_cap.return_value.get.return_value = 1920  # width/height

        # パス結合（ログファイルと出力ファイル）
        mock_path_join.side_effect = ["test_log.csv", "output1.mp4", "output2.mp4"]

        # process_videoの戻り値
        mock_results = MagicMock()
        mock_results.get_summary.return_value = {
            "frame_count": 100,
            "avg_detection_time": 10.0,
            "avg_tracking_time": 2.0,
            "avg_fps": 30.0,
            "avg_objects_detected": 2.5,
            "avg_objects_tracked": 2.0,
            "avg_memory_usage": 500.0,
            "avg_detection_conf": 0.85,
            "max_objects_detected": 5,
            "max_objects_tracked": 4,
        }
        mock_process_video.return_value = mock_results

        # 関数実行
        compare_resolutions("test.mp4", "output_dir", "yolov11n.pt", False, "cpu")

        # ディレクトリが作成されたことを確認
        mock_makedirs.assert_called_once_with("output_dir", exist_ok=True)

        # YOLOモデルがロードされたことを確認
        mock_yolo.assert_called_once()

        # 2つの解像度でprocess_videoが呼ばれたことを確認
        assert mock_process_video.call_count == 2

        # ByteTrackが2回（解像度ごとに）初期化されたことを確認
        assert mock_bytetrack.call_count == 2


# process_videoとcompare_resolutionsは複雑でモックが多いため、
# 結合テストやコンポーネントテストとして別途実装するか、
# より小さな単位でテストすることを推奨
