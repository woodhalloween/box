import sys
from pathlib import Path
from unittest.mock import patch

# テスト対象のスクリプトが配置されているディレクトリをsys.pathに追加
# これにより、テスト実行時に 'src' モジュールを正しくインポートできる
script_dir = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(script_dir))

import argparse

import cv2
import numpy as np
import pytest

from scripts.detect_long_stay import LongStayBatchProcessor, LongStayDetector, main


# --- Fixtures ---
@pytest.fixture
def mock_detector_dependencies(mocker):
    """LongStayDetectorの依存関係をモック化するフィクスチャ"""
    mock_model = mocker.Mock()
    mock_model.ckpt_path = "dummy/path/yolov8n.pt"  # TypeError: expected str... の修正
    mocker.patch("scripts.detect_long_stay.load_yolo_model", return_value=mock_model)
    mocker.patch("scripts.detect_long_stay.initialize_bytetrack", return_value=mocker.Mock())
    mocker.patch("scripts.detect_long_stay.update_stay_times", return_value=({}, [], 10.0))
    mocker.patch(
        "scripts.detect_long_stay.process_frame_for_tracking",
        return_value=(np.array([[10, 10, 50, 50, 1]]), 10.0, 5.0, 1, 1, None),
    )
    mocker.patch("scripts.detect_long_stay.draw_tracking_info", return_value=np.zeros((100, 100, 3), dtype=np.uint8))
    mocker.patch.object(LongStayDetector, "_draw_overlay", return_value=np.zeros((100, 100, 3), dtype=np.uint8))


@pytest.fixture
def mock_cv2(mocker):
    """cv2関連の関数とクラスをモック化するフィクスチャ"""
    mock_video_capture = mocker.patch("cv2.VideoCapture")
    mock_cap_instance = mock_video_capture.return_value
    mock_cap_instance.isOpened.return_value = True
    mock_cap_instance.get.side_effect = [1920, 1080, 30, 300]  # width, height, fps, frame_count
    # 5フレーム分だけダミー画像を返し、その後(False, None)を返す
    mock_cap_instance.read.side_effect = [(True, np.zeros((1080, 1920, 3), dtype=np.uint8))] * 5 + [(False, None)]
    mocker.patch("cv2.VideoWriter")
    mocker.patch("cv2.imshow")
    mocker.patch("cv2.waitKey", return_value=27)  # ESCキーが押されたことにする
    mocker.patch("cv2.destroyAllWindows")


# --- LongStayDetector Tests ---
@pytest.mark.usefixtures("mock_detector_dependencies", "mock_cv2")
class TestLongStayDetector:
    def test_init(self):
        """初期化のテスト"""
        detector = LongStayDetector()
        assert detector.stay_threshold_sec == 5.0
        assert detector.conf == 0.3

    def test_process_video_basic_run(self, tmp_path):
        """process_videoが正常に最後まで実行されるかのテスト"""
        detector = LongStayDetector()
        input_file = "dummy.mp4"
        output_file = tmp_path / "output.mp4"

        # ファイルが存在するかのように見せかける
        with patch("os.path.exists", return_value=True):
            event_count = detector.process_video(str(input_file), str(output_file))

        assert event_count == 0  # update_stay_timesが空の通知を返すため
        assert cv2.VideoCapture.return_value.read.call_count == 6  # 5フレーム + 終了

    def test_process_video_perf_log(self, tmp_path, mocker):
        """パフォーマンスログが有効な場合のテスト"""
        mocker.patch("scripts.detect_long_stay.initialize_perf_log", return_value=str(tmp_path / "perf.csv"))
        mock_open = mocker.patch("builtins.open", mocker.mock_open())

        detector = LongStayDetector()
        input_file = "dummy.mp4"
        output_file = tmp_path / "output.mp4"

        with patch("os.path.exists", return_value=True):
            detector.process_video(str(input_file), str(output_file), enable_perf_log=True)

        # ログファイルが書き込み用にオープンされたかを確認
        assert mock_open.call_count > 0
        assert "perf.csv" in str(mock_open.call_args_list[0].args[0])

    def test_process_video_file_not_found(self, capsys):
        """入力ファイルが見つからない場合のテスト"""
        detector = LongStayDetector()
        with patch("os.path.exists", return_value=False):
            event_count = detector.process_video("nonexistent.mp4", "output.mp4")
        assert event_count == 0
        captured = capsys.readouterr()
        assert "エラー: 入力ファイルが見つかりません" in captured.out


# --- LongStayBatchProcessor Tests ---
@pytest.mark.usefixtures("mock_detector_dependencies")
class TestLongStayBatchProcessor:
    @pytest.fixture
    def setup_dirs(self, tmp_path):
        """バッチ処理用のディレクトリ構造を作成するフィクスチャ"""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        (input_dir / "video1.mp4").touch()
        (input_dir / "subdir").mkdir()
        (input_dir / "subdir" / "video2.avi").touch()
        (input_dir / "not_a_video.txt").touch()
        return input_dir, output_dir

    def test_run_batch(self, setup_dirs, mocker):
        """バッチ処理の実行テスト"""
        input_dir, output_dir = setup_dirs
        mock_process_video = mocker.patch.object(LongStayDetector, "process_video", return_value=2)

        processor = LongStayBatchProcessor(input_dir, output_dir, detector_options={})
        processor.run()

        # 2つの動画ファイルに対してprocess_videoが呼ばれたか
        assert mock_process_video.call_count == 2
        # ログファイルが作成され、書き込まれているか
        assert processor.log_file.exists()
        with open(processor.log_file) as f:
            content = f.read()
            assert "video1.mp4" in content
            assert "subdir/video2.avi" in content
            assert "Success" in content

    def test_run_batch_keyboard_interrupt(self, setup_dirs, mocker):
        """キーボード割り込みのテスト"""
        input_dir, output_dir = setup_dirs
        mocker.patch.object(LongStayDetector, "process_video", side_effect=KeyboardInterrupt)

        processor = LongStayBatchProcessor(input_dir, output_dir, detector_options={})
        processor.run()

        # 最初のファイル処理で中断される
        assert LongStayDetector.process_video.call_count == 1
        with open(processor.log_file) as f:
            assert "Interrupted" in f.read()


# --- main function Tests ---
@pytest.fixture
def mock_main_dependencies(mocker):
    """main関数の依存関係をモック化し、モックオブジェクトを返す"""
    mocks = {
        "LongStayDetector": mocker.patch("scripts.detect_long_stay.LongStayDetector", autospec=True),
        "LongStayBatchProcessor": mocker.patch("scripts.detect_long_stay.LongStayBatchProcessor", autospec=True),
    }
    mocker.patch("pathlib.Path.is_dir")
    mocker.patch("pathlib.Path.is_file")
    mocker.patch("pathlib.Path.mkdir")  # ディレクトリ作成をモック化してファイルシステムへの依存をなくす
    return mocks


@pytest.mark.usefixtures("mock_main_dependencies")
class TestMainFunction:
    def test_main_file_processing(self, mocker, mock_main_dependencies):
        """ファイル入力時のmain関数のテスト"""
        Path.is_dir.return_value = False
        Path.is_file.return_value = True
        mocker.patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(input="a.mp4", output="b.mp4", **self.get_default_args()),
        )

        main()

        mock_detector = mock_main_dependencies["LongStayDetector"]
        mock_batch_processor = mock_main_dependencies["LongStayBatchProcessor"]

        # LongStayDetectorクラスが一度インスタンス化されたことを確認
        mock_detector.assert_called_once()
        # インスタンスのprocess_videoメソッドが一度呼ばれたことを確認
        detector_instance = mock_detector.return_value
        detector_instance.process_video.assert_called_once()
        mock_batch_processor.assert_not_called()

    def test_main_directory_processing(self, mocker, mock_main_dependencies):
        """ディレクトリ入力時のmain関数のテスト"""
        Path.is_dir.return_value = True
        mocker.patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(input="in_dir", output="out_dir", **self.get_default_args()),
        )

        main()

        mock_detector = mock_main_dependencies["LongStayDetector"]
        mock_batch_processor = mock_main_dependencies["LongStayBatchProcessor"]

        # LongStayBatchProcessorクラスが一度インスタンス化されたことを確認
        mock_batch_processor.assert_called_once()
        # インスタンスのrunメソッドが一度呼ばれたことを確認
        batch_processor_instance = mock_batch_processor.return_value
        batch_processor_instance.run.assert_called_once()

    def test_main_invalid_path(self, mocker, capsys):
        """無効なパス入力時のmain関数のテスト"""
        Path.is_dir.return_value = False
        Path.is_file.return_value = False
        mocker.patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(input="invalid", output="out", **self.get_default_args()),
        )

        main()

        captured = capsys.readouterr()
        assert "エラー: 指定されたパスが見つかりません" in captured.out

    def get_default_args(self):
        """テスト用のデフォルト引数を返すヘルパー"""
        return {
            "model": "yolov8n.pt",
            "enable_perf_log": False,
            "enable_batch_perf_log": False,
            "enable_video_display": False,
            "device": "",
            "stay_threshold_sec": 5.0,
            "move_threshold_px": 30.0,
            "conf": 0.3,
            "enable_pose": False,
        }
