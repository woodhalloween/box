import sys
from pathlib import Path

# テスト対象のスクリプトが配置されているディレクトリをsys.pathに追加
# これにより、テスト実行時に 'src' モジュールを正しくインポートできる
# Ruff (E402) の規約に従い、sys.pathの操作と他のimportの間に空白行を設ける
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

import argparse
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from ultralytics import YOLO

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

    def test_init_model_load_fail(self, mocker):
        """モデルのロードに失敗した場合の初期化テスト"""
        mocker.patch("scripts.detect_long_stay.load_yolo_model", return_value=None)
        with pytest.raises(ValueError, match="YOLOモデルのロードに失敗しました"):
            LongStayDetector()

    @pytest.mark.usefixtures("mock_cv2")
    def test_init_pose_enabled(self, capsys):
        """姿勢推定を有効にした場合の初期化テスト"""
        detector = LongStayDetector(enable_pose=True)
        assert detector.enable_pose is True
        captured = capsys.readouterr()
        assert "姿勢推定モードが有効です" in captured.out

    @pytest.mark.usefixtures("mock_cv2")
    def test_process_video_basic_run(self, tmp_path, capsys):
        """process_videoが正常に最後まで実行されるかのテスト"""
        detector = LongStayDetector()
        input_file = str(tmp_path / "dummy.mp4")
        Path(input_file).touch()
        output_file = tmp_path / "output.mp4"

        detector.process_video(input_file, str(output_file))

        captured = capsys.readouterr()
        assert "処理完了" in captured.out
        assert cv2.VideoCapture.return_value.read.call_count == 6  # 5フレーム + 終了

    def test_process_video_perf_log(self, tmp_path, mocker):
        """パフォーマンスログが有効な場合のテスト"""
        mocker.patch("scripts.detect_long_stay.initialize_perf_log", return_value=str(tmp_path / "perf.csv"))
        mock_open = mocker.patch("builtins.open", mocker.mock_open())

        detector = LongStayDetector()
        input_file = str(tmp_path / "dummy.mp4")
        Path(input_file).touch()
        output_file = tmp_path / "output.mp4"

        detector.process_video(str(input_file), str(output_file), enable_perf_log=True)

        # ログファイルが書き込み用にオープンされたかを確認
        assert mock_open.call_count > 0
        assert "perf.csv" in str(mock_open.call_args_list[0].args[0])

    def test_process_video_file_not_found(self, capsys):
        """入力ファイルが見つからない場合のテスト"""
        detector = LongStayDetector()
        # process_video内でos.path.existsをチェックするようになったため、patchは不要
        detector.process_video("nonexistent.mp4", "output.mp4")
        captured = capsys.readouterr()
        assert "エラー: 入力ファイルが見つかりません" in captured.out

    @pytest.mark.usefixtures("mock_cv2")
    def test_process_video_cannot_open(self, capsys):
        """動画ファイルが開けない場合のテスト"""
        cv2.VideoCapture.return_value.isOpened.return_value = False
        detector = LongStayDetector()
        input_file = "dummy.mp4"
        with patch("os.path.exists", return_value=True):
            detector.process_video(input_file, "output.mp4")
        captured = capsys.readouterr()
        assert "エラー: 動画ファイルを開けません" in captured.out

    @pytest.mark.usefixtures("mock_cv2")
    def test_process_video_with_notifications(self, mocker, capsys):
        """長時間滞在の通知が発生するケースのテスト"""
        # update_stay_timesが通知を返すようにモック化
        mocker.patch(
            "scripts.detect_long_stay.update_stay_times", return_value=({}, [{"id": 1, "duration": 5.1}], 10.0)
        )
        detector = LongStayDetector()
        input_file = "dummy.mp4"
        with patch("os.path.exists", return_value=True):
            detector.process_video(input_file, "output.mp4")
        captured = capsys.readouterr()
        assert "Frame 1:" in captured.out  # 通知が出力されることを確認

    @pytest.mark.usefixtures("mock_cv2")
    def test_process_video_keyboard_interrupt(self, mocker, capsys):
        """処理中にキーボード割り込みが発生してもfinallyが実行されるテスト"""
        mocker.patch("scripts.detect_long_stay.process_frame_for_tracking", side_effect=KeyboardInterrupt)
        detector = LongStayDetector()
        input_file = "dummy.mp4"
        with pytest.raises(KeyboardInterrupt), patch("os.path.exists", return_value=True):
            detector.process_video(input_file, "output.mp4")

        # 例外が発生しても、finallyブロックの出力は行われる
        captured = capsys.readouterr()
        assert "処理完了" in captured.out

    @pytest.mark.usefixtures("mock_cv2")
    def test_process_video_no_output_no_display(self, mocker, tmp_path):
        """動画出力も画面表示も無効な場合のテスト"""
        detector = LongStayDetector(enable_video_display=False)
        input_file = str(tmp_path / "dummy.mp4")
        Path(input_file).touch()

        # `process_frame_for_tracking`が呼ばれることを確認するためのスパイ
        process_frame_spy = mocker.spy(sys.modules["scripts.detect_long_stay"], "process_frame_for_tracking")

        # output_file=Noneで呼び出す
        detector.process_video(input_file, output_file=None)

        # 描画関連の処理がスキップされても、フレーム処理は行われることを確認
        assert process_frame_spy.call_count > 0

    @pytest.mark.usefixtures("mock_cv2")
    def test_model_name_fallback(self, mocker, tmp_path):
        """モデルにckpt_pathがない場合のフォールバックテスト"""
        # CI環境でのpsutilエラーを回避しつつ、後続処理に必要なキーをダミーで提供
        mocker.patch(
            "src.tracking.bytetrack_utils.get_system_info",
            return_value={
                "os": "mock_os",
                "os_version": "1.0",
                "python_version": "3.10",
                "cpu": "mock_cpu",
                "cpu_cores": 1,
                "ram_total": 1,
                "cpu_threads": 2,
            },
        )

        # ckpt_pathを持たないモデルをモック
        mock_model_no_path = mocker.Mock(spec=YOLO)
        del mock_model_no_path.ckpt_path
        mocker.patch("scripts.detect_long_stay.load_yolo_model", return_value=mock_model_no_path)

        # ログファイル書き込みの確認
        mock_open = mocker.patch("builtins.open", mocker.mock_open())
        detector = LongStayDetector()
        input_file = str(tmp_path / "dummy.mp4")
        Path(input_file).touch()
        detector.process_video(input_file, str(tmp_path / "output.mp4"), enable_perf_log=True)

        # ログに 'yolo_model' と書き込まれているか確認
        write_calls = mock_open.return_value.write.call_args_list
        log_content = "".join([call[0][0] for call in write_calls])
        assert "yolo_model" in log_content

    @pytest.mark.usefixtures("mock_cv2")
    def test_video_display_loop(self, mocker):
        """動画表示ループが正常に動作し、ESCキーで終了するかのテスト"""
        # waitKeyが最初は1を返し、最後に27(ESC)を返すように設定
        mocker.patch("cv2.waitKey", side_effect=[1, 1, 1, 1, 27])
        imshow_spy = mocker.spy(cv2, "imshow")

        detector = LongStayDetector(enable_video_display=True)
        input_file = "dummy.mp4"
        with patch("os.path.exists", return_value=True):
            detector.process_video(input_file, output_file=None)

        # 4フレーム分表示され、5回目のwaitKeyでループを抜けることを確認
        assert imshow_spy.call_count == 5

    @pytest.mark.usefixtures("mock_cv2")
    def test_perf_log_initialization_fails(self, mocker, tmp_path, capsys):
        """パフォーマンスログの初期化に失敗するケースをテスト"""
        mock_init_log = mocker.patch("scripts.detect_long_stay.initialize_perf_log", return_value=None)
        mock_open = mocker.patch("builtins.open", mocker.mock_open())

        detector = LongStayDetector()
        input_file = str(tmp_path / "dummy.mp4")
        Path(input_file).touch()

        detector.process_video(input_file, None, enable_perf_log=True)

        # initialize_perf_log は呼ばれるが、open は呼ばれないことを確認
        mock_init_log.assert_called_once()
        mock_open.assert_not_called()

    @pytest.mark.usefixtures("mock_cv2")
    def test_perf_log_write_error_in_loop(self, mocker, tmp_path, capsys):
        """ループ中のパフォーマンスログ書き込みでIOErrorが発生するケース"""
        # このテストケースでのみ、FPS=1として毎フレームログが書き込まれるようにする
        mock_video_capture = mocker.patch("cv2.VideoCapture")
        mock_cap_instance = mock_video_capture.return_value
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.get.side_effect = [1920, 1080, 1.0, 5]  # width, height, fps=1, frame_count
        mock_cap_instance.read.side_effect = [(True, np.zeros((1080, 1920, 3), dtype=np.uint8))] * 5 + [(False, None)]

        log_path = tmp_path / "perf.csv"
        mocker.patch("scripts.detect_long_stay.initialize_perf_log", return_value=str(log_path))

        # openをモックし、2回目の呼び出しでIOErrorを発生させる
        mock_open = mocker.patch("builtins.open", mocker.mock_open())
        mock_open.side_effect = [
            mocker.DEFAULT,  # 1回目(ヘッダー)は通常の振る舞い
            OSError("Disk full"),  # 2回目(ループ内)でエラー
        ]

        detector = LongStayDetector()
        input_file = str(tmp_path / "dummy.mp4")
        Path(input_file).touch()

        detector.process_video(input_file, None, enable_perf_log=True)
        captured = capsys.readouterr()
        assert "パフォーマンスログファイルへの書き込み中にエラーが発生しました" in captured.out

    @pytest.mark.usefixtures("mock_cv2")
    def test_zero_loop_time(self, mocker, tmp_path):
        """ループ処理時間がゼロになるエッジケースのテスト"""
        # time.time()が常に同じ値を返すようにモック
        mocker.patch("time.time", return_value=12345.0)

        detector = LongStayDetector()
        input_file = str(tmp_path / "dummy.mp4")
        Path(input_file).touch()

        # エラーなく実行が完了することを確認
        detector.process_video(input_file, None)

    def test_long_video_processing(self, mocker, tmp_path):
        """長尺動画を想定したテスト（進捗表示やバッファのテスト）"""
        # 40フレームの動画をモック
        mock_video_capture = mocker.patch("cv2.VideoCapture")
        mock_cap_instance = mock_video_capture.return_value
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.get.side_effect = [1920, 1080, 30.0, 40]  # fpsをfloatに
        read_results = [(True, np.zeros((1080, 1920, 3), dtype=np.uint8))] * 40 + [(False, None)]
        mock_cap_instance.read.side_effect = read_results
        mocker.patch("cv2.VideoWriter")
        # process_frame_for_trackingが空のリストを返すようにモック
        mocker.patch(
            "scripts.detect_long_stay.process_frame_for_tracking",
            return_value=([], 0, 0, 0, 0, None),  # keypointsをNoneに
        )

        detector = LongStayDetector(enable_video_display=False)
        input_file = str(tmp_path / "dummy.mp4")
        Path(input_file).touch()

        with patch("builtins.print") as mock_print:
            detector.process_video(input_file, str(tmp_path / "output.mp4"))
            # 進捗表示が呼ばれたかを確認
            assert any("進捗" in str(call) for call in mock_print.call_args_list)


@pytest.mark.usefixtures("mock_cv2")
def test_draw_overlay(mocker):
    """_draw_overlayメソッドのテスト"""
    # このテストでは_draw_overlay自体をテストしたいため、
    # クラスから独立させて、必要なものだけモックする
    mocker.patch("scripts.detect_long_stay.load_yolo_model")
    mocker.patch("scripts.detect_long_stay.initialize_bytetrack")

    # _draw_overlay内のcv2関数をモック
    mock_rect = mocker.patch("cv2.rectangle")
    mock_text = mocker.patch("cv2.putText")

    # _draw_overlayはプライベートメソッドなので、インスタンスを作って呼び出す
    detector = LongStayDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # ケース1：通常時
    detector._draw_overlay(
        frame, 1, 100, 30.0, {"detection": 10, "tracking": 5, "stay_check": 2}, {"detected": 2, "tracked": 2}
    )
    assert mock_rect.call_count == 1
    assert mock_text.call_count == 3

    # モックの呼び出し履歴をリセット
    mock_rect.reset_mock()
    mock_text.reset_mock()

    # ケース2：長時間滞在者あり
    detector.stay_info = {1: {"stay_duration": 6.0}}
    detector._draw_overlay(
        frame, 1, 100, 30.0, {"detection": 10, "tracking": 5, "stay_check": 2}, {"detected": 2, "tracked": 2}
    )
    # 長時間滞在者用の四角とテキストが追加で描画される
    assert mock_rect.call_count == 1 + 1
    assert mock_text.call_count == 3 + 2


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

    def test_init_fail_not_a_directory(self, tmp_path):
        """入力パスがディレクトリでない場合の初期化テスト"""
        input_file = tmp_path / "not_a_dir.txt"
        input_file.touch()
        with pytest.raises(ValueError, match="入力パスはディレクトリである必要があります"):
            LongStayBatchProcessor(input_file, tmp_path / "output", {})

    def test_run_batch_no_videos(self, setup_dirs, capsys):
        """処理対象の動画がない場合のテスト"""
        input_dir, output_dir = setup_dirs
        # 動画ファイルを削除
        for p in input_dir.rglob("*"):
            if p.is_file():
                p.unlink()

        processor = LongStayBatchProcessor(input_dir, output_dir, detector_options={})
        processor.run()
        captured = capsys.readouterr()
        assert "処理対象の動画ファイルが見つかりませんでした" in captured.out

    def test_run_batch_generic_exception(self, setup_dirs, mocker, capsys):
        """処理中に予期せぬ例外が発生するテスト"""
        input_dir, output_dir = setup_dirs
        mocker.patch.object(LongStayDetector, "process_video", side_effect=Exception("Unexpected error"))

        processor = LongStayBatchProcessor(input_dir, output_dir, detector_options={})
        processor.run()

        captured = capsys.readouterr()
        assert "[エラー]" in captured.out
        assert "Unexpected error" in captured.out

        with open(processor.log_file) as f:
            assert "Failed" in f.read()

    def test_main_generic_exception(self, mocker, capsys):
        """main関数で予期せぬ例外が発生するテスト"""
        mocker.patch("pathlib.Path.exists", return_value=True)  # パスは存在するものとする
        mocker.patch("pathlib.Path.is_dir", return_value=False)  # ディレクトリではない
        mocker.patch("pathlib.Path.is_file", return_value=True)  # ファイルである
        # LongStayDetectorの初期化でエラーを発生させる
        mocker.patch("scripts.detect_long_stay.LongStayDetector", side_effect=Exception("Init failed"))
        mocker.patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(input="a.mp4", **self.get_default_args()),
        )
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "予期せぬエラーが発生しました" in captured.out

    def test_main_invalid_path(self, mocker, capsys):
        """無効なパス入力時のmain関数のテスト"""
        mocker.patch("pathlib.Path.exists", return_value=False)
        mocker.patch(
            "argparse.ArgumentParser.parse_args",
            # main関数が必要とするすべての引数を渡す
            return_value=argparse.Namespace(input="invalid", **self.get_default_args()),
        )
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "指定されたパスが見つかりません" in captured.out

    def get_default_args(self):
        """テスト用のデフォルト引数を返すヘルパー"""
        return {
            "model": "yolov8n.pt",
            "output": "output",
            "enable_perf_log": False,
            "enable_batch_perf_log": False,
            "enable_video_display": False,
            "device": "",
            "stay_threshold_sec": 5.0,
            "move_threshold_px": 30.0,
            "conf": 0.3,
            "enable_pose": False,
        }


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
        mocker.patch("pathlib.Path.exists", return_value=True)
        Path.is_dir.return_value = False
        Path.is_file.return_value = True
        mocker.patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(input="a.mp4", **self.get_default_args()),
        )

        main()

        mock_detector = mock_main_dependencies["LongStayDetector"]
        mock_batch_processor = mock_main_dependencies["LongStayBatchProcessor"]

        # ファイル処理が呼ばれることを確認
        mock_detector.return_value.process_video.assert_called_once_with(
            "a.mp4",
            "output",  # get_default_argsから取得
            enable_perf_log=False,
        )
        assert mock_batch_processor.call_count == 0

    def test_main_file_processing_with_options(self, mocker, mock_main_dependencies):
        """オプション付きファイル入力時のmain関数のテスト"""
        mocker.patch("pathlib.Path.exists", return_value=True)
        Path.is_dir.return_value = False
        Path.is_file.return_value = True
        mocker.patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(
                input="a.mp4",
                output="b.mp4",
                model="yolov8s.pt",
                enable_perf_log=True,
                enable_batch_perf_log=True,
                enable_video_display=True,
                device="cpu",
                stay_threshold_sec=10.0,
                move_threshold_px=50.0,
                conf=0.5,
                enable_pose=True,
            ),
        )

        main()

        mock_detector = mock_main_dependencies["LongStayDetector"]
        mock_batch_processor = mock_main_dependencies["LongStayBatchProcessor"]

        # ファイル処理が呼ばれることを確認
        mock_detector.return_value.process_video.assert_called_once_with(
            "a.mp4",
            "b.mp4",
            enable_perf_log=True,
        )
        assert mock_batch_processor.call_count == 0

    def test_main_directory_processing(self, mocker, mock_main_dependencies):
        """ディレクトリ入力時のmain関数のテスト"""
        mocker.patch("pathlib.Path.exists", return_value=True)
        Path.is_dir.return_value = True
        mocker.patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(input="in_dir", **self.get_default_args()),
        )

        main()

        mock_batch_processor = mock_main_dependencies["LongStayBatchProcessor"]

        # ディレクトリ処理が呼ばれることを確認
        mock_batch_processor.assert_called_once()
        # runメソッドが一度呼ばれたことを確認
        mock_batch_processor.return_value.run.assert_called_once()

    def test_main_invalid_path(self, mocker, capsys, mock_main_dependencies):
        """無効なパスが指定された場合のmain関数のテスト"""
        mocker.patch("pathlib.Path.exists", return_value=False)
        mocker.patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(input="invalid", **self.get_default_args()),
        )

        # main() を実行し、SystemExitが発生することを確認
        with pytest.raises(SystemExit):
            main()

        # DetectorやProcessorが呼ばれないことを確認
        assert mock_main_dependencies["LongStayDetector"].call_count == 0
        assert mock_main_dependencies["LongStayBatchProcessor"].call_count == 0

        captured = capsys.readouterr()
        assert "指定されたパスが見つかりません" in captured.out

    def test_main_generic_exception(self, mocker, capsys, mock_main_dependencies):
        """main関数で予期せぬ例外が発生するケース"""
        mocker.patch("pathlib.Path.exists", return_value=True)
        mocker.patch("pathlib.Path.is_dir", return_value=False)  # ディレクトリではない
        mocker.patch("pathlib.Path.is_file", return_value=True)  # ファイルである
        mocker.patch("scripts.detect_long_stay.LongStayDetector", side_effect=Exception("Test Exception"))
        mocker.patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(input="a.mp4", **self.get_default_args()),
        )

        with pytest.raises(SystemExit):
            main()

        captured = capsys.readouterr()
        assert "予期せぬエラーが発生しました" in captured.out
        assert "Test Exception" in captured.out

    def get_default_args(self):
        """テスト用のデフォルト引数を返すヘルパー"""
        return {
            "model": "yolov8n.pt",
            "output": "output",
            "enable_perf_log": False,
            "enable_batch_perf_log": False,
            "enable_video_display": False,
            "device": "",
            "stay_threshold_sec": 5.0,
            "move_threshold_px": 30.0,
            "conf": 0.3,
            "enable_pose": False,
        }
