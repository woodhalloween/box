import os
import time
from pathlib import Path

import numpy as np
import pytest

from src.tracking.bytetrack_utils import get_system_info, initialize_perf_log, update_stay_times

# テスト用のダミー入力ファイル名とモデルパス
DUMMY_INPUT_FILE = "test_video.mp4"
DUMMY_MODEL_PATH = "test_model.pt"


def test_initialize_perf_log_enabled(tmp_path):
    """パフォーマンスログが有効な場合のテスト"""
    log_dir = tmp_path / "output/logs"  # tmp_path を使うことでテスト固有のディレクトリになる
    # initialize_perf_log内で output/logs を作成するため、ここでは作成不要

    perf_log_file, perf_log_f, perf_log_writer = initialize_perf_log(
        enable_perf_log=True,
        input_file=DUMMY_INPUT_FILE,
        model_path=DUMMY_MODEL_PATH,
        log_type="test_log",
    )

    assert perf_log_file is not None
    assert perf_log_f is not None
    assert perf_log_writer is not None

    # ファイルパスの検証 (tmp_path 以下に作成されることを期待)
    # initialize_perf_log は Path("output/logs") を基準にするため、
    # テスト実行時のカレントディレクトリに output/logs が掘られ、その中にログファイルができる。
    # tmp_path を直接 initialize_perf_log に渡す改修をするか、
    # ここではカレントディレクトリ基準でチェックする。
    # 今回はカレントディレクトリ基準でチェックします。

    # 一時的に出力先を固定するため、関数を少し変更するか、
    # もしくは出力されるファイル名を予測してチェックします。
    # ここでは、関数が返すファイルパスが期待通りかを確認します。

    # initialize_perf_log 内で Path("output/logs") を使うため、
    # pytest実行時のカレントディレクトリに output/logs ができます。
    # そのため、テスト後はこのディレクトリをクリーンアップする必要があります。
    # より良いのは、initialize_perf_log が出力先ディレクトリを引数で受け取るようにすることです。
    # 今回は現状のままでテストします。

    log_file_path = Path(perf_log_file)
    assert log_file_path.exists()
    assert log_file_path.name.startswith(
        f"log_{Path(DUMMY_INPUT_FILE).stem}_test_log_{Path(DUMMY_MODEL_PATH).stem}_"
    )
    assert log_file_path.parent.name == "logs"
    assert log_file_path.parent.parent.name == "output"

    # ヘッダーの検証 (一部)
    with open(perf_log_file) as f:
        lines = f.readlines()
        assert "# System Information" in lines[0]
        assert "Frame" in lines[-1]  # 最後の行がヘッダー行

    if perf_log_f:
        perf_log_f.close()

    # クリーンアップ (pytest の tmp_path を使っていれば不要だが、今回は手動)
    # ただし、tmp_pathを使っても initialize_perf_log がカレントディレクトリに作るため意味がない
    # このテスト後処理は改善の余地あり
    if Path("output/logs").exists() and Path(perf_log_file).is_relative_to(Path("output/logs")):
        os.remove(perf_log_file)
        # 他のファイルがなければディレクトリも削除
        if not os.listdir(Path("output/logs")):
            os.rmdir(Path("output/logs"))
        if not os.listdir(Path("output")) and Path("output").exists():
            os.rmdir(Path("output"))


def test_initialize_perf_log_disabled(tmp_path):
    """パフォーマンスログが無効な場合のテスト"""
    # tmp_path はこのテストでは直接使われないが、pytestの慣習として引数に含める
    perf_log_file, perf_log_f, perf_log_writer = initialize_perf_log(
        enable_perf_log=False,
        input_file=DUMMY_INPUT_FILE,
        model_path=DUMMY_MODEL_PATH,
        log_type="test_log",
    )
    assert perf_log_file is None
    assert perf_log_f is None
    assert perf_log_writer is None


def test_initialize_perf_log_file_creation_error(tmp_path, monkeypatch):
    """パフォーマンスログファイル作成エラーのテスト"""

    # 強制的にOSErrorを発生させるために、openをモックする
    def mock_open(*args, **kwargs):
        raise OSError("Test error: Cannot open file")

    monkeypatch.setattr("builtins.open", mock_open)

    # outputディレクトリが存在しない場合でも正しくNoneが返ることを確認
    # (initialize_perf_log内でmkdirするため、このテストケースは実際には通りにくいが、
    #  open自体が失敗するシナリオとして残す)
    if Path("output/logs").exists():
        import shutil

        shutil.rmtree("output")  # 強制的に削除

    perf_log_file, perf_log_f, perf_log_writer = initialize_perf_log(
        enable_perf_log=True,
        input_file=DUMMY_INPUT_FILE,
        model_path=DUMMY_MODEL_PATH,
        log_type="test_log",
    )

    assert perf_log_file is None
    assert perf_log_f is None
    assert perf_log_writer is None

    # monkeypatchが元に戻るので、他のテストに影響はない
    # output ディレクトリが作られていないことを確認
    assert not Path("output").exists()


# `get_system_info` は外部ライブラリに依存しているため、簡単な呼び出しテストのみ
def test_get_system_info():
    """get_system_info関数の基本的な動作テスト"""
    info = get_system_info()
    assert "os" in info
    assert "python_version" in info
    assert "cpu_cores" in info
    assert "ram_total" in info


# update_stay_times のテスト
@pytest.fixture
def initial_stay_info():
    return {}


@pytest.fixture
def sample_tracks():
    # [x1, y1, x2, y2, track_id, conf, cls_id]
    # track_id 1: center (50, 50), height 100
    # track_id 2: center (150, 150), height 80
    return np.array(
        [
            [0, 0, 100, 100, 1, 0.9, 0],
            [100, 100, 200, 180, 2, 0.8, 0],
        ]
    )


def test_update_stay_times_no_stay(initial_stay_info, sample_tracks):
    """滞在なし（初回フレーム）のテスト"""
    stay_info = initial_stay_info
    current_time = time.time()
    move_threshold_px = 20.0
    stay_threshold_sec = 5.0

    updated_stay_info, notifications, stay_check_time_ms = update_stay_times(
        sample_tracks, stay_info, current_time, move_threshold_px, stay_threshold_sec
    )

    assert 1 in updated_stay_info
    assert 2 in updated_stay_info
    assert updated_stay_info[1]["stay_duration"] == 0
    assert not notifications
    assert stay_check_time_ms >= 0


def test_update_stay_times_short_stay(sample_tracks):
    """短時間滞在のテスト"""
    stay_info = {}
    current_time = time.time()
    move_threshold_px = 20.0
    stay_threshold_sec = 5.0

    # Frame 1
    stay_info, _, _ = update_stay_times(
        sample_tracks, stay_info, current_time, move_threshold_px, stay_threshold_sec
    )

    # Frame 2 (1秒後、同じ位置)
    time.sleep(0.1)  # 実際の時間経過を模倣（ただしテスト時間を短縮するため0.1秒）
    current_time_2 = current_time + 1.0
    updated_stay_info, notifications, _ = update_stay_times(
        sample_tracks, stay_info, current_time_2, move_threshold_px, stay_threshold_sec
    )

    assert updated_stay_info[1]["stay_duration"] > 0  # わずかに増加
    assert updated_stay_info[1]["stay_duration"] < stay_threshold_sec
    assert not notifications


def test_update_stay_times_long_stay_and_notification(sample_tracks):
    """長時間滞在と通知のテスト"""
    stay_info = {}
    current_time = time.time()
    move_threshold_px = 1.0  # ほぼ動かない設定
    stay_threshold_sec = 0.1  # 短い閾値でテスト

    # Frame 1
    stay_info, _, _ = update_stay_times(
        sample_tracks, stay_info, current_time, move_threshold_px, stay_threshold_sec
    )

    # Frame 2 (閾値を超える時間後、同じ位置)
    # time.sleep(stay_threshold_sec + 0.1) # CI環境などで不安定になるため time.sleep は避ける
    current_time_2 = current_time + stay_threshold_sec + 0.1

    updated_stay_info, notifications, _ = update_stay_times(
        sample_tracks, stay_info, current_time_2, move_threshold_px, stay_threshold_sec
    )

    assert updated_stay_info[1]["stay_duration"] >= stay_threshold_sec
    assert (
        len(notifications) > 0
    )  # sample_tracks に複数のトラックIDがあるので、それぞれ通知される可能性がある
    assert notifications[0]["id"] == 1 or notifications[0]["id"] == 2
    assert updated_stay_info[1]["notified"]

    # Frame 3 (さらに時間経過、通知済みなので新たな通知はなし)
    current_time_3 = current_time_2 + 0.1
    updated_stay_info_3, notifications_3, _ = update_stay_times(
        sample_tracks, stay_info, current_time_3, move_threshold_px, stay_threshold_sec
    )
    assert len(notifications_3) == 0  # 既に通知されているので新たな通知はない


def test_update_stay_times_move_resets_stay(sample_tracks):
    """移動による滞在時間リセットのテスト"""
    stay_info = {}
    current_time = time.time()
    move_threshold_px = 10.0
    stay_threshold_sec = 0.1

    # Frame 1 (滞在開始)
    stay_info, _, _ = update_stay_times(
        sample_tracks, stay_info, current_time, move_threshold_px, stay_threshold_sec
    )
    # time.sleep(stay_threshold_sec + 0.1)
    current_time_2 = current_time + stay_threshold_sec + 0.1
    stay_info, notifications, _ = update_stay_times(
        sample_tracks, stay_info, current_time_2, move_threshold_px, stay_threshold_sec
    )
    assert stay_info[1]["stay_duration"] >= stay_threshold_sec
    assert len(notifications) > 0

    # Frame 2 (ID 1が大きく移動)
    moved_tracks = sample_tracks.copy()
    moved_tracks[0, 0] += 50  # x1 を大きく変更して移動を模倣
    moved_tracks[0, 2] += 50  # x2 も同様に

    # time.sleep(0.1)
    current_time_3 = current_time_2 + 0.1
    updated_stay_info, notifications_2, _ = update_stay_times(
        moved_tracks, stay_info, current_time_3, move_threshold_px, stay_threshold_sec
    )

    assert updated_stay_info[1]["stay_duration"] == 0  # 移動したのでリセット
    assert not updated_stay_info[1]["notified"]
    assert len(notifications_2) == 0  # リセットされたので通知なし


def test_update_stay_times_track_lost(sample_tracks):
    """追跡が途切れたIDの削除テスト"""
    stay_info = {}
    current_time = time.time()
    move_threshold_px = 20.0
    stay_threshold_sec = 5.0

    # Frame 1 (ID 1, 2 を登録)
    stay_info, _, _ = update_stay_times(
        sample_tracks, stay_info, current_time, move_threshold_px, stay_threshold_sec
    )
    assert 1 in stay_info
    assert 2 in stay_info

    # Frame 2 (ID 1 のみ存在)
    tracks_id1_only = np.array([sample_tracks[0]])
    time.sleep(0.1)  # わずかな時間経過
    current_time_2 = current_time + 0.1
    updated_stay_info, _, _ = update_stay_times(
        tracks_id1_only, stay_info, current_time_2, move_threshold_px, stay_threshold_sec
    )

    assert 1 in updated_stay_info
    assert 2 not in updated_stay_info  # ID 2 はロストしたため削除
