import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from boxmot.trackers.bytetrack.bytetrack import ByteTrack
from ultralytics import YOLO

from src.tracking.bytetrack_utils import (
    DetectionResults,
    draw_tracking_info,
    get_system_info,
    initialize_bytetrack,
    initialize_perf_log,
    load_yolo_model,
    process_frame_for_tracking,
    resize_frame,
    update_stay_times,
)

# テスト用のダミー入力ファイル名とモデルパス
DUMMY_INPUT_FILE = "test_video.mp4"
DUMMY_MODEL_PATH = "test_model.pt"


def test_initialize_perf_log_enabled(tmp_path):
    """Test case when performance log is enabled"""

    perf_log_file = initialize_perf_log(
        enable_perf_log=True,
        input_file=DUMMY_INPUT_FILE,
        model_path=DUMMY_MODEL_PATH,
        log_type="test_log",
    )

    assert perf_log_file is not None

    log_file_path = Path(perf_log_file)
    assert log_file_path.exists()
    assert log_file_path.name.startswith(f"log_{Path(DUMMY_INPUT_FILE).stem}_test_log_{Path(DUMMY_MODEL_PATH).stem}_")
    assert log_file_path.parent.name == "logs"
    assert log_file_path.parent.parent.name == "output"

    # ✅ Now read after flush/close
    with open(perf_log_file) as f:
        lines = f.readlines()
        assert lines  # ensure file is not empty
        assert "# System Information" in lines[0]
        assert "Frame" in lines[-1]

    # 🧹 Cleanup
    if Path("output/logs").exists() and Path(perf_log_file).is_relative_to(Path("output/logs")):
        os.remove(perf_log_file)
        if not os.listdir("output/logs"):
            os.rmdir("output/logs")
        if not os.listdir("output"):
            os.rmdir("output")


def test_initialize_perf_log_disabled(tmp_path):
    """パフォーマンスログが無効な場合のテスト"""
    # tmp_path はこのテストでは直接使われないが、pytestの慣習として引数に含める
    perf_log_file = initialize_perf_log(
        enable_perf_log=False,
        input_file=DUMMY_INPUT_FILE,
        model_path=DUMMY_MODEL_PATH,
        log_type="test_log",
    )
    assert perf_log_file is None


def test_initialize_perf_log_file_creation_error(mocker):
    # Patch get_system_info to avoid psutil usage (prevent collateral errors)
    mocker.patch(
        "src.tracking.bytetrack_utils.get_system_info",
        return_value={
            "os": "Linux",
            "os_version": "5.10",
            "python_version": "3.10.0",
            "cpu": "FakeCPU",
            "cpu_cores": 4,
            "cpu_threads": 8,
            "ram_total": 16,
        },
    )

    # mocker.patch を使用して builtins.open をモック化
    mocker.patch("builtins.open", side_effect=OSError("Test error: Cannot open file"))

    result = initialize_perf_log(
        enable_perf_log=True,
        input_file="test_video.mp4",
        model_path="test_model.pt",
        log_type="test_log",
    )

    assert result is None


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
    stay_info, _, _ = update_stay_times(sample_tracks, stay_info, current_time, move_threshold_px, stay_threshold_sec)

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
    stay_info, _, _ = update_stay_times(sample_tracks, stay_info, current_time, move_threshold_px, stay_threshold_sec)

    # Frame 2 (閾値を超える時間後、同じ位置)
    # time.sleep(stay_threshold_sec + 0.1) # CI環境などで不安定になるため time.sleep は避ける
    current_time_2 = current_time + stay_threshold_sec + 0.1

    updated_stay_info, notifications, _ = update_stay_times(
        sample_tracks, stay_info, current_time_2, move_threshold_px, stay_threshold_sec
    )

    assert updated_stay_info[1]["stay_duration"] >= stay_threshold_sec
    assert len(notifications) > 0  # sample_tracks に複数のトラックIDがあるので、それぞれ通知される可能性がある
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
    stay_info, _, _ = update_stay_times(sample_tracks, stay_info, current_time, move_threshold_px, stay_threshold_sec)
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
    stay_info, _, _ = update_stay_times(sample_tracks, stay_info, current_time, move_threshold_px, stay_threshold_sec)
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


def test_update_stay_times_empty_relevant_history(sample_tracks):
    """update_stay_timesでrelevant_historyが空になるエッジケースのテスト（ブランチカバレッジ）"""
    stay_info = {}
    move_threshold_px = 20.0
    stay_threshold_sec = 5.0

    # Frame 1 at t=0
    time_1 = time.time()
    stay_info, _, _ = update_stay_times(sample_tracks, stay_info, time_1, move_threshold_px, stay_threshold_sec)

    # Frame 2 at t=2.0 (history_seconds=1.0より未来)
    time_2 = time_1 + 2.0
    updated_stay_info, _, _ = update_stay_times(sample_tracks, stay_info, time_2, move_threshold_px, stay_threshold_sec)

    # このテストは主に、relevant_historyが空になった場合にエラーなく処理が継続されることを保証します
    assert 1 in updated_stay_info
    assert updated_stay_info[1]["stay_duration"] > 0


def test_load_yolo_model_success(mocker):
    """YOLOモデルのロード成功時のテスト"""
    mock_yolo = mocker.patch("src.tracking.bytetrack_utils.YOLO", autospec=True)
    mock_model_instance = mock_yolo.return_value
    mocker.patch.object(mock_model_instance, "to")

    model = load_yolo_model("dummy_model.pt", device="cpu")

    mock_yolo.assert_called_once_with("dummy_model.pt")
    mock_model_instance.to.assert_called_once_with("cpu")
    assert model is mock_model_instance


def test_load_yolo_model_failure(mocker):
    """YOLOモデルのロード失敗時のテスト"""
    mocker.patch("src.tracking.bytetrack_utils.YOLO", side_effect=Exception("Load error"))

    model = load_yolo_model("invalid_model.pt")

    assert model is None


def test_initialize_bytetrack(mocker):
    """ByteTrackトラッカーの初期化テスト"""
    mock_bytetrack = mocker.patch("src.tracking.bytetrack_utils.ByteTrack", autospec=True)

    tracker = initialize_bytetrack(frame_rate=60)

    mock_bytetrack.assert_called_once_with(track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=60)
    assert tracker is mock_bytetrack.return_value


def test_resize_frame():
    """リサイズ関数のテスト"""
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    resized = resize_frame(dummy_frame, 1280, 720)
    assert resized.shape == (720, 1280, 3)


@pytest.fixture
def mock_model(mocker):
    """process_frame_for_trackingで使うYOLOモデルのモック"""
    mock_yolo_instance = mocker.Mock(spec=YOLO)
    mock_result = mocker.Mock()

    # boxesのモック設定
    mock_boxes = mocker.MagicMock()  # MagicMockを使用して__len__を扱えるようにする
    mock_boxes.xyxy.cpu.return_value.numpy.return_value = [np.array([10, 20, 60, 80])]
    mock_boxes.conf.cpu.return_value.numpy.return_value = [0.9]
    mock_boxes.cls.cpu.return_value.numpy.return_value = [0]
    # MagicMockがリストのように振る舞うように設定
    mock_boxes.__len__.return_value = 1

    # keypointsのモック設定
    mock_keypoints = mocker.Mock()

    mock_result.boxes = mock_boxes
    mock_result.keypoints = mock_keypoints
    mock_yolo_instance.predict.return_value = [mock_result]
    return mock_yolo_instance


@pytest.fixture
def mock_tracker(mocker):
    """process_frame_for_trackingで使うByteTrackのモック"""
    mock_tracker_instance = mocker.Mock(spec=ByteTrack)
    # [x1, y1, x2, y2, track_id, conf, cls]
    tracked_data = np.array([[10, 20, 60, 80, 1, 0.9, 0]])
    mock_tracker_instance.update.return_value = tracked_data
    return mock_tracker_instance


def test_process_frame_for_tracking_pose_disabled(mock_model, mock_tracker):
    """フレーム処理と追跡のテスト（姿勢推定なし）"""
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    tracks, det_time, track_time, num_det, num_track, keypoints = process_frame_for_tracking(
        dummy_frame, mock_model, mock_tracker, conf=0.5, enable_pose=False
    )

    mock_model.predict.assert_called_once()
    mock_tracker.update.assert_called_once()

    assert len(tracks) == 1
    assert tracks[0, 4] == 1  # track_id
    assert det_time >= 0
    assert track_time >= 0
    assert num_det == 1
    assert num_track == 1
    assert keypoints is None  # enable_pose=False の場合は None


def test_process_frame_for_tracking_pose_enabled(mock_model, mock_tracker):
    """フレーム処理と追跡のテスト（姿勢推定あり）"""
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    tracks, det_time, track_time, num_det, num_track, keypoints = process_frame_for_tracking(
        dummy_frame, mock_model, mock_tracker, conf=0.5, enable_pose=True
    )
    assert keypoints is not None  # enable_pose=True の場合はオブジェクトが返る


def test_draw_tracking_info(mocker):
    """描画関数のテスト"""
    mock_rectangle = mocker.patch("cv2.rectangle")
    mock_put_text = mocker.patch("cv2.putText")
    mock_line = mocker.patch("cv2.line")
    mock_circle = mocker.patch("cv2.circle")

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame_copy_for_assert = frame.copy()
    tracks = np.array([[10, 20, 60, 80, 1, 0.9, 0]])
    stay_info = {1: {"stay_duration": 5.5, "person_height": 60}}

    # キーポイントのモック
    mock_kpts = mocker.Mock()
    kpts_data = np.array([[[5, 5], [15, 15], [25, 25]] + [[0, 0]] * 14])  # 3点だけ有効
    mock_kpts.xy.cpu.return_value.numpy.return_value = kpts_data

    # ケース1: 基本的な描画
    draw_tracking_info(frame.copy(), tracks)
    # np.arrayの比較はnp.array_equalで行う
    assert np.array_equal(mock_rectangle.call_args[0][0], frame_copy_for_assert)
    assert mock_rectangle.call_args[0][1:] == ((10, 20), (60, 80), (0, 255, 0), 2)
    # putTextも同様にnp.arrayを比較しないようにする
    assert np.array_equal(mock_put_text.call_args[0][0], frame_copy_for_assert)
    assert mock_put_text.call_args[0][1:] == ("ID: 1", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ケース2: 滞在時間と姿勢推定を有効
    draw_tracking_info(
        frame.copy(), tracks, keypoints=mock_kpts, enable_pose=True, show_duration=True, stay_info=stay_info
    )

    # 最後の呼び出し(latest call)を検証
    latest_call = mock_put_text.call_args
    assert np.array_equal(latest_call[0][0], frame_copy_for_assert)
    assert latest_call[0][1:] == ("ID:1 滞在:5.5s 高さ:60px", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    assert mock_line.call_count > 0
    assert mock_circle.call_count > 0


def test_detection_results_basic():
    """DetectionResultsクラスの基本的なテスト"""
    results = DetectionResults()
    results.add_frame_result(
        detection_time=10,
        tracking_time=5,
        fps=30,
        objects_detected=2,
        objects_tracked=2,
        memory_usage=1024,
        detection_conf=0.9,
    )
    results.add_frame_result(
        detection_time=12,
        tracking_time=7,
        fps=28,
        objects_detected=3,
        objects_tracked=2,
        memory_usage=1030,
        detection_conf=0.8,
    )

    summary = results.get_summary()

    assert summary["frame_count"] == 2
    assert summary["avg_detection_time"] == 11
    assert summary["avg_tracking_time"] == 6
    assert summary["avg_fps"] == 29
    assert summary["avg_objects_detected"] == 2.5
    assert summary["avg_objects_tracked"] == 2
    assert summary["avg_memory_usage"] == 1027
    assert summary["avg_detection_conf"] == pytest.approx(0.85)
    assert summary["max_objects_detected"] == 3
    assert summary["max_objects_tracked"] == 2


def test_detection_results_empty():
    """DetectionResultsが空の場合のテスト（エッジケース）"""
    results = DetectionResults()
    summary = results.get_summary()

    assert summary["frame_count"] == 0
    assert summary["avg_detection_time"] == 0
    assert summary["max_objects_tracked"] == 0


def test_load_yolo_model_no_device(mocker):
    """YOLOモデルのロード（デバイス指定なし）のテスト（ブランチカバレッジ）"""
    mock_yolo = mocker.patch("src.tracking.bytetrack_utils.YOLO", autospec=True)
    mock_model_instance = mock_yolo.return_value
    mock_to = mocker.patch.object(mock_model_instance, "to")

    model = load_yolo_model("dummy_model.pt", device="")

    mock_yolo.assert_called_once_with("dummy_model.pt")
    mock_to.assert_not_called()
    assert model is mock_model_instance


def test_draw_tracking_info_more_tracks_than_kpts(mocker):
    """キーポイントより追跡IDが多い場合の描画テスト（ブランチカバレッジ）"""
    mock_rectangle = mocker.patch("cv2.rectangle")
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    # 2 tracks
    tracks = np.array([[10, 20, 60, 80, 1, 0.9, 0], [100, 120, 160, 180, 2, 0.9, 0]])

    # Only 1 keypoint data
    mock_kpts = mocker.Mock()
    kpts_data = np.array([[[5, 5], [15, 15], [25, 25]] + [[0, 0]] * 14])  # 1 person
    mock_kpts.xy.cpu.return_value.numpy.return_value = kpts_data

    # This should run without error and cover the `if i < num_kpts:` branch
    # for both True (i=0) and False (i=1)
    draw_tracking_info(frame, tracks, keypoints=mock_kpts, enable_pose=True)

    # Assert that rectangle is drawn for both tracks
    assert mock_rectangle.call_count == 2


def test_initialize_perf_log_long_stay_type(tmp_path):
    """initialize_perf_logでlog_type='long_stay'を指定した場合のテスト（ブランチカバレッジ）"""
    perf_log_file = initialize_perf_log(
        enable_perf_log=True,
        input_file=DUMMY_INPUT_FILE,
        model_path=DUMMY_MODEL_PATH,
        log_type="long_stay",
    )
    assert perf_log_file is not None
    log_file_path = Path(perf_log_file)
    assert log_file_path.exists()

    with open(log_file_path) as f:
        header = f.readlines()[-1]
        assert "Stay_Check_Time_ms" in header

    # Cleanup
    os.remove(perf_log_file)
    if not os.listdir(log_file_path.parent):
        os.rmdir(log_file_path.parent)
    if not os.listdir(log_file_path.parent.parent):
        os.rmdir(log_file_path.parent.parent)
