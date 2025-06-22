import time

import numpy as np
import pytest

from src.tracking.bytetrack_utils import update_stay_times


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


def test_update_stay_times_fallback_to_last_pos():
    """Should fallback to last_pos when relevant_history is empty (older than 1 sec)."""
    import time

    from src.tracking.bytetrack_utils import update_stay_times

    current_time = time.time()
    old_time = current_time - 2.0  # More than 1 second old

    # Simulate a single tracked object with stale history
    stay_info = {
        42: {
            "start_time": old_time,
            "last_pos": (100.0, 100.0),
            "stay_duration": 0.0,
            "notified": False,
            "person_height": 180,
            "history": [(old_time, (90.0, 90.0), 180)],
        }
    }

    # Simulate track with same ID and current position
    tracks = np.array([[90, 90, 110, 110, 42, 0.9, 0]])
    move_threshold_px = 50.0
    stay_threshold_sec = 1.0

    updated_stay_info, notifications, _ = update_stay_times(
        tracks, stay_info, current_time, move_threshold_px, stay_threshold_sec
    )

    assert 42 in updated_stay_info
    assert updated_stay_info[42]["stay_duration"] >= 0
    assert updated_stay_info[42]["history"][-2][1] == (90.0, 90.0)  # From original stale entry
    assert updated_stay_info[42]["history"][-1][1] != (90.0, 90.0)  # New position was added
