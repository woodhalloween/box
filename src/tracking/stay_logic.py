import time
from datetime import datetime
import numpy as np

def update_stay_times(tracks, stay_info, current_time, move_threshold_px, stay_threshold_sec):
    """Update stay duration for each tracked object and check for long stays."""
    stay_check_start = time.time()
    current_track_ids = set()
    notifications = []

    for track in tracks:
        # トラックの形式: [x1, y1, x2, y2, track_id, conf, cls_id, ...]
        x1, y1, x2, y2, track_id = track[:5]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        track_id = int(track_id)

        current_track_ids.add(track_id)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        current_pos = (center_x, center_y)

        # 人物の高さを計算（遠近感の基準として使用）
        person_height = y2 - y1

        # 基準高さの設定（画面の半分の高さを基準と仮定）
        reference_height = 200  # この値は環境によって調整可能

        # 正規化係数の計算（小さい人物ほど係数が大きくなる）
        normalization_factor = reference_height / max(person_height, 1)  # ゼロ除算防止

        if track_id not in stay_info:
            stay_info[track_id] = {
                "start_time": current_time,
                "last_pos": current_pos,
                "stay_duration": 0,
                "notified": False,
                "person_height": person_height,
                "history": [(current_time, current_pos, person_height)],
            }
        else:
            last_time, last_pos, _ = stay_info[track_id]["history"][-1]
            time_diff = current_time - last_time

            # 移動距離の計算 (正規化された距離)
            # dist_moved = np.sqrt(
            #     (current_pos[0] - last_pos[0]) ** 2 + (current_pos[1] - last_pos[1]) ** 2
            # )
            # dist_moved_normalized = dist_moved * normalization_factor

            # 過去1秒間の平均位置と比較して移動判定（より安定した判定のため）
            history_seconds = 1.0  # 過去何秒間の履歴を参照するか
            relevant_history = [
                p
                for t, p, _ in stay_info[track_id]["history"]
                if current_time - t <= history_seconds
            ]
            if not relevant_history:
                relevant_history = [last_pos]  # 履歴が足りない場合は最後の位置を使用

            avg_prev_x = np.mean([p[0] for p in relevant_history])
            avg_prev_y = np.mean([p[1] for p in relevant_history])
            avg_prev_pos = (avg_prev_x, avg_prev_y)

            dist_moved_from_avg = np.sqrt(
                (current_pos[0] - avg_prev_pos[0]) ** 2 + (current_pos[1] - avg_prev_pos[1]) ** 2
            )
            dist_moved_normalized = dist_moved_from_avg * normalization_factor

            # 閾値も正規化
            # dynamic_move_threshold = move_threshold_px / normalization_factor
            # print(f"ID:{track_id}, dist_norm:{dist_moved_normalized:.2f}, thresh_norm:{dynamic_move_threshold:.2f}, factor:{normalization_factor:.2f}, height:{person_height}")

            if dist_moved_normalized < move_threshold_px:  # 正規化後の閾値と比較
                stay_info[track_id]["stay_duration"] += time_diff
            else:
                # 移動があった場合は滞在時間をリセットし、開始時刻と位置を更新
                stay_info[track_id]["start_time"] = current_time
                stay_info[track_id]["stay_duration"] = 0
                stay_info[track_id]["notified"] = False  # 通知状態もリセット

            stay_info[track_id]["last_pos"] = current_pos
            stay_info[track_id]["person_height"] = person_height  # 高さを更新
            stay_info[track_id]["history"].append((current_time, current_pos, person_height))
            # 古い履歴を削除（例: 過去5秒分のみ保持）
            max_history_duration = 5.0
            stay_info[track_id]["history"] = [
                h
                for h in stay_info[track_id]["history"]
                if current_time - h[0] <= max_history_duration
            ]

            # 長時間滞在の通知チェック
            if (
                stay_info[track_id]["stay_duration"] >= stay_threshold_sec
                and not stay_info[track_id]["notified"]
            ):
                notifications.append(
                    {
                        "id": track_id,
                        "duration": stay_info[track_id]["stay_duration"],
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                stay_info[track_id]["notified"] = True

    # 追跡が途切れたIDをstay_infoから削除
    # （必要に応じて、すぐ消さずに一定時間後に消すなどの工夫も可能）
    lost_ids = set(stay_info.keys()) - current_track_ids
    for lost_id in lost_ids:
        del stay_info[lost_id]

    stay_check_time_ms = (time.time() - stay_check_start) * 1000
    return stay_info, notifications, stay_check_time_ms
