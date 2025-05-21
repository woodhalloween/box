import cv2

def resize_frame(frame, target_width, target_height):
    """フレームを指定サイズにリサイズ"""
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def draw_tracking_info(frame, tracks, show_duration=False, stay_info=None):
    """トラッキング情報を描画

    Args:
        frame: フレーム画像（BGRフォーマット）
        tracks: 追跡情報 [x1, y1, x2, y2, track_id, ...]
        show_duration: 滞在時間を表示するかどうか (default: False)
        stay_info: 滞在情報の辞書 (show_durationがTrueの場合必要)
    """
    for track in tracks:
        # トラックの形式: [x1, y1, x2, y2, track_id, conf, cls_id, ...]
        x1, y1, x2, y2, track_id = track[:5]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        track_id = int(track_id)

        # バウンディングボックスの描画
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # IDの表示
        if show_duration and stay_info and track_id in stay_info:
            stay_duration = stay_info[track_id]["stay_duration"]
            person_height = stay_info[track_id]["person_height"]
            label = f"ID:{track_id} 滞在:{stay_duration:.1f}s 高さ:{person_height}px"

            # 長時間滞在の場合は色を変える
            if stay_duration >= 4.0:  # デフォルトのしきい値
                color = (0, 0, 255)  # 赤色
            else:
                color = (0, 255, 0)  # 緑色

            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame
