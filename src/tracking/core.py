import time
import numpy as np
from boxmot.trackers.bytetrack.bytetrack import ByteTrack
from ultralytics import YOLO

def process_frame_for_tracking(frame_rgb, model, tracker):
    """1フレームを処理して検出と追跡を行う"""
    detection_start_time = time.time()
    # 人物クラス(0)のみを検出
    results = model.predict(frame_rgb, verbose=False, classes=[0])
    detection_time_ms = (time.time() - detection_start_time) * 1000
    result = results[0]

    boxes = result.boxes
    # 検出結果をboxmot用の形式に変換
    dets_for_tracker = []
    avg_conf = 0.0

    if len(boxes) > 0:
        total_conf = 0.0
        for i in range(len(boxes)):
            box = boxes[i].xyxy.cpu().numpy()[0]  # [x1, y1, x2, y2]
            conf = float(boxes[i].conf.cpu().numpy()[0])
            cls = int(boxes[i].cls.cpu().numpy()[0])
            total_conf += conf

            # [x1, y1, x2, y2, conf, class]の形式
            dets_for_tracker.append([box[0], box[1], box[2], box[3], conf, cls])

        dets_for_tracker = np.array(dets_for_tracker)
        avg_conf = total_conf / len(boxes) if len(boxes) > 0 else 0
    else:
        dets_for_tracker = np.empty((0, 6))

    tracking_start_time = time.time()
    # ByteTrackのupdateメソッドにnumpy配列を渡す
    tracks = tracker.update(dets_for_tracker, frame_rgb)
    tracking_time_ms = (time.time() - tracking_start_time) * 1000

    return tracks, detection_time_ms, tracking_time_ms, len(boxes), len(tracks), avg_conf


def initialize_bytetrack(frame_rate=30):
    """ByteTrackトラッカーを初期化

    Args:
        frame_rate (int, optional): トラッカーのフレームレート. Defaults to 30.
    """
    # track_thresh: 追跡を開始するための信頼度の閾値。デフォルトは0.5。
    # track_buffer: 追跡が途切れた後、IDを保持するフレーム数。デフォルトは30。
    # match_thresh: 追跡と検出を関連付けるためのIoUの閾値。デフォルトは0.8。
    # frame_rate: ビデオのフレームレート。
    return ByteTrack(track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=frame_rate)


def load_yolo_model(model_path, device=""):
    """YOLOモデルをロード"""
    try:
        model = YOLO(model_path)
        if device:  # 'cpu' or 'mps' or '0' (for CUDA GPU 0)
            model.to(device)
        print(f"YOLOモデル '{model_path}' を正常にロードしました。")
        return model
    except Exception as e:
        print(f"エラー: YOLOモデル '{model_path}' のロードに失敗しました: {e}")
        return None
