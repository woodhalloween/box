import cv2
import numpy as np
import time
import os
import pandas as pd
from collections import deque
from ultralytics import YOLO

class HeadDetector:
    """
    YOLOを使用した頭部検出クラス
    人物の頭部を検出し、首振り（左右の動き）を検出する
    """
    def __init__(self, 
                model_path="yolo11n.pt", 
                confidence=0.35, 
                movement_threshold=0.08, 
                consecutive_frames=3, 
                detection_cooldown=2.0,
                history_size=40,
                min_event_duration=0.5):
        """
        頭部検出器の初期化
        Args:
            model_path: YOLOモデルのパス
            confidence: 検出信頼度閾値
            movement_threshold: 頭部移動と判定する閾値（画像幅に対する比率）
            consecutive_frames: 首振りと判定する連続フレーム数
            detection_cooldown: 検出後のクールダウン時間（秒）
            history_size: 履歴に保存するフレーム数
            min_event_duration: 首振りイベントの最小持続時間（秒）
        """
        # YOLOモデルの読み込み
        self.model = YOLO(model_path)
        self.confidence = confidence
        
        # 検出パラメータ
        self.movement_threshold = movement_threshold
        self.consecutive_frames = consecutive_frames
        self.detection_cooldown = detection_cooldown
        self.min_event_duration = min_event_duration
        
        # 首振り状態管理用
        self.head_positions = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)
        self.turning_start_time = None  # 首振り開始時間
        self.is_turning = False   # 現在首振り中か
        self.last_detection_time = 0  # 最後に首振りを検出した時間
        self.current_movement_count = 0  # 現在の動きカウント
        
        # ログデータ
        self.log_data = []
        
        # 性能評価用の変数
        self.true_positives = 0
        self.false_positives = 0
        self.total_events = 0
        
    def detect_head(self, frame, frame_idx=0, display=True):
        """
        フレームから頭部を検出し、結果を返す
        Args:
            frame: 検出を行うビデオフレーム
            frame_idx: フレームインデックス（ログ用）
            display: 表示用のデータを生成するかどうか
        Returns:
            tuple: (head_detected, head_data, display_frame, head_turning)
        """
        # YOLOで人物検出
        start_time = time.time()
        result = self.model(frame, conf=self.confidence, verbose=False)[0]
        
        # 検出結果処理
        heads = []
        for box in result.boxes:
            if box.cls.cpu().numpy()[0] == 0:  # クラス0は'person'
                if box.conf.cpu().numpy()[0] > self.confidence:
                    # バウンディングボックスの座標を取得
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                    conf = box.conf.cpu().numpy()[0]
                    
                    # 頭部は人物の上部1/6と仮定
                    head_h = (y2 - y1) / 6
                    head_x = (x1 + x2) / 2  # 頭部の中心X座標
                    head_y = y1 + head_h / 2  # 頭部の中心Y座標
                    head_w = (x2 - x1) * 0.8  # 頭部の幅は人物幅の80%と仮定
                    
                    heads.append((head_x, head_y, head_w, head_h, conf))
        
        # 見つかった頭部の中で最も信頼度の高いものを選択
        best_head = None
        best_conf = 0
        for head in heads:
            if head[4] > best_conf:
                best_head = head
                best_conf = head[4]
        
        processing_time = (time.time() - start_time) * 1000  # ミリ秒に変換
        
        # 頭部が検出された場合、履歴に追加
        if best_head:
            self.head_positions.append((best_head[0], best_head[1]))
            self.timestamps.append(time.time())
            
            # 首振りの検出
            turning_detected = self._detect_head_turning(frame.shape[1])
            
            # 首振り状態の更新
            head_turning, event_started, event_ended, event_duration = self._update_turning_state(turning_detected, frame_idx, time.time())
            
            # ログデータの記録
            self.log_data.append({
                'timestamp': time.time(),
                'head_x': best_head[0],
                'head_y': best_head[1],
                'head_turning_detected': head_turning,
                'processing_time': processing_time,
                'confidence': best_conf
            })
            
            return best_head, heads, head_turning, processing_time
        
        return None, heads, False, processing_time
    
    def _detect_head_turning(self, frame_width):
        """
        履歴からの頭部位置データを使用して首振りを検出
        Args:
            frame_width: フレームの幅（頭部移動の割合計算に使用）
        Returns:
            bool: 首振りが検出された場合はTrue
        """
        current_time = time.time()
        
        # クールダウン時間中は検出しない
        if current_time - self.last_detection_time < self.detection_cooldown:
            return False
        
        # 履歴が少ない場合は検出しない
        if len(self.head_positions) < self.consecutive_frames + 2:  # さらに履歴数を減らす
            return False
        
        # 頭部の左右（X軸）移動を計算
        x_movements = []
        for i in range(1, len(self.head_positions)):
            if i < len(self.timestamps) and i-1 < len(self.timestamps):
                dt = self.timestamps[i] - self.timestamps[i-1]
                if dt > 0:
                    # 単位時間あたりの移動量（画像幅に対する割合）
                    dx = (self.head_positions[i][0] - self.head_positions[i-1][0]) / frame_width
                    velocity = dx / dt  # 速度（画像幅に対する割合/秒）
                    x_movements.append(velocity)
        
        if len(x_movements) < self.consecutive_frames:
            return False
        
        # 首振り検出ロジック - 単純化版
        # 1. 十分な大きさの動きが連続して発生しているか
        recent_movements = x_movements[-self.consecutive_frames:]
        significant_movements = sum(1 for m in recent_movements if abs(m) > self.movement_threshold)
        
        # 2. 方向転換判定を単純化
        max_movement = max(abs(m) for m in recent_movements) if recent_movements else 0
        
        # 動きカウントを更新
        if significant_movements >= 1 or max_movement > self.movement_threshold * 1.5:
            self.current_movement_count += 1
        else:
            # 動きがなければカウントをリセット（ただし0以下にはしない）
            self.current_movement_count = max(0, self.current_movement_count - 0.5)
        
        # 条件: 連続した有意な動きがある場合に検出
        if self.current_movement_count >= self.consecutive_frames:
            self.last_detection_time = current_time
            self.current_movement_count = 0  # 検出後はカウントをリセット
            return True
        
        return False
    
    def _update_turning_state(self, turning_detected, frame_idx, timestamp):
        """
        首振り状態を更新し、イベントの開始・終了を管理
        Args:
            turning_detected: 現在のフレームで首振りが検出されたか
            frame_idx: 現在のフレームインデックス
            timestamp: 現在のタイムスタンプ
        Returns:
            tuple: (head_turning, event_started, event_ended, event_duration)
        """
        event_started = False
        event_ended = False
        event_duration = 0
        head_turning = self.is_turning
        
        # 首振り開始
        if turning_detected and not self.is_turning:
            self.is_turning = True
            head_turning = True
            self.turning_start_time = timestamp
            event_started = True
        
        # 首振り終了
        elif not turning_detected and self.is_turning:
            self.is_turning = False
            head_turning = False
            event_duration = timestamp - self.turning_start_time
            
            # 最小持続時間を超えた場合のみイベントとして報告
            if event_duration >= self.min_event_duration:
                event_ended = True
            else:
                # 短すぎるイベントは無視する
                event_started = False  # 開始通知も取り消す
            
            self.turning_start_time = None
        
        return head_turning, event_started, event_ended, event_duration
    
    def draw_head_detection(self, frame, head, all_heads=None, head_turning=False):
        """
        検出された頭部をフレームに描画
        Args:
            frame: 描画する画像
            head: メインの頭部 (x, y, w, h, conf)
            all_heads: 検出された全ての頭部 (オプション)
            head_turning: 首振り検出フラグ
        Returns:
            frame: 描画された画像
        """
        if head:
            x, y, w, h, conf = head
            
            # 頭部の矩形を描画
            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)
            
            # メインの頭部を緑色で描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Head: {conf:.2f}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 首振り検出時の表示
            if head_turning:
                cv2.putText(frame, "Head Turning Detected!", 
                           (frame.shape[1] // 2 - 180, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                # 赤い枠を描画
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 5)
            
            # 頭部の軌跡を描画
            if len(self.head_positions) > 1:
                pts = np.array([[int(p[0]), int(p[1])] for p in self.head_positions], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (255, 0, 0), 2)
        
        # 他の検出された頭部があれば青色で描画
        if all_heads:
            for h in all_heads:
                if h != head:  # メインの頭部以外
                    x, y, w, h, _ = h
                    x1 = int(x - w/2)
                    y1 = int(y - h/2)
                    x2 = int(x + w/2)
                    y2 = int(y + h/2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        return frame
    
    def save_logs(self, filename):
        """
        検出ログをCSVファイルに保存
        Args:
            filename: 保存するCSVファイルのパス
        """
        df = pd.DataFrame(self.log_data)
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
    
    def calculate_metrics(self):
        """
        検出精度メトリクスを計算
        Returns:
            dict: 精度、再現率、F1スコアを含む辞書
        """
        if self.total_events == 0:
            return {"precision": 0, "recall": 0, "f1_score": 0}
            
        if self.true_positives + self.false_positives == 0:
            precision = 0
        else:
            precision = self.true_positives / (self.true_positives + self.false_positives)
            
        recall = self.true_positives / self.total_events
        
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
            
        return {
            "precision": precision * 100,
            "recall": recall * 100,
            "f1_score": f1_score * 100
        }


def process_video(video_path, output_path=None, show_preview=True, log_dir="logs", detector=None):
    """
    ビデオを処理し、頭部を検出する
    Args:
        video_path: 入力ビデオのパス
        output_path: 出力ビデオのパス（指定しない場合は保存しない）
        show_preview: プレビューを表示するかどうか
        log_dir: ログファイルを保存するディレクトリ
        detector: カスタム検出器のインスタンス（指定しない場合はデフォルト設定で作成）
    Returns:
        list: 検出ログデータ
    """
    cap = cv2.VideoCapture(video_path)
    
    # 検出器の作成
    if detector is None:
        detector = HeadDetector()
    
    # ビデオのプロパティを取得
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 出力ビデオの設定
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    frame_count = 0
    total_processing_time = 0
    max_processing_time = 0
    
    # 検出された首振りの情報を保持
    turning_events = []
    current_turning_event = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # 頭部検出の実行
        head, all_heads, head_turning, processing_time = detector.detect_head(frame, frame_count - 1, show_preview)
        
        # 処理時間の記録
        total_processing_time += processing_time
        max_processing_time = max(max_processing_time, processing_time)
        
        # 検出結果の描画
        if head:
            frame = detector.draw_head_detection(frame, head, all_heads, head_turning)
            
            # 首振りイベントの記録
            if head_turning:
                if current_turning_event is None:
                    current_turning_event = {
                        'start_frame': frame_count,
                        'start_time': time.time()
                    }
            elif current_turning_event is not None:
                # 首振りイベントの終了
                current_turning_event['end_frame'] = frame_count - 1
                current_turning_event['end_time'] = time.time()
                current_turning_event['duration'] = current_turning_event['end_time'] - current_turning_event['start_time']
                turning_events.append(current_turning_event)
                current_turning_event = None
        
        # 平均処理時間と要件（500ms以内）の達成状況を表示
        avg_time = total_processing_time / frame_count
        cv2.putText(frame, f"Avg: {avg_time:.1f}ms (Target: 500ms)", (10, frame_height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if avg_time <= 500 else (0, 0, 255), 2)
        
        # 出力ビデオに書き込み
        if output_path:
            out.write(frame)
            
        # プレビューの表示
        if show_preview:
            cv2.imshow('Head Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # 最後に首振りイベントが終了していない場合は終了させる
    if current_turning_event is not None:
        current_turning_event['end_frame'] = frame_count
        current_turning_event['end_time'] = time.time()
        current_turning_event['duration'] = current_turning_event['end_time'] - current_turning_event['start_time']
        turning_events.append(current_turning_event)
    
    # リソースの解放
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    # 性能情報の表示
    print(f"処理したフレーム数: {frame_count}")
    print(f"平均処理時間: {total_processing_time / frame_count:.2f}ms")
    print(f"最大処理時間: {max_processing_time:.2f}ms")
    print(f"目標時間（500ms）内: {'達成' if total_processing_time / frame_count <= 500 else '未達成'}")
    
    # 首振りイベントの情報を表示
    print(f"\n検出された首振りイベント数: {len(turning_events)}")
    if turning_events:
        total_duration = sum(event['duration'] for event in turning_events)
        avg_duration = total_duration / len(turning_events)
        print(f"平均持続時間: {avg_duration:.2f}秒")
        for i, event in enumerate(turning_events):
            print(f"イベント{i+1}: フレーム {event['start_frame']} - {event['end_frame']} " 
                  f"(持続時間: {event['duration']:.2f}秒)")
    
    # ログを保存
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"head_detection_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    detector.save_logs(log_filename)
    
    # 首振りイベントのログも保存
    if turning_events:
        events_df = pd.DataFrame(turning_events)
        events_log_filename = os.path.join(log_dir, f"turning_events_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        events_df.to_csv(events_log_filename, index=False)
    
    return detector.log_data


if __name__ == "__main__":
    video_path = "data/videos/WIN_20250319_10_03_53_Pro.mp4"
    output_path = "data/output/head_detection_output.mp4"
    log_data = process_video(video_path, output_path, show_preview=True, log_dir="logs") 