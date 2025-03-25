import cv2
import numpy as np
import time
import os
import pandas as pd
from collections import deque
import mediapipe as mp

class StickFigureDetector:
    """
    MediaPipeを使用した棒人間モデル検出クラス
    人物の骨格を検出し、首振り（頭部の左右の動き）を検知する
    """
    def __init__(self, 
                confidence=0.5, 
                movement_threshold=0.05, 
                consecutive_frames=3, 
                detection_cooldown=2.0,
                history_size=40,
                min_event_duration=0.5):
        """
        棒人間検出器の初期化
        Args:
            confidence: 検出信頼度閾値
            movement_threshold: 首振りと判定する移動量の閾値（画像幅に対する比率）
            consecutive_frames: 首振りと判定する連続フレーム数
            detection_cooldown: 検出後のクールダウン時間（秒）
            history_size: 履歴に保存するフレーム数
            min_event_duration: 首振りイベントの最小持続時間（秒）
        """
        # MediaPipeモデルの初期化
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=Lite 1=Full 2=Heavy
            smooth_landmarks=True,
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 検出パラメータ
        self.confidence = confidence
        self.movement_threshold = movement_threshold
        self.consecutive_frames = consecutive_frames
        self.detection_cooldown = detection_cooldown
        self.min_event_duration = min_event_duration
        
        # 頭部位置の履歴
        self.head_positions = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)
        
        # 首振り検出状態
        self.turning_start_time = None
        self.is_turning = False
        self.last_detection_time = 0
        self.current_movement_count = 0
        
        # ログデータ
        self.log_data = []
        
    def detect_pose(self, frame, frame_idx=0, display=True):
        """
        フレームから骨格を検出し、頭部位置を特定
        Args:
            frame: 検出を行うビデオフレーム
            frame_idx: フレームインデックス（ログ用）
            display: 表示用のデータを生成するかどうか
        Returns:
            tuple: (head_detected, landmarks, display_frame, head_turning)
        """
        start_time = time.time()
        
        # MediaPipeはRGBを想定しているのでBGR->RGBに変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        processing_time = (time.time() - start_time) * 1000  # ミリ秒に変換
        
        head_detected = False
        head_turning = False
        landmarks = None
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks
            
            # 頭部位置の取得（鼻のランドマークを使用）
            nose = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            left_ear = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR]
            
            # 正規化座標から画像座標に変換
            frame_height, frame_width = frame.shape[:2]
            nose_x, nose_y = int(nose.x * frame_width), int(nose.y * frame_height)
            left_ear_x, left_ear_y = int(left_ear.x * frame_width), int(left_ear.y * frame_height)
            right_ear_x, right_ear_y = int(right_ear.x * frame_width), int(right_ear.y * frame_height)
            
            # 頭部の中心位置を計算（鼻と両耳の中心）
            head_x = (nose_x + (left_ear_x + right_ear_x) / 2) / 2
            head_y = (nose_y + (left_ear_y + right_ear_y) / 2) / 2
            
            # 信頼度が閾値を超えていれば検出成功とみなす
            if nose.visibility > self.confidence and \
               left_ear.visibility > self.confidence * 0.7 and \
               right_ear.visibility > self.confidence * 0.7:
                head_detected = True
                
                # 頭部位置を履歴に追加
                self.head_positions.append((head_x, head_y))
                self.timestamps.append(time.time())
                
                # 首振りの検出
                turning_detected = self._detect_head_turning(frame_width)
                
                # 首振り状態の更新
                head_turning, event_started, event_ended, event_duration = self._update_turning_state(
                    turning_detected, frame_idx, time.time())
                
                # ログデータの記録
                self.log_data.append({
                    'timestamp': time.time(),
                    'head_x': head_x,
                    'head_y': head_y,
                    'head_turning_detected': head_turning,
                    'processing_time': processing_time,
                    'confidence': nose.visibility
                })
        
        if display and results.pose_landmarks:
            # 骨格を画像に描画
            annotated_frame = frame.copy()
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # 首振り検出時の表示
            if head_turning:
                cv2.putText(annotated_frame, "Head Turning Detected!", 
                           (frame_width // 2 - 180, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                # 赤い枠を描画
                cv2.rectangle(annotated_frame, (0, 0), (frame_width-1, frame.shape[0]-1), (0, 0, 255), 5)
            
            # 頭部の軌跡を描画
            if len(self.head_positions) > 1:
                pts = np.array([[int(p[0]), int(p[1])] for p in self.head_positions], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], False, (255, 0, 0), 2)
                
            return head_detected, landmarks, annotated_frame, head_turning
        
        return head_detected, landmarks, frame, head_turning
    
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
        if len(self.head_positions) < self.consecutive_frames + 2:
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
        
        # 首振り検出ロジック
        # 1. 十分な大きさの動きが連続して発生しているか
        recent_movements = x_movements[-self.consecutive_frames:]
        significant_movements = sum(1 for m in recent_movements if abs(m) > self.movement_threshold)
        
        # 2. 最大移動量
        max_movement = max(abs(m) for m in recent_movements) if recent_movements else 0
        
        # 動きカウントを更新
        if significant_movements >= 1 or max_movement > self.movement_threshold * 1.5:
            self.current_movement_count += 1
        else:
            # 動きがなければカウントをリセット
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
                # 短すぎるイベントは無視
                event_started = False
            
            self.turning_start_time = None
        
        return head_turning, event_started, event_ended, event_duration
        
    def save_logs(self, filename):
        """
        検出ログをCSVファイルに保存
        Args:
            filename: 保存するCSVファイルのパス
        """
        if not self.log_data:
            return
            
        df = pd.DataFrame(self.log_data)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        
    def release(self):
        """リソースを解放"""
        self.pose.close()


def process_video(video_path, output_path=None, show_preview=True, log_dir="logs", detector=None):
    """
    ビデオを処理し、骨格を検出して首振りを検知する
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
        detector = StickFigureDetector()
    
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
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 骨格検出の実行
            start_time = time.time()
            head_detected, landmarks, display_frame, head_turning = detector.detect_pose(frame, frame_count - 1, show_preview)
            processing_time = (time.time() - start_time) * 1000
            
            # 処理時間の記録
            total_processing_time += processing_time
            max_processing_time = max(max_processing_time, processing_time)
            
            # 首振りイベントの記録
            if head_turning and current_turning_event is None:
                current_turning_event = {
                    'start_frame': frame_count,
                    'start_time': time.time()
                }
            elif not head_turning and current_turning_event is not None:
                # 首振りイベントの終了
                current_turning_event['end_frame'] = frame_count - 1
                current_turning_event['end_time'] = time.time()
                current_turning_event['duration'] = current_turning_event['end_time'] - current_turning_event['start_time']
                turning_events.append(current_turning_event)
                current_turning_event = None
            
            # 平均処理時間と要件（500ms以内）の達成状況を表示
            avg_time = total_processing_time / frame_count
            cv2.putText(display_frame, f"Avg: {avg_time:.1f}ms (Target: 500ms)", (10, frame_height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if avg_time <= 500 else (0, 0, 255), 2)
            
            # 出力ビデオに書き込み
            if output_path:
                out.write(display_frame)
                
            # プレビューの表示
            if show_preview:
                cv2.imshow('Stick Figure Detection', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except KeyboardInterrupt:
        print("処理が中断されました")
    finally:
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
        detector.release()
    
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
    log_filename = os.path.join(log_dir, f"stick_figure_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    detector.save_logs(log_filename)
    
    # 首振りイベントのログも保存
    if turning_events:
        events_df = pd.DataFrame(turning_events)
        events_log_filename = os.path.join(log_dir, f"stick_figure_events_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        events_df.to_csv(events_log_filename, index=False)
    
    return detector.log_data


if __name__ == "__main__":
    video_path = "WIN_20250319_10_03_53_Pro.mp4"
    output_path = "data/output/stick_figure_detection.mp4"
    log_data = process_video(video_path, output_path, show_preview=True, log_dir="logs") 