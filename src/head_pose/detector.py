import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from scipy.signal import savgol_filter
import pandas as pd
import os

class HeadPoseDetector:
    """
    頭部姿勢検出クラス
    MediaPipeのFaceMeshを使用して顔のランドマークを検出し、頭部の姿勢（特にyaw角）を推定
    """
    def __init__(self, max_history=40, yaw_threshold=10, consecutive_frames=2, detection_cooldown=1.0):
        """
        初期化
        Args:
            max_history (int): 履歴に保存するフレーム数
            yaw_threshold (float): 首振りと判定する角度変化の閾値（度）
            consecutive_frames (int): 首振りと判定する連続フレーム数
            detection_cooldown (float): 検出後のクールダウン時間（秒）
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.yaw_history = deque(maxlen=max_history)
        self.time_history = deque(maxlen=max_history)
        self.yaw_threshold = yaw_threshold
        self.consecutive_frames = consecutive_frames
        self.detection_cooldown = detection_cooldown
        self.log_data = []
        
        # 性能評価用の変数
        self.true_positives = 0
        self.false_positives = 0
        self.total_events = 0
        
        # 最後の検出時刻
        self.last_detection_time = 0
        
    def calculate_head_pose(self, image):
        """
        画像から頭部姿勢を計算
        Args:
            image: 入力画像（OpenCV形式）
        Returns:
            Tuple: (検出成功フラグ, yaw角, 首振り検出フラグ, 処理時間(ms))
        """
        start_time = time.time()
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # 顔の主要なランドマークを取得
            nose = np.array([face_landmarks.landmark[1].x, face_landmarks.landmark[1].y, face_landmarks.landmark[1].z])
            left_eye = np.array([face_landmarks.landmark[33].x, face_landmarks.landmark[33].y, face_landmarks.landmark[33].z])
            right_eye = np.array([face_landmarks.landmark[263].x, face_landmarks.landmark[263].y, face_landmarks.landmark[263].z])
            
            # yaw角（左右の回転）を計算
            eye_center = (left_eye + right_eye) / 2
            eye_to_nose = nose - eye_center
            yaw = np.arctan2(eye_to_nose[0], eye_to_nose[2]) * 180 / np.pi
            
            # 履歴に追加
            self.yaw_history.append(yaw)
            self.time_history.append(time.time())
            
            # 首振りの検出
            head_turning = self._detect_head_turning()
            
            processing_time = (time.time() - start_time) * 1000  # ミリ秒に変換
            
            # ログデータの記録
            self.log_data.append({
                'timestamp': time.time(),
                'yaw_angle': yaw,
                'head_turning_detected': head_turning,
                'processing_time': processing_time
            })
            
            return True, yaw, head_turning, processing_time
            
        return False, None, False, (time.time() - start_time) * 1000

    def _detect_head_turning(self):
        """
        履歴からのyaw角データを使用して首振りを検出
        Returns:
            bool: 首振りが検出された場合はTrue
        """
        current_time = time.time()
        
        # クールダウン時間中は検出しない
        if current_time - self.last_detection_time < self.detection_cooldown:
            return False
            
        if len(self.yaw_history) < self.consecutive_frames + 3:
            return False
            
        # Savitzky-Golayフィルタでノイズを除去
        if len(self.yaw_history) >= 5:
            yaw_array = np.array(list(self.yaw_history))
            try:
                # 5点で2次の多項式でフィルタリング
                smoothed_yaw = savgol_filter(yaw_array, 5, 2)
                
                # 角速度の計算（度/秒）
                yaw_velocity = []
                for i in range(1, len(smoothed_yaw)):
                    if i < len(self.time_history) and i-1 < len(self.time_history):
                        dt = self.time_history[i] - self.time_history[i-1]
                        if dt > 0:
                            velocity = (smoothed_yaw[i] - smoothed_yaw[i-1]) / dt
                            yaw_velocity.append(velocity)
                
                if len(yaw_velocity) < self.consecutive_frames:
                    return False
                
                # より高度な首振り検出ロジック
                # 1. 連続した閾値超えを検出
                num_significant_changes = sum(1 for v in yaw_velocity[-self.consecutive_frames-2:-2] if abs(v) > self.yaw_threshold)
                
                # 2. 方向の変化も検出（左右の振り）
                sign_changes = 0
                for i in range(1, len(yaw_velocity)):
                    if (yaw_velocity[i-1] > 0 and yaw_velocity[i] < 0) or (yaw_velocity[i-1] < 0 and yaw_velocity[i] > 0):
                        sign_changes += 1
                
                # 必要条件: 十分な数の大きな変化と方向転換の両方が必要
                if num_significant_changes >= self.consecutive_frames and sign_changes >= 1:
                    self.last_detection_time = current_time
                    return True
                
            except ValueError:
                # フィルタリングに失敗した場合（データポイントが少ない場合など）
                pass
                
        return False
        
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
        
    def __del__(self):
        """デストラクタ：リソースの解放"""
        self.face_mesh.close()


def process_video(video_path, output_path=None, show_preview=True, log_dir="logs", detector=None):
    """
    ビデオを処理し、頭部姿勢を検出する
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
        # 顔が小さいシーンでも検出できるようパラメータを調整
        detector = HeadPoseDetector(max_history=40, yaw_threshold=10, consecutive_frames=2)
    
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
        
        # 頭部姿勢の計算
        success, yaw, head_turning, processing_time = detector.calculate_head_pose(frame)
        
        # 処理時間の記録
        total_processing_time += processing_time
        max_processing_time = max(max_processing_time, processing_time)
        
        if success:
            # 結果を画面に表示
            cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {processing_time:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 首振り検出の表示を強化
            if head_turning:
                # 首振り検出時は目立つ表示
                cv2.putText(frame, "Head Turning Detected!", (frame_width // 2 - 150, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                # 赤い枠を描画
                cv2.rectangle(frame, (0, 0), (frame_width, frame_height), (0, 0, 255), 5)
                
                # 検出イベントの記録
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
            cv2.imshow('Head Pose Detection', frame)
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
    log_filename = os.path.join(log_dir, f"head_pose_detection_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    detector.save_logs(log_filename)
    
    # 首振りイベントのログも保存
    if turning_events:
        events_df = pd.DataFrame(turning_events)
        events_log_filename = os.path.join(log_dir, f"turning_events_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        events_df.to_csv(events_log_filename, index=False)
    
    return detector.log_data


if __name__ == "__main__":
    video_path = "data/videos/WIN_20250319_10_03_53_Pro.mp4"
    output_path = "data/output/head_pose_output.mp4"
    log_data = process_video(video_path, output_path, show_preview=True, log_dir="logs") 