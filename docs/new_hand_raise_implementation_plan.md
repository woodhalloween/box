# 手挙げ検出 実装計画書

## 1. 目的

「手を挙げている」人物を検出するアルゴリズムを実装し、動画ファイルに適用して結果を出力する一連の機能を提供する。

## 2. 実装方針

既存の姿勢推定 (`PoseEstimator`) や動画処理 (`VideoProcessor`) の基盤を活用し、新たに手挙げ検出ロジックを独立したモジュールとして実装する。これにより、保守性と再利用性を高める。

## 3. 作成ファイル

本計画では、以下の2つの主要ファイルを作成する。

1.  **`src/hand_raise_detector.py`**: 手挙げ検出アルゴリズムの中核ロジックを実装するモジュール。
2.  **`run_hand_raise.py`**: 動画の読み込み、検出の実行、結果の出力を統括するスクリプト。

## 4. 各ファイルの実装詳細

### 4.1. `src/hand_raise_detector.py`

手挙げの判定と状態追跡を担当するクラスを定義する。

```python
from typing import Dict, Any, List
import numpy as np

class HandRaiseDetector:
    """
    人物のキーポイント情報から手挙げ状態を検出し、追跡するクラス。
    """

    def __init__(self, visibility_threshold: float = 0.5, min_frames: int = 5):
        """
        Args:
            visibility_threshold (float): キーポイントの信頼度スコアの閾値。
            min_frames (int): 手挙げ状態と確定するための最小連続フレーム数。
        """
        self.visibility_threshold = visibility_threshold
        self.min_frames = min_frames
        self.track_data: Dict[str, int] = {}  # e.g. {"left_hand_up": count, "right_hand_up": count}

    def detect(self, landmarks: np.ndarray | None) -> Dict[str, bool]:
        """
        1フレーム分のランドマークから手挙げ状態を判定・更新する。
        MediaPipeは人物IDを返さないため、フレーム内の最初の人物を対象とする。

        Args:
            landmarks (np.ndarray | None): 1人の人物のランドマークデータ。

        Returns:
            Dict[str, bool]: 左右の腕の手挙げ状態（確定済み）。
        """
        # 実装ロジック
        # 1. landmarksがNoneならリセットして終了
        # 2. 左右の腕について、肩・手首のランドマークを取得
        # 3. visibilityが閾値以上かチェック
        # 4. 手首が肩より上にあれば、対応する腕のカウンターをインクリメント
        # 5. そうでなければカウンターをリセット
        # 6. カウンターがmin_frames以上であればTrueを返す
        pass

```

### 4.2. `run_hand_raise.py`

`src`配下のモジュールを組み合わせて、手挙げ検出パイプラインを実行する。

```python
import argparse
import cv2
from src.video_processor import VideoProcessor
from src.pose_estimator import PoseEstimator
from src.hand_raise_detector import HandRaiseDetector
from src.drawing_utils import DrawingUtils
from src.io_utils import CSVWriter

def main():
    """
    動画を処理し、手挙げ検出を実行して結果を保存するメイン関数。
    """
    # 1. argparseで入力動画パス、出力パスなどを設定
    
    # 2. 各クラスのインスタンスを生成
    # pose_estimator = PoseEstimator(model_complexity=1)
    # hand_raise_detector = HandRaiseDetector()
    # csv_writer = CSVWriter("output/results.csv")
    
    # 3. VideoProcessorのコールバック関数を定義
    # def process_frame(frame, frame_number):
    #     # 3-1. 姿勢推定を実行
    #     landmarks = pose_estimator.estimate(frame)
    #
    #     # 3-2. 手挙げ検出を実行
    #     hand_statuses = hand_raise_detector.detect(landmarks)
    #
    #     # 3-3. 結果を描画し、CSVに書き込み
    #     if landmarks is not None and (hand_statuses["left"] or hand_statuses["right"]):
    #         # MediaPipeはbboxを直接返さないため、ランドマークから簡易的に計算するか、
    #         # もしくは描画しない選択も考えられる。
    #         # DrawingUtils.draw_landmarks(frame, landmarks)
    #         csv_writer.write(...)
    #
    #     return frame
    
    # 4. VideoProcessorを実行
    # processor = VideoProcessor("input.mp4", "output.mp4", callback=process_frame)
    # processor.process()

if __name__ == "__main__":
    main()

```
