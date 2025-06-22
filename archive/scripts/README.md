# scriptsフォルダ構成

このフォルダには、動画解析や物体追跡に関連する様々なスクリプトが含まれています。

## カテゴリ別ファイル一覧

### 1. トラッキング関連
- `track_video.py` - 基本的なビデオトラッキング
- `track_video_deepsort.py` - DeepSORTを使用したビデオトラッキング
- `detect_long_stay.py` - 長時間滞在検出

### 2. ByteTrack関連
- `detect_long_stay_bytetrack.py` - ByteTrackを使用した長時間滞在検出
- `compare_fps_yolov11_bytetrack.py` - YOLOv11とByteTrackのFPS比較
- `compare_resolution_yolov11_bytetrack.py` - 解像度によるYOLOv11とByteTrackの比較

### 3. YOLO関連
- `yolo_mac_compatible.py` - Mac互換のYOLO実装
- `compare_yolo_versions.py` - 異なるYOLOバージョンの比較
- `exam-yolo11-pose-estimation.py` - YOLOv11を使用したポーズ推定

### 4. トラッカー検証
- `check_botsort.py` - BotSORTトラッカーの検証
- `check_strongsort.py` - StrongSORTトラッカーの検証
- `check_boxmot.py` - BoxMOTの検証
- `check_tracker_output.py` - トラッカー出力の確認

### 5. ランドマーク/ポーズ検出
- `run_stick_figure_detection.py` - 棒人間の検出
- `run_head_pose.py` - 頭部姿勢の検出
- `run_head_detection.py` - 頭部検出
- `visualize_landmarks.py` - ランドマークの可視化
- `visualize_all_landmarks.py` - すべてのランドマークの可視化
- `create_landmark_video.py` - ランドマーク付きビデオの作成

### 6. 解析ツール
- `analyze_log.py` - ログ解析
- `analyze_head_turning.py` - 頭部回転の詳細解析
- `analyze_head_turning_simple.py` - 頭部回転の簡易解析
- `analyze_landmark_metrics.py` - ランドマークメトリクスの解析
- `export_landmark_metrics.py` - ランドマークメトリクスのエクスポート

### 7. ユーティリティ
- `clip_video.py` - ビデオクリッピングツール

## 推奨される整理方法

より良い管理のために、上記のカテゴリごとにサブフォルダを作成することをお勧めします：

```
scripts/
├── tracking/
├── bytetrack/
├── yolo/
├── tracker_tests/
├── landmarks/
├── analytics/
└── utils/
``` 