# YOLO単体 vs ByteTrack ID付与比較実装　実装計画書

## 1. 実装概要

### 目的
YOLO単体でのID付与（簡易版・高度版）とByteTrackでのID付与の性能と精度を比較する実装を行う。

### 比較対象
1. **YOLO簡易ID付与**: 位置ベースの簡単なID割り当て
2. **YOLO高度ID付与**: 特徴量マッチングを使った高度なID割り当て
3. **ByteTrack ID付与**: ByteTrackライブラリによるID割り当て

## 2. アーキテクチャ設計

### 2.1 システム構成図
```
comparison/
├── main_comparison.py          # メイン実行ファイル
├── config/
│   └── comparison_config.yaml  # 設定ファイル
├── id_methods/
│   ├── __init__.py
│   ├── base_tracker.py         # 基底トラッカークラス
│   ├── yolo_simple_tracker.py  # YOLO簡易トラッカー
│   ├── yolo_advanced_tracker.py # YOLO高度トラッカー
│   └── bytetrack_wrapper.py    # ByteTrackラッパー
├── evaluators/
│   ├── __init__.py
│   └── comparison_evaluator.py # 比較評価クラス
├── utils/
│   ├── __init__.py
│   ├── visualization.py        # 可視化ユーティリティ
│   └── metrics.py             # メトリクス計算
└── output/                    # 出力ディレクトリ
    ├── comparison_results.csv
    ├── performance_summary.json
    └── comparison_video.mp4
```

### 2.2 クラス設計

#### BaseTracker（基底クラス）
```python
class BaseTracker:
    def __init__(self, model_path, confidence=0.3)
    def detect_and_track(self, frame) -> List[Track]
    def get_performance_metrics(self) -> Dict
    def reset(self)
```

#### YoloSimpleTracker（YOLO簡易）
```python
class YoloSimpleTracker(BaseTracker):
    def __init__(self, distance_threshold=50, max_disappeared=10)
    def _assign_ids_by_distance(self, detections, prev_tracks)
    def _update_disappeared_tracks(self)
```

#### YoloAdvancedTracker（YOLO高度）
```python
class YoloAdvancedTracker(BaseTracker):
    def __init__(self, appearance_weight=0.3, position_weight=0.7)
    def _extract_features(self, frame, bbox)
    def _calculate_similarity(self, detection, track)
    def _hungarian_assignment(self, cost_matrix)
```

#### ByteTrackWrapper（ByteTrack）
```python
class ByteTrackWrapper(BaseTracker):
    def __init__(self, track_thresh=0.5, track_buffer=30)
    def _convert_to_bytetrack_format(self, detections)
    def _convert_from_bytetrack_format(self, tracks)
```

## 3. 実装手順

### Phase 1: 基盤実装（1-2日）
1. ディレクトリ構造の作成
2. 基底クラス`BaseTracker`の実装
3. 共通ユーティリティの実装
4. 設定ファイルの作成

### Phase 2: ID付与手法の実装（3-4日）
1. `YoloSimpleTracker`の実装
   - 位置ベースの距離計算
   - 簡易的なID割り当てロジック
2. `YoloAdvancedTracker`の実装
   - 特徴量抽出（CNN特徴量）
   - 外観と位置の複合マッチング
3. `ByteTrackWrapper`の実装
   - 既存ByteTrackとの連携
   - フォーマット変換

### Phase 3: 評価システムの実装（2日）
1. `ComparisonEvaluator`の実装
   - 各手法の並列実行
   - メトリクス収集
   - 結果比較
2. 可視化機能の実装
   - リアルタイム比較表示
   - 結果動画の生成

### Phase 4: メイン統合とテスト（1-2日）
1. `main_comparison.py`の実装
2. 総合テストの実行
3. 結果の検証と調整

## 4. 実装詳細

### 4.1 YOLO簡易ID付与の実装詳細

#### アルゴリズム
1. YOLOで人物検出
2. 前フレームの追跡対象との距離計算
3. 最近傍マッチングでID割り当て
4. 新規検出は新しいID付与

#### 特徴
- 軽量で高速
- シンプルな実装
- 遮蔽に弱い

### 4.2 YOLO高度ID付与の実装詳細

#### アルゴリズム
1. YOLOで人物検出
2. バウンディングボックス内の特徴量抽出
3. 外観類似度 + 位置類似度の計算
4. ハンガリアン法でマッチング
5. ID割り当てと更新

#### 特徴
- 高精度だが重い処理
- 遮蔽にある程度対応
- 再出現時の再識別可能

### 4.3 ByteTrackとの比較ポイント

#### 評価項目
1. **精度**
   - ID一貫性（同一人物のIDが変わらない）
   - IDスイッチ回数
   - 断片化率（1つの軌跡が複数に分かれる）

2. **性能**
   - 処理時間（フレーム当たり）
   - メモリ使用量
   - 実現可能FPS

3. **堅牢性**
   - 遮蔽時の追跡継続
   - 一時的な消失後の再識別
   - 混雑シーンでの性能

## 5. データフロー

### 5.1 フレーム処理フロー
```
入力フレーム
    ↓
YOLO検出（共通）
    ↓
┌──────────────┬──────────────┬──────────────┐
│ YOLO簡易     │ YOLO高度     │ ByteTrack    │
│ ID付与       │ ID付与       │ ID付与       │
└──────────────┴──────────────┴──────────────┘
    ↓
評価メトリクス計算
    ↓
結果比較・可視化
    ↓
ログ出力・動画保存
```

### 5.2 評価データフロー
```
各手法の結果
    ↓
ID一貫性チェック
    ↓
性能メトリクス計算
    ↓
比較分析
    ↓
レポート生成
```

## 6. 設定パラメータ

### 6.1 共通パラメータ
- `confidence_threshold`: 0.3
- `model_path`: "yolo11n.pt"
- `frame_rate`: 30
- `test_duration`: 60秒

### 6.2 手法別パラメータ
#### YOLO簡易
- `distance_threshold`: 50px
- `max_disappeared_frames`: 10

#### YOLO高度
- `appearance_weight`: 0.3
- `position_weight`: 0.7
- `iou_threshold`: 0.3

#### ByteTrack
- `track_thresh`: 0.5
- `track_buffer`: 30
- `match_thresh`: 0.8

## 7. 出力仕様

### 7.1 リアルタイム表示
- 3分割画面で各手法を並列表示
- ID番号、信頼度、手法名を表示
- 処理時間とFPSをオーバーレイ

### 7.2 ログファイル
#### comparison_results.csv
```
frame_number,timestamp,method_type,object_id,bbox_x1,bbox_y1,bbox_x2,bbox_y2,confidence,processing_time_ms,memory_usage_mb
```

#### performance_summary.json
```json
{
  "yolo_simple": {
    "avg_processing_time": 15.2,
    "avg_fps": 28.5,
    "total_id_switches": 5,
    "tracking_accuracy": 82.3
  },
  "yolo_advanced": {...},
  "bytetrack": {...}
}
```

## 8. テストシナリオ

### 8.1 基本テスト
- 単一人物の追跡
- 複数人物の追跡
- 人物の一時的な消失

### 8.2 高難度テスト
- 遮蔽発生時の追跡
- 混雑シーンでの追跡
- 類似した外観の人物

## 9. 成功基準

### 9.1 機能的成功基準
- 3つの手法すべてが正常に動作
- リアルタイム比較表示が可能
- 各種メトリクスが正確に計算される

### 9.2 性能的成功基準
- 最低15FPS以上で動作
- メモリ使用量1GB以下
- ID一貫性85%以上（ByteTrackとの比較）

## 10. リスク要因と対策

### 10.1 技術的リスク
- **リスク**: YOLO高度版の処理が重すぎる
- **対策**: 特徴量の次元削減、処理の最適化

- **リスク**: 評価メトリクスの計算が複雑
- **対策**: 段階的な実装、簡易版から開始

### 10.2 データ的リスク
- **リスク**: テスト動画が適切でない
- **対策**: 複数のテスト動画を準備、段階的テスト

## 11. 今後の拡張可能性

### 11.1 追加比較手法
- DeepSORT
- FairMOT
- CenterTrack

### 11.2 評価指標の拡張
- MOTA/MOTP
- CLEAR MOT metrics
- Identity metrics 