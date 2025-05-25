# YOLO単体 vs ByteTrack ID付与比較システム

このシステムは、YOLO単体でのID付与とByteTrackでのID付与の性能と精度を比較評価するためのツールです。

## 概要

### 比較対象
1. **YOLO簡易ID付与**: 位置ベースの簡単なID割り当て
2. **ByteTrack ID付与**: ByteTrackライブラリによるID割り当て

### 評価項目
- **精度**: ID一貫性、IDスイッチ回数、追跡継続性
- **性能**: 処理時間、メモリ使用量、FPS
- **信頼性**: 追跡安定性、遮蔽処理

## ディレクトリ構成

```
comparison/
├── main_comparison.py          # メイン実行ファイル
├── config/
│   └── comparison_config.yaml  # 設定ファイル
├── id_methods/                 # ID付与手法
│   ├── __init__.py
│   ├── base_tracker.py         # 基底クラス
│   ├── yolo_simple_tracker.py  # YOLO簡易版
│   └── bytetrack_wrapper.py    # ByteTrackラッパー
├── evaluators/                 # 評価システム
│   ├── __init__.py
│   └── comparison_evaluator.py # 比較評価クラス
├── output/                     # 結果出力
│   ├── comparison_results_*.csv
│   └── comparison_summary_*.json
└── README.md                   # このファイル
```

## インストール

### 必要なライブラリ

```bash
pip install ultralytics opencv-python numpy scipy scikit-learn psutil
pip install boxmot  # ByteTrack用
```

### YOLOモデル
システムが自動的に`yolo11n.pt`をダウンロードします。

## 使用方法

### 1. 基本的な使用方法

```bash
# 動画ファイルを指定して実行
python comparison/main_comparison.py path/to/your/video.mp4

# オプション指定の例
python comparison/main_comparison.py video.mp4 \
  --model yolo11n.pt \
  --confidence 0.3 \
  --max-frames 1800 \
  --output-dir results
```

### 2. デモ実行

引数なしで実行するとデモモードになります：

```bash
python comparison/main_comparison.py
```

### 3. コマンドラインオプション

| オプション | 説明 | デフォルト値 |
|-----------|------|------------|
| `video_path` | 入力動画ファイルのパス | (必須) |
| `--model` | YOLOモデルのパス | `yolo11n.pt` |
| `--confidence` | 検出信頼度閾値 | `0.3` |
| `--device` | 実行デバイス | `""` (自動) |
| `--max-frames` | 処理する最大フレーム数 | 全フレーム |
| `--display` | リアルタイム表示を有効 | 無効 |
| `--output-dir` | 結果出力ディレクトリ | `comparison/output` |
| `--prefix` | 出力ファイル名のプレフィックス | `comparison` |

### 4. 設定ファイルの利用

`config/comparison_config.yaml`でデフォルト設定を変更できます。

## 出力結果

### 1. CSV詳細結果 (`comparison_results_*.csv`)

各フレームの詳細データ：
- フレーム番号、タイムスタンプ
- 各手法の処理時間、FPS、トラック数、検出数、ID切り替え回数

### 2. JSON サマリー (`comparison_summary_*.json`)

各手法の総合評価：
```json
{
  "tracking_metrics": {
    "YOLO_Simple": {
      "avg_processing_time_ms": "平均処理時間(ms)",
      "avg_fps": "平均FPS",
      "id_switches": "ID切り替え回数",
      "track_consistency": "トラック一貫性(0-1)"
    },
    "ByteTrack": { "..." }
  }
}
```

### 3. コンソール出力

実行時に比較結果のサマリーが表示されます：

```
================================================================================
YOLO vs ByteTrack ID付与比較結果
================================================================================

【YOLO_Simple】
  処理フレーム数: [処理したフレーム数]
  平均処理時間: [処理時間] ms
  平均FPS: [フレームレート]
  平均メモリ使用量: [メモリ使用量] MB
  総検出数: [検出総数]
  総トラック数: [トラック総数]
  ID切り替え回数: [ID切り替え回数]
  トラック一貫性: [一貫性スコア(0-1)]

【ByteTrack】
  処理フレーム数: [処理したフレーム数]
  平均処理時間: [処理時間] ms
  平均FPS: [フレームレート]
  平均メモリ使用量: [メモリ使用量] MB
  総検出数: [検出総数]
  総トラック数: [トラック総数]
  ID切り替え回数: [ID切り替え回数]
  トラック一貫性: [一貫性スコア(0-1)]
================================================================================
```

## アルゴリズムの特徴

### YOLO簡易ID付与
- **手法**: 位置ベースの最近傍マッチング
- **長所**: 高速処理、低メモリ使用量、シンプル実装
- **短所**: 遮蔽に弱い、外観モデルなし
- **想定用途**: リアルタイム性を重視する軽量アプリケーション

### ByteTrack
- **手法**: BYTE association algorithm with Kalman filtering
- **長所**: 高精度な追跡性能、遮蔽に堅牢、混雑シーンに対応
- **短所**: 複雑な実装、高い計算コスト、パラメータ調整が必要
- **想定用途**: 精度を重視する本格的な追跡システム

## トラブルシューティング

### 1. 動画ファイルが開けない
- ファイルパスが正しいか確認
- OpenCVがサポートする形式か確認（mp4, avi等）

### 2. メモリ不足エラー
- `--max-frames`で処理フレーム数を制限
- より軽量なYOLOモデル（yolo11n.pt）を使用

### 3. ByteTrackエラー
- boxmotライブラリが正しくインストールされているか確認
- `pip install boxmot`を実行

### 4. GPU使用エラー
- `--device cpu`でCPU実行に切り替え
- CUDA環境が正しく設定されているか確認

## 拡張可能性

### 新しいトラッカーの追加
1. `BaseTracker`を継承したクラスを作成
2. `ComparisonEvaluator._initialize_trackers()`に追加
3. 設定ファイルにパラメータを追加

### 新しい評価メトリクスの追加
1. `TrackingMetrics`データクラスに項目追加
2. `ComparisonEvaluator.calculate_final_metrics()`で計算ロジック実装

## ライセンス

このプロジェクトは研究・開発目的で作成されています。

## 参考文献

- [YOLO11](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [BoxMOT](https://github.com/mikel-brostrom/yolov8_tracking) 