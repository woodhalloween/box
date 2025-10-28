# アノテーション（動画ラベル付け）システム

このディレクトリには、動画に対して手作業でラベル付けを行い、検出アルゴリズムの性能を評価するためのツールが含まれています。

## 📋 概要

動画を再生しながらキーボード操作でリアルタイムにラベル付けを行い、正解ラベルを作成します。その後、MediaPipeやYOLO11などの検出結果と比較して、定量的な評価指標（Precision、Recall、F1-Score、IoUなど）を計算します。

## 🗂️ ディレクトリ構成

```
annotation/
├── README.md                           # このファイル
├── video_labeling_tool.py              # ラベル付けツール本体
├── LABELING_GUIDE.md                   # 詳細な使用ガイド
├── EVALUATION_SYSTEM_SUMMARY.md        # システム概要
├── evaluation/                         # 評価モジュール
│   ├── __init__.py
│   ├── result_loader.py                # 検出結果読み込み
│   ├── metrics_calculator.py           # 評価指標計算
│   ├── evaluate_detections.py          # 評価実行スクリプト
│   └── results/                        # 評価レポート保存先
├── ground_truth/                       # 正解ラベル保存先
│   └── *.csv
└── videos/                             # アノテーション対象動画（オプション）
```

## 🚀 クイックスタート

### 1. ラベル付けツールの実行

```bash
cd /Users/takahashiryoutarou/Desktop/国宝さん一覧/box/annotation

python3 video_labeling_tool.py \
  "../data/raw/国宝さん手をあげる2.mp4" \
  "ground_truth/国宝さん手をあげる2_labels.csv"
```

**キーボード操作:**
- `1`: 首振りイベントの開始/終了をトグル
- `Space`: 一時停止/再生
- `←/→`: 5秒巻き戻し/早送り
- `q`: 終了して保存

### 2. 評価の実行

```bash
python3 evaluation/evaluate_detections.py \
  "ground_truth/国宝さん手をあげる2_labels.csv" \
  --video "../data/raw/国宝さん手をあげる2.mp4" \
  --mediapipe "../data/raw/results/国宝さん手をあげる2_mediapipe_detection.csv" \
  --output "evaluation/results/国宝さん手をあげる2_evaluation.md"
```

### 3. 複数手法の比較

```bash
python3 evaluation/evaluate_detections.py \
  "ground_truth/国宝さん手をあげる2_labels.csv" \
  --video "../data/raw/国宝さん手をあげる2.mp4" \
  --mediapipe "../data/raw/results/国宝さん手をあげる2_mediapipe_detection.csv" \
  --yolo11 "../results/movement_analysis.csv" \
  --output "evaluation/results/comparison.md"
```

## 📊 評価指標

### フレーム単位
- **Precision**: 検出したフレームのうち、正解だった割合
- **Recall**: 正解フレームのうち、検出できた割合
- **F1-Score**: PrecisionとRecallの調和平均
- **Accuracy**: 全フレームのうち、正しく分類できた割合

### イベント単位
- **IoU**: イベント区間の重なり度合い（Intersection over Union）
- **True Positive / False Positive / False Negative**: イベント検出の正確性

### 時間的重なり
- **重なり率**: 正解イベントの時間をどれだけカバーできているか

## 📚 詳細ドキュメント

- **[LABELING_GUIDE.md](LABELING_GUIDE.md)**: 詳細な使用方法とトラブルシューティング
- **[EVALUATION_SYSTEM_SUMMARY.md](EVALUATION_SYSTEM_SUMMARY.md)**: システムの完全な説明

## 🎯 ワークフロー

1. **ラベル付け**: `video_labeling_tool.py` で正解ラベルを作成
2. **検出実行**: MediaPipeやYOLO11で動画を処理（既存のスクリプト使用）
3. **評価**: `evaluation/evaluate_detections.py` で性能を評価
4. **改善**: 評価結果をもとにアルゴリズムを改善
5. **再評価**: 改善後のアルゴリズムを再度評価

## ⚠️ 注意事項

- ラベル付けツールはGUIを使用するため、ターミナルから実行してください
- 動画ファイルのパスは絶対パスまたは相対パスで指定できます
- 正解ラベルは `ground_truth/` ディレクトリに保存されます
- 評価レポートは `evaluation/results/` ディレクトリに保存されます

## 🔧 トラブルシューティング

### ラベル付けツールが起動しない

```bash
# OpenCVがインストールされているか確認
pip install opencv-python
```

### 評価スクリプトでエラーが出る

```bash
# evaluation ディレクトリから実行
cd evaluation
python3 evaluate_detections.py ...
```

詳細は [LABELING_GUIDE.md](LABELING_GUIDE.md) を参照してください。


