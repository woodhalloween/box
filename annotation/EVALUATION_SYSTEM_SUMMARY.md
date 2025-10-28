# 動画ラベル付けと評価システム 完成報告

## 📋 概要

動画に対して手作業でラベル付けを行い、既存の検出アルゴリズム（MediaPipe、YOLO11）の性能を定量的に評価するシステムを実装しました。

## 🎯 実装内容

### 1. ラベル付けツール (`video_labeling_tool.py`)

動画を再生しながらキーボード操作でリアルタイムにラベル付けを行うGUIツール。

**主な機能:**
- OpenCVによる動画再生
- キーボード操作でイベントの開始/終了を記録
  - `1`: 首振りイベントのトグル
  - `Space`: 一時停止/再生
  - `←/→`: 5秒巻き戻し/早送り
  - `q`: 終了して保存
- リアルタイムの状態表示（フレーム番号、時刻、記録状態）
- CSV形式での保存

**出力形式:**
```csv
start_frame,end_frame,start_time,end_time,event_type
150,250,5.0,8.33,HEAD_SHAKE
400,500,13.33,16.67,HEAD_SHAKE
```

### 2. 検出結果読み込みモジュール (`evaluation/result_loader.py`)

MediaPipeとYOLO11の検出結果を統一フォーマットに変換するモジュール。

**主な機能:**
- 正解ラベルCSVの読み込み
- MediaPipe検出結果の読み込み（複数のカラム名形式に対応）
- YOLO11検出結果の読み込み
- 連続するフレームを1つのイベントにまとめる処理
- イベントタイプによるフィルタリング
- データサマリーの表示

### 3. 評価指標計算モジュール (`evaluation/metrics_calculator.py`)

フレーム単位とイベント単位の評価指標を計算するモジュール。

**フレーム単位の評価:**
- Precision（精度）: 検出したフレームのうち、正解だった割合
- Recall（再現率）: 正解フレームのうち、検出できた割合
- F1-Score: PrecisionとRecallの調和平均
- Accuracy（正解率）: 全フレームのうち、正しく分類できた割合
- True Positive / False Positive / False Negative / True Negative

**イベント単位の評価:**
- IoU（Intersection over Union）ベースのマッチング
- 複数のIoU閾値（0.3, 0.5, 0.7）での評価
- True Positive / False Positive / False Negative
- Mean IoU（平均IoU）

**時間的重なりの統計:**
- 正解総時間 / 予測総時間
- 重なり時間 / 重なり率

### 4. 評価実行スクリプト (`evaluation/evaluate_detections.py`)

正解ラベルと検出結果を比較し、評価レポートを生成するスクリプト。

**主な機能:**
- 動画情報の自動取得（総フレーム数、FPS）
- 複数の検出手法の同時評価
- Markdown形式の比較レポート生成
- 詳細な評価指標の表示

## 📊 使用例

### ステップ1: ラベル付け

```bash
cd /Users/takahashiryoutarou/Desktop/国宝さん一覧/box/experiments

python3 video_labeling_tool.py \
  "../data/raw/国宝さん手をあげる2.mp4" \
  "ground_truth/国宝さん手をあげる2_labels.csv"
```

### ステップ2: 評価実行

```bash
python3 evaluation/evaluate_detections.py \
  "ground_truth/国宝さん手をあげる2_labels.csv" \
  --video "../data/raw/国宝さん手をあげる2.mp4" \
  --mediapipe "../data/raw/results/国宝さん手をあげる2_mediapipe_detection_20251022_092013.csv" \
  --output "evaluation/results/国宝さん手をあげる2_evaluation.md"
```

### ステップ3: 結果確認

```bash
cat evaluation/results/国宝さん手をあげる2_evaluation.md
```

## 📈 評価レポートのサンプル

```markdown
# 検出手法の比較結果

**正解ラベル:** `ground_truth/国宝さん手をあげる2_labels.csv`
**評価日時:** 2025-10-26 23:21:26
**総フレーム数:** 857
**FPS:** 23.99

## 📊 フレーム単位の評価

| 手法 | Precision | Recall | F1-Score | Accuracy |
|------|-----------|--------|----------|----------|
| MediaPipe | 0.00% | 0.00% | 0.00% | 64.53% |

## 🎯 イベント単位の評価 (IoU >= 0.5)

| 手法 | Precision | Recall | F1-Score | Mean IoU |
|------|-----------|--------|----------|----------|
| MediaPipe | 0.00% | 0.00% | 0.00% | 0.00% |

## ⏱️  時間的重なり

| 手法 | 正解総時間(秒) | 予測総時間(秒) | 重なり時間(秒) | 重なり率(正解基準) |
|------|---------------|---------------|---------------|-------------------|
| MediaPipe | 10.00 | 0.00 | 0.00 | 0.00% |
```

## 🗂️ ファイル構成

```
experiments/
├── video_labeling_tool.py          # ラベル付けツール本体
├── evaluation/
│   ├── __init__.py
│   ├── result_loader.py            # 検出結果読み込み
│   ├── metrics_calculator.py       # 評価指標計算
│   ├── evaluate_detections.py      # 評価実行スクリプト
│   └── results/                    # 評価レポート保存先
│       └── *.md
├── ground_truth/                   # 正解ラベル保存先
│   └── *.csv
├── LABELING_GUIDE.md              # 詳細な使用ガイド
└── EVALUATION_SYSTEM_SUMMARY.md   # このファイル
```

## ✅ テスト結果

### デモデータでの評価

- **正解ラベル**: 3つの首振りイベント（計10秒）
- **MediaPipe検出結果**: 1つのイベント（1フレームのみ）
- **評価結果**: 
  - フレーム単位: Precision 0%, Recall 0%, Accuracy 64.53%
  - イベント単位: TP=0, FP=1, FN=3
  - 時間的重なり: 0秒（重なりなし）

評価システムが正常に動作し、定量的な評価指標が計算できることを確認しました。

## 🎓 評価指標の解釈

### フレーム単位の評価

- **高いPrecision**: 検出した箇所はほぼ正解（誤検出が少ない）
- **高いRecall**: 正解箇所をほぼ検出できている（見逃しが少ない）
- **高いF1-Score**: PrecisionとRecallのバランスが良い

### イベント単位の評価

- **高いPrecision**: 検出したイベントはほぼ正解（誤検出イベントが少ない）
- **高いRecall**: 正解イベントをほぼ検出できている（見逃しイベントが少ない）
- **高いMean IoU**: イベント区間の重なりが大きい（時間的に正確）

### 時間的重なり

- **高い重なり率**: 正解イベントの時間をカバーできている
- **予測総時間 ≈ 正解総時間**: 過検出・過少検出が少ない

## 🔧 今後の拡張

### 1. イベントタイプの追加

現在は首振り（HEAD_SHAKE）のみですが、以下のイベントも追加可能:
- 手を上げる（HAND_RAISE）
- うなずき（HEAD_NOD）
- 体の揺れ（TORSO_SWAY）

### 2. 複数人の対応

person_idを追加して、複数人のラベル付けに対応。

### 3. 自動ラベル付け支援

既存の検出結果を初期値として表示し、修正のみ行う機能。

### 4. 可視化機能

- 正解ラベルと検出結果を動画上に同時表示
- タイムライン表示（正解 vs 予測）
- 混同行列（Confusion Matrix）の可視化

## 📚 関連ドキュメント

- [LABELING_GUIDE.md](LABELING_GUIDE.md): 詳細な使用ガイド
- [README.md](README.md): 外部レポジトリ実験の概要
- [QUICK_START.md](QUICK_START.md): 外部検出スクリプトのクイックスタート
- [EXTERNAL_COMPARISON_REPORT.md](EXTERNAL_COMPARISON_REPORT.md): 外部手法の比較レポート

## 🎉 まとめ

動画ラベル付けツールと評価システムの実装が完了しました。

**実装した機能:**
1. ✅ リアルタイムラベル付けツール（OpenCV + キーボード操作）
2. ✅ 検出結果読み込みモジュール（MediaPipe/YOLO11対応）
3. ✅ 評価指標計算モジュール（フレーム単位・イベント単位）
4. ✅ 評価実行スクリプト（Markdownレポート生成）
5. ✅ 詳細な使用ガイド

**次のステップ:**
1. 実際の動画でラベル付けを行う
2. MediaPipeとYOLO11の検出結果を評価
3. 評価結果をもとにアルゴリズムを改善

このシステムを使用することで、検出アルゴリズムの性能を客観的に評価し、改善の方向性を明確にすることができます。

