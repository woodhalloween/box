# 動画ラベル付けガイド

このガイドでは、動画ラベル付けツールの使い方と、評価システムの実行方法を説明します。

## 📋 目次

1. [ラベル付けツールの使い方](#ラベル付けツールの使い方)
2. [評価システムの実行](#評価システムの実行)
3. [トラブルシューティング](#トラブルシューティング)

---

## ラベル付けツールの使い方

### 基本的な使い方

```bash
cd /Users/takahashiryoutarou/Desktop/国宝さん一覧/box/experiments

python3 video_labeling_tool.py \
  "../data/raw/国宝さん手をあげる2.mp4" \
  "ground_truth/国宝さん手をあげる2_labels.csv"
```

### キーボード操作

| キー | 機能 |
|------|------|
| `1` | 首振りイベントの開始/終了をトグル |
| `Space` | 一時停止/再生 |
| `←` | 5秒巻き戻し |
| `→` | 5秒早送り |
| `q` | 終了して保存 |

### ラベル付けの流れ

1. **ツールを起動**
   - 動画が再生されます
   - 画面上部に現在のフレーム番号と時刻が表示されます

2. **首振りイベントを記録**
   - 首振りが始まったら `1` キーを押す
   - 画面に `[REC] HEAD SHAKE` と表示されます
   - 首振りが終わったら再度 `1` キーを押す
   - ラベルが自動的に保存されます

3. **巻き戻し/早送り**
   - 見逃した場合は `←` キーで5秒巻き戻し
   - 早送りしたい場合は `→` キーで5秒早送り
   - `Space` キーで一時停止して、正確な位置を確認できます

4. **終了**
   - `q` キーを押すと、ラベルがCSVファイルに保存されます
   - 記録中のイベントがあれば、自動的に終了されます

### 出力ファイル形式

CSVファイルは以下の形式で保存されます:

```csv
start_frame,end_frame,start_time,end_time,event_type
150,250,5.0,8.33,HEAD_SHAKE
400,500,13.33,16.67,HEAD_SHAKE
```

- `start_frame`: イベント開始フレーム番号
- `end_frame`: イベント終了フレーム番号
- `start_time`: イベント開始時刻（秒）
- `end_time`: イベント終了時刻（秒）
- `event_type`: イベントタイプ（`HEAD_SHAKE`）

---

## 評価システムの実行

### 前提条件

1. 正解ラベルCSVが作成済み
2. MediaPipeまたはYOLO11の検出結果CSVが存在する

### MediaPipeの結果と比較

```bash
cd /Users/takahashiryoutarou/Desktop/国宝さん一覧/box/experiments

python3 evaluation/evaluate_detections.py \
  "ground_truth/国宝さん手をあげる2_labels.csv" \
  --video "../data/raw/国宝さん手をあげる2.mp4" \
  --mediapipe "../data/raw/results/国宝さん手をあげる2_mediapipe.csv" \
  --event-type "HEAD_SHAKE" \
  --output "evaluation/results/国宝さん手をあげる2_mediapipe_evaluation.md"
```

### YOLO11の結果と比較

```bash
python3 evaluation/evaluate_detections.py \
  "ground_truth/国宝さん手をあげる2_labels.csv" \
  --video "../data/raw/国宝さん手をあげる2.mp4" \
  --yolo11 "../results/movement_analysis.csv" \
  --event-type "HEAD_SHAKE" \
  --output "evaluation/results/国宝さん手をあげる2_yolo11_evaluation.md"
```

### 両方の手法を比較

```bash
python3 evaluation/evaluate_detections.py \
  "ground_truth/国宝さん手をあげる2_labels.csv" \
  --video "../data/raw/国宝さん手をあげる2.mp4" \
  --mediapipe "../data/raw/results/国宝さん手をあげる2_mediapipe.csv" \
  --yolo11 "../results/movement_analysis.csv" \
  --event-type "HEAD_SHAKE" \
  --output "evaluation/results/国宝さん手をあげる2_comparison.md"
```

### 評価指標の説明

#### フレーム単位の評価

- **Precision（精度）**: 検出したフレームのうち、正解だった割合
- **Recall（再現率）**: 正解フレームのうち、検出できた割合
- **F1-Score**: PrecisionとRecallの調和平均
- **Accuracy（正解率）**: 全フレームのうち、正しく分類できた割合

#### イベント単位の評価

- **IoU（Intersection over Union）**: イベント区間の重なり度合い
- **True Positive**: 正しく検出されたイベント数
- **False Positive**: 誤検出されたイベント数
- **False Negative**: 検出漏れしたイベント数

#### 時間的重なり

- **正解総時間**: 正解ラベルのイベント総時間
- **予測総時間**: 検出結果のイベント総時間
- **重なり時間**: 正解と予測が重なっている時間
- **重なり率**: 重なり時間 / 正解総時間

---

## トラブルシューティング

### ラベル付けツールが起動しない

**症状**: `動画ファイルを開けません` というエラー

**解決方法**:
- 動画ファイルのパスが正しいか確認
- 動画ファイルが存在するか確認
- OpenCVがインストールされているか確認: `pip install opencv-python`

### 矢印キーが反応しない

**症状**: `←` `→` キーで巻き戻し/早送りができない

**解決方法**:
- ターミナルによってキーコードが異なる場合があります
- 一時停止（`Space`）してから、動画ウィンドウをクリックして再度試してください

### 評価スクリプトでエラーが出る

**症状**: `ModuleNotFoundError: No module named 'result_loader'`

**解決方法**:
```bash
cd /Users/takahashiryoutarou/Desktop/国宝さん一覧/box/experiments/evaluation
python3 evaluate_detections.py ...
```

または、PYTHONPATHを設定:
```bash
export PYTHONPATH="/Users/takahashiryoutarou/Desktop/国宝さん一覧/box/experiments/evaluation:$PYTHONPATH"
```

### MediaPipe/YOLO11の結果が読み込めない

**症状**: `⚠️  MediaPipeの結果ファイルが見つかりません`

**解決方法**:
- ファイルパスが正しいか確認
- MediaPipe/YOLO11のスクリプトを実行して、結果CSVを生成してください

---

## 実行例

### 完全なワークフロー

```bash
# 1. experiments ディレクトリに移動
cd /Users/takahashiryoutarou/Desktop/国宝さん一覧/box/experiments

# 2. ラベル付けツールを実行
python3 video_labeling_tool.py \
  "../data/raw/国宝さん手をあげる2.mp4" \
  "ground_truth/国宝さん手をあげる2_labels.csv"

# 3. MediaPipeで検出（既に実行済みの場合はスキップ）
python3 mediapipe_head_detection.py \
  "../data/raw/国宝さん手をあげる2.mp4" \
  "results/国宝さん手をあげる2_mediapipe.mp4"

# 4. 評価を実行
python3 evaluation/evaluate_detections.py \
  "ground_truth/国宝さん手をあげる2_labels.csv" \
  --video "../data/raw/国宝さん手をあげる2.mp4" \
  --mediapipe "../data/raw/results/国宝さん手をあげる2_mediapipe.csv" \
  --event-type "HEAD_SHAKE" \
  --output "evaluation/results/国宝さん手をあげる2_evaluation.md"

# 5. 評価レポートを確認
cat evaluation/results/国宝さん手をあげる2_evaluation.md
```

---

## ヒント

### 効率的なラベル付け

1. **最初は通しで見る**: 動画を一度通しで見て、首振りの回数や特徴を把握
2. **2回目で記録**: 2回目の再生で実際にラベル付けを行う
3. **確認**: 最後にもう一度見直して、漏れがないか確認

### 正確なラベル付け

- 首振りの「開始」と「終了」を正確に記録することが重要
- 迷った場合は、少し広めに記録する（開始を早め、終了を遅めに）
- 一時停止（`Space`）を活用して、正確な位置を確認

### 複数の動画をラベル付け

```bash
# 国宝さん手をあげる1
python3 video_labeling_tool.py \
  "../data/raw/国宝さん手をあげる1.mp4" \
  "ground_truth/国宝さん手をあげる1_labels.csv"

# 国宝さん手をあげる2
python3 video_labeling_tool.py \
  "../data/raw/国宝さん手をあげる2.mp4" \
  "ground_truth/国宝さん手をあげる2_labels.csv"

# 国宝さん手をあげる3
python3 video_labeling_tool.py \
  "../data/raw/国宝さん手をあげる3.mp4" \
  "ground_truth/国宝さん手をあげる3_labels.csv"
```

---

## 参考資料

- [experiments/README.md](README.md): 外部レポジトリ実験の概要
- [experiments/QUICK_START.md](QUICK_START.md): 外部検出スクリプトのクイックスタート
- [experiments/EXTERNAL_COMPARISON_REPORT.md](EXTERNAL_COMPARISON_REPORT.md): 外部手法の比較レポート

