# Human Activity Analyzer

このプロジェクトは、動画内の人物の骨格ランドマークを解析し、挙手や滞在時間などの状態を検出・可視化するシステムです。MediaPipe Pose を用いた骨格推定結果に対して複数の解析器を組み合わせ、連続フレームの判定を通じて安定した検出結果を提供します。

## 主な機能

- **骨格推定 (MediaPipe Pose)**: 33 点ランドマークから姿勢を取得し、後段の解析に利用します。
- **挙手検出**: 肩と手首のランドマークをもとに「手が肩より上にある状態」を連続フレーム数で評価し、誤検出を抑制します。
- **滞在時間検出**: 腰位置の軌跡を追跡し、指定範囲内で留まり続けた場合に長期滞在としてアラートを生成します。スパイク検出や正規化など高度な判定にも対応します。
- **可視化とエクスポート**: 元映像へ検出結果や骨格を描画しつつ、フレームごとのステータスを CSV に出力します。
- **設定ファイル対応**: `config.yaml` から各種しきい値や動作を上書き可能です。存在しない場合は安全なデフォルト値が使用されます。

## セットアップ

依存関係は [Poetry](https://python-poetry.org/) で管理しています。Poetry が未インストールの場合は公式手順に従って導入してください。

```bash
# 解析に必要な依存関係をインストール
poetry install --with tracking,head_pose
```

## 使用方法

メインの解析スクリプトは `run_hand_raise.py` です。

```bash
poetry run python run_hand_raise.py --input data/videos/your_video.mp4
```

実行すると以下の処理が行われます。

- 指定動画をフレーム単位で読み込み、骨格推定と解析を実施
- 挙手状態および滞在ステータスのテキストをフレームへ描画
- 必要に応じてウィンドウ表示・処理済み動画の書き出し・CSV 出力を実行

### 主なオプション

- `--output_video PATH` : 処理済み動画の出力先を指定（未指定時は `<入力名>_processed.mp4`）。
- `--output_csv PATH` : CSV 出力先を指定（未指定時は `<入力名>_results.csv`）。
- `--no_display` : ウィンドウ表示を無効化。
- `--no_video_output` : 動画の書き出しを無効化。
- `--no_csv_output` : CSV の書き出しを無効化。
- `--draw_skeleton` : ランドマークと骨格線を描画。

### 設定ファイル (`config.yaml`)

`config.yaml` が存在する場合、以下のキーでしきい値を上書きできます。いずれも未設定時はデフォルト値が使用されます。

- `hand_raise.visibility_threshold`
- `hand_raise.min_consecutive_frames`
- `dwell_time_detector.stay_threshold_sec`
- `dwell_time_detector.confidence_threshold`
- `dwell_time_detector.advanced_detection.spike_threshold`
- `dwell_time_detector.advanced_detection.stability_threshold_px`
- `dwell_time_detector.advanced_detection.grace_period_sec`
- `dwell_time_detector.use_normalization`
- `dwell_time_detector.normalization_base`

## 出力

- 処理済み動画 (`*_processed.mp4`) : 描画済みフレームを動画として保存。
- 結果 CSV (`*_results.csv`) : フレーム番号、タイムスタンプ、左右挙手のフラグ、滞在検出の内部状態などを記録。
- ログ : `logs/` 配下に各種ログが保存される場合があります。

## プロジェクト構造

```
human-activity-analyzer/
├── run_hand_raise.py                # エントリーポイント
├── src/
│   ├── run_hand_raise.py            # CLI と解析処理本体（挙手＋滞在）
│   ├── main_detector.py             # 統合版エントリ（分類/滞在/姿勢）
│   ├── video_processor.py           # 統合パイプライン制御
│   ├── pose_estimator.py            # MediaPipe Pose のラッパー
│   ├── drawing_utils.py             # 高レベル描画
│   ├── movement_analyzer.py         # 動き解析の補助
│   ├── head_shake_detector.py       # 首振り検出
│   ├── detectors/
│   │   └── hand_raise_detector.py   # 挙手検出（連続フレーム安定化）
│   ├── analysis/
│   │   ├── dwell_time_detector.py   # 滞在検出コア
│   │   ├── posture_monitor.py       # 前傾割合監視
│   │   └── user_classifier.py       # 膝角のユーザー分類
│   ├── io/
│   │   ├── csv_writer.py            # 結果CSV出力
│   │   └── drawing.py               # 低レベル描画
│   ├── pose/
│   │   ├── definitions.py           # ランドマーク定義/骨格
│   │   └── utils.py                 # 可視性・正規化ユーティリティ
│   ├── config.py                    # 設定読み込み
│   ├── definitions.py               # 共通定義/型
│   └── io_utils.py                  # I/O ユーティリティ
├── tests/                           # 単体・結合テストとダミー資産
├── data/                            # 入力データ（例: 動画）
├── output/                          # 書き出された動画等
├── results/                         # 解析結果（CSV/動画）
├── logs/
├── models/                          # 学習済み重み（任意）
├── weights/                         # 同上
├── docs/                            # 設計・試験ドキュメント
├── diagrams/                        # フロー図・構成図
├── config.yaml                      # 任意。しきい値設定
├── pyproject.toml
└── README.md
```

## 実行手順

### run_hand_raise.py（挙手＋滞在のシンプル実行）

1. 依存関係をインストール

```bash
poetry install --with tracking,head_pose
```

2. 入力動画を用意して実行

```bash
# ルート直下のエントリスクリプトを実行
poetry run python run_hand_raise.py --input data/videos/your_video.mp4 --draw_skeleton

# 表示をオフにしたい場合
poetry run python run_hand_raise.py --input data/videos/your_video.mp4 --no_display
```

3. 出力
- 動画: デフォルトで `<入力名>_processed.mp4`（同ディレクトリ）
- CSV: デフォルトで `<入力名>_results.csv`
- 実行中に表示ウィンドウで `q` で中断

ヒント: `config.yaml` があれば `hand_raise.*` と `dwell_time_detector.*` の値でしきい値を上書きできます。

#### 主なオプション（実装準拠）

```text
必須:
  --input PATH

任意:
  --output_video PATH     処理済み動画の出力先
  --output_csv PATH       結果CSVの出力先
  --no_display            ウィンドウ表示を無効化
  --no_video_output       動画の書き出しを無効化
  --no_csv_output         CSV の書き出しを無効化
  --draw_skeleton         ランドマークと骨格線を描画
```

追加例:

```bash
# 出力先を明示し、表示を無効化（バッチ処理向け）
poetry run python run_hand_raise.py \
  --input data/videos/your_video.mp4 \
  --output_video output/your_video_processed.mp4 \
  --output_csv output/your_video_results.csv \
  --no_display

# 解析中の骨格を重ねて確認
poetry run python run_hand_raise.py --input data/videos/your_video.mp4 --draw_skeleton
```

### main_detector.py（統合版：分類/滞在/姿勢）

1. 依存関係をインストール（上と同じ）

```bash
poetry install --with tracking,head_pose
```

2. モジュールとして実行（相対インポートに対応）

```bash
poetry run python -m src.main_detector \
  --video data/videos/your_video.mp4 \
  --output-video output/processed.mp4 \
  --output-csv output/results.csv
```

3. 主な調整項目（例）
- 滞在検出: `--stay-threshold`, `--spike-threshold`, `--stability-threshold`, `--grace-period`
- 姿勢監視: `--pm-duration`, `--pm-threshold`
- ユーザー分類: `--uc-threshold`, `--uc-window`

ヒント: `config.yaml` があれば上記のデフォルト値は自動的に読み込まれます（該当キーがある場合）。

注意: `src.main_detector` は相対インポート（`from .config import config`）を使用しているため、`-m` でモジュール実行してください。

#### 主なオプション（実装準拠）

```text
必須:
  --video PATH

任意（出力/表示など）:
  --output-csv PATH       結果CSVの出力先
  --output-video PATH     処理済み動画の出力先
  --disable-japanese      描画テキストを英語化

任意（滞在検出）:
  --stay-threshold FLOAT
  --spike-threshold FLOAT
  --stability-threshold FLOAT
  --grace-period FLOAT

任意（姿勢監視）:
  --pm-duration FLOAT
  --pm-threshold FLOAT

任意（ユーザー分類）:
  --uc-threshold FLOAT
  --uc-window INT
```

追加例:

```bash
# 英語UIでしきい値を調整しつつ実行
poetry run python -m src.main_detector \
  --video data/videos/your_video.mp4 \
  --output-video output/processed.mp4 \
  --output-csv output/results.csv \
  --disable-japanese \
  --stay-threshold 12.0 \
  --spike-threshold 1.8 \
  --stability-threshold 60.0 \
  --grace-period 2.0 \
  --pm-duration 90.0 \
  --pm-threshold 0.6 \
  --uc-threshold 95.0 \
  --uc-window 7
```

設定キー（`config.yaml` 例）:

```yaml
dwell_time_detector:
  stay_threshold_sec: 10.0
  advanced_detection:
    spike_threshold: 1.5
    stability_threshold_px: 50.0
    grace_period_sec: 1.5
posture_monitor:
  monitoring_duration_sec: 60.0
  alert_threshold_ratio: 0.7
user_classifier:
  threshold_deg: 90.0
  moving_window_seconds: 5
```

## 旧バージョンのスクリプト

`src/detect_joint_movement_with_hip_stay.py` や `archive/` 以下には、前傾検知や YOLO を用いた長期滞在検知など過去の実験的機能が含まれています。必要に応じて `poetry install --with tracking,head_pose` を実行し、ご自身の責任で利用してください。

## ライセンス

Copyright (c) 2025 Sibyl Inc.
