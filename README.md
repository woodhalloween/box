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
├── run_hand_raise.py        # エントリーポイント
├── src/
│   ├── run_hand_raise.py    # CLI と解析処理本体
│   ├── detectors/
│   │   └── hand_raise_detector.py
│   ├── analysis/
│   │   └── dwell_time_detector.py
│   ├── drawing_utils.py
│   ├── pose_estimator.py
│   └── ...
├── data/                    # 入力データ（例: 動画）
├── output/                  # 書き出された動画等
├── logs/
├── config.yaml              # 任意。しきい値設定
├── pyproject.toml
└── README.md
```

## 旧バージョンのスクリプト

`src/detect_joint_movement_with_hip_stay.py` や `archive/` 以下には、前傾検知や YOLO を用いた長期滞在検知など過去の実験的機能が含まれています。必要に応じて `poetry install --with tracking,head_pose` を実行し、ご自身の責任で利用してください。

## ライセンス

Copyright (c) 2025 Sibyl Inc.
