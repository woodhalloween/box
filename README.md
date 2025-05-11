# human-activity-analyzer (旧: 頭部姿勢検出システム)

このプロジェクトは、動画内の人物の行動を分析するためのシステムです。
元々は店舗内での顧客体験向上のための頭部姿勢検出システムとして開発が開始されましたが、現在はより広範な人物行動分析、特に長時間滞在の検知機能を中心に開発を進めています。

## 機能

-   リアルタイム人物検出・追跡
-   指定時間以上同じ場所に留まる人物の検知と通知
-   頭部姿勢検出 (初期機能)
-   検出精度の評価と記録

## セットアップ

```bash
# 依存関係のインストール
poetry install

# (オプション) 頭部姿勢検出グループの依存関係をインストール
# poetry install --with head_pose
```

## 使用方法

### 長時間滞在検出スクリプトの実行例

```bash
# 基本的な実行 (デフォルト設定: YOLOv8nモデル使用、滞在閾値6秒、移動閾値20ピクセル)
poetry run python scripts/detect_long_stay.py --video data/videos/your_video.mp4 --output data/output/output_video.mp4 --log_dir logs/

# モデルや閾値を指定して実行
poetry run python scripts/detect_long_stay.py --video data/videos/your_video.mp4 --output data/output/output_video.mp4 --log_dir logs/ --yolo_model models/yolov8n-pose.pt --stay-threshold 10 --move-threshold 15
```

### (旧機能) 頭部姿勢検出スクリプトの実行例

```bash
# 基本的な実行方法
poetry run python scripts/run_head_pose.py --video data/videos/your_video.mp4

# 出力ファイルを指定
poetry run python scripts/run_head_pose.py --video data/videos/your_video.mp4 --output data/output/output_head_pose.mp4

# プレビューなしで実行
poetry run python scripts/run_head_pose.py --video data/videos/your_video.mp4 --no-preview
```

## プロジェクト構造の概要

```
human-activity-analyzer/
├── .git/                   # Gitリポジトリ管理用
├── .venv/                  # Python仮想環境 (Poetry管理)
├── config/                 # 設定ファイル (YAML定義書など)
│   ├── *.yaml
├── data/                   # データファイル
│   ├── videos/             # 入力ビデオファイル
│   ├── output/             # スクリプトによる出力ファイル (処理済みビデオ、ログなど)
│   ├── clips/              # (動画クリップなど、必要に応じて使用)
│   └── raw/                # (生データなど、必要に応じて使用)
├── diagrams/               # 設計図やフローチャート (Mermaidなど)
│   ├── detect_long_stay_flow.md
│   └── README.md
├── logs/                   # 各種処理のログファイル (CSV形式など)
│   └── *.csv
├── models/                 # 機械学習モデルファイル
│   ├── yolov8n.pt
│   ├── yolov8n-pose.pt
│   └── ... (*.ptファイル)
├── scripts/                # メインとなるPythonスクリプト群
│   ├── detect_long_stay.py # 主要な長時間滞在検出スクリプト
│   ├── exam-yolo11-pose-estimation.py # YOLOv11姿勢推定サンプル
│   ├── run_head_pose.py    # (旧)頭部姿勢検出スクリプト
│   └── ... (その他の分析・補助スクリプト)
├── src/                    # (将来的にモジュール化する場合のソースコード置き場候補)
├── やること/                 # (タスク管理用ディレクトリ、必要に応じて整理)
├── .gitignore              # Gitで無視するファイルを指定
├── poetry.lock             # 依存関係のロックファイル
├── pyproject.toml          # プロジェクト設定と依存関係の定義 (Poetry)
└── README.md               # このファイル (プロジェクトの説明書)
```

## 性能評価 (長時間滞在検出)

(長時間滞在検出機能に関する性能目標や評価結果をここに記述)

## 性能評価 (頭部姿勢検出 - 旧機能)

このシステムは以下の性能指標を目標としています：

-   応答時間: 500ms以内
-   検出精度 (True Positive Rate): 90%以上
-   誤検出率 (False Positive Rate): 5%以下
-   F1スコア: 92%以上

## ライセンス

Copyright (c) 2025 Sibyl Inc. (または適切な組織名/個人名) 