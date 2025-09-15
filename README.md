# Human Activity Analyzer

このプロジェクトは、動画内の人物の行動を分析するためのシステムです。骨格推定技術を用いて関節の動きを詳細に分析し、特定の行動パターン（前傾姿勢、長期滞ade在、しゃがみ込みなど）を検知することを目的としています。

## 機能

現在のバージョンでは、`src/detect_joint_movement_with_hip_stay.py` スクリプトを通じて以下の機能を提供します。

-   **骨格推定**: MediaPipeを利用して、動画内の人物の主要な関節（肩、肘、腰、膝など）の位置を検出します。
-   **関節角度・状態分析**: 検出した骨格情報から、各関節の角度をリアルタイムに計算し、動きの状態（静止、屈曲、伸展）を判定します。
-   **前傾姿勢検知**: 体幹の傾きや首の角度から前傾姿勢をスコアリングし、一定時間以上継続した場合にアラートを出力します。
-   **腰基準の長期滞在検知**: 腰の中心位置を追跡し、指定された範囲内での移動が少ない場合に「長期滞在」として検知します。移動距離は人物の大きさ（体幹長など）に基づいて正規化することも可能です。
-   **膝角度の監視**: 膝の角度が一定以下になった状態（しゃがみ込みなど）を検知し、通知します。
-   **結果の可視化と出力**: 分析結果（骨格、関節角度、各種アラート）を元動画に描画して保存すると同時に、フレームごとの詳細な分析データをCSVファイルに出力します。

## セットアップ

本プロジェクトは [Poetry](https://python-poetry.org/) を使用して依存関係を管理しています。

```bash
# Poetryがインストールされていない場合はインストール
# (https://python-poetry.org/docs/#installation)

# 必要なライブラリグループをインストール
poetry install --with tracking,head_pose
```

## 使用方法

### 関節運動と長期滞在の統合分析

`src/detect_joint_movement_with_hip_stay.py` を使用して、動画の統合的な分析を実行します。

```bash
poetry run python src/detect_joint_movement_with_hip_stay.py --video [ビデオファイルのパス]
```

#### 主なオプション

-   `--output-csv [パス]`: 分析結果を保存するCSVファイルのパスを指定します。
-   `--output-video [パス]`: 分析結果を描画した動画の保存先パスを指定します。
-   `--hip-move-threshold [ピクセル数]`: 「移動」と判定する腰の移動距離の閾値をピクセル単位で指定します。（デフォルト: 25.0）
-   `--hip-stay-threshold [秒数]`: 「長期滞在」と判定する時間の閾値を秒単位で指定します。（デフォルト: 60.0）
-   `--hip-normalize`: このフラグを付けると、移動距離を人物のスケール（体幹長など）で正規化して判定します。これにより、カメラからの距離が変化しても安定した検知が可能になります。
-   `--hip-norm-base [torso|shoulder|screen]`: 正規化の基準となる体の部位を選択します。（デフォルト: torso）

#### 実行例

```bash
# 基本的な実行（結果はoutput/ディレクトリに保存されます）
poetry run python src/detect_joint_movement_with_hip_stay.py --video data/videos/your_video.mp4

# 正規化を有効にして、長期滞在の閾値を30秒に設定
poetry run python src/detect_joint_movement_with_hip_stay.py \
    --video data/videos/your_video.mp4 \
    --hip-normalize \
    --hip-stay-threshold 30.0
```

## プロジェクト構造

```
human-activity-analyzer/
├── .venv/                  # Python仮想環境 (Poetry管理)
├── archive/                # 旧バージョンのスクリプトや実験コード
├── data/
│   ├── videos/             # 入力ビデオ
│   └── ...
├── output/                 # スクリプトによる出力ファイル (処理済みビデオ, CSVなど)
├── src/                    # メインのソースコード
│   ├── pose/
│   │   └── definitions.py
│   ├── definitions.py
│   ├── detect_joint_movement_with_hip_stay.py  # 主要スクリプト
│   ├── drawing_utils.py
│   ├── movement_analyzer.py
│   └── pose_estimator.py
├── .gitignore
├── poetry.lock
├── pyproject.toml          # プロジェクト設定と依存関係の定義 (Poetry)
└── README.md
```

---

## 旧機能 (Archive)

### 長時間滞在検出 (YOLOベース)

YOLOv8を利用した物体追跡ベースの長時間滞在検出機能です。

```bash
# poetry install --with tracking を実行しておく必要があります
poetry run python archive/src/detect_long_stay_main.py --input data/videos/your_video.mp4
```

### 頭部姿勢検出

MediaPipeを利用した頭部姿勢の検出機能です。

```bash
# poetry install --with head_pose を実行しておく必要があります
poetry run python archive/scripts/landmarks/run_head_pose.py --video data/videos/your_video.mp4
```

## ライセンス

Copyright (c) 2025 Sibyl Inc. 