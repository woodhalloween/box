# 頭部姿勢検出システム

このプロジェクトは店舗内での顧客体験向上のための頭部姿勢検出システムです。来店客が店員を探す際の「首振り」動作を検出し、迅速な対応を可能にします。

## 機能

- リアルタイム頭部姿勢検出
- 首振り動作の自動検出
- 500ms以内の応答時間
- 検出精度の評価と記録

## セットアップ

```bash
# 依存関係のインストール
poetry install

# 頭部姿勢検出グループの依存関係をインストール
poetry install --with head_pose
```

## 使用方法

```bash
# 基本的な実行方法
poetry run python run_head_pose.py --video 動画ファイルのパス

# 出力ファイルを指定
poetry run python run_head_pose.py --video 動画ファイルのパス --output 出力ファイルのパス

# プレビューなしで実行
poetry run python run_head_pose.py --video 動画ファイルのパス --no-preview
```

## プロジェクト構造

```
box/
├── head_pose/            # 頭部姿勢検出モジュール
│   ├── __init__.py
│   ├── detector.py       # 頭部姿勢検出の実装
│   └── weights/          # モデルの重み（必要な場合）
├── tracking/             # 物体追跡モジュール（別機能）
│   ├── __init__.py
│   └── weights/          # 追跡モデルの重み
├── data/                 # データディレクトリ
│   ├── videos/           # 入力動画
│   └── output/           # 出力動画
├── logs/                 # ログファイル
├── pyproject.toml        # 依存関係の定義
└── README.md             # このファイル
```

## 性能評価

このシステムは以下の性能指標を目標としています：

- 応答時間: 500ms以内
- 検出精度 (True Positive Rate): 90%以上
- 誤検出率 (False Positive Rate): 5%以下
- F1スコア: 92%以上

## ライセンス

Copyright (c) 2025 Your Company 