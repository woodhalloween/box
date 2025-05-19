#!/bin/bash

# 開発用の依存関係をインストール
echo "開発用の依存関係をインストール中..."
pip install -r requirements-dev.txt

# テストを実行
echo "テストを実行中..."
python -m pytest tests/ -v --cov=scripts

# カバレッジレポートを表示
echo "カバレッジレポート:"
python -m pytest tests/ --cov=scripts --cov-report=term-missing 