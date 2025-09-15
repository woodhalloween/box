#!/bin/bash
# プロジェクトのルートディレクトリにいることを確認
# (このスクリプトがプロジェクトルートから実行されることを想定)

echo "テストを実行中..."
poetry run pytest --cov=src --cov-report=html --cov-report=term tests/

echo "カバレッジレポート:"
poetry run pytest --cov=src --cov-report=term-missing tests/ 