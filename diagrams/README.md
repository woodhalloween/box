# Mermaid図表ディレクトリ

このディレクトリには、プロジェクト内の各種処理フローやアルゴリズムを視覚化するためのMermaid記法によるフローチャートやダイアグラムが保存されています。

## ファイル一覧

- `detect_long_stay_flow.md` - 滞在時間検知スクリプト（detect_long_stay.py）の処理フロー図

## Mermaidについて

Mermaidは、テキストベースでダイアグラムを定義するための構文およびレンダリングツールです。GitHubやGitLab、Notion、VS Code（拡張機能）など多くの環境でサポートされています。

- [Mermaid公式サイト](https://mermaid.js.org/)
- [Mermaid Live Editor](https://mermaid.live/) - オンラインでMermaidコードを編集・プレビューできるツール

## 使用方法

各Markdown (.md) ファイルには、Mermaid記法のコードブロックが含まれています。GitHubなどのMermaidをサポートする環境で表示すると、自動的に図として描画されます。

ローカルで図を確認したい場合は、以下のいずれかの方法が利用できます：

1. VS Codeの場合: [Markdown Preview Mermaid Support](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid) 拡張機能をインストール
2. Webブラウザの場合: [Mermaid Live Editor](https://mermaid.live/) にコードをコピー&ペースト
3. その他のマークダウンエディタで、Mermaidをサポートしているものを利用 