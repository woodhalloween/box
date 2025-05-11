# コード品質管理システム

このドキュメントでは、プロジェクトに導入したコード品質管理システムのフロー図を示します。

## システム構成図

```mermaid
flowchart TD
    subgraph 開発環境["ローカル開発環境"]
        IDE["VS Code"]
        Code["Python コード"]
        IDE -- "編集" --> Code
        Code -- "保存" --> Ruff_local["Ruff (ローカル)"]
        Ruff_local -- "リント・フォーマット" --> Code
        
        subgraph 設定ファイル["設定ファイル"]
            pyproject["pyproject.toml<br>- Ruff の設定<br>- Poetry 依存関係"]
            vscode_settings[".vscode/settings.json<br>- VS Code と Ruff の連携設定"]
            actions_yml[".github/workflows/python-ci.yml<br>- CI パイプライン定義"]
        end
    end
    
    subgraph GitHub["GitHub リポジトリ"]
        Git["コード リポジトリ"]
        
        subgraph Actions["GitHub Actions"]
            CI["CI パイプライン"]
            
            subgraph jobs["ジョブ"]
                lint["lint ジョブ<br>1. リポジトリをチェックアウト<br>2. Python 3.11 をセットアップ<br>3. Poetry をインストール<br>4. 依存関係をインストール<br>5. Ruff でリントチェック<br>6. Ruff でフォーマットチェック"]
                test["test ジョブ<br>1. リポジトリをチェックアウト<br>2. Python 3.11 をセットアップ<br>3. Poetry をインストール<br>4. 依存関係をインストール<br>5. pytest でテスト実行"]
            end
            
            CI --> lint
            CI --> test
        end
    end
    
    Code -- "git コミット & プッシュ" --> Git
    Git -- "トリガー" --> CI
    pyproject -- "設定を参照" --> Ruff_local
    pyproject -- "設定を参照" --> lint
    vscode_settings -- "設定を参照" --> IDE
    actions_yml -- "設定を参照" --> CI
```

## ワークフローシーケンス図

```mermaid
sequenceDiagram
    actor Dev as 開発者
    participant Editor as VS Code
    participant Git as Gitリポジトリ
    participant Actions as GitHub Actions
    
    Dev->>Editor: コード編集
    Dev->>Editor: 保存
    activate Editor
    Editor->>Editor: Ruffによるリント・フォーマット<br>(settings.jsonの設定に基づく)
    Note over Editor: source.fixAll<br>source.organizeImports
    deactivate Editor
    
    Dev->>Git: コミット & プッシュ
    Git->>Actions: プッシュイベント発生
    activate Actions
    
    par Lint ジョブ
        Actions->>Actions: リポジトリをチェックアウト
        Actions->>Actions: Python 3.11 をセットアップ
        Actions->>Actions: Poetry をインストール
        Actions->>Actions: 依存関係をインストール
        Actions->>Actions: リントチェック<br>poetry run ruff check .
        Actions->>Actions: フォーマットチェック<br>poetry run ruff format --check .
    and Test ジョブ
        Actions->>Actions: リポジトリをチェックアウト
        Actions->>Actions: Python 3.11 をセットアップ
        Actions->>Actions: Poetry をインストール
        Actions->>Actions: 依存関係をインストール
        Note over Actions: testsディレクトリがある場合のみ
        Actions->>Actions: テスト実行<br>poetry run pytest
    end
    
    Actions-->>Dev: CI結果通知
    deactivate Actions
```

## システム概要

このシステムは主に3つの重要な設定から構成されています：

1. **Ruff の設定** (`pyproject.toml`):
   - Python コードのリントとフォーマットのルールを定義
   - 最大行長、対象 Python バージョン、有効なリントルールを設定
   - コードスタイルの一貫性を保証

2. **GitHub Actions ワークフロー** (`.github/workflows/python-ci.yml`):
   - コードがプッシュされるたびに自動的に実行される CI パイプライン
   - リントとフォーマットチェックを行う **lint** ジョブ
   - 自動テストを実行する **test** ジョブ
   - 問題がある場合は早期に発見し通知

3. **VS Code の設定** (`.vscode/settings.json`):
   - ローカル開発環境での Ruff の挙動を定義
   - ファイル保存時の自動フォーマットと修正を設定
   - エディタ上でのリアルタイムフィードバックを提供

これにより、コードの品質を常に高い水準に維持するための継続的な検証システムが実現されています。開発中はリアルタイムにフィードバックを受け取り、コミット時には自動的に検証が行われます。 