# `detect_joint_movement_with_hip_stay.py` リファクタリング計画書

## 1. 目的

`src/detect_joint_movement_with_hip_stay.py` は、単一ファイル内に複数の責務（データ構造定義、分析ロジック、ファイルI/O、描画処理、メイン処理）が混在しており、単一責任の原則に違反している。
このリファクタリングでは、責務ごとにファイルを分割し、コードの保守性、可読性、再利用性を向上させることを目的とする。

## 2. ディレクトリ構成の変更案

```
src/
├── analysis/
│   ├── __init__.py
│   ├── hip_stay_detector.py      # HipBasedStayDetector, HipStayState, HipStayInfo を移動
│   ├── knee_angle_monitor.py     # KneeAngleMonitor を移動
│   └── posture_monitor.py        # PostureMonitor, PostureSnapshot を移動
├── video_processing/
│   ├── __init__.py
│   ├── drawing.py                # 描画関連の関数を移動・統合
│   └── io.py                     # CSV/ビデオの読み書き関連関数を移動
├── __init__.py
├── definitions.py
├── drawing_utils.py              # video_processing/drawing.py に統合後、削除検討
├── head_shake_detector.py
├── movement_analyzer.py
├── pose/
├── pose_estimator.py
└── run_integrated_analysis.py    # 新しいメイン実行スクリプト
```

### 2.1. 各ファイルの責務

- **`src/analysis/hip_stay_detector.py`**: 人物の腰の位置を基準とし、移動が少ない場合に「滞在」として検知するロジックを管理する。
- **`src/analysis/knee_angle_monitor.py`**: 膝の角度を時系列で監視し、特定の閾値を下回る状態が続いた場合に通知するロジックを管理する。
- **`src/analysis/posture_monitor.py`**: 体の傾きや関節の状態から「前傾姿勢」を判定し、その状態が長時間続いた場合にアラートを出すロジックを管理する。
- **`src/video_processing/drawing.py`**: 分析結果（骨格、関節角度、アラートメッセージ等）をビデオフレーム上に描画するための関数群を提供する。
- **`src/video_processing/io.py`**: ビデオファイルの読み込み、分析結果のCSVファイルへの書き出し、注釈付きビデオの書き出しといった入出力処理を担当する。
- **`src/run_integrated_analysis.py`**: リファクタリングによって分割された各モジュール（分析、描画、I/O）を統括し、ビデオ解析の処理フロー全体を実行するメインスクリプト。

## 3. リファクタリング手順

### Step 1: 分析ロジックの分離

`PostureMonitor`, `HipBasedStayDetector`, `KneeAngleMonitor` および関連するデータクラスを `src/analysis/` ディレクトリ配下の各ファイルに移動する。

- **`src/analysis/posture_monitor.py`**:
  - `PostureSnapshot` データクラス
  - `PostureMonitor` クラス
- **`src/analysis/hip_stay_detector.py`**:
  - `HipStayInfo` データクラス
  - `HipStayState` Enum
  - `HipBasedStayDetector` クラス
- **`src/analysis/knee_angle_monitor.py`**:
  - `KneeAngleMonitor` クラス

### Step 2: ビデオ関連処理の分離

ファイル入出力と描画に関する関数を `src/video_processing/` ディレクトリに分離する。

- **`src/video_processing/io.py`**:
  - `setup_csv_writer`
  - `setup_video_writer`
  - `write_results_to_csv`
- **`src/video_processing/drawing.py`**:
  - `draw_posture_alerts`
  - `draw_hip_stay_info`
  - `draw_head_shake_info`
  - 既存の `src/drawing_utils.py` の内容もレビューし、こちらに統合する。

### Step 3: メインスクリプトの再構築

1. 新しいメインスクリプト `src/run_integrated_analysis.py` を作成する。
2. `detect_joint_movement_with_hip_stay.py` 内の `process_video` 関数と `main` 関数を移植する。
3. 分離したモジュール（analysis, video_processing）をインポートし、処理フローを再構築する。
4. `process_video` は、各分析器やプロセッサを初期化し、ビデオフレームのループ内でそれらを呼び出すという、処理全体の統括に専念する形に整理する。

### Step 4: クリーンアップ

1. すべての機能が `src/run_integrated_analysis.py` と新モジュールに移行したことを確認する。
2. 元の `src/detect_joint_movement_with_hip_stay.py` ファイルを削除する。
3. `src/drawing_utils.py` が `src/video_processing/drawing.py` に完全に統合された場合、`src/drawing_utils.py` を削除する。
4. すべてのファイルのインポートパスが正しく更新されていることを確認する。

## 4. 期待される効果

- **保守性の向上**: 各機能が独立したファイルに分割されるため、修正箇所を特定しやすくなる。
- **可読性の向上**: 1ファイルあたりのコード量が減り、各ファイルの責務が明確になるため、コードが理解しやすくなる。
- **再利用性の向上**: `PostureMonitor` や `HipBasedStayDetector` などの分析モジュールを、他のスクリプトから容易に再利用できるようになる。
- **テスト容易性の向上**: 機能ごとにファイルが分かれているため、単体テストが書きやすくなる。
