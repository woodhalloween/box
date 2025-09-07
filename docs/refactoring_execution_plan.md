# `detect_joint_movement_with_hip_stay.py` リファクタリング実行計画と機能チェックリスト

## 1. はじめに

### 1.1. 目的

本ドキュメントは、`src/detect_joint_movement_with_hip_stay.py` のリファクタリングを安全かつ計画的に進めるための実行計画と、機能のデグレードを防ぐためのチェックリストを定義する。

リファクタリングの主な目的は以下の通り。

- **ビジネスロジックの明確化**: 新しい要件（車椅子利用者の検知、滞在時間に基づく店員への通知）をコードに反映し、各クラスの責務を明確にする。
- **保守性の向上**: 単一責任の原則に基づき、巨大なファイルを機能ごとに小さなモジュールへ分割する。
- **テスト容易性の向上**: 分割されたモジュールごとに独立した単体テストを作成・保守しやすくする。

### 1.2. 対象ファイル

- **ソースコード**: `src/detect_joint_movement_with_hip_stay.py`
- **テストコード**: `tests/unit/src/test_detect_joint_movement_with_hip_stay.py`

---

## 2. リファクタリング実行計画

リファクタリングは、**「小さなステップでの変更」**と**「各ステップ完了後のテスト実行」**を繰り返すことで、安全に進める。

### フェーズ1: 既存コードの分割とテストの追従

このフェーズでは、既存の機能を変えずに、コードを責務ごとにファイル分割することに注力する。

**Step 1.1: `KneeAngleMonitor` の分離**
- **作業内容**:
    1. `src/analysis` ディレクトリを作成する。
    2. `KneeAngleMonitor` クラスを `src/analysis/user_classifier.py` に移動し、クラス名を `UserClassifier` に変更する。
    3. `tests/unit/src/analysis` ディレクトリを作成する。
    4. `KneeAngleMonitor` に関連するテストを `tests/unit/src/test_detect_joint_movement_with_hip_stay.py` から `tests/unit/src/analysis/test_user_classifier.py` に移動し、新しいクラス名とパスに合わせて修正する。
    5. 元の `detect_joint_movement_with_hip_stay.py` では、新しい `UserClassifier` をインポートして使用するように修正する。
- **完了確認**: `poetry run pytest` を実行し、全てのテストがパスすること。

**Step 1.2: `HipBasedStayDetector` の分離**
- **作業内容**:
    1. `HipBasedStayDetector` クラスと関連するEnum/データクラスを `src/analysis/dwell_time_detector.py` に移動する。
    2. 関連するテストを `tests/unit/src/analysis/test_dwell_time_detector.py` に移動・修正する。
    3. 元のファイルで新しい `DwellTimeDetector` をインポートして使用するように修正する。
- **完了確認**: `poetry run pytest` を実行し、全てのテストがパスすること。

**Step 1.3: `PostureMonitor` の分離**
- **作業内容**:
    1. `PostureMonitor` クラスを `src/analysis/posture_monitor.py` に移動する。
    2. 関連するテストを `tests/unit/src/analysis/test_posture_monitor.py` に移動・修正する。
    3. 元のファイルで新しい `PostureMonitor` をインポートして使用するように修正する。
- **完了確認**: `poetry run pytest` を実行し、全てのテストがパスすること。

**Step 1.4: I/O関連処理の分離**
- **作業内容**:
    1. `src/io` ディレクトリを作成する。
    2. 描画関連の関数 (`draw_...`) を `src/io/video_drawer.py` に移動する。
    3. CSV書き出し関数 (`write_results_to_csv`) を `src/io/csv_writer.py` に移動する。
    4. 関連するテストがあれば `tests/unit/src/io/` 配下に移動・修正する（なければ、この機会に簡単なテストを追加する）。
- **完了確認**: `poetry run pytest` を実行し、全てのテストがパスすること。

**Step 1.5: メイン処理ループの分離**
- **作業内容**:
    1. ビデオ処理のメインループ (`process_video` 関数) を `src/main_processor.py` に移動する。
    2. `detect_joint_movement_with_hip_stay.py` は、最終的に `main_processor.py` を呼び出すためのエントリーポイントスクリプト（`if __name__ == "__main__":` のみ）となる。
    3. 統合テストを `tests/integration/` に作成し、`main_processor.py` がファイルI/Oから各分析モジュールの連携までを正しく実行できることを確認する。
- **完了確認**: `poetry run pytest` を実行し、全てのテスト（単体・統合）がパスすること。

### フェーズ2: 新ビジネスロジックの実装

フェーズ1でクリーンになったコードベースに対し、新しいビジネスロジックを追加する。

**Step 2.1: `UserClassifier` のロジック更新**
- **作業内容**:
    1. `UserClassifier` のロジックを「膝角度の移動平均が特定の閾値以下の場合に座位（車椅子利用者の可能性）と判定する」ように修正する。
    2. `tests/unit/src/analysis/test_user_classifier.py` に、立位・座位・姿勢変化のシナリオを追加し、分類ロジックを徹底的にテストする。
- **完了確認**: `test_user_classifier.py` のテストが全てパスすること。

**Step 2.2: 通知システムの導入**
- **作業内容**:
    1. `src/notification/notifier.py` を作成し、通知を行うクラス（例: `StaffNotifier`）を定義する。
    2. `main_processor.py` で、`UserClassifier` が座位と判定し、かつ `DwellTimeDetector` の滞在時間が閾値を超えた場合に `StaffNotifier` を呼び出すロジックを追加する。
    3. 統合テストで `StaffNotifier` をモックし、正しい条件で通知メソッドが呼び出されることを検証する。
- **完了確認**: `poetry run pytest` を実行し、全てのテストがパスすること。

---

## 3. 機能チェックリスト

リファクタリングの各ステップ後、以下の機能が損なわれていないことを既存のテストスイートで確認する。

| カテゴリ | 機能 | 確認方法 | ステータス |
| :--- | :--- | :--- | :--- |
| **コア機能** | **人物検出** | `PoseEstimator` の単体テストがパスする。 | `[ ]` |
| | **膝角度の計算** | `pose.utils` の単体テストがパスする。 | `[ ]` |
| | **滞在検知（腰基準）** | `DwellTimeDetector` (旧`HipBasedStayDetector`) の単体テストがパスする。 | `[ ]` |
| | **滞在時間の計測** | `DwellTimeDetector` の単体テストがパスする。 | `[ ]` |
| | **前傾姿勢の検知** | `PostureMonitor` の単体テストがパスする。 | `[ ]` |
| | **座位状態の判定** | `UserClassifier` (旧`KneeAngleMonitor`) の単体テストがパスする。 | `[ ]` |
| **I/O** | **ビデオファイルの読込み** | 統合テストでサンプルビデオが正常に処理される。 | `[ ]` |
| | **結果のCSV出力** | 統合テストで生成されたCSVの内容が期待通りである。 | `[ ]` |
| | **結果のビデオ描画** | 統合テストで生成されたビデオに出力が描画されている。 | `[ ]` |
| **ビジネス** | **通知トリガー** | 統合テストで、特定の条件（座位＋長時間滞在）で通知モジュールが呼び出されることをモックで確認する。（フェーズ2で実装） | `[ ]` |
