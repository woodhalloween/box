# PRコードレビューレポート

## 対象
- ブランチ: `feature/torso-detection`
- 最新2コミット: c4cacad (HEAD), 204a539
- 変更ファイル数: 8ファイル
- 追加・変更行数: +455, -6

---

## 📋 変更概要

### 主な追加機能
1. **体幹揺れ検知モジュール** (`TorsoSwayDetector`) の追加・簡素化
2. **体幹インジケーター描画** (`draw_torso_indicators`) の実装
3. **体幹揺れアラートのUI描画パイプライン** の統合
4. **CSV出力** に体幹揺れメトリクスを統合
5. **デザイン書** の作成・管理

---

## 🔍 コードレビュー詳細

### ✅ 良い点

#### 1. 良好なアーキテクチャ設計
- **関心の分離**: 検知ロジック (`TorsoSwayDetector`) と描画ロジック (`draw_torso_indicators`) が明確に分離
- **パイプライン統合**: `video_processor.py` で一貫性のあるデータフロー構築
- **後方互換性**: 既存パラメータを受け取りつつ、未使用のものは `**_compat` で吸収

#### 2. シンプルな実装
```python
# src/analysis/torso_sway_detector.py (L41-62)
# 状態機械(state machine)を使用した明確な判定ロジック
# - 「しきい値以上が連続 on_frames → ON」
# - 「しきい値未満が連続 off_frames → OFF」
```
- 複雑な周波数解析・平滑化なしで実装 → 保守性 ⬆️

#### 3. 包括的なテストカバレッジ
```
tests/unit/src/test_torso_sway_detector.py
✅ test_torso_sway_detector_detects_lateral_sine (PASSED)
✅ test_torso_sway_detector_ap_sine_detects (PASSED)
```
- 新機能に対するユニットテストが実装済み

#### 4. ドキュメント整備
- `docs/torso_sway_detection_design.md` で設計意図・フローチャート等が明記
- 日本語コメント充実で実装意図が明確

#### 5. UI/UX統合
```python
# src/io/drawing.py (L286-294)
# 体幹揺れアラートを画面下部に表示（重なり回避）
# カラーコード: 赤 (0, 0, 255)
```
- ユーザー向けの視覚的フィードバックが適切

---

### ⚠️ 改善が必要な点

#### 1. **[高優先度] Ruff指摘への対応**

```python
# src/analysis/torso_sway_detector.py (L1-4)
# ❌ 現在:
from typing import Dict  # 不要: Python 3.10+ では dict を使用

# ✅ 修正案:
# from typing import Dict を削除
# L72: Dict[str, bool] → dict[str, bool] に変更
```

**Ruff指摘:**
- UP006: `Dict` は廃止予定 → `dict` を使用
- UP009: `Dict[str, bool]` は廃止予定 → `dict[str, bool]` を使用
- F401: 未使用のインポート

**修正方法:**
```bash
python -m ruff check src/analysis/torso_sway_detector.py --fix
python -m ruff format src/analysis/torso_sway_detector.py
```

#### 2. **[中優先度] パラメータ管理の一貫性**

```python
# src/video_processor.py (L166-178)
# TorsoSwayDetector初期化で多数のパラメータをハードコード
self.sway_detector = TorsoSwayDetector(
    fps=self.fps,
    window_sec=8.0,           # ← ハードコード
    smooth_sec=0.5,           # ← ハードコード
    amp_th_lat=10.0,          # ← ハードコード
    ...
)
```

**問題点:** 
- `process_video()` 関数では `sway_*` パラメータを受け取る (L502-512)
- 一方、`VideoProcessor` クラスではハードコード
- 不整合: 設定がどちらで制御されるのか不明瞭

**改善案:**
```python
# config.yaml に追加
torso_sway:
  window_sec: 8.0
  smooth_sec: 0.5
  amp_th_lat: 10.0
  amp_th_ap: 8.0
  on_sec: 1.2
  off_sec: 0.7
```

#### 3. **[中優先度] エラーハンドリングの強化**

```python
# src/io/drawing.py (L155-156)
except (IndexError, TypeError):
    pass  # エラーが発生した場合は何もしない
```

**問題:** 静かにエラーを無視している → デバッグ困難

**改善案:**
```python
except (IndexError, TypeError) as e:
    import logging
    logging.debug(f"Error drawing torso indicators: {e}")
```

#### 4. **[中優先度] 型注釈の完全性**

```python
# src/video_processor.py (L237, L300, L398)
sway_flags  # 型注釈なし

# ✅ 修正案:
sway_flags: dict[str, bool] | None = ...
```

#### 5. **[低優先度] 重複コード**

```python
# src/video_processor.py (L237-261) と process_frame (L360-382)
# 同じロジック: 体幹揺れペイロード構築が2箇所に存在

# 推奨: 共通メソッドに抽出
def _prepare_torso_sway_payload(self, sway_flags: dict) -> dict:
    """体幹揺れペイロードを生成"""
    return {
        "lateral": {
            "amp": 0.0,
            "freq": 0.0,
            "cycles": 0,
            "sway": bool(sway_flags.get("lateral", False)),
            "level": "none",
        },
        "ap": {
            "amp": 0.0,
            "freq": 0.0,
            "cycles": 0,
            "sway": bool(sway_flags.get("ap", False)),
            "level": "none",
        },
    }
```

---

### 📊 コード品質指標

| 項目 | 状態 | 詳細 |
|------|------|------|
| Ruff Linter | ⚠️ 3件指摘 | `Dict` import未使用・型の廃止予定警告 |
| Ruff Formatter | ⚠️ 1ファイル | `torso_sway_detector.py` の再フォーマット必要 |
| テストカバレッジ | ✅ 良好 | 新機能に対するユニットテスト実装済み (2/2 PASSED) |
| 型注釈 | ⚠️ 部分的 | 一部の戻り値に型注釈欠落 |
| ドキュメント | ✅ 良好 | 設計書・フローチャート・Mermaid図が充実 |
| 関心の分離 | ✅ 優秀 | ロジック・描画・I/Oが明確に分離 |

---

## 🔧 修正実施ロードマップ

### 必須修正 (Pre-merge必須)

**Ruff指摘を解決:**
```bash
cd /Users/takahashiryoutarou/Desktop/国宝さん一覧/box
python -m ruff check src/analysis/torso_sway_detector.py --fix
python -m ruff format src/analysis/torso_sway_detector.py
```

**修正内容:**
1. `from typing import Dict` 行を削除
2. `Dict[str, bool]` → `dict[str, bool]` に変更

### 推奨修正 (Post-merge課題)

- [ ] **パラメータ管理の統一** → config.yaml 読み込み
- [ ] **重複コード抽出** → 共通メソッド化 (`_prepare_torso_sway_payload`)
- [ ] **エラーハンドリング** → logging追加
- [ ] **型注釈の完全化** → 残存する型注釈欠落を解消

---

## 🎯 マージ判定

### 総合評価: ✅ **条件付きマージ推奨**

**マージ前チェックリスト:**
- [ ] Ruff 指摘 3件を修正
- [ ] テスト実行確認 (pytest)
- [ ] コミット前 ruff check 再実行

**推奨コマンド:**
```bash
# 1. 必須修正を実施
python -m ruff check src/analysis/torso_sway_detector.py --fix
python -m ruff format src/analysis/torso_sway_detector.py

# 2. テスト実行確認
python -m pytest tests/unit/src/test_torso_sway_detector.py -v

# 3. 全体チェック
python -m ruff check src/
python -m ruff format src/ --check

# 4. ステージング
git add src/analysis/torso_sway_detector.py
git commit -m "Fix: ruff lint issues for torso sway detector (Dict→dict)"

# 5. マージ実行
git merge feature/torso-detection
```

---

## 📝 確認事項

| 項目 | 状態 | 注釈 |
|------|------|------|
| 機能要件の実装 | ✅ | 体幹揺れ検知・UI表示・CSV出力すべて完了 |
| ユニットテスト | ✅ | 2/2 PASSED |
| デザイン書 | ✅ | torso_sway_detection_design.md 作成 |
| UI/UXフィードバック | ✅ | 画面下部にアラート表示 |
| 設定管理一貫性 | ⚠️ | 改善推奨 (Post-merge) |
| 静的解析ツール対応 | 🔴 | 修正必須 (Pre-merge)

