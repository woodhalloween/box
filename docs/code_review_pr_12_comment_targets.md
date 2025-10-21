# PR12 コメント対象の具体的な行番号

## 📍 コメント対象ファイルのサマリー

| 優先度 | ファイル | 行番号 | 内容 |
|--------|---------|--------|------|
| ✅実装済 | src/video_processor.py | 454 | show_progress パラメータ実装済み |
| 中 | src/video_processing/estimate_total_frames.py | 19, 45 | デフォルト推定値の設定 |
| 低 | src/video_processor.py | 224-229 | 過防御的チェックの削除 |
| 低 | src/io/csv_writer.py | 62-63 | 動的フィールド生成システム |

---

## 1️⃣ **estimate_total_frames.py の2箇所**

### 第1箇所
**ファイル:** `src/video_processing/estimate_total_frames.py`
**行番号:** 19行目

```python
18 |    if not show_progress:
19 |        return None  ← コメント対象
```

### 第2箇所
**行番号:** 45行目

```python
42 |    except Exception:
43 |        pass
44 |
45 |    return None  ← コメント対象
```

**改善提案コメント内容:**
```
💡 改善提案：デフォルト推定値を設定

現在、以下で None を返却するため進捗率表示ができません：
- L19: show_progress=False の場合
- L45: 例外発生時

改善案：
デフォルト推定値を返す（例：3600フレーム）

これでプログレスバーが常に進捗率を表示できます
```

---

## 2️⃣ **video_processor.py の過防御的チェック**

**ファイル:** `src/video_processor.py`
**行番号:** 224-229行目

```python
224 |        hand_statuses = None
225 |        if hasattr(self, "hand_raise_detector") and self.hand_raise_detector:
226 |            try:
227 |                hand_statuses = self.hand_raise_detector.detect(landmarks)
228 |            except Exception:
229 |                hand_statuses = None
```

**改善提案コメント内容:**
```
💡 改善提案：過防御的なチェック削除

問題点：
- L225: hasattr() チェックが過度
- L228: 例外を無視でデバッグ困難

理由：hand_raise_detector は _setup_modules()（L163-164）で常に初期化される

改善案：
以下のようにシンプルに：

hand_statuses = self.hand_raise_detector.detect(landmarks)

エラーが発生した場合は適切に例外が発生します
```

---

## 3️⃣ **csv_writer.py の拡張性**

**ファイル:** `src/io/csv_writer.py`
**行番号:** 62-63行目 及び 21-76行全体

```python
21 |def setup_csv_writer(csv_file: IO) -> csv.DictWriter:
22 |    """CSVライターをセットアップする"""
23 |    fieldnames = [
...
62 |        "left_hand_raised",    ← コメント対象
63 |        "right_hand_raised",   ← コメント対象
...
76 |    ]
```

**改善提案コメント内容:**
```
💡 改善提案：動的フィールド生成システム

現在の問題：
新しい検知器追加のたびに setup_csv_writer() を修正する必要

改善案：
1. 基底クラスに get_csv_field_names() メソッド追加
2. setup_csv_writer でこれを呼び出す

例：
def setup_csv_writer(csv_file: IO, detectors) -> csv.DictWriter:
    fieldnames = []
    for detector in detectors:
        fieldnames.extend(detector.get_csv_field_names())

これで新検知器追加時に csv_writer.py 修正が不要
```

---

## 4️⃣ **video_processor.py の show_progress**

**ファイル:** `src/video_processor.py`
**行番号:** 454行目

```python
454 |    show_progress: bool = False,  # Whether to show tqdm progress bar
```

**コメント内容:**
```
✅ 良い点：既に実装済み！

process_video() に show_progress パラメータが追加されており、
ユーザー向けAPIで進捗表示が制御可能です

- L454: パラメータ定義
- L365: run_pipeline()で使用
```

---

## 🎯 GitHub コメント方法

### 手順
1. PR12の "Files changed" タブをクリック
2. コメント対象ファイルを探す
3. 行番号の横の "+" アイコンをクリック
4. コメント内容を入力して "Comment" をクリック

### 複数行コメント（Span Comment）
1. 開始行番号をクリック
2. Shift を押しながら終了行番号をクリック
3. "+" アイコンをクリック
