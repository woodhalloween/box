# テストカバレッジ改善計画書

本ドキュメントは、コードレビューにて指摘されたテストカバレッジの不足箇所を解消するための計画を定めるものです。

---

### `src/movement_analyzer.py`

| No. | 分類 | 指摘内容 | 対応方針 | 優先度 | 完了 |
|:---:|:---|:---|:---|:---:|:---:|
| 1 | 🧪テスト | else 分岐が未テスト | `head_horizontal_angle <= 0` となるデータでテストし、`HEAD_LEFT_TURN` が返ることを確認します。 | 高 | ☐ |
| 2 | 🧪テスト | `if Angle.HEAD_VERTICAL_NOD...` の if 文全体が未テスト | `previous_angles` に `HEAD_VERTICAL_NOD` が含まれる場合と含まれない場合の両方をテストします。 | 高 | ☐ |
| 3 | ♻️リファクタリング<br>🧪テスト | `horizontal_angle` の正規化処理が未テストかつ共通化可能 | 角度を -180 ~ +180 に正規化する共通関数を作成し、重複コードを置き換えます。その上で、新関数をテストします。 | 高 | ☐ |
| 4 | ♻️リファクタリング | `vertical_angle` の正規化処理が共通化可能 | No.3 で作成した共通関数を使用するように修正します。 | 高 | ☐ |
| 5 | 🧪テスト | except ブロックが未テスト | `IndexError`, `TypeError`, `ZeroDivisionError` が発生する不正なデータを渡し、例外が捕捉されることを確認します。 | 高 | ☐ |

### `src/head_shake_detector.py`

| No. | 分類 | 指摘内容 | 対応方針 | 優先度 | 完了 |
|:---:|:---|:---|:---|:---:|:---:|
| 6 | 🧪テスト | `if avg_confidence...` の if 文が未テスト | 平均信頼度がしきい値を下回る場合と上回る場合の両方をテストします。 | 高 | ☐ |
| 7 | 🧪テスト | `if ear_dist < 1e-6:` の if 文が未テスト | `ear_dist` がゼロに近い微小な値になるデータを渡し、この分岐に入ることを確認します。 | 高 | ☐ |
| 8 | 🧪テスト | `if neck_length_approx < 1e-6:` の if 文が未テスト | `neck_length_approx` がゼロに近い微小な値になるデータを渡し、この分岐に入ることを確認します。 | 高 | ☐ |
| 9 | 🧪テスト | except 文が未テスト | `IndexError`, `TypeError`, `ZeroDivisionError` が発生する不正なデータを渡し、例外が捕捉されることを確認します。 | 高 | ☐ |
| 10 | 🧪テスト | `if len(values) < ...` の if 文が未テスト | `values` の要素数が条件を満たす場合と満たさない場合の両方をテストします。 | 高 | ☐ |
| 11 | 🧪テスト | `if landmarks is None:` の if 文が未テスト | `landmarks` に `None` を渡して、早期 return されることを確認します。 | 高 | ☐ |
| 12 | 🧪テスト | `_analyze_horizontal_movement` 関数全体が未テスト | 内部状態をセットアップした後、この関数が期待される `MovementState` を返すことを確認します。 | 高 | ☐ |
| 13 | 🧪テスト | `_analyze_vertical_movement` 関数全体が未テスト | 内部状態をセットアップした後、この関数が期待される `MovementState` を返すことを確認します。 | 高 | ☐ |
| 14 | 🧪テスト | `check_alerts` 関数全体が未テスト | アラートが生成される状態とされない状態をシミュレートし、戻り値を確認します。 | 高 | ☐ |
| 15 | 🧪テスト | `get_status` 関数全体が未テスト | オブジェクトの内部状態が正しく辞書として返されることを確認します。 | 高 | ☐ |

### `src/detect_joint_movement_with_hip_stay.py`

| No. | 分類 | 指摘内容 | 対応方針 | 優先度 | 完了 |
|:---:|:---|:---|:---|:---:|:---:|
| 16 | ♻️リファクタリング | `PostureSnapshot` クラスをファイル分離 | `src/classes/posture_snapshot.py` のようなファイルにクラスを移動し、`import` を修正します。 | 高 | ☐ |
| 17 | ♻️リファクタリング | `HipStayInfo` クラスをファイル分離 | `src/classes/hip_stay_info.py` のようなファイルにクラスを移動し、`import` を修正します。 | 高 | ☐ |
| 18 | ♻️リファクタリング | `PostureMonitor` クラスをファイル分離 | `src/classes/posture_monitor.py` のようなファイルにクラスを移動し、`import` を修正します。 | 高 | ☐ |
| 19 | 🧪テスト | `is_leaning` の return 部が未テスト | `forward_score` が 0.5 を超える場合と超えない場合で、返り値が正しいことを確認します。 | 高 | ✅ |
| 20 | 🧪テスト | `if not self.posture_history:` の if 文内が未テスト | `posture_history` が空の状態で関数を呼び出し、早期 return されることを確認します。 | 高 | ✅ |
| 21 | ♻️リファクタリング | `HipStayState` Enum をファイル分離 | `src/classes/hip_stay_state.py` のようなファイルに Enum を移動し、`import` を修正します。 | 高 | ☐ |
| 22 | ♻️リファクタリング | `HipBasedStayDetector` クラスをファイル分離 | `src/classes/hip_based_stay_detector.py` のようなファイルにクラスを移動し、`import` を修正します。 | 高 | ☐ |
| 23 | 🧪テスト | `visible` 内の except 句 (`TypeError`, `ValueError`) が未テスト | `pt[3]` に `float` 変換できない値（文字列など）を渡し、例外が捕捉され `False` が返ることを確認します。 | 高 | ✅ |
| 24 | 🧪テスト | `if not (visible(l_sh) and visible(r_sh)):` の if 文内が未テスト | 肩のランドマークが visible でない（信頼度が低い）データを渡し、この分岐に入ることを確認します。 | 高 | ✅ |
| 25 | 🧪テスト | `torso_len` 計算の except 文内が未テスト | `landmarks` が不正な形式（`IndexError`）や `None` （`TypeError`）の場合に `None` が返ることを確認します。 | 高 | ✅ |
| 26 | 🧪テスト | `draw_posture_alerts` 関数全体が未テスト | ダミーの画像データ等を渡し、関数がエラーなく実行されることを確認します。 | 中 | ✅ |
| 27 | 🧪テスト | `draw_hip_stay_info` 関数全体が未テスト | ダミーの画像データ等を渡し、関数がエラーなく実行されることを確認します。 | 中 | ✅ |
| 28 | 🧪テスト | `draw_head_shake_info` 関数全体が未テスト | ダミーの画像データ等を渡し、関数がエラーなく実行されることを確認します。 | 中 | ✅ |
| 29 | 🧪テスト | `process_video` 関数全体が未テスト | 関連関数をモック化し、一連の処理がエラーなく呼び出されることを確認します。 | 中 | ✅ |
| 30 | 🧪テスト | `main` 関数全体が未テスト | コマンドライン引数をシミュレートし、`process_video` が呼び出されることを確認します。 | 低 | ✅ |