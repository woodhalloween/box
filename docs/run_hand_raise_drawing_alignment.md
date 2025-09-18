# run_hand_raise 描画コード共通化方針

## 背景
`src/run_hand_raise.py` では、骨格描画や手挙げ状態、滞在状態のオーバーレイを直接扱っており、特に `draw_dwell_status` がスクリプト内に定義されています。一方で、既存の描画ヘルパーとして `src/drawing_utils.py` と `src/io/drawing.py` の2種類が存在します。今後、`run_hand_raise` における描画処理を共通化するため、どちらのヘルパーと統合すべきかを評価しました。

## 選定対象の比較
### `src/drawing_utils.py`
- 既に `run_hand_raise` から `draw_hand_raise_status`、`draw_japanese_text`、`draw_landmarks` が再利用されており、依存関係が自然に成立しています。【F:src/run_hand_raise.py†L12-L14】【F:src/drawing_utils.py†L220-L257】
- ファイル単体の責務は骨格・角度分析に関する描画であり、独立したユーティリティとして `run_hand_raise` の単体スクリプトに組み込みやすい構成です。
- `draw_dwell_status` を追加する場合も、既存のフォント描画や色設定のヘルパーを流用でき、最小限の依存追加で済みます。

### `src/io/drawing.py`
- `draw_detection_info` の内部で滞在検知や首振り検知など複数モジュールの状態をまとめて描画しており、包括的な UI を提供します。【F:src/io/drawing.py†L95-L189】
- ただし `UserClassifier` や `HeadShakeDetector` など多数の分析コンポーネントに依存しており、`run_hand_raise` の軽量な CLI に組み込むには過剰な依存関係を招きます。【F:src/io/drawing.py†L10-L14】
- `draw_japanese_text` や `draw_landmarks` など、`drawing_utils` と重複するヘルパーを自身で保持しており、二重管理を解消するためにも共通化の第一候補ではありません。

## 方針
以上の比較から、`run_hand_raise` の描画共通化は `src/drawing_utils.py` への集約が最適と判断します。今後の実装では、`draw_dwell_status` を `drawing_utils` に移設し、既存のテスト群を活用しつつ `run_hand_raise` からの呼び出しを統一する計画です。`src/io/drawing.py` については別途段階的に `drawing_utils` へ依存を寄せることで、描画関連コードの重複排除を図ることを検討します。
