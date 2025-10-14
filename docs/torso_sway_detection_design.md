## 体幹の左右・前後揺れ検知 設計（簡素版）

### 目的
- 側屈 `Angle.LATERAL_TILT` と体幹傾き `Angle.BODY_TILT` から、体幹の左右/前後揺れをシンプルに検出。
- 判定は「絶対角度がしきい値を連続フレーム数以上」でON、下回りが連続でOFF。

### 入力
- 角度系列（フレーム毎）
  - 左右：`Angle.LATERAL_TILT` の角度（度）
  - 前後：`ap = |180 - BODY_TILT|` を揺れ量とする
- タイムスタンプ `t`（秒）: 互換のため受け取るが使わない

### 判定
- しきい値：左右 `th_lat`、前後 `th_ap`
- ON: `val >= threshold` が `on_frames` 連続
- OFF: `val < threshold` が `off_frames` 連続
- None 入力は 0 とみなし OFF 側にカウント

### パラメータ初期値
- `th_lat=10°`, `th_ap=8°`
- `on_sec=1.2s`, `off_sec=0.7s`（FPSでフレーム換算）

### 出力
- `{"lateral": bool, "ap": bool}`（CSV 互換のため amp/freq/cycles/level は 0/"none" で埋める）

### フローチャート
```mermaid
flowchart LR
I[角度入力] --> L[lat=|lateral|]
I --> A[ap=|180-body_tilt|]
L --> LB{>=th_lat}
A --> AB{>=th_ap}
LB -->|yes| L1[lat_over++ ; lat_under=0]
LB -->|no|  L0[lat_under++ ; lat_over=0]
AB -->|yes| A1[ap_over++ ; ap_under=0]
AB -->|no|  A0[ap_under++ ; ap_over=0]
L1 --> LS{!state && over>=on -> ON}
L0 --> LE{state && under>=off -> OFF}
A1 --> AS{!state && over>=on -> ON}
A0 --> AE{state && under>=off -> OFF}
```

### 実装ポイント
- `src/analysis/torso_sway_detector.py` を簡素ロジックに置換。
- `video_processor` ではフラグを受け取り、CSV ペイロードを 0/"none" で整形。


