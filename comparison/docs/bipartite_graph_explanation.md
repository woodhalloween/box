# 二部グラフの最小重み完全マッチング：わかりやすい解説

## 🎯 基本概念をステップ別に理解

### 1. 二部グラフとは？

**二部グラフ**は、グラフの頂点を**2つのグループに分割**できるグラフのことです。

```mermaid
graph LR
    subgraph "グループA（左側）"
        A1[検出1]
        A2[検出2]
        A3[検出3]
    end
    
    subgraph "グループB（右側）"
        B1[トラック1]
        B2[トラック2]
        B3[トラック3]
    end
    
    A1 --- B1
    A1 --- B2
    A1 --- B3
    A2 --- B1
    A2 --- B2
    A2 --- B3
    A3 --- B1
    A3 --- B2
    A3 --- B3
    
    style A1 fill:#FFE082
    style A2 fill:#FFE082
    style A3 fill:#FFE082
    style B1 fill:#81C784
    style B2 fill:#81C784
    style B3 fill:#81C784
```

**重要な特徴**：
- 左側のグループ（検出）と右側のグループ（トラック）
- 同じグループ内の頂点同士は**直接つながらない**
- 異なるグループ間のみ線（エッジ）で接続

### 2. 重み付きグラフ

各エッジに**コスト（重み）**が付いています：

```mermaid
graph LR
    subgraph "検出結果"
        D1[検出1<br/>人物A]
        D2[検出2<br/>人物B]
        D3[検出3<br/>人物C]
    end
    
    subgraph "既存トラック"
        T1[トラック1<br/>前回の人物A]
        T2[トラック2<br/>前回の人物B]
        T3[トラック3<br/>前回の人物C]
    end
    
    D1 ---|"コスト: 0.1<br/>（類似度高）"| T1
    D1 ---|"コスト: 0.8<br/>（類似度低）"| T2
    D1 ---|"コスト: 0.9<br/>（類似度低）"| T3
    
    D2 ---|"コスト: 0.7"| T1
    D2 ---|"コスト: 0.2"| T2
    D2 ---|"コスト: 0.6"| T3
    
    D3 ---|"コスト: 0.9"| T1
    D3 ---|"コスト: 0.8"| T2
    D3 ---|"コスト: 0.1"| T3
    
    style D1 fill:#FFE082
    style T1 fill:#FFE082
    style D2 fill:#A5D6A7
    style T2 fill:#A5D6A7
    style D3 fill:#FFCDD2
    style T3 fill:#FFCDD2
```

### 3. 完全マッチングとは？

**完全マッチング**とは、すべての頂点が**ちょうど1つずつ**ペアになることです：

```mermaid
graph LR
    subgraph "マッチング前"
        A1[検出1] 
        A2[検出2]
        A3[検出3]
        B1[トラック1]
        B2[トラック2] 
        B3[トラック3]
        
        A1 -.-> B1
        A1 -.-> B2
        A1 -.-> B3
        A2 -.-> B1
        A2 -.-> B2
        A2 -.-> B3
        A3 -.-> B1
        A3 -.-> B2
        A3 -.-> B3
    end
    
    ARROW[⬇️ ハンガリアン法]
    
    subgraph "マッチング後"
        C1[検出1] === D1[トラック1]
        C2[検出2] === D2[トラック2]
        C3[検出3] === D3[トラック3]
    end
    
    style C1 fill:#FFE082
    style D1 fill:#FFE082
    style C2 fill:#A5D6A7
    style D2 fill:#A5D6A7
    style C3 fill:#FFCDD2
    style D3 fill:#FFCDD2
```

### 4. 最小重み完全マッチング

**全体のコストが最小**になるマッチングを見つけることです：

## 🔍 具体例で理解する

### 例：3人の人物追跡

現在のフレームで3人を検出し、前フレームの3つのトラックとマッチングしたい場合：

#### ステップ1：コスト行列の作成

| | トラック1 | トラック2 | トラック3 |
|---|----------|----------|----------|
| **検出1** | 0.1 | 0.8 | 0.9 |
| **検出2** | 0.7 | 0.2 | 0.6 |
| **検出3** | 0.9 | 0.8 | 0.1 |

#### ステップ2：可能なマッチングの検討

```mermaid
graph TD
    ROOT[マッチング組み合わせ]
    
    ROOT --> OPTION1[組み合わせ1<br/>1→1, 2→2, 3→3<br/>コスト: 0.1+0.2+0.1=0.4]
    ROOT --> OPTION2[組み合わせ2<br/>1→1, 2→3, 3→2<br/>コスト: 0.1+0.6+0.8=1.5]
    ROOT --> OPTION3[組み合わせ3<br/>1→2, 2→1, 3→3<br/>コスト: 0.8+0.7+0.1=1.6]
    ROOT --> DOTS[...]
    
    OPTION1 --> BEST[最小コスト: 0.4<br/>最適解！]
    
    style OPTION1 fill:#4CAF50
    style BEST fill:#E8F5E8
```

#### ステップ3：最適解の発見

```mermaid
graph LR
    subgraph "最適マッチング"
        D1[検出1] === T1[トラック1] 
        D2[検出2] === T2[トラック2]
        D3[検出3] === T3[トラック3]
    end
    
    D1 -.->|"コスト: 0.1"| COST1[類似度高]
    D2 -.->|"コスト: 0.2"| COST2[類似度高]
    D3 -.->|"コスト: 0.1"| COST3[類似度高]
    
    COST1 --> TOTAL[総コスト: 0.4]
    COST2 --> TOTAL
    COST3 --> TOTAL
    
    style D1 fill:#FFE082
    style T1 fill:#FFE082
    style D2 fill:#A5D6A7
    style T2 fill:#A5D6A7
    style D3 fill:#FFCDD2
    style T3 fill:#FFCDD2
    style TOTAL fill:#E8F5E8
```

## ⚡ O(n³)の計算複雑度とは？

### 計算量の意味

**O(n³)**は、処理対象数が**n個**の時、計算回数が**n³に比例**することを意味します：

```mermaid
graph LR
    subgraph "計算量の変化"
        A[n=3人<br/>計算回数: 27回]
        B[n=5人<br/>計算回数: 125回]
        C[n=10人<br/>計算回数: 1000回]
        D[n=20人<br/>計算回数: 8000回]
    end
    
    A --> B --> C --> D
    
    style D fill:#FF9800
```

### 実際の処理時間への影響

```mermaid
graph TD
    subgraph "人数と処理時間"
        P1[3人: 0.1ms]
        P2[5人: 0.5ms]
        P3[10人: 2ms]
        P4[20人: 8ms]
        P5[50人: 125ms]
    end
    
    P1 --> P2 --> P3 --> P4 --> P5
    
    style P5 fill:#FFCDD2
```

### なぜO(n³)なのか？

ハンガリアン法の処理ステップ：

```mermaid
graph TD
    STEP1[行列の各行を処理] --> STEP2[各列を処理]
    STEP2 --> STEP3[マッチング検証]
    STEP3 --> CHECK{完了?}
    CHECK -->|No| STEP1
    CHECK -->|Yes| DONE[完了]
    
    STEP1 -.->|"O(n)"| NOTE1[n回の処理]
    STEP2 -.->|"O(n)"| NOTE2[n回の処理]
    STEP3 -.->|"O(n)"| NOTE3[最大n回の反復]
    
    style DONE fill:#4CAF50
```

## 🚀 人物追跡での実際の応用

### YOLO高度版での実装例

```python
def hungarian_assignment_example():
    """ハンガリアン法の実装例"""
    
    # ステップ1: 類似度行列の計算
    similarity_matrix = calculate_similarity_matrix(detections, tracks)
    # 例: [[0.9, 0.2, 0.1],
    #      [0.3, 0.8, 0.4], 
    #      [0.1, 0.2, 0.9]]
    
    # ステップ2: コスト行列への変換（類似度 → コスト）
    cost_matrix = 1.0 - similarity_matrix
    # 例: [[0.1, 0.8, 0.9],
    #      [0.7, 0.2, 0.6],
    #      [0.9, 0.8, 0.1]]
    
    # ステップ3: ハンガリアン法実行
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    # 結果: [(0,0), (1,1), (2,2)] - 最小コストの組み合わせ
    
    # ステップ4: マッチング結果の適用
    for det_idx, track_idx in zip(row_indices, col_indices):
        assign_detection_to_track(detections[det_idx], tracks[track_idx])
```

### 処理フロー図

```mermaid
sequenceDiagram
    participant F as フレーム
    participant D as 検出結果
    participant S as 類似度計算
    participant H as ハンガリアン法
    participant M as マッチング
    
    F->>D: 新フレーム処理
    D->>S: 検出結果（3人）
    Note over S: 特徴量抽出<br/>位置・外観情報
    S->>S: 類似度行列計算<br/>3×3=9個のペア
    S->>H: コスト行列作成
    Note over H: O(n³)最適化<br/>全組み合わせ検討
    H->>M: 最小コスト解
    M->>F: ID付き追跡結果
```

## 💡 実測結果による評価

### ハンガリアン法の性能特性

```mermaid
graph TD
    subgraph "単純シーン（WIN動画）"
        SIMPLE1[検出数: 少]
        SIMPLE2[特徴量: 安定]
        SIMPLE3[結果: 19回ID切り替え]
    end
    
    subgraph "複雑シーン（フロア動画）"
        COMPLEX1[検出数: 多]
        COMPLEX2[特徴量: ノイズ多]
        COMPLEX3[結果: 164回ID切り替え]
    end
    
    SIMPLE1 --> SIMPLE2 --> SIMPLE3
    COMPLEX1 --> COMPLEX2 --> COMPLEX3
    
    SIMPLE3 --> DEGRADATION[8.6倍の性能劣化]
    COMPLEX3 --> DEGRADATION
    
    style COMPLEX3 fill:#FFCDD2
    style DEGRADATION fill:#FFCDD2
```

### 他手法との比較

| 手法 | マッチング方式 | 複雑度 | 実際の性能 |
|------|---------------|--------|------------|
| **ハンガリアン法** | 最適化 | O(n³) | 164回ID切り替え |
| **最近傍法** | 貪欲法 | O(n²) | 164回ID切り替え |
| **カルマン+IoU** | 予測ベース | O(n) | 12回ID切り替え |

## 🎯 重要な学び

### 1. 理論と実践のギャップ

```mermaid
graph LR
    THEORY[理論的最適性<br/>ハンガリアン法] --> GAP[実践的効果]
    PRACTICE[時系列考慮<br/>カルマンフィルター] --> GAP
    
    GAP --> RESULT[カルマンフィルターの圧勝<br/>99.7% vs 95.3%]
    
    style PRACTICE fill:#4CAF50
    style RESULT fill:#E8F5E8
```

### 2. 計算複雑度の実世界への影響

- **O(n³)のハンガリアン法**: 理論最適だが重い計算
- **O(n)のカルマンフィルター**: 高速で実用的な予測

### 3. 適用シーンによる使い分け

```mermaid
flowchart TD
    SCENARIO{適用シーン} --> STATIC{静的マッチング}
    SCENARIO --> DYNAMIC{動的追跡}
    
    STATIC --> HUNGARIAN[ハンガリアン法<br/>一度限りの最適マッチング]
    DYNAMIC --> KALMAN[カルマンフィルター<br/>連続的な状態推定]
    
    HUNGARIAN --> USE1[配送最適化<br/>作業割り当て]
    KALMAN --> USE2[人物追跡<br/>軌跡予測]
    
    style KALMAN fill:#4CAF50
    style USE2 fill:#E8F5E8
```

## 🎯 まとめ

**二部グラフの最小重み完全マッチング**とは：

1. **二部グラフ**: 2つのグループ間の関係を表現
2. **重み**: 各ペアのコスト（類似度の逆）
3. **完全マッチング**: 全員が1対1でペアになる
4. **最小重み**: 総コストが最小の組み合わせ
5. **O(n³)**: 人数の3乗に比例する計算量

**人物追跡での実際の効果**：
- ✅ 理論的には最適解
- ❌ 時系列の文脈を無視
- ❌ 複雑シーンで大幅劣化
- ❌ 計算コストが高い

**結論**: 人物追跡では**継続的な予測（カルマンフィルター）**の方が、**瞬間的な最適化（ハンガリアン法）**よりも実用的で高性能！ 