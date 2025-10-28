# 実装ドキュメント

このディレクトリには、プロジェクトの主要な実装に関するドキュメントが含まれています。

## ドキュメント一覧

### MediaPipe Face Mesh 頭部方向検知システム

- **[実装完了報告](./mediapipe_head_turn_implementation_summary.md)**
  - MediaPipe Face Meshを使用した高精度な頭部方向検知システムの実装報告
  - 実装の詳細、使用方法、テスト結果、今後の改善点をまとめています

## 関連ドキュメント

### 要件定義

- [MediaPipe頭部方向検知 要件定義書](../requirements/mediapipe_head_turn_requirements.md)

### 設計

- [MediaPipe頭部方向検知 詳細設計書](../design/mediapipe_head_turn_design.md)

### 分析

- [ヨー角分析レポート](../analysis/yaw_angle_analysis.md)

## 実装コード

### コア実装

- `src/detectors/mediapipe_head_turn_detector.py`: MediaPipe検出器本体
- `src/definitions.py`: MovementState拡張
- `src/io/csv_writer.py`: CSV出力拡張
- `src/io/drawing.py`: 描画機能拡張
- `src/video_processor.py`: VideoProcessor統合

### スクリプト

- `scripts/analyze_annotations_for_threshold.py`: 閾値最適化
- `scripts/test_mediapipe_head_turn.py`: テストスクリプト

## クイックスタート

```bash
# テスト実行
python3 scripts/test_mediapipe_head_turn.py

# 閾値分析
python3 scripts/analyze_annotations_for_threshold.py
```

## 更新履歴

- 2025-01-29: MediaPipe Face Mesh頭部方向検知システム実装完了

