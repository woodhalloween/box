"""
hand_raise_detector.py

レガシー実装との互換性を維持するために提供されるラッパーモジュール。

`HandRaiseDetector` クラスは新しい実装 `hand_raise_refactored.HandRaiseDetector`
を再エクスポートし、既存コードからのインポートパスを維持する。
"""

from __future__ import annotations

from .hand_raise_refactored import HandRaiseDetector

__all__ = ["HandRaiseDetector"]
