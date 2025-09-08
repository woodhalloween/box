"""
デバッグ用のCSV書き込み機能を提供するモジュール。
analysis_resultsが空かどうかを記録する。
"""
from __future__ import annotations

import csv
from typing import IO, Any

from ..definitions import Angle


def setup_debug_csv_writer(csv_file: IO):
    """デバッグ用CSVライターをセットアップする"""
    fieldnames = [
        "timestamp",
        "frame_number",
        "analysis_results_empty",
        "landmark_detected",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    return writer


def write_debug_results_to_csv(
    csv_writer,
    timestamp: float,
    frame_number: int,
    analysis_results: dict[Angle, dict[str, Any]],
    landmarks: Any | None,
):
    """デバッグ結果をCSVに書き込む"""
    row = {
        "timestamp": timestamp,
        "frame_number": frame_number,
        "analysis_results_empty": not bool(analysis_results),
        "landmark_detected": landmarks is not None,
    }
    csv_writer.writerow(row)
