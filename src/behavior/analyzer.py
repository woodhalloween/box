# src/behavior/analyzer.py

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# --- データ構造の定義 (型ヒントで明確化) ---


@dataclass
class CustomerState:
    """各顧客の状態を管理するデータクラス"""

    track_id: int
    status: str = "OUTSIDE_AREA"  # 'INSIDE_AREA', 'OUTSIDE_AREA', 'LEFT_AREA'
    position_history: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=30))
    last_seen_time: float = field(default_factory=time.time)
    re_entry_notified: bool = False


# --- メインクラス ---


class BehaviorAnalyzer:
    """顧客の行動を分析し、購買予兆を検出するクラス"""

    def __init__(
        self,
        aoi_coords: list[int],
        disappear_threshold_sec: float = 5.0,
    ) -> None:
        """
        Args:
            aoi_coords (List[int]): 監視エリアの座標 [x_min, y_min, x_max, y_max]
            disappear_threshold_sec (float): 人物が消えたと判断する秒数
        """
        self.aoi = aoi_coords
        self.disappear_threshold = disappear_threshold_sec
        self.customer_states: dict[int, CustomerState] = {}

    def analyze_frame(self, tracks: list[list[Any]], current_time: float) -> list[str]:
        """
        フレームごとの追跡結果を分析し、通知リストを返す

        Args:
            tracks (List[List[Any]]): ByteTrackからの追跡結果
            current_time (float): 現在のタイムスタンプ

        Returns:
            List[str]: 検知された行動に関する通知文字列のリスト
        """
        notifications = []
        current_track_ids = {int(track[4]) for track in tracks}

        # 1. 各人物の状態を更新
        for track in tracks:
            track_id = int(track[4])
            position = self._get_center_position(track)

            if track_id not in self.customer_states:
                self.customer_states[track_id] = CustomerState(track_id=track_id)

            state = self.customer_states[track_id]
            state.position_history.append(position)
            state.last_seen_time = current_time

            # 2. 再入店ロジックの判定
            re_entry_notification = self._check_re_entry(state, position)
            if re_entry_notification:
                notifications.append(re_entry_notification)

            # TODO: Uターンロジックの判定をここに追加

        # 3. 画面から消えた人物をクリーンアップ
        self._cleanup_disappeared(current_track_ids, current_time)

        return notifications

    def _get_center_position(self, track: list[Any]) -> tuple[float, float]:
        """トラック情報から中心座標を計算する"""
        x1, y1, x2, y2 = track[:4]
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _is_in_aoi(self, position: tuple[float, float]) -> bool:
        """指定された座標がAOI内にあるか判定する"""
        x, y = position
        x_min, y_min, x_max, y_max = self.aoi
        return x_min <= x <= x_max and y_min <= y <= y_max

    def _check_re_entry(self, state: CustomerState, position: tuple[float, float]) -> str | None:
        """再入店（商品棚へ戻る動き）を検知する"""
        is_currently_in_aoi = self._is_in_aoi(position)

        if state.status == "LEFT_AREA" and is_currently_in_aoi and not state.re_entry_notified:
            state.re_entry_notified = True  # 通知は1回のみ
            state.status = "INSIDE_AREA"
            return f"【予兆:再訪】ID:{state.track_id} がエリアに戻りました。"

        if state.status == "INSIDE_AREA" and not is_currently_in_aoi:
            state.status = "LEFT_AREA"
        elif state.status == "OUTSIDE_AREA" and is_currently_in_aoi:
            state.status = "INSIDE_AREA"

        return None

    def _cleanup_disappeared(self, current_track_ids: set[int], current_time: float) -> None:
        """長期間見失った人物のデータを削除する"""
        disappeared_ids = [
            track_id
            for track_id, state in self.customer_states.items()
            if track_id not in current_track_ids and (current_time - state.last_seen_time) > self.disappear_threshold
        ]
        for track_id in disappeared_ids:
            del self.customer_states[track_id]
