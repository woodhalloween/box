"""
拡張可視化モジュール
顔認識統合システム用の高度な可視化機能
"""

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class EnhancedVisualizer:
    """
    顔認識統合システム用の拡張可視化クラス
    """

    def __init__(self, config: dict[str, Any]):
        """
        可視化器の初期化

        Args:
            config: 設定辞書
        """
        self.config = config
        vis_config = config.get("visualization", {})

        # 表示設定
        self.show_face_boxes = vis_config.get("show_face_boxes", True)
        self.show_person_boxes = vis_config.get("show_person_boxes", True)
        self.show_id_labels = vis_config.get("show_id_labels", True)
        self.show_confidence = vis_config.get("show_confidence", True)
        self.show_recognition_status = vis_config.get("show_recognition_status", True)

        # 色設定
        self.face_box_color = tuple(vis_config.get("face_box_color", [255, 0, 0]))  # 青
        self.person_box_color = tuple(vis_config.get("person_box_color", [0, 255, 0]))  # 緑
        self.known_person_color = tuple(vis_config.get("known_person_color", [0, 255, 255]))  # 黄
        self.unknown_person_color = tuple(vis_config.get("unknown_person_color", [0, 0, 255]))  # 赤

        # フォント設定
        self.font_scale = vis_config.get("font_scale", 0.6)
        self.font_thickness = vis_config.get("font_thickness", 2)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # ステータス表示設定
        self.info_panel_height = 150
        self.info_panel_color = (50, 50, 50)  # 暗いグレー
        self.text_color = (255, 255, 255)  # 白

        logger.info("EnhancedVisualizer initialized")

    def draw_enhanced_tracks(
        self,
        frame: np.ndarray,
        enhanced_tracks: list[dict[str, Any]],
        detected_faces: list[dict[str, Any]] = None,
        performance_metrics: dict[str, Any] = None,
    ) -> np.ndarray:
        """
        拡張追跡結果をフレームに描画

        Args:
            frame: 入力フレーム (BGR)
            enhanced_tracks: 拡張追跡結果
            detected_faces: 検出された顔（オプション）
            performance_metrics: パフォーマンスメトリクス（オプション）

        Returns:
            描画されたフレーム
        """
        result_frame = frame.copy()

        # 1. 顔ボックスの描画
        if detected_faces and self.show_face_boxes:
            result_frame = self._draw_face_boxes(result_frame, detected_faces)

        # 2. 人物追跡ボックスとIDの描画
        if enhanced_tracks:
            result_frame = self._draw_person_tracks(result_frame, enhanced_tracks)

        # 3. 情報パネルの描画
        if performance_metrics:
            result_frame = self._draw_info_panel(result_frame, enhanced_tracks, performance_metrics)

        return result_frame

    def _draw_face_boxes(
        self, frame: np.ndarray, detected_faces: list[dict[str, Any]]
    ) -> np.ndarray:
        """顔ボックスを描画"""
        for face in detected_faces:
            x1, y1, x2, y2 = face["bbox"]
            confidence = face.get("confidence", 0.0)
            quality_score = face.get("quality_score", 0.0)

            # 顔ボックスを描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.face_box_color, 1)

            # 信頼度を表示
            if self.show_confidence:
                label = f"Face: {confidence:.2f}"
                label_size = cv2.getTextSize(label, self.font, self.font_scale * 0.7, 1)[0]
                cv2.rectangle(
                    frame,
                    (x1, y1 - label_size[1] - 5),
                    (x1 + label_size[0], y1),
                    self.face_box_color,
                    -1,
                )
                cv2.putText(
                    frame, label, (x1, y1 - 5), self.font, self.font_scale * 0.7, (255, 255, 255), 1
                )

        return frame

    def _draw_person_tracks(
        self, frame: np.ndarray, enhanced_tracks: list[dict[str, Any]]
    ) -> np.ndarray:
        """人物追跡ボックスとIDを描画"""
        for track in enhanced_tracks:
            x1, y1, x2, y2 = track["bbox"]
            stable_id = track["stable_id"]
            person_name = track["person_name"]
            is_known = track["is_known_person"]
            face_confidence = track["face_confidence"]
            tracking_source = track["tracking_source"]

            # 人物の状態に応じて色を決定
            if is_known:
                box_color = self.known_person_color
                text_color = self.known_person_color
            else:
                box_color = self.unknown_person_color
                text_color = self.unknown_person_color

            # 人物ボックスを描画
            if self.show_person_boxes:
                thickness = 3 if is_known else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

            # IDとラベルを描画
            if self.show_id_labels:
                # メインラベル（ID + 名前）
                if is_known:
                    main_label = f"ID {stable_id}: {person_name}"
                else:
                    main_label = f"ID {stable_id}: Unknown"

                # 信頼度ラベル
                confidence_label = ""
                if self.show_confidence and face_confidence > 0:
                    confidence_label = f" ({face_confidence:.2f})"

                # 追跡ソースインジケータ
                source_indicator = ""
                if self.show_recognition_status:
                    if tracking_source == "face_recognition":
                        source_indicator = " [FR]"
                    elif tracking_source == "face_detection":
                        source_indicator = " [FD]"
                    else:
                        source_indicator = " [PO]"

                full_label = main_label + confidence_label + source_indicator

                # ラベル背景の描画
                label_size = cv2.getTextSize(
                    full_label, self.font, self.font_scale, self.font_thickness
                )[0]
                label_bg_y1 = y1 - label_size[1] - 10
                label_bg_y2 = y1

                cv2.rectangle(
                    frame, (x1, label_bg_y1), (x1 + label_size[0] + 10, label_bg_y2), box_color, -1
                )

                # ラベルテキストの描画
                cv2.putText(
                    frame,
                    full_label,
                    (x1 + 5, y1 - 5),
                    self.font,
                    self.font_scale,
                    (0, 0, 0),
                    self.font_thickness,
                )

                # 追加情報（複数行表示）
                if is_known and len(person_name) > 15:  # 長い名前の場合は省略
                    short_name = person_name[:12] + "..."
                    cv2.putText(
                        frame,
                        f"ID {stable_id}: {short_name}",
                        (x1 + 5, y1 - 5),
                        self.font,
                        self.font_scale,
                        (0, 0, 0),
                        self.font_thickness,
                    )

        return frame

    def _draw_info_panel(
        self,
        frame: np.ndarray,
        enhanced_tracks: list[dict[str, Any]],
        performance_metrics: dict[str, Any],
    ) -> np.ndarray:
        """情報パネルを描画"""
        height, width = frame.shape[:2]

        # パネル背景
        panel_y1 = height - self.info_panel_height
        cv2.rectangle(frame, (0, panel_y1), (width, height), self.info_panel_color, -1)

        # 統計情報の準備
        total_tracks = len(enhanced_tracks)
        known_persons = sum(1 for track in enhanced_tracks if track["is_known_person"])
        unknown_persons = total_tracks - known_persons

        # テキスト情報
        info_lines = [
            f"Total Tracks: {total_tracks}",
            f"Known Persons: {known_persons}",
            f"Unknown Persons: {unknown_persons}",
            f"Processing Time: {performance_metrics.get('total_processing_time', 0):.1f}ms",
            f"Face Detection: {performance_metrics.get('face_detection_time', 0):.1f}ms",
            f"Face Recognition: {performance_metrics.get('face_recognition_time', 0):.1f}ms",
        ]

        # テキストの描画
        y_offset = panel_y1 + 20
        line_height = 20

        for i, line in enumerate(info_lines):
            x_pos = 10 if i < 3 else width // 2 + 10
            y_pos = y_offset + (i % 3) * line_height

            cv2.putText(
                frame, line, (x_pos, y_pos), self.font, self.font_scale * 0.8, self.text_color, 1
            )

        # 既知の人物リスト
        known_persons_list = [track for track in enhanced_tracks if track["is_known_person"]]
        if known_persons_list:
            list_x = width - 300
            list_y = panel_y1 + 20

            cv2.putText(
                frame,
                "Known Persons:",
                (list_x, list_y),
                self.font,
                self.font_scale * 0.8,
                self.text_color,
                1,
            )

            for i, track in enumerate(known_persons_list[:5]):  # 最大5人表示
                person_info = f"ID {track['stable_id']}: {track['person_name'][:15]}"
                cv2.putText(
                    frame,
                    person_info,
                    (list_x, list_y + (i + 1) * 18),
                    self.font,
                    self.font_scale * 0.6,
                    self.known_person_color,
                    1,
                )

        return frame

    def draw_tracking_trails(
        self, frame: np.ndarray, track_history: dict[int, Any], max_trail_length: int = 30
    ) -> np.ndarray:
        """追跡軌跡を描画"""
        for stable_id, history in track_history.items():
            if len(history) < 2:
                continue

            # 軌跡の点を取得
            trail_points = []
            for entry in list(history)[-max_trail_length:]:
                bbox = entry.get("bbox")
                if bbox:
                    x1, y1, x2, y2 = bbox
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    trail_points.append((center_x, center_y))

            # 軌跡線を描画
            if len(trail_points) >= 2:
                for i in range(1, len(trail_points)):
                    # 透明度を距離に応じて調整
                    alpha = i / len(trail_points)
                    color = tuple(int(c * alpha) for c in self.person_box_color)

                    cv2.line(frame, trail_points[i - 1], trail_points[i], color, 2)

                # 最新位置にマーカー
                latest_point = trail_points[-1]
                cv2.circle(frame, latest_point, 3, self.person_box_color, -1)

        return frame

    def create_debug_view(
        self,
        frame: np.ndarray,
        detected_faces: list[dict[str, Any]],
        recognition_results: list[dict[str, Any]],
    ) -> np.ndarray:
        """デバッグビューを作成"""
        debug_frame = frame.copy()

        # 顔検出結果の詳細表示
        for i, face in enumerate(detected_faces):
            x1, y1, x2, y2 = face["bbox"]
            confidence = face.get("confidence", 0.0)
            quality_score = face.get("quality_score", 0.0)

            # 詳細ボックス
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # 詳細情報
            info_text = f"Face {i}: C={confidence:.3f}, Q={quality_score:.3f}"
            cv2.putText(debug_frame, info_text, (x1, y1 - 10), self.font, 0.5, (255, 255, 255), 1)

            # ランドマーク表示
            landmarks = face.get("landmarks", {})
            for landmark_name, (lx, ly) in landmarks.items():
                cv2.circle(debug_frame, (lx, ly), 2, (0, 255, 255), -1)

        # 認識結果の表示
        if len(recognition_results) > 0:
            result_text_y = 30
            for i, result in enumerate(recognition_results):
                result_text = f"Recognition {i}: {result['name']} ({result['confidence']:.3f})"
                cv2.putText(
                    debug_frame, result_text, (10, result_text_y), self.font, 0.6, (0, 255, 0), 1
                )
                result_text_y += 25

        return debug_frame

    def save_debug_image(
        self,
        frame: np.ndarray,
        filename: str,
        detected_faces: list[dict[str, Any]] = None,
        enhanced_tracks: list[dict[str, Any]] = None,
    ) -> bool:
        """デバッグ画像を保存"""
        try:
            debug_config = self.config.get("logging", {})
            if not debug_config.get("enable_debug_images", False):
                return False

            debug_path = debug_config.get("debug_images_path", "data/debug_images/")
            import os

            os.makedirs(debug_path, exist_ok=True)

            # デバッグ情報を追加
            debug_frame = frame.copy()

            if detected_faces:
                debug_frame = self._draw_face_boxes(debug_frame, detected_faces)

            if enhanced_tracks:
                debug_frame = self._draw_person_tracks(debug_frame, enhanced_tracks)

            # ファイル保存
            full_path = os.path.join(debug_path, filename)
            cv2.imwrite(full_path, debug_frame)

            return True

        except Exception as e:
            logger.error(f"Error saving debug image: {e}")
            return False

    def create_summary_visualization(self, stats: dict[str, Any]) -> np.ndarray:
        """統計サマリーの可視化を作成"""
        # 固定サイズのサマリー画像
        summary_width, summary_height = 600, 400
        summary_frame = np.zeros((summary_height, summary_width, 3), dtype=np.uint8)
        summary_frame.fill(40)  # 暗いグレー背景

        # タイトル
        title = "Face Recognition Tracking Summary"
        title_size = cv2.getTextSize(title, self.font, 1.0, 2)[0]
        title_x = (summary_width - title_size[0]) // 2
        cv2.putText(summary_frame, title, (title_x, 40), self.font, 1.0, (255, 255, 255), 2)

        # 統計情報の表示
        y_offset = 80
        line_height = 30

        stat_lines = [
            f"Total Frames Processed: {stats.get('frame_count', 0)}",
            f"Active Tracks: {stats.get('active_tracks', 0)}",
            f"Known Persons: {stats.get('known_persons', 0)}",
            f"Average FPS: {stats.get('avg_fps', 0):.1f}",
            f"Average Processing Time: {stats.get('avg_processing_time_ms', 0):.1f}ms",
            f"Face Detection Avg: {stats.get('avg_detection_time_ms', 0):.1f}ms",
            f"Face Recognition Avg: {stats.get('avg_recognition_time_ms', 0):.1f}ms",
        ]

        for i, line in enumerate(stat_lines):
            cv2.putText(
                summary_frame,
                line,
                (20, y_offset + i * line_height),
                self.font,
                0.7,
                (200, 200, 200),
                1,
            )

        return summary_frame
