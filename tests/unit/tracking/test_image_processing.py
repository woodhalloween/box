from unittest.mock import patch

import numpy as np

from src.tracking.bytetrack_utils import (
    draw_tracking_info,
    resize_frame,
)


class TestImageProcessing:
    """画像処理関連の関数のテスト"""

    def test_resize_frame(self):
        """resize_frame関数のテスト"""
        # テスト用の簡単な画像配列を作成
        test_frame = np.zeros((100, 200, 3), dtype=np.uint8)
        test_frame[40:60, 80:120] = 255  # 中央に白い四角形を描画

        # リサイズ
        resized_frame = resize_frame(test_frame, 100, 50)

        # 期待されるサイズを確認
        assert resized_frame.shape == (50, 100, 3)

    @patch("cv2.rectangle")
    @patch("cv2.putText")
    def test_draw_tracking_info(self, mock_put_text, mock_rectangle):
        """draw_tracking_info関数のテスト"""
        # テスト用の簡単な画像配列とトラック情報
        test_frame = np.zeros((100, 200, 3), dtype=np.uint8)
        test_tracks = [
            [10, 20, 50, 60, 1, 0.9, 0],  # x1, y1, x2, y2, track_id, conf, cls_id
            [70, 30, 120, 80, 2, 0.8, 0],
        ]

        result_frame = draw_tracking_info(test_frame, test_tracks)

        # cv2.rectangleとcv2.putTextが各トラックに対して呼ばれたことを確認
        assert mock_rectangle.call_count == 2
        assert mock_put_text.call_count == 2

        # フレームが変更されずに返されたことを確認
        assert result_frame is test_frame

    @patch("cv2.rectangle")
    @patch("cv2.putText")
    def test_draw_tracking_info_with_stay_info(self, mock_put_text, mock_rectangle):
        """Should draw stay_duration and person_height when show_duration is True and stay_info is provided."""
        import numpy as np

        from src.tracking.bytetrack_utils import draw_tracking_info

        # Prepare test data
        test_frame = np.zeros((100, 200, 3), dtype=np.uint8)
        test_tracks = [
            [10, 20, 50, 60, 1, 0.9, 0],  # (centered at 30, 40)
            [70, 30, 120, 80, 2, 0.8, 0],
        ]
        stay_info = {
            1: {"stay_duration": 3.5, "person_height": 180},  # Should be green
            2: {"stay_duration": 4.2, "person_height": 170},  # Should be red
        }

        # Execute function
        result_frame = draw_tracking_info(frame=test_frame, tracks=test_tracks, show_duration=True, stay_info=stay_info)

        # Assertions
        assert result_frame is test_frame
        assert mock_rectangle.call_count == 2
        assert mock_put_text.call_count == 2

        # Extract arguments used in putText
        args1 = mock_put_text.call_args_list[0][0]  # for track_id 1
        args2 = mock_put_text.call_args_list[1][0]  # for track_id 2

        label_1 = args1[1]
        color_1 = args1[5]
        label_2 = args2[1]
        color_2 = args2[5]

        assert "ID:1" in label_1 and "滞在:3.5s" in label_1
        assert color_1 == (0, 255, 0)  # green for < 4.0s

        assert "ID:2" in label_2 and "滞在:4.2s" in label_2
        assert color_2 == (0, 0, 255)  # red for >= 4.0s
