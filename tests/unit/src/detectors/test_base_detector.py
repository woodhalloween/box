"""
test_base_detector.py

base_detector.py の完全なテストカバレッジを提供するモジュール。
"""

from collections import deque

import numpy as np
import pytest

from src.detectors.base_detector import PostureAndMotionDetectorBase


# テスト用の具象クラス
class ConcreteDetector(PostureAndMotionDetectorBase):
    """テスト用の具象実装クラス"""

    def get_status(self):
        """get_statusの具象実装"""
        return {"status": "test"}


class TestPostureAndMotionDetectorBase:
    """PostureAndMotionDetectorBase クラスのテストスイート"""

    # ==================== Initialization Tests ====================

    def test_init_default_confidence_threshold(self):
        """デフォルトの信頼度閾値で初期化できることを確認"""
        detector = ConcreteDetector()
        assert detector.confidence_threshold == 0.5
        assert isinstance(detector._history_deques, dict)
        assert len(detector._history_deques) == 0

    def test_init_custom_confidence_threshold(self):
        """カスタム信頼度閾値で初期化できることを確認"""
        detector = ConcreteDetector(confidence_threshold=0.8)
        assert detector.confidence_threshold == 0.8

    # ==================== History Management Tests ====================

    def test_create_history_deque(self):
        """履歴dequeを作成・登録できることを確認"""
        detector = ConcreteDetector()
        dq = detector._create_history_deque("test_deque", maxlen=10)

        assert isinstance(dq, deque)
        assert dq.maxlen == 10
        assert "test_deque" in detector._history_deques
        assert detector._history_deques["test_deque"] is dq

    def test_create_multiple_history_deques(self):
        """複数の履歴dequeを作成できることを確認"""
        detector = ConcreteDetector()
        dq1 = detector._create_history_deque("deque1", maxlen=5)  # noqa: F841
        dq2 = detector._create_history_deque("deque2", maxlen=10)  # noqa: F841

        assert len(detector._history_deques) == 2
        assert detector._history_deques["deque1"].maxlen == 5
        assert detector._history_deques["deque2"].maxlen == 10

    def test_clear_all_history(self):
        """すべての履歴dequeをクリアできることを確認"""
        detector = ConcreteDetector()
        dq1 = detector._create_history_deque("deque1", maxlen=5)  # noqa: F841
        dq2 = detector._create_history_deque("deque2", maxlen=5)  # noqa: F841

        dq1.append(1)
        dq1.append(2)
        dq2.append(3)
        dq2.append(4)

        detector._clear_all_history()

        assert len(dq1) == 0
        assert len(dq2) == 0

    def test_clear_history_existing_deque(self):
        """指定された履歴dequeをクリアできることを確認"""
        detector = ConcreteDetector()
        dq1 = detector._create_history_deque("deque1", maxlen=5)  # noqa: F841
        dq2 = detector._create_history_deque("deque2", maxlen=5)  # noqa: F841

        dq1.append(1)
        dq2.append(2)

        detector._clear_history("deque1")

        assert len(dq1) == 0
        assert len(dq2) == 1

    def test_clear_history_nonexistent_deque(self):
        """存在しないdequeをクリアしようとしてもエラーにならないことを確認"""
        detector = ConcreteDetector()
        detector._clear_history("nonexistent")  # エラーが発生しないことを確認

    # ==================== Landmark Utilities Tests ====================

    def test_check_landmark_visibility_above_threshold(self):
        """閾値以上の信頼度を持つランドマークを確認"""
        landmark = np.array([0.5, 0.5, 0.5, 0.8])
        result = PostureAndMotionDetectorBase._check_landmark_visibility(landmark, 0.5)
        assert result is True

    def test_check_landmark_visibility_at_threshold(self):
        """閾値と等しい信頼度を持つランドマークを確認"""
        landmark = np.array([0.5, 0.5, 0.5, 0.5])
        result = PostureAndMotionDetectorBase._check_landmark_visibility(landmark, 0.5)
        assert result is True

    def test_check_landmark_visibility_below_threshold(self):
        """閾値未満の信頼度を持つランドマークを確認"""
        landmark = np.array([0.5, 0.5, 0.5, 0.3])
        result = PostureAndMotionDetectorBase._check_landmark_visibility(landmark, 0.5)
        assert result is False

    def test_check_landmark_visibility_index_error(self):
        """不正なインデックスでIndexErrorが発生する場合"""
        landmark = np.array([0.5, 0.5])  # 要素が足りない
        result = PostureAndMotionDetectorBase._check_landmark_visibility(landmark, 0.5)
        assert result is False

    def test_check_landmark_visibility_type_error(self):
        """型エラーが発生する場合"""
        landmark = np.array([0.5, 0.5, 0.5, "invalid"])
        result = PostureAndMotionDetectorBase._check_landmark_visibility(landmark, 0.5)
        assert result is False

    def test_check_landmark_visibility_value_error(self):
        """値エラーが発生する場合"""
        landmark = np.array([0.5, 0.5, 0.5, np.nan])
        result = PostureAndMotionDetectorBase._check_landmark_visibility(landmark, 0.5)
        assert result is False

    def test_extract_landmark_2d(self):
        """正規化座標からピクセル座標への変換を確認"""
        landmark = np.array([0.5, 0.3, 0.0, 0.9])
        frame_shape = (480, 640)
        x, y = PostureAndMotionDetectorBase._extract_landmark_2d(landmark, frame_shape)

        assert x == 320.0  # 0.5 * 640
        assert y == 144.0  # 0.3 * 480

    def test_extract_landmark_2d_corner_cases(self):
        """端の座標の変換を確認"""
        landmark_origin = np.array([0.0, 0.0, 0.0, 1.0])
        landmark_far = np.array([1.0, 1.0, 0.0, 1.0])
        frame_shape = (100, 200)

        x1, y1 = PostureAndMotionDetectorBase._extract_landmark_2d(landmark_origin, frame_shape)
        assert x1 == 0.0
        assert y1 == 0.0

        x2, y2 = PostureAndMotionDetectorBase._extract_landmark_2d(landmark_far, frame_shape)
        assert x2 == 200.0
        assert y2 == 100.0

    def test_calculate_distance_2d(self):
        """2点間の距離計算を確認"""
        point1 = (0.0, 0.0)
        point2 = (3.0, 4.0)
        distance = PostureAndMotionDetectorBase._calculate_distance_2d(point1, point2)
        assert distance == 5.0

    def test_calculate_distance_2d_same_point(self):
        """同じ点の距離がゼロであることを確認"""
        point1 = (5.0, 7.0)
        point2 = (5.0, 7.0)
        distance = PostureAndMotionDetectorBase._calculate_distance_2d(point1, point2)
        assert distance == 0.0

    def test_calculate_distance_2d_negative_coordinates(self):
        """負の座標での距離計算を確認"""
        point1 = (-3.0, -4.0)
        point2 = (0.0, 0.0)
        distance = PostureAndMotionDetectorBase._calculate_distance_2d(point1, point2)
        assert distance == 5.0

    def test_validate_landmarks_valid(self):
        """有効なランドマークの検証"""
        detector = ConcreteDetector()
        landmarks = np.array([[0.5, 0.5, 0.5, 0.9], [0.6, 0.6, 0.6, 0.8]])
        assert detector._validate_landmarks(landmarks) is True

    def test_validate_landmarks_none(self):
        """Noneランドマークの検証"""
        detector = ConcreteDetector()
        assert detector._validate_landmarks(None) is False

    def test_validate_landmarks_empty(self):
        """空のランドマークの検証"""
        detector = ConcreteDetector()
        landmarks = np.array([])
        assert detector._validate_landmarks(landmarks) is False

    # ==================== Pattern Detection Tests ====================

    def test_detect_oscillation_pattern_too_short(self):
        """データが短すぎる場合、振動を検出しない"""
        values = [1.0, 2.0, 1.0]
        result = PostureAndMotionDetectorBase._detect_oscillation_pattern(values, threshold=0.5)
        assert result is False

    def test_detect_oscillation_pattern_no_extrema(self):
        """極値がない場合、振動を検出しない"""
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        result = PostureAndMotionDetectorBase._detect_oscillation_pattern(values, threshold=0.5)
        assert result is False

    def test_detect_oscillation_pattern_below_threshold(self):
        """閾値未満の値のみの場合、振動を検出しない"""
        values = [0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1]
        result = PostureAndMotionDetectorBase._detect_oscillation_pattern(values, threshold=0.5)
        assert result is False

    def test_detect_oscillation_pattern_only_peaks(self):
        """極大値のみの場合、振動を検出しない"""
        values = [0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5]
        result = PostureAndMotionDetectorBase._detect_oscillation_pattern(values, threshold=0.5)
        assert result is False

    def test_detect_oscillation_pattern_only_valleys(self):
        """極小値のみの場合、振動を検出しない"""
        values = [0.5, -1.0, 0.5, 0.5, -1.0, 0.5, 0.5, -1.0, 0.5]
        result = PostureAndMotionDetectorBase._detect_oscillation_pattern(values, threshold=0.5)
        assert result is False

    def test_detect_oscillation_pattern_insufficient_extrema(self):
        """極値が不十分な場合、振動を検出しない"""
        values = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = PostureAndMotionDetectorBase._detect_oscillation_pattern(values, threshold=0.5, min_extrema=4)
        assert result is False

    def test_detect_oscillation_pattern_valid_oscillation(self):
        """有効な振動パターンを検出"""
        values = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0]
        result = PostureAndMotionDetectorBase._detect_oscillation_pattern(values, threshold=0.5)
        assert result is True

    def test_detect_oscillation_pattern_complex_oscillation(self):
        """複雑な振動パターンを検出"""
        values = [0.0, 2.0, 0.5, -1.5, 0.3, 1.8, 0.2, -1.2, 0.1, 1.5, 0.0]
        result = PostureAndMotionDetectorBase._detect_oscillation_pattern(values, threshold=0.5)
        assert result is True

    def test_detect_oscillation_pattern_custom_min_extrema(self):
        """カスタムmin_extremaパラメータでの振動検出"""
        values = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        result = PostureAndMotionDetectorBase._detect_oscillation_pattern(values, threshold=0.5, min_extrema=2)
        assert result is True

    # ==================== Abstract Method Tests ====================

    def test_get_status_concrete_implementation(self):
        """具象クラスのget_status実装を確認"""
        detector = ConcreteDetector()
        status = detector.get_status()
        assert isinstance(status, dict)
        assert status["status"] == "test"

    def test_cannot_instantiate_abstract_class(self):
        """抽象クラスを直接インスタンス化できないことを確認"""
        with pytest.raises(TypeError):
            PostureAndMotionDetectorBase()
