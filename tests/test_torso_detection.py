"""体幹検知機能のテスト"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.dwell_time_detector import DwellTimeDetector
from src.definitions import Angle, MovementState
from src.movement_analyzer import MovementAnalyzer
from src.pose.definitions import BodyPart


@pytest.fixture
def sample_landmarks():
    """テスト用のランドマークデータを生成"""
    landmarks = np.zeros((33, 4))
    # 基本的な可視性を設定
    landmarks[:, 3] = 0.9

    # 肩の位置（正規化座標）
    landmarks[BodyPart.LEFT_SHOULDER] = [0.4, 0.3, 0.0, 0.9]
    landmarks[BodyPart.RIGHT_SHOULDER] = [0.6, 0.3, 0.0, 0.9]

    # 腰の位置（正規化座標）
    landmarks[BodyPart.LEFT_HIP] = [0.4, 0.6, 0.0, 0.9]
    landmarks[BodyPart.RIGHT_HIP] = [0.6, 0.6, 0.0, 0.9]

    # 鼻の位置
    landmarks[BodyPart.NOSE] = [0.5, 0.2, 0.0, 0.9]

    return landmarks


@pytest.fixture
def movement_analyzer():
    """MovementAnalyzerのインスタンスを生成"""
    return MovementAnalyzer(confidence_threshold=0.7)


@pytest.fixture
def dwell_time_detector():
    """DwellTimeDetectorのインスタンスを生成"""
    return DwellTimeDetector(
        stay_threshold_sec=5.0,
        confidence_threshold=0.5,
        spike_threshold=1.5,
        spike_window_sec=1.0,
        stability_threshold_px=50.0,
        stability_window_sec=2.0,
        grace_period_sec=1.5,
        confirmation_ratio=0.5,
        use_normalization=True,
        normalization_base="torso",
    )


class TestMovementAnalyzer:
    """MovementAnalyzerのテスト"""

    def test_body_tilt_calculation(self, movement_analyzer, sample_landmarks):
        """体幹傾き（BODY_TILT）の計算をテスト"""
        results = movement_analyzer.analyze(sample_landmarks)

        assert Angle.BODY_TILT in results
        assert "angle" in results[Angle.BODY_TILT]
        assert "state" in results[Angle.BODY_TILT]
        assert isinstance(results[Angle.BODY_TILT]["angle"], float)
        assert isinstance(results[Angle.BODY_TILT]["state"], MovementState)

    def test_neck_trunk_angle_calculation(self, movement_analyzer, sample_landmarks):
        """頸部-体幹角度（NECK_TRUNK_ANGLE）の計算をテスト"""
        results = movement_analyzer.analyze(sample_landmarks)

        assert Angle.NECK_TRUNK_ANGLE in results
        assert "angle" in results[Angle.NECK_TRUNK_ANGLE]
        assert "state" in results[Angle.NECK_TRUNK_ANGLE]

    def test_lateral_tilt_calculation(self, movement_analyzer, sample_landmarks):
        """側屈（LATERAL_TILT）の計算をテスト"""
        results = movement_analyzer.analyze(sample_landmarks)

        assert Angle.LATERAL_TILT in results
        assert "angle" in results[Angle.LATERAL_TILT]
        assert "state" in results[Angle.LATERAL_TILT]

    def test_forward_tilt_detection(self, movement_analyzer):
        """前傾姿勢の検出をテスト"""
        # 前傾姿勢のランドマークを作成
        landmarks = np.zeros((33, 4))
        landmarks[:, 3] = 0.9

        # 肩を前に傾ける
        landmarks[BodyPart.LEFT_SHOULDER] = [0.4, 0.4, 0.2, 0.9]  # z座標を前に
        landmarks[BodyPart.RIGHT_SHOULDER] = [0.6, 0.4, 0.2, 0.9]
        landmarks[BodyPart.LEFT_HIP] = [0.4, 0.6, 0.0, 0.9]
        landmarks[BodyPart.RIGHT_HIP] = [0.6, 0.6, 0.0, 0.9]
        landmarks[BodyPart.NOSE] = [0.5, 0.3, 0.3, 0.9]

        results = movement_analyzer.analyze(landmarks)

        # 前傾が検出されることを確認（角度が150度以下）
        assert results[Angle.BODY_TILT]["angle"] <= 150


class TestDwellTimeDetector:
    """DwellTimeDetectorのテスト"""

    def test_extract_hip_center(self, dwell_time_detector, sample_landmarks):
        """ヒップ中心抽出のテスト"""
        frame_shape = (1080, 1920, 3)
        hip_data = dwell_time_detector.extract_hip_center(sample_landmarks, frame_shape)

        assert hip_data is not None
        hip_x, hip_y, confidence = hip_data
        assert isinstance(hip_x, float)
        assert isinstance(hip_y, float)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_compute_person_scale_torso(self, dwell_time_detector, sample_landmarks):
        """体幹長による正規化スケールの計算をテスト"""
        frame_shape = (1080, 1920, 3)
        scale = dwell_time_detector._compute_person_scale(sample_landmarks, frame_shape)

        assert scale is not None
        assert scale > 0

    def test_compute_person_scale_shoulder(self, sample_landmarks):
        """肩幅による正規化スケールの計算をテスト"""
        detector = DwellTimeDetector(normalization_base="shoulder", use_normalization=True)
        frame_shape = (1080, 1920, 3)
        scale = detector._compute_person_scale(sample_landmarks, frame_shape)

        assert scale is not None
        assert scale > 0

    def test_compute_person_scale_screen(self, sample_landmarks):
        """画面高さによる正規化スケールの計算をテスト"""
        detector = DwellTimeDetector(normalization_base="screen", use_normalization=True)
        frame_shape = (1080, 1920, 3)
        scale = detector._compute_person_scale(sample_landmarks, frame_shape)

        assert scale is not None
        assert scale == 1080  # 画面高さ

    def test_update_staying_state(self, dwell_time_detector, sample_landmarks):
        """滞在状態の更新をテスト"""
        frame_shape = (1080, 1920, 3)
        timestamp = 0.0

        # 初回更新
        alert = dwell_time_detector.update(sample_landmarks, frame_shape, timestamp)
        assert alert is None  # 初回はアラートなし

        status = dwell_time_detector.get_current_status()
        assert status["hip_position"] is not None
        assert status["stay_duration"] == 0.0
        assert not status["is_long_stay"]

    def test_long_stay_detection(self, dwell_time_detector, sample_landmarks):
        """長期滞在検知のテスト"""
        frame_shape = (1080, 1920, 3)

        # 5秒間滞在をシミュレート（閾値は5.0秒）
        for i in range(6):
            timestamp = float(i)
            alert = dwell_time_detector.update(sample_landmarks, frame_shape, timestamp)

            if i < 5:
                assert alert is None or "Movement" in alert
            else:
                # 5秒経過後は長期滞在アラートが出るはず
                status = dwell_time_detector.get_current_status()
                assert status["is_long_stay"]

    def test_movement_detection(self, dwell_time_detector, sample_landmarks):
        """移動検知のテスト"""
        frame_shape = (1080, 1920, 3)

        # 初期位置
        dwell_time_detector.update(sample_landmarks, frame_shape, 0.0)

        # 大きく移動
        moved_landmarks = sample_landmarks.copy()
        moved_landmarks[BodyPart.LEFT_HIP] = [0.2, 0.6, 0.0, 0.9]
        moved_landmarks[BodyPart.RIGHT_HIP] = [0.4, 0.6, 0.0, 0.9]

        # 移動を検知するまで更新
        for i in range(1, 10):
            timestamp = float(i) * 0.1
            alert = dwell_time_detector.update(moved_landmarks, frame_shape, timestamp)
            if alert and "Movement" in alert:
                break

        status = dwell_time_detector.get_current_status()
        # 移動が検知されたら滞在時間がリセットされる
        assert status["stay_duration"] < 1.0


class TestIntegration:
    """統合テスト"""

    def test_full_analysis_pipeline(self, movement_analyzer, dwell_time_detector, sample_landmarks):  # noqa: PLR0913
        """完全な分析パイプラインのテスト"""
        frame_shape = (1080, 1920, 3)
        timestamp = 0.0

        # 姿勢分析
        analysis_results = movement_analyzer.analyze(sample_landmarks)
        assert Angle.BODY_TILT in analysis_results
        assert Angle.NECK_TRUNK_ANGLE in analysis_results
        assert Angle.LATERAL_TILT in analysis_results

        # 滞在検知
        _ = dwell_time_detector.update(sample_landmarks, frame_shape, timestamp)
        dwell_status = dwell_time_detector.get_current_status()

        assert dwell_status["hip_position"] is not None
        assert dwell_status["confidence"] > 0.0

    def test_missing_landmarks_handling(self, movement_analyzer, dwell_time_detector):  # noqa: PLR0913
        """ランドマーク欠損時の処理をテスト"""
        # 信頼度が低いランドマーク
        landmarks = np.zeros((33, 4))
        landmarks[:, 3] = 0.1  # 低い信頼度

        frame_shape = (1080, 1920, 3)

        # 姿勢分析（信頼度が低いので結果が少ない）
        analysis_results = movement_analyzer.analyze(landmarks)
        # 基本的な関節角度は計算されないはず
        assert len(analysis_results) == 0 or all(
            angle not in analysis_results for angle in [Angle.RIGHT_ELBOW, Angle.LEFT_ELBOW]
        )

        # 滞在検知（信頼度が低いのでNone）
        hip_data = dwell_time_detector.extract_hip_center(landmarks, frame_shape)
        assert hip_data is None
