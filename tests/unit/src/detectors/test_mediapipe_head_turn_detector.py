"""
tests/unit/src/detectors/test_mediapipe_head_turn_detector.py

MediaPipe Face Mesh頭部方向検出器のユニットテスト。
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

# プロジェクトルートをsys.pathに追加
sys.path.append(str(Path(__file__).resolve().parents[4]))

from src.detectors.mediapipe_head_turn_detector import MediaPipeFaceMeshHeadTurnDetector

# --- テスト用の設定 ---
YAW_THRESHOLD_RIGHT = 30.0
YAW_THRESHOLD_LEFT = -30.0
MIN_CONSECUTIVE_FRAMES = 3
COOLDOWN_SEC = 10.0
CONFIDENCE_THRESHOLD = 0.5


# --- テスト用のモックデータ生成ヘルパー ---
def create_mock_face_landmarks(nose_x: float = 0.5, left_eye_x: float = 0.4, right_eye_x: float = 0.6):
    """テスト用のMediaPipe Face Meshランドマークをモックする。

    Args:
        nose_x: 鼻先のx座標（0.0〜1.0の正規化座標）
        left_eye_x: 左目外側のx座標
        right_eye_x: 右目外側のx座標

    Returns:
        Mock: MediaPipe Face Meshのランドマークをシミュレートするモックオブジェクト
    """
    mock_landmarks = Mock()

    # ランドマークリストをモック
    mock_landmarks.landmark = []

    # 468個のランドマークを作成（必要な部分のみ実装）
    for _ in range(468):
        landmark = Mock()
        landmark.x = 0.5
        landmark.y = 0.5
        landmark.z = 0.0
        mock_landmarks.landmark.append(landmark)

    # 必要なランドマークを設定
    mock_landmarks.landmark[1].x = nose_x  # NOSE_TIP
    mock_landmarks.landmark[1].y = 0.5

    mock_landmarks.landmark[33].x = left_eye_x  # LEFT_EYE_OUTER
    mock_landmarks.landmark[33].y = 0.5

    mock_landmarks.landmark[263].x = right_eye_x  # RIGHT_EYE_OUTER
    mock_landmarks.landmark[263].y = 0.5

    return mock_landmarks


def create_mock_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """テスト用のダミーフレームを生成する。

    Args:
        width: フレームの幅
        height: フレームの高さ

    Returns:
        np.ndarray: ダミーのBGRフレーム
    """
    return np.zeros((height, width, 3), dtype=np.uint8)


@pytest.fixture
def detector():
    """MediaPipeFaceMeshHeadTurnDetectorのインスタンスを生成するpytestフィクスチャ。"""
    with patch("src.detectors.mediapipe_head_turn_detector.mp.solutions.face_mesh.FaceMesh"):
        return MediaPipeFaceMeshHeadTurnDetector(
            yaw_threshold_right=YAW_THRESHOLD_RIGHT,
            yaw_threshold_left=YAW_THRESHOLD_LEFT,
            min_consecutive_frames=MIN_CONSECUTIVE_FRAMES,
            cooldown_sec=COOLDOWN_SEC,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )


# --- 初期化テスト ---
def test_initialization(detector):
    """初期化時のパラメータが正しく設定されることをテストする。"""
    assert detector.yaw_threshold_right == YAW_THRESHOLD_RIGHT
    assert detector.yaw_threshold_left == YAW_THRESHOLD_LEFT
    assert detector.min_consecutive_frames == MIN_CONSECUTIVE_FRAMES
    assert detector.cooldown_sec == COOLDOWN_SEC
    assert detector.confidence_threshold == CONFIDENCE_THRESHOLD


def test_initial_state(detector):
    """初期状態が正しく設定されることをテストする。"""
    status = detector.get_status()
    assert status["face_detected"] is False
    assert status["yaw_angle"] == 0.0
    assert status["direction"] == "正面"
    assert status["consecutive_frames"] == 0
    assert status["confidence"] == 0.0
    assert status["is_sustained"] is False


# --- ヨー角計算テスト ---
def test_calculate_yaw_angle_center():
    """鼻が中央にある場合、ヨー角が0度付近であることをテストする。"""
    with patch("src.detectors.mediapipe_head_turn_detector.mp.solutions.face_mesh.FaceMesh"):
        detector = MediaPipeFaceMeshHeadTurnDetector()

    # 鼻が目の中点にある（正面）
    face_landmarks = create_mock_face_landmarks(nose_x=0.5, left_eye_x=0.4, right_eye_x=0.6)
    yaw_angle, confidence = detector.calculate_yaw_angle(face_landmarks)

    assert abs(yaw_angle) < 1.0  # ほぼ0度
    assert confidence == 1.0


def test_calculate_yaw_angle_right():
    """鼻が右にずれている場合、正のヨー角が計算されることをテストする。"""
    with patch("src.detectors.mediapipe_head_turn_detector.mp.solutions.face_mesh.FaceMesh"):
        detector = MediaPipeFaceMeshHeadTurnDetector()

    # 鼻が右にずれている（右向き）
    face_landmarks = create_mock_face_landmarks(nose_x=0.6, left_eye_x=0.4, right_eye_x=0.6)
    yaw_angle, confidence = detector.calculate_yaw_angle(face_landmarks)

    assert yaw_angle > 0  # 正の角度
    assert confidence == 1.0


def test_calculate_yaw_angle_left():
    """鼻が左にずれている場合、負のヨー角が計算されることをテストする。"""
    with patch("src.detectors.mediapipe_head_turn_detector.mp.solutions.face_mesh.FaceMesh"):
        detector = MediaPipeFaceMeshHeadTurnDetector()

    # 鼻が左にずれている（左向き）
    face_landmarks = create_mock_face_landmarks(nose_x=0.4, left_eye_x=0.4, right_eye_x=0.6)
    yaw_angle, confidence = detector.calculate_yaw_angle(face_landmarks)

    assert yaw_angle < 0  # 負の角度
    assert confidence == 1.0


def test_calculate_yaw_angle_zero_eye_distance():
    """目間距離がゼロの場合、ゼロ除算エラーが起きず0度を返すことをテストする。"""
    with patch("src.detectors.mediapipe_head_turn_detector.mp.solutions.face_mesh.FaceMesh"):
        detector = MediaPipeFaceMeshHeadTurnDetector()

    # 左右の目が同じ位置（異常ケース）
    face_landmarks = create_mock_face_landmarks(nose_x=0.5, left_eye_x=0.5, right_eye_x=0.5)
    yaw_angle, confidence = detector.calculate_yaw_angle(face_landmarks)

    assert yaw_angle == 0.0
    assert confidence == 1.0


# --- 方向分類テスト ---
def test_classify_direction_front(detector):
    """閾値内の角度が正面と分類されることをテストする。"""
    assert detector._classify_direction(0.0) == "正面"
    assert detector._classify_direction(10.0) == "正面"
    assert detector._classify_direction(-10.0) == "正面"


def test_classify_direction_right(detector):
    """閾値以上の正の角度が右向きと分類されることをテストする。"""
    assert detector._classify_direction(YAW_THRESHOLD_RIGHT) == "右向き"
    assert detector._classify_direction(YAW_THRESHOLD_RIGHT + 10.0) == "右向き"


def test_classify_direction_left(detector):
    """閾値以下の負の角度が左向きと分類されることをテストする。"""
    assert detector._classify_direction(YAW_THRESHOLD_LEFT) == "左向き"
    assert detector._classify_direction(YAW_THRESHOLD_LEFT - 10.0) == "左向き"


# --- 連続フレーム更新テスト ---
def test_update_consecutive_frames_same_direction(detector):
    """同じ方向が続く場合、連続フレーム数が増加することをテストする。"""
    # 初回
    count = detector._update_consecutive_frames("右向き")
    assert count == 1
    assert detector._head_state.current_direction == "右向き"

    # 2回目
    count = detector._update_consecutive_frames("右向き")
    assert count == 2

    # 3回目
    count = detector._update_consecutive_frames("右向き")
    assert count == 3
    assert detector._head_state.is_sustained is True


def test_update_consecutive_frames_direction_change(detector):
    """方向が変わった場合、連続フレーム数がリセットされることをテストする。"""
    # 右向きを2フレーム
    detector._update_consecutive_frames("右向き")
    detector._update_consecutive_frames("右向き")
    assert detector._head_state.consecutive_frames == 2

    # 左向きに変更
    count = detector._update_consecutive_frames("左向き")
    assert count == 1
    assert detector._head_state.current_direction == "左向き"


def test_sustained_flag_below_threshold(detector):
    """連続フレーム数が閾値未満の場合、持続フラグがFalseであることをテストする。"""
    for _ in range(MIN_CONSECUTIVE_FRAMES - 1):
        detector._update_consecutive_frames("右向き")

    assert detector._head_state.is_sustained is False


def test_sustained_flag_at_threshold(detector):
    """連続フレーム数が閾値に達した場合、持続フラグがTrueになることをテストする。"""
    for _ in range(MIN_CONSECUTIVE_FRAMES):
        detector._update_consecutive_frames("右向き")

    assert detector._head_state.is_sustained is True


# --- 検出テスト (detect メソッド) ---
def test_detect_no_face(detector):
    """顔が検出されない場合の動作をテストする。"""
    frame = create_mock_frame()

    # MediaPipe Face Meshが顔を検出しないようにモック
    detector.face_mesh.process = Mock(return_value=Mock(multi_face_landmarks=None))

    result = detector.detect(frame, timestamp=0.0)

    assert result["face_detected"] is False
    assert result["yaw_angle"] == 0.0
    assert result["direction"] == "正面"


def test_detect_face_center(detector):
    """顔が正面を向いている場合の検出をテストする。"""
    frame = create_mock_frame()

    # 正面を向いた顔のモック
    mock_face = create_mock_face_landmarks(nose_x=0.5, left_eye_x=0.4, right_eye_x=0.6)
    detector.face_mesh.process = Mock(return_value=Mock(multi_face_landmarks=[mock_face]))

    result = detector.detect(frame, timestamp=0.0)

    assert result["face_detected"] is True
    assert abs(result["yaw_angle"]) < 1.0
    assert result["direction"] == "正面"


def test_detect_face_right_turn(detector):
    """顔が右を向いている場合の検出をテストする。"""
    frame = create_mock_frame()

    # 右を向いた顔のモック（鼻が大きく右にずれている）
    mock_face = create_mock_face_landmarks(nose_x=0.7, left_eye_x=0.4, right_eye_x=0.6)
    detector.face_mesh.process = Mock(return_value=Mock(multi_face_landmarks=[mock_face]))

    # 連続フレームで検出
    for _ in range(MIN_CONSECUTIVE_FRAMES):
        result = detector.detect(frame, timestamp=0.0)

    assert result["face_detected"] is True
    assert result["yaw_angle"] > YAW_THRESHOLD_RIGHT
    assert result["direction"] == "右向き"
    assert result["is_sustained"] is True


def test_detect_face_left_turn(detector):
    """顔が左を向いている場合の検出をテストする。"""
    frame = create_mock_frame()

    # 左を向いた顔のモック（鼻が大きく左にずれている）
    mock_face = create_mock_face_landmarks(nose_x=0.3, left_eye_x=0.4, right_eye_x=0.6)
    detector.face_mesh.process = Mock(return_value=Mock(multi_face_landmarks=[mock_face]))

    # 連続フレームで検出
    for _ in range(MIN_CONSECUTIVE_FRAMES):
        result = detector.detect(frame, timestamp=0.0)

    assert result["face_detected"] is True
    assert result["yaw_angle"] < YAW_THRESHOLD_LEFT
    assert result["direction"] == "左向き"
    assert result["is_sustained"] is True


# --- 持続的方向転換検知テスト (check_sustained_turn) ---
def test_check_sustained_turn_not_sustained(detector):
    """持続フラグがFalseの場合、Noneが返されることをテストする。"""
    detector._head_state.is_sustained = False
    result = detector.check_sustained_turn(timestamp=0.0)
    assert result is None


def test_check_sustained_turn_front_facing(detector):
    """正面を向いている場合、Noneが返されることをテストする。"""
    detector._head_state.is_sustained = True
    detector._head_state.current_direction = "正面"
    result = detector.check_sustained_turn(timestamp=0.0)
    assert result is None


def test_check_sustained_turn_right(detector):
    """右向きが持続している場合、検知イベントが返されることをテストする。"""
    detector._head_state.is_sustained = True
    detector._head_state.current_direction = "右向き"
    detector._head_state.consecutive_frames = MIN_CONSECUTIVE_FRAMES
    detector._head_state.yaw_angle = 45.0

    result = detector.check_sustained_turn(timestamp=10.0)

    assert result is not None
    assert result["detected"] is True
    assert result["direction"] == "右向き"
    assert result["frames"] == MIN_CONSECUTIVE_FRAMES
    assert result["timestamp"] == 10.0
    assert result["yaw_angle"] == 45.0


def test_check_sustained_turn_left(detector):
    """左向きが持続している場合、検知イベントが返されることをテストする。"""
    detector._head_state.is_sustained = True
    detector._head_state.current_direction = "左向き"
    detector._head_state.consecutive_frames = MIN_CONSECUTIVE_FRAMES
    detector._head_state.yaw_angle = -45.0

    result = detector.check_sustained_turn(timestamp=10.0)

    assert result is not None
    assert result["detected"] is True
    assert result["direction"] == "左向き"


def test_check_sustained_turn_cooldown(detector):
    """クールダウン期間内は再検知しないことをテストする。"""
    detector._head_state.is_sustained = True
    detector._head_state.current_direction = "右向き"
    detector._head_state.consecutive_frames = MIN_CONSECUTIVE_FRAMES

    # 最初の検知
    result1 = detector.check_sustained_turn(timestamp=10.0)
    assert result1 is not None

    # クールダウン期間内の検知（抑制される）
    result2 = detector.check_sustained_turn(timestamp=15.0)  # 5秒後
    assert result2 is None

    # クールダウン期間外の検知（再検知される）
    result3 = detector.check_sustained_turn(timestamp=21.0)  # 11秒後
    assert result3 is not None


# --- リセットテスト ---
def test_reset(detector):
    """リセット後、状態が初期化されることをテストする。"""
    # 状態を変更
    detector._head_state.face_detected = True
    detector._head_state.yaw_angle = 45.0
    detector._head_state.current_direction = "右向き"
    detector._head_state.consecutive_frames = 5
    detector._head_state.is_sustained = True

    # リセット
    detector.reset()

    # 状態が初期化されたことを確認
    status = detector.get_status()
    assert status["face_detected"] is False
    assert status["yaw_angle"] == 0.0
    assert status["direction"] == "正面"
    assert status["consecutive_frames"] == 0
    assert status["is_sustained"] is False


# --- ステータス取得テスト ---
def test_get_status_with_sustained_right(detector):
    """持続的な右向きの状態が正しく返されることをテストする。"""
    detector._head_state.face_detected = True
    detector._head_state.yaw_angle = 45.0
    detector._head_state.current_direction = "右向き"
    detector._head_state.consecutive_frames = MIN_CONSECUTIVE_FRAMES
    detector._head_state.is_sustained = True
    detector._head_state.confidence = 0.95

    status = detector.get_status()

    assert status["face_detected"] is True
    assert status["yaw_angle"] == 45.0
    assert status["direction"] == "右向き"
    assert status["consecutive_frames"] == MIN_CONSECUTIVE_FRAMES
    assert status["is_sustained"] is True
    assert "SUSTAINED_RIGHT" in status["sustained_direction"]


def test_get_status_with_sustained_left(detector):
    """持続的な左向きの状態が正しく返されることをテストする。"""
    detector._head_state.face_detected = True
    detector._head_state.yaw_angle = -45.0
    detector._head_state.current_direction = "左向き"
    detector._head_state.consecutive_frames = MIN_CONSECUTIVE_FRAMES
    detector._head_state.is_sustained = True
    detector._head_state.confidence = 0.95

    status = detector.get_status()

    assert status["face_detected"] is True
    assert status["yaw_angle"] == -45.0
    assert status["direction"] == "左向き"
    assert status["is_sustained"] is True
    assert "SUSTAINED_LEFT" in status["sustained_direction"]


def test_get_status_no_sustained_direction(detector):
    """持続的でない場合、sustained_directionが空文字列であることをテストする。"""
    detector._head_state.face_detected = True
    detector._head_state.yaw_angle = 10.0
    detector._head_state.current_direction = "正面"
    detector._head_state.consecutive_frames = 1
    detector._head_state.is_sustained = False

    status = detector.get_status()

    assert status["sustained_direction"] == ""


# --- エラーハンドリングテスト ---
def test_calculate_yaw_angle_with_invalid_landmarks(detector):
    """不正なランドマークデータでもエラーにならず、デフォルト値を返すことをテストする。"""
    # 不正なランドマーク（Noneや不完全なデータ）
    invalid_landmarks = None
    yaw_angle, confidence = detector.calculate_yaw_angle(invalid_landmarks)

    assert yaw_angle == 0.0
    assert confidence == 0.0


# --- 統合テスト ---
def test_full_detection_workflow(detector):
    """検出から持続的方向転換検知までの一連のワークフローをテストする。"""
    frame = create_mock_frame()

    # 右向きの顔のモック
    mock_face = create_mock_face_landmarks(nose_x=0.7, left_eye_x=0.4, right_eye_x=0.6)
    detector.face_mesh.process = Mock(return_value=Mock(multi_face_landmarks=[mock_face]))

    # MIN_CONSECUTIVE_FRAMES分だけ検出
    for i in range(MIN_CONSECUTIVE_FRAMES):
        result = detector.detect(frame, timestamp=float(i))

    # 最後のフレームで持続的方向転換が検知されるべき
    assert result["is_sustained"] is True

    # 持続的方向転換の確認
    turn_event = detector.check_sustained_turn(timestamp=float(MIN_CONSECUTIVE_FRAMES))
    assert turn_event is not None
    assert turn_event["detected"] is True
    assert turn_event["direction"] == "右向き"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
