"""src/drawing_utils.pyのテスト"""

import numpy as np
import pytest
from mediapipe.python.solutions.pose import PoseLandmark
from PIL import Image

from src.definitions import Angle, MovementState
from src.drawing_utils import (
    draw_analysis_results,
    draw_hand_raise_status,
    draw_japanese_text,
    draw_landmarks,
)


@pytest.fixture
def dummy_image():
    """テスト用のダミー画像を生成するフィクスチャ"""
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_draw_landmarks(mocker, dummy_image):
    """draw_landmarksがcv2の描画関数を正しく呼び出すかテストする"""
    mock_circle = mocker.patch("cv2.circle")
    mock_line = mocker.patch("cv2.line")

    # 33個のランドマークを持つダミーデータ
    landmarks = np.random.rand(33, 2)
    draw_landmarks(dummy_image, landmarks)

    # circleがランドマークの数だけ呼ばれることを確認
    assert mock_circle.call_count == 33
    # lineがCONNECTIONSの数だけ呼ばれることを確認
    # CONNECTIONSの数は12
    assert mock_line.call_count == 12


def test_draw_japanese_text(mocker, dummy_image):
    """draw_japanese_textがPillowの描画関数を正しく呼び出すかテストする"""
    mocker.patch("cv2.cvtColor", side_effect=lambda x, y: x)  # cvtColorをモック化
    mock_fromarray = mocker.patch("PIL.Image.fromarray", return_value=Image.new("RGB", (100, 100)))
    mock_draw = mocker.patch("PIL.ImageDraw.Draw")
    mock_truetype = mocker.patch("PIL.ImageFont.truetype")

    draw_japanese_text(dummy_image, "テスト", (10, 10), 20, (255, 255, 255))

    mock_fromarray.assert_called_once()
    mock_draw.return_value.text.assert_called_once_with(
        (10, 10), "テスト", font=mock_truetype.return_value, fill=(255, 255, 255)
    )


def test_draw_japanese_text_font_not_found(mocker, dummy_image):
    """draw_japanese_textでフォントが見つからない場合にデフォルトフォントを試みるかテストする"""
    mocker.patch("cv2.cvtColor", side_effect=lambda x, y: x)
    mocker.patch("PIL.Image.fromarray", return_value=Image.new("RGB", (100, 100)))
    mocker.patch("PIL.ImageDraw.Draw")
    mocker.patch("PIL.ImageFont.truetype", side_effect=OSError("Font not found"))
    mock_load_default = mocker.patch("PIL.ImageFont.load_default")

    draw_japanese_text(dummy_image, "テスト", (10, 10), 20, (255, 255, 255))

    mock_load_default.assert_called_once()


def test_draw_analysis_results(mocker, dummy_image):
    """draw_analysis_resultsがテキスト描画関数を正しく呼び出すかテストする"""
    mock_draw_jp = mocker.patch("src.drawing_utils.draw_japanese_text", return_value=dummy_image)
    landmarks = np.random.rand(33, 3)

    results = {
        Angle.RIGHT_ELBOW: {"angle": 90.0, "state": MovementState.FLEXION},
        Angle.BODY_TILT: {"angle": 10.0, "state": MovementState.FORWARD_TILT},
    }

    draw_analysis_results(dummy_image, results, landmarks, fps=30.0)

    # FPS表示 + 結果2つの計3回呼ばれる
    assert mock_draw_jp.call_count == 3


def test_draw_analysis_results_english(mocker, dummy_image):
    """draw_analysis_resultsが英語モードでcv2.putTextを呼び出すかテストする"""
    mock_put_text = mocker.patch("cv2.putText")
    mock_draw_jp = mocker.patch("src.drawing_utils.draw_japanese_text", return_value=dummy_image)
    landmarks = np.random.rand(33, 3)

    results = {Angle.RIGHT_ELBOW: {"angle": 90.0, "state": MovementState.FLEXION}}

    draw_analysis_results(dummy_image, results, landmarks, disable_japanese=True)

    # 日本語描画はFPS表示の1回だけ呼ばれる
    mock_draw_jp.assert_called_once()
    # 英語の結果表示でputTextが1回呼ばれる
    mock_put_text.assert_called_once()


def test_draw_analysis_results_debug_lines(mocker, dummy_image):
    """debug描画のラインと色指定が正しく行われるか確認する"""

    draw_calls: list[tuple[str, tuple[int, int, int]]] = []

    def fake_draw(image, text, position, font_size, color):
        draw_calls.append((text, color))
        return image

    line_calls: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []

    def fake_line(image, start, end, color, thickness):
        line_calls.append((start, end, color))
        return image

    mocker.patch("src.drawing_utils.draw_japanese_text", side_effect=fake_draw)
    mocker.patch("cv2.line", side_effect=fake_line)

    landmarks = np.zeros((33, 3), dtype=np.float32)
    landmarks[PoseLandmark.LEFT_SHOULDER.value] = [0.4, 0.4, 0.0]
    landmarks[PoseLandmark.RIGHT_SHOULDER.value] = [0.6, 0.4, 0.0]
    landmarks[PoseLandmark.LEFT_HIP.value] = [0.45, 0.7, 0.0]
    landmarks[PoseLandmark.RIGHT_HIP.value] = [0.55, 0.7, 0.0]
    landmarks[PoseLandmark.NOSE.value] = [0.5, 0.3, 0.0]

    results = {
        Angle.BODY_TILT: {"angle": 12.0, "state": MovementState.FORWARD_TILT},
        Angle.NECK_TRUNK_ANGLE: {"angle": 25.0, "state": MovementState.HUNCH},
        Angle.LATERAL_TILT: {"angle": 5.0, "state": MovementState.RIGHT_TILT},
    }

    draw_analysis_results(dummy_image, results, landmarks)

    # FPS行 + 3つの角度で4回描画される
    assert len(draw_calls) == 4
    # BODY_TILT は緑、NECK_TRUNK_ANGLE は専用色、LATERAL_TILT はデフォルト色
    assert draw_calls[1][1] == (0, 128, 0)
    assert draw_calls[2][1] == (11, 134, 184)
    assert draw_calls[3][1] == (139, 0, 0)

    # 体幹ライン2本、頸部ライン1本、側屈ライン2本が描画される
    colors = [color for *_rest, color in line_calls]
    assert colors.count((0, 255, 0)) == 1
    assert colors.count((255, 0, 0)) == 1
    assert colors.count((0, 255, 255)) == 1
    assert colors.count((128, 0, 128)) == 1
    assert colors.count((255, 255, 0)) == 1


def test_draw_hand_raise_status_colors(mocker, dummy_image):
    """手挙げ描画で左右の色分けが想定通りか確認する"""

    calls: list[tuple[str, tuple[int, int, int]]] = []

    def fake_draw(image, text, position, font_size, color):
        calls.append((text, color))
        return image

    mocker.patch("src.drawing_utils.draw_japanese_text", side_effect=fake_draw)

    statuses = {"left_hand_raised": True, "right_hand_raised": False}
    draw_hand_raise_status(dummy_image, statuses, font_size=18, position=(5, 5))

    assert calls == [
        ("左手 挙手", (0, 255, 0)),
        ("右手 挙手", (128, 128, 128)),
    ]
