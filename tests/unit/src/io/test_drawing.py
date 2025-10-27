"""src/io/drawing.pyのテスト"""

import numpy as np
import pytest
from PIL import Image

from src.definitions import Angle, MovementState
from src.io.drawing import (
    draw_analysis_results,
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
    mock_draw_jp = mocker.patch("src.io.drawing.draw_japanese_text", return_value=dummy_image)
    landmarks = np.random.rand(33, 3)

    results = {
        Angle.RIGHT_ELBOW: {"angle": 90.0, "state": MovementState.FLEXION},
        Angle.BODY_TILT: {"angle": 10.0, "state": MovementState.FORWARD_TILT},
    }

    draw_analysis_results(dummy_image, results, None, landmarks, fps=30.0)

    # FPS表示 + 結果2つの計3回呼ばれる（hand_statusesがNoneなので手の挙上表示なし）
    assert mock_draw_jp.call_count == 3


def test_draw_analysis_results_english(mocker, dummy_image):
    """draw_analysis_resultsが英語モードでcv2.putTextを呼び出すかテストする"""
    mock_put_text = mocker.patch("cv2.putText")
    mock_draw_jp = mocker.patch("src.io.drawing.draw_japanese_text", return_value=dummy_image)
    landmarks = np.random.rand(33, 3)

    results = {Angle.RIGHT_ELBOW: {"angle": 90.0, "state": MovementState.FLEXION}}

    draw_analysis_results(dummy_image, results, None, landmarks, disable_japanese=True)

    # 日本語描画はFPS表示の1回だけ呼ばれる（hand_statusesがNoneなので手の挙上表示なし）
    mock_draw_jp.assert_called_once()
    # 英語の結果表示でputTextが1回呼ばれる
    mock_put_text.assert_called_once()
