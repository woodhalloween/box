"""src/io/drawing.pyのテスト"""

import numpy as np
import pytest
from PIL import Image

from src.definitions import Angle, MovementState
from src.io.drawing import (
    draw_analysis_results,
    draw_color_frame,
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


def test_draw_color_frame_comma_separated_rgb(dummy_image):
    """draw_color_frameがカンマ区切りのRGB形式を正しく解析するかテストする"""
    color_str = "255,0,0"  # Red in RGB
    result = draw_color_frame(dummy_image, color_str, alpha=0.5, border_width=10)

    # 結果の形状と型を確認
    assert result.shape == dummy_image.shape
    assert result.dtype == np.uint8
    assert isinstance(result, np.ndarray)


def test_draw_color_frame_hex_format(dummy_image):
    """draw_color_frameが16進数形式の色を正しく解析するかテストする"""
    color_str = "#FF0000"  # Red in hex
    result = draw_color_frame(dummy_image, color_str, alpha=0.5, border_width=10)

    assert result.shape == dummy_image.shape
    assert result.dtype == np.uint8


def test_draw_color_frame_hex_with_leading_hash(dummy_image):
    """draw_color_frameが#で始まる16進数形式を正しく処理するかテストする"""
    color_str = "#00FF00"  # Green in hex
    result = draw_color_frame(dummy_image, color_str)

    assert result.shape == dummy_image.shape
    assert result.dtype == np.uint8


def test_draw_color_frame_invalid_color_value_error(dummy_image, capsys):
    """draw_color_frameが無効な色文字列（ValueError）の場合にフォールバックするかテストする"""
    color_str = "invalid,color,string"
    result = draw_color_frame(dummy_image, color_str)

    # エラーメッセージが出力されることを確認
    captured = capsys.readouterr()
    assert "Warning: Could not parse color" in captured.out

    # 結果は正常に返される（赤色でフォールバック）
    assert result.shape == dummy_image.shape
    assert result.dtype == np.uint8


def test_draw_color_frame_invalid_color_index_error(dummy_image, capsys):
    """draw_color_frameが不完全な色文字列（IndexError）の場合にフォールバックするかテストする"""
    color_str = "255,0"  # 不完全なRGB値
    result = draw_color_frame(dummy_image, color_str)

    # エラーメッセージが出力されることを確認
    captured = capsys.readouterr()
    assert "Warning: Could not parse color" in captured.out

    # 結果は正常に返される
    assert result.shape == dummy_image.shape
    assert result.dtype == np.uint8


def test_draw_color_frame_invalid_hex_format(dummy_image, capsys):
    """draw_color_frameが無効な16進数形式の場合にフォールバックするかテストする"""
    color_str = "#GG0000"  # 無効な16進数文字
    result = draw_color_frame(dummy_image, color_str)

    # エラーメッセージが出力されることを確認
    captured = capsys.readouterr()
    assert "Warning: Could not parse color" in captured.out

    # 結果は正常に返される
    assert result.shape == dummy_image.shape
    assert result.dtype == np.uint8


def test_draw_color_frame_short_hex_string(dummy_image, capsys):
    """draw_color_frameが短い16進数文字列（IndexError）の場合にフォールバックするかテストする"""
    color_str = "#FF"  # 短すぎる16進数文字列
    result = draw_color_frame(dummy_image, color_str)

    # エラーメッセージが出力されることを確認
    captured = capsys.readouterr()
    assert "Warning: Could not parse color" in captured.out

    # 結果は正常に返される
    assert result.shape == dummy_image.shape
    assert result.dtype == np.uint8


def test_draw_color_frame_empty_string(dummy_image, capsys):
    """draw_color_frameが空文字列の場合にフォールバックするかテストする"""
    color_str = ""
    result = draw_color_frame(dummy_image, color_str)

    # エラーメッセージが出力されることを確認
    captured = capsys.readouterr()
    assert "Warning: Could not parse color" in captured.out

    # 結果は正常に返される
    assert result.shape == dummy_image.shape
    assert result.dtype == np.uint8


def test_draw_color_frame_with_spaces_in_rgb(dummy_image):
    """draw_color_frameがスペースを含むRGB文字列を正しく解析するかテストする"""
    color_str = " 255 , 0 , 0 "  # スペースを含む
    result = draw_color_frame(dummy_image, color_str)

    assert result.shape == dummy_image.shape
    assert result.dtype == np.uint8


def test_draw_color_frame_calls_cv2_rectangle(mocker, dummy_image):
    """draw_color_frameがcv2.rectangleを正しく呼び出すかテストする"""
    mock_rectangle = mocker.patch("cv2.rectangle")

    draw_color_frame(dummy_image, "255,0,0", border_width=30)

    # 4つの矩形が描画される（上、下、左、右）
    assert mock_rectangle.call_count == 4


def test_draw_color_frame_border_width_values(dummy_image):
    """draw_color_frameが異なるborder_width値で動作するかテストする"""
    for border_width in [1, 10, 30, 50]:
        result = draw_color_frame(dummy_image, "255,0,0", border_width=border_width)
        assert result.shape == dummy_image.shape
        assert result.dtype == np.uint8


def test_draw_color_frame_alpha_values(dummy_image):
    """draw_color_frameが異なるalpha値で動作するかテストする"""
    for alpha in [0.0, 0.3, 0.5, 1.0]:
        result = draw_color_frame(dummy_image, "255,0,0", alpha=alpha)
        assert result.shape == dummy_image.shape
        assert result.dtype == np.uint8


def test_draw_color_frame_large_border_width(dummy_image):
    """draw_color_frameがフレームサイズより大きなborder_widthでも動作するかテストする"""
    # border_widthがフレームサイズより大きい場合でもクラッシュしないことを確認
    result = draw_color_frame(dummy_image, "255,0,0", border_width=150)
    assert result.shape == dummy_image.shape
    assert result.dtype == np.uint8


def test_draw_color_frame_small_frame(mocker):
    """draw_color_frameが小さいフレームでも動作するかテストする"""
    small_frame = np.zeros((50, 50, 3), dtype=np.uint8)
    mock_rectangle = mocker.patch("cv2.rectangle")

    result = draw_color_frame(small_frame, "255,0,0", border_width=10)

    assert result.shape == small_frame.shape
    assert result.dtype == np.uint8
    # 4つの矩形が描画されることを確認
    assert mock_rectangle.call_count == 4


def test_draw_color_frame_different_colors(dummy_image):
    """draw_color_frameが異なる色で動作するかテストする"""
    colors = [
        "0,255,0",  # Green
        "0,0,255",  # Blue
        "255,255,0",  # Yellow
        "#00FF00",  # Green in hex
        "#0000FF",  # Blue in hex
    ]

    for color in colors:
        result = draw_color_frame(dummy_image, color)
        assert result.shape == dummy_image.shape
        assert result.dtype == np.uint8


def test_draw_color_frame_returns_uint8(dummy_image):
    """draw_color_frameがuint8型を返すかテストする"""
    result = draw_color_frame(dummy_image, "255,0,0")
    assert result.dtype == np.uint8


def test_draw_color_frame_preserves_frame_shape(dummy_image):
    """draw_color_frameがフレームの形状を保持するかテストする"""
    original_shape = dummy_image.shape
    result = draw_color_frame(dummy_image, "255,0,0")
    assert result.shape == original_shape


def test_draw_color_frame_default_parameters(dummy_image):
    """draw_color_frameがデフォルトパラメータで動作するかテストする"""
    result = draw_color_frame(dummy_image, "255,0,0")
    assert result.shape == dummy_image.shape
    assert result.dtype == np.uint8


def test_draw_color_frame_border_actually_drawn(dummy_image):
    """draw_color_frameが実際にボーダーを描画するかテストする（ピクセル値の変化を確認）"""
    # 元のフレームは全て0（黒）
    original_sum = dummy_image.sum()

    # ボーダーを描画
    result = draw_color_frame(dummy_image, "255,0,0", alpha=1.0, border_width=10)

    # ボーダーが描画されていれば、フレームの値が変化しているはず
    result_sum = result.sum()
    # ボーダーが描画されていれば、sumは増加する（alpha=1.0なので完全に上書き）
    # ボーダー領域には色が適用されるため、合計値は増加する
    assert result_sum > original_sum
