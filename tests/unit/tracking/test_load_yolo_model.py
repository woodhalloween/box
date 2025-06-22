from unittest.mock import MagicMock, patch

import pytest

from src.tracking.bytetrack_utils import (
    load_yolo_model,
)


@patch("src.tracking.bytetrack_utils.YOLO")  # Patch once for all methods
class TestLoadYOLOModel:
    def test_load_success_default_device(self, mock_yolo, capsys):
        """Should load model successfully with default device (no .to call)."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        result = load_yolo_model("model.pt")
        out = capsys.readouterr().out

        assert result == mock_model
        assert "正常にロードしました" in out

    @pytest.mark.parametrize("device", ["cpu", "0"])
    def test_load_success_with_device(self, mock_yolo, capsys, device):
        """Should load YOLO model and transfer to specified device."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        result = load_yolo_model("model.pt", device)
        out = capsys.readouterr().out

        mock_model.to.assert_called_once_with(device)
        assert result == mock_model
        assert f"'{device}'" not in out  # just to confirm device isn’t mistakenly printed

    def test_load_model_constructor_failure(self, mock_yolo, capsys):
        """Should return None and print error if YOLO constructor fails."""
        mock_yolo.side_effect = Exception("model load error")

        result = load_yolo_model("invalid.pt")
        out = capsys.readouterr().out

        assert result is None
        assert "ロードに失敗しました" in out
        assert "model load error" in out

    def test_model_to_device_failure(self, mock_yolo, capsys):
        """Should return None and print error if model.to(device) fails."""
        mock_model = MagicMock()
        mock_model.to.side_effect = Exception("device error")
        mock_yolo.return_value = mock_model

        result = load_yolo_model("model.pt", device="gpu42")
        out = capsys.readouterr().out

        assert result is None
        assert "ロードに失敗しました" in out
        assert "device error" in out
