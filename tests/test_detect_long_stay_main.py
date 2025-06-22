import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.detect_long_stay_main import main, parse_arguments


def test_parse_arguments_defaults(monkeypatch):
    # Simulate: prog --input input.mp4
    monkeypatch.setattr(sys, "argv", ["prog", "--input", "input.mp4"])
    args = parse_arguments()

    # Required
    assert args.input == "input.mp4"

    # Defaults
    assert args.output == "output/tracked_video.mp4"
    assert args.model == "yolov8n.pt"
    assert args.enable_perf_log is False
    assert args.enable_video_display is False
    assert args.device == ""
    assert pytest.approx(args.stay_threshold_sec, rel=1e-6) == 5.0
    assert pytest.approx(args.move_threshold_px, rel=1e-6) == 30.0
    assert pytest.approx(args.conf, rel=1e-6) == 0.3


def test_parse_arguments_all_flags_and_values(monkeypatch):
    # Simulate all options provided
    argv = [
        "prog",
        "--input",
        "in.mp4",
        "--output",
        "out.mp4",
        "--model",
        "custom_model.pt",
        "--enable_perf_log",
        "--enable_video_display",
        "--device",
        "cuda:0",
        "--stay_threshold_sec",
        "12.34",
        "--move_threshold_px",
        "56.78",
        "--conf",
        "0.91",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = parse_arguments()

    # Required and overrides
    assert args.input == "in.mp4"
    assert args.output == "out.mp4"
    assert args.model == "custom_model.pt"
    assert args.enable_perf_log is True
    assert args.enable_video_display is True
    assert args.device == "cuda:0"

    # Numeric overrides
    assert pytest.approx(args.stay_threshold_sec, rel=1e-6) == 12.34
    assert pytest.approx(args.move_threshold_px, rel=1e-6) == 56.78
    assert pytest.approx(args.conf, rel=1e-6) == 0.91


def test_parse_arguments_missing_required(monkeypatch):
    # No --input provided should trigger argparse error
    monkeypatch.setattr(sys, "argv", ["prog"])
    with pytest.raises(SystemExit) as excinfo:
        parse_arguments()
        # Argparse uses exit code 2 for argument errors
        assert excinfo.value.code == 2


def test_main_traversal_with_minimum_args(monkeypatch):
    # Simulate minimum CLI call
    monkeypatch.setattr(sys, "argv", ["prog", "--input", "input.mp4"])

    with (
        patch("src.detect_long_stay_main.run_long_stay_detection") as mock_run,
        patch.object(Path, "mkdir") as mock_mkdir,
    ):
        main()

        # 📁 mkdir should be called to ensure output path
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # ✅ run_long_stay_detection should be invoked with expected structure
        called_args = mock_run.call_args.kwargs
        assert called_args["input_path"] == "input.mp4"
        assert called_args["output_path"] == "output/tracked_video.mp4"
        assert called_args["model_path"] == "yolov8n.pt"
        assert called_args["device"] == ""
        assert called_args["stay_threshold"] == 5.0
        assert called_args["move_threshold"] == 30.0
        assert called_args["conf"] == 0.3
        assert called_args["enable_perf_log"] is False
        assert called_args["enable_video_display"] is False


def test_main_traversal_with_all_args(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--input",
            "input.mp4",
            "--output",
            "logs/out.mp4",
            "--model",
            "yolov8s.pt",
            "--device",
            "cuda:1",
            "--stay_threshold_sec",
            "15.5",
            "--move_threshold_px",
            "22.2",
            "--conf",
            "0.99",
            "--enable_perf_log",
            "--enable_video_display",
        ],
    )

    with (
        patch("src.detect_long_stay_main.run_long_stay_detection") as mock_run,
        patch.object(Path, "mkdir") as mock_mkdir,
    ):
        main()

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        args = mock_run.call_args.kwargs
        assert args["input_path"] == "input.mp4"
        assert args["output_path"] == "logs/out.mp4"
        assert args["model_path"] == "yolov8s.pt"
        assert args["device"] == "cuda:1"
        assert args["stay_threshold"] == 15.5
        assert args["move_threshold"] == 22.2
        assert args["conf"] == 0.99
        assert args["enable_perf_log"] is True
        assert args["enable_video_display"] is True
