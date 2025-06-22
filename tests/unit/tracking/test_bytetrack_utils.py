import os
from pathlib import Path
from unittest.mock import patch

from src.tracking.bytetrack_utils import get_system_info, initialize_perf_log

# テスト用のダミー入力ファイル名とモデルパス
DUMMY_INPUT_FILE = "test_video.mp4"
DUMMY_MODEL_PATH = "test_model.pt"


def test_initialize_perf_log_enabled(tmp_path):
    """Test case when performance log is enabled"""

    perf_log_file = initialize_perf_log(
        enable_perf_log=True,
        input_file=DUMMY_INPUT_FILE,
        model_path=DUMMY_MODEL_PATH,
        log_type="test_log",
    )

    assert perf_log_file is not None

    log_file_path = Path(perf_log_file)
    assert log_file_path.exists()
    assert log_file_path.name.startswith(f"log_{Path(DUMMY_INPUT_FILE).stem}_test_log_{Path(DUMMY_MODEL_PATH).stem}_")
    assert log_file_path.parent.name == "logs"
    assert log_file_path.parent.parent.name == "output"

    # ✅ Now read after flush/close
    with open(perf_log_file) as f:
        lines = f.readlines()
        assert lines  # ensure file is not empty
        assert "# System Information" in lines[0]
        assert "Frame" in lines[-1]

    # 🧹 Cleanup
    if Path("output/logs").exists() and Path(perf_log_file).is_relative_to(Path("output/logs")):
        os.remove(perf_log_file)
        if not os.listdir("output/logs"):
            os.rmdir("output/logs")
        if not os.listdir("output"):
            os.rmdir("output")


def test_initialize_perf_log_disabled(tmp_path):
    """パフォーマンスログが無効な場合のテスト"""
    # tmp_path はこのテストでは直接使われないが、pytestの慣習として引数に含める
    perf_log_file = initialize_perf_log(
        enable_perf_log=False,
        input_file=DUMMY_INPUT_FILE,
        model_path=DUMMY_MODEL_PATH,
        log_type="test_log",
    )
    assert perf_log_file is None


def test_initialize_perf_log_file_creation_error(mocker):
    # Patch get_system_info to avoid psutil usage (prevent collateral errors)
    mocker.patch(
        "src.tracking.bytetrack_utils.get_system_info",
        return_value={
            "os": "Linux",
            "os_version": "5.10",
            "python_version": "3.10.0",
            "cpu": "FakeCPU",
            "cpu_cores": 4,
            "cpu_threads": 8,
            "ram_total": 16,
        },
    )

    # Patch ONLY the built-in open inside the specific context
    with patch("builtins.open", side_effect=OSError("Test error: Cannot open file")):
        result = initialize_perf_log(
            enable_perf_log=True,
            input_file="test_video.mp4",
            model_path="test_model.pt",
            log_type="test_log",
        )

    assert result is None


def test_initialize_perf_log_with_long_stay_column(mocker):
    """Should insert 'Stay_Check_Time_ms' column when log_type is 'long_stay'."""

    # Patch system info
    mocker.patch(
        "src.tracking.bytetrack_utils.get_system_info",
        return_value={
            "os": "Linux",
            "os_version": "5.10",
            "python_version": "3.10.0",
            "cpu": "FakeCPU",
            "cpu_cores": 4,
            "cpu_threads": 8,
            "ram_total": 16,
        },
    )

    perf_log_file = initialize_perf_log(
        enable_perf_log=True,
        input_file=DUMMY_INPUT_FILE,
        model_path=DUMMY_MODEL_PATH,
        log_type="long_stay",
    )

    assert perf_log_file is not None
    log_file_path = Path(perf_log_file)
    assert log_file_path.exists()

    with open(log_file_path, encoding="utf-8") as f:
        lines = f.readlines()
        header = [col.strip() for col in lines[-1].strip().split(",")]
        assert "Stay_Check_Time_ms" in header
        assert header.index("Stay_Check_Time_ms") == 4  # Confirm correct insertion point

    # 🧹 Cleanup
    try:
        os.remove(log_file_path)
        logs_dir = log_file_path.parent
        if logs_dir.exists() and not any(logs_dir.iterdir()):
            logs_dir.rmdir()
        output_dir = logs_dir.parent
        if output_dir.exists() and not any(output_dir.iterdir()):
            output_dir.rmdir()
    except Exception:
        pass


# `get_system_info` は外部ライブラリに依存しているため、簡単な呼び出しテストのみ
def test_get_system_info():
    """get_system_info関数の基本的な動作テスト"""
    info = get_system_info()
    assert "os" in info
    assert "python_version" in info
    assert "cpu_cores" in info
    assert "ram_total" in info
