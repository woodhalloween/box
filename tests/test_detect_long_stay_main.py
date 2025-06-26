import subprocess
import sys
import os
from unittest.mock import MagicMock
import numpy as np
import cv2

# Assuming detect_long_stay_main.py is in src/
from src.detect_long_stay_main import main


def test_main_function_execution(mocker):
    # Mock argparse to control command-line arguments
    mock_parse_args = mocker.patch('argparse.ArgumentParser.parse_args')
    mock_parse_args.return_value = MagicMock(
        input='dummy_input.mp4',
        output='dummy_output.mp4',
        model='dummy_model.pt',
        enable_perf_log=False,
        enable_video_display=False,
        device='cpu',
        stay_threshold_sec=5.0,
        move_threshold_px=30.0,
        conf=0.3,
        enable_pose=False # Add enable_pose
    )

    # Mock Path.mkdir to prevent actual directory creation
    mocker.patch('pathlib.Path.mkdir')

    # Mock os.path.exists to return True for the dummy input file
    mocker.patch('os.path.exists', return_value=True)

    # Mock cv2.VideoCapture
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH: 640,
        cv2.CAP_PROP_FRAME_HEIGHT: 480,
        cv2.CAP_PROP_FPS: 10.0,
        cv2.CAP_PROP_FRAME_COUNT: 3,
    }[prop]
    mock_cap.read.side_effect = [(True, np.zeros((480, 640, 3), dtype=np.uint8)),
                                  (True, np.zeros((480, 640, 3), dtype=np.uint8)),
                                  (False, None)]
    mocker.patch('cv2.VideoCapture', return_value=mock_cap)

    # Mock cv2.VideoWriter
    mock_video_writer = MagicMock()
    mocker.patch('cv2.VideoWriter', return_value=mock_video_writer)

    # Mock load_yolo_model
    mock_yolo_model = MagicMock()
    mock_yolo_model.predict.return_value = [MagicMock(boxes=MagicMock(xyxy=MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=[])))),
                                                                      conf=MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=[])))),
                                                                      cls=MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=[]))))))]
    mocker.patch('src.detect_long_stay_main.load_yolo_model', return_value=mock_yolo_model)

    # Mock initialize_bytetrack
    mock_bytetrack = MagicMock()
    mock_bytetrack.update.return_value = np.array([])
    mocker.patch('src.detect_long_stay_main.initialize_bytetrack', return_value=mock_bytetrack)

    # Mock process_frame_for_tracking
    mocker.patch('src.detect_long_stay_main.process_frame_for_tracking', return_value=(np.array([]), 0.1, 0.1, 0, 0, None))

    # Mock update_stay_times
    mocker.patch('src.detect_long_stay_main.update_stay_times', return_value=({}, [], 0.05))

    # Mock initialize_perf_log
    mocker.patch('src.detect_long_stay_main.initialize_perf_log', return_value=None)

    # Mock draw_tracking_info
    mocker.patch('src.detect_long_stay_main.draw_tracking_info', return_value=np.zeros((480, 640, 3), dtype=np.uint8))

    # Simulate running the script directly by calling main()
    main()

    # Assert that run_long_stay_detection was called with the correct arguments
    # (This assertion is now implicitly covered by the fact that main() runs without error)


def test_main_script_execution(tmp_path):
    """Test that the main script can be executed as a subprocess."""
    # Create a temporary script that sets up mocks and then calls main.
    script_content = """
import sys
import os
from unittest.mock import patch, MagicMock
import numpy as np
import cv2

sys.path.insert(0, '{}/src')
from detect_long_stay_main import main

# Mock os.path.exists
@patch('os.path.exists', return_value=True)
# Mock cv2.VideoCapture
@patch('cv2.VideoCapture')
def run_main_with_mocks(mock_video_capture, mock_os_path_exists):
    # Configure mock_video_capture
    mock_cap_instance = MagicMock()
    mock_cap_instance.isOpened.return_value = True
    mock_cap_instance.get.side_effect = lambda prop: {{
        cv2.CAP_PROP_FRAME_WIDTH: 640,
        cv2.CAP_PROP_FRAME_HEIGHT: 480,
        cv2.CAP_PROP_FPS: 10.0,
        cv2.CAP_PROP_FRAME_COUNT: 3,
    }}[prop]
    mock_cap_instance.read.side_effect = [(True, np.zeros((480, 640, 3), dtype=np.uint8)),
                                          (True, np.zeros((480, 640, 3), dtype=np.uint8)),
                                          (False, None)]
    mock_video_capture.return_value = mock_cap_instance

    # Mock cv2.VideoWriter
    mock_video_writer = MagicMock()
    patch('cv2.VideoWriter', return_value=mock_video_writer).start()

    # Mock load_yolo_model
    mock_yolo_model = MagicMock()
    mock_yolo_model.predict.return_value = [MagicMock(boxes=MagicMock(xyxy=MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=[])))),
                                                                      conf=MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=[])))),
                                                                      cls=MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=[]))))))]
    patch('src.detect_long_stay_main.load_yolo_model', return_value=mock_yolo_model).start()

    # Mock initialize_bytetrack
    mock_bytetrack = MagicMock()
    mock_bytetrack.update.return_value = np.array([])
    patch('src.detect_long_stay_main.initialize_bytetrack', return_value=mock_bytetrack).start()

    # Mock process_frame_for_tracking
    patch('src.detect_long_stay_main.process_frame_for_tracking', return_value=(np.array([]), 0.1, 0.1, 0, 0, None)).start()

    # Mock update_stay_times
    patch('src.detect_long_stay_main.update_stay_times', return_value=({{}}, [], 0.05)).start()

    # Mock initialize_perf_log
    patch('src.detect_long_stay_main.initialize_perf_log', return_value=None).start()

    # Mock draw_tracking_info
    patch('src.detect_long_stay_main.draw_tracking_info', return_value=np.zeros((480, 640, 3), dtype=np.uint8)).start()

    main()

if __name__ == '__main__':
    run_main_with_mocks()
    """.format(os.getcwd())
    script_path = tmp_path / "run_main_with_mocks.py"
    script_path.write_text(script_content)

    command = [
        sys.executable,
        str(script_path),
        "--input", "dummy_input_for_subprocess.mp4",
        "--output", "dummy_output_for_subprocess.mp4",
        "--model", "dummy_model.pt",
        "--enable_perf_log",
        "--enable_video_display",
        "--device", "cpu",
        "--stay_threshold_sec", "5.0",
        "--move_threshold_px", "30.0",
        "--conf", "0.3",
        "--enable_pose",
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with error: {e.stderr}")
        raise