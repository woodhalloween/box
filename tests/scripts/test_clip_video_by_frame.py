import os
import subprocess
import unittest

import cv2


class TestClipVideoByFrame(unittest.TestCase):
    DUMMY_VIDEO_PATH = "tests/assets/dummy_video.mp4"
    OUTPUT_VIDEO_PATH = "tests/assets/output_clip.mp4"
    SCRIPT_PATH = "scripts/clip_video_by_frame.py"

    @classmethod
    def setUpClass(cls):
        """テストクラスの開始時に一度だけ呼ばれる"""
        # ダミー動画が存在することを確認
        assert os.path.exists(cls.DUMMY_VIDEO_PATH), f"Dummy video not found at {cls.DUMMY_VIDEO_PATH}"

    def tearDown(self):
        """各テストメソッドの実行後に呼ばれる"""
        # テストで生成された出力ファイルを削除
        if os.path.exists(self.OUTPUT_VIDEO_PATH):
            os.remove(self.OUTPUT_VIDEO_PATH)

    def run_script(self, args):
        """スクリプトをサブプロセスで実行し、結果を返す"""
        command = ["python3", self.SCRIPT_PATH] + args
        return subprocess.run(command, capture_output=True, text=True)

    def get_frame_count(self, video_path):
        """動画の総フレーム数を取得する"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count

    def test_clip_successfully(self):
        """動画が正常に切り抜かれるかテストする"""
        start_frame = 30
        end_frame = 60
        expected_frames = end_frame - start_frame + 1

        result = self.run_script(
            [
                "--input",
                self.DUMMY_VIDEO_PATH,
                "--start",
                str(start_frame),
                "--end",
                str(end_frame),
                "--output",
                self.OUTPUT_VIDEO_PATH,
            ]
        )

        self.assertEqual(result.returncode, 0, f"Script failed with error: {result.stderr}")
        self.assertTrue(os.path.exists(self.OUTPUT_VIDEO_PATH), "Output video was not created.")

        actual_frames = self.get_frame_count(self.OUTPUT_VIDEO_PATH)
        self.assertEqual(actual_frames, expected_frames, "The frame count of the clipped video is incorrect.")

    def test_invalid_input_file(self):
        """存在しない入力ファイルを指定した場合のテスト"""
        result = self.run_script(
            ["--input", "non_existent_video.mp4", "--start", "0", "--end", "10", "--output", self.OUTPUT_VIDEO_PATH]
        )

        self.assertNotEqual(result.returncode, 0, "Script should fail for non-existent input file.")
        self.assertIn("does not exist", result.stderr.lower())

    def test_invalid_frame_range(self):
        """不正なフレーム範囲を指定した場合のテスト"""
        result = self.run_script(
            ["--input", self.DUMMY_VIDEO_PATH, "--start", "100", "--end", "50", "--output", self.OUTPUT_VIDEO_PATH]
        )

        self.assertNotEqual(result.returncode, 0, "Script should fail for invalid frame range.")
        self.assertIn("end frame must be greater than", result.stderr.lower())
