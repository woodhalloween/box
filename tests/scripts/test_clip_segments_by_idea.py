import os
import shutil
import subprocess
import unittest

import cv2
import pandas as pd


class TestClipSegmentsByIdea(unittest.TestCase):
    DUMMY_VIDEO_PATH = "tests/assets/dummy_video.mp4"
    DUMMY_CSV_PATH = "tests/assets/dummy_ideas.csv"
    OUTPUT_DIR = "tests/assets/output_segments"
    SCRIPT_PATH = "scripts/clip_segments_by_idea.py"

    def setUp(self):
        """各テストの前に呼ばれる"""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def tearDown(self):
        """各テストの後に呼ばれる"""
        # 出力ディレクトリとその中身をすべて削除
        if os.path.isdir(self.OUTPUT_DIR):
            shutil.rmtree(self.OUTPUT_DIR)

    def run_script(self, args):
        """スクリプトをサブプロセスで実行し、結果を返す"""
        command = ["python3", self.SCRIPT_PATH] + args
        return subprocess.run(command, capture_output=True, text=True)

    def get_frame_count(self, video_path):
        """動画の総フレーム数を取得する"""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()
        return frame_count

    def test_clip_successfully(self):
        """正常にセグメントが切り抜かれるかテストする"""
        result = self.run_script(
            ["--video-path", self.DUMMY_VIDEO_PATH, "--csv-path", self.DUMMY_CSV_PATH, "--output-dir", self.OUTPUT_DIR]
        )

        self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")

        # CSVファイルから期待される出力を読み込む
        df = pd.read_csv(self.DUMMY_CSV_PATH)
        expected_files_count = {}
        for _, row in df.iterrows():
            idea = row["idea"]

            # ideaごとのサブディレクトリが存在するかチェック
            idea_dir = os.path.join(self.OUTPUT_DIR, idea)
            self.assertTrue(os.path.isdir(idea_dir), f"Sub-directory for '{idea}' was not created.")

            count = expected_files_count.get(idea, 0) + 1
            expected_files_count[idea] = count

            video_basename = os.path.splitext(os.path.basename(self.DUMMY_VIDEO_PATH))[0]
            filename = f"{video_basename}_{row['start_frame']}_{row['end_frame']}.mp4"
            filepath = os.path.join(idea_dir, filename)

            self.assertTrue(os.path.exists(filepath), f"Output file was not created: {filepath}")

            expected_frames = row["end_frame"] - row["start_frame"] + 1
            actual_frames = self.get_frame_count(filepath)
            self.assertEqual(actual_frames, expected_frames, f"Frame count mismatch for {filename}")

    def test_missing_csv_column(self):
        """CSVの必須カラムが欠けている場合のテスト"""
        # 'idea' カラムのない不正なCSVを作成
        bad_csv_path = "tests/assets/bad_ideas.csv"
        with open(bad_csv_path, "w") as f:
            f.write("start_frame,end_frame\n1,2\n")

        result = self.run_script(
            ["--video-path", self.DUMMY_VIDEO_PATH, "--csv-path", bad_csv_path, "--output-dir", self.OUTPUT_DIR]
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required columns", result.stderr.lower())

        os.remove(bad_csv_path)

    def test_non_existent_input(self):
        """存在しない入力ファイルやCSVを指定した場合のテスト"""
        result = self.run_script(
            ["--video-path", "non_existent.mp4", "--csv-path", self.DUMMY_CSV_PATH, "--output-dir", self.OUTPUT_DIR]
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not found", result.stderr.lower())
