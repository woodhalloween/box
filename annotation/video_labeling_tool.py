#!/usr/bin/env python3
"""
動画ラベル付けツール

動画を再生しながらキーボード操作でリアルタイムにラベル付けを行います。
首振りなどのイベントの開始/終了を記録し、CSV形式で保存します。

使い方:
    python3 video_labeling_tool.py <video_path> <output_csv>

キーボード操作:
    1: 首振りイベントの開始/終了をトグル
    Space: 一時停止/再生
    ←: 5秒巻き戻し
    →: 5秒早送り
    q: 終了して保存
"""

import csv
import sys
from datetime import timedelta
from pathlib import Path

import cv2


class VideoLabelingTool:
    def __init__(self, video_path, output_csv):
        self.video_path = video_path
        self.output_csv = output_csv

        # 動画読み込み
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"動画ファイルを開けません: {video_path}")

        # 動画情報取得
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 状態管理
        self.current_frame = 0
        self.paused = False
        self.recording_head_shake = False
        self.head_shake_start_frame = None

        # ラベルデータ
        self.labels = []

        print("=== 動画情報 ===")
        print(f"ファイル: {video_path}")
        print(f"FPS: {self.fps:.2f}")
        print(f"総フレーム数: {self.total_frames}")
        print(f"解像度: {self.width}x{self.height}")
        print(f"総時間: {self._format_time(self.total_frames / self.fps)}")
        print()
        print("=== 操作方法 ===")
        print("1: 首振りイベントの開始/終了をトグル")
        print("Space: 一時停止/再生")
        print("←: 5秒巻き戻し")
        print("→: 5秒早送り")
        print("q: 終了して保存")
        print()

    def _format_time(self, seconds):
        """秒数を HH:MM:SS.mmm 形式に変換"""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def _get_current_time(self):
        """現在のフレームの時刻を取得"""
        return self.current_frame / self.fps

    def _draw_info(self, frame):
        """フレームに情報を描画"""
        overlay = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 背景
        cv2.rectangle(overlay, (10, 10), (self.width - 10, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # フレーム情報
        current_time = self._get_current_time()
        cv2.putText(frame, f"Frame: {self.current_frame}/{self.total_frames}", (20, 40), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {self._format_time(current_time)}", (20, 70), font, 0.7, (255, 255, 255), 2)

        # 再生状態
        status = "PAUSED" if self.paused else "PLAYING"
        color = (0, 165, 255) if self.paused else (0, 255, 0)
        cv2.putText(frame, f"Status: {status}", (20, 100), font, 0.7, color, 2)

        # 記録状態
        if self.recording_head_shake:
            start_time = self.head_shake_start_frame / self.fps
            duration = current_time - start_time
            cv2.putText(frame, f"[REC] HEAD SHAKE - Duration: {duration:.2f}s", (20, 130), font, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Ready to label (Press '1' for head shake)", (20, 130), font, 0.6, (200, 200, 200), 1)

        # ラベル数
        cv2.putText(frame, f"Total Labels: {len(self.labels)}", (20, 160), font, 0.7, (255, 255, 0), 2)

        return frame

    def _toggle_head_shake(self):
        """首振りイベントの記録を開始/終了"""
        if not self.recording_head_shake:
            # 記録開始
            self.recording_head_shake = True
            self.head_shake_start_frame = self.current_frame
            print(f"✅ 首振り記録開始: Frame {self.current_frame} ({self._format_time(self._get_current_time())})")
        else:
            # 記録終了
            self.recording_head_shake = False
            end_frame = self.current_frame
            start_time = self.head_shake_start_frame / self.fps
            end_time = end_frame / self.fps

            # ラベルを保存
            label = {
                "start_frame": self.head_shake_start_frame,
                "end_frame": end_frame,
                "start_time": start_time,
                "end_time": end_time,
                "event_type": "HEAD_SHAKE",
            }
            self.labels.append(label)

            duration = end_time - start_time
            print(
                f"⏹️  首振り記録終了: Frame {self.head_shake_start_frame}-{end_frame} "
                f"({self._format_time(start_time)} - {self._format_time(end_time)}, {duration:.2f}s)"
            )
            print(f"   総ラベル数: {len(self.labels)}")

    def _seek_frames(self, offset):
        """指定フレーム数だけシーク"""
        new_frame = max(0, min(self.current_frame + offset, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        self.current_frame = new_frame
        print(f"⏩ Seek to Frame {self.current_frame} ({self._format_time(self._get_current_time())})")

    def _save_labels(self):
        """ラベルをCSVファイルに保存"""
        # 出力ディレクトリを作成
        output_path = Path(self.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # CSV書き込み
        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["start_frame", "end_frame", "start_time", "end_time", "event_type"])
            writer.writeheader()
            writer.writerows(self.labels)

        print(f"\n✅ ラベルを保存しました: {self.output_csv}")
        print(f"   総ラベル数: {len(self.labels)}")

    def run(self):
        """ラベル付けツールを実行"""
        cv2.namedWindow("Video Labeling Tool", cv2.WINDOW_NORMAL)

        try:
            while True:
                if not self.paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("動画の終端に到達しました")
                        break

                    self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                else:
                    # 一時停止中は現在のフレームを再表示
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                # 情報を描画
                frame = self._draw_info(frame)

                # 表示
                cv2.imshow("Video Labeling Tool", frame)

                # キー入力処理
                key = cv2.waitKey(30 if not self.paused else 100) & 0xFF

                if key == ord("q"):
                    print("\n終了します...")
                    break
                if key == ord("1"):
                    self._toggle_head_shake()
                elif key == ord(" "):
                    self.paused = not self.paused
                    status = "一時停止" if self.paused else "再生"
                    print(f"▶️  {status}")
                elif key == 81 or key == 2:  # 左矢印
                    self._seek_frames(-int(5 * self.fps))
                elif key == 83 or key == 3:  # 右矢印
                    self._seek_frames(int(5 * self.fps))

        finally:
            # 記録中のイベントがあれば自動終了
            if self.recording_head_shake:
                print("\n⚠️  記録中のイベントを自動終了します")
                self._toggle_head_shake()

            # ラベルを保存
            if self.labels:
                self._save_labels()
            else:
                print("\n⚠️  ラベルが1つも作成されませんでした")

            # クリーンアップ
            self.cap.release()
            cv2.destroyAllWindows()


def main():
    if len(sys.argv) != 3:
        print("使い方: python3 video_labeling_tool.py <video_path> <output_csv>")
        print()
        print("例:")
        print(
            "  python3 video_labeling_tool.py ../data/raw/国宝さん手をあげる2.mp4 "
            "ground_truth/国宝さん手をあげる2_labels.csv"
        )
        sys.exit(1)

    video_path = sys.argv[1]
    output_csv = sys.argv[2]

    # ビデオファイルの存在確認
    if not Path(video_path).exists():
        print(f"エラー: 動画ファイルが見つかりません: {video_path}")
        sys.exit(1)

    # ラベル付けツールを実行
    tool = VideoLabelingTool(video_path, output_csv)
    tool.run()


if __name__ == "__main__":
    main()
