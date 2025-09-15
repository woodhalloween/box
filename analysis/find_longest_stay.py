import pandas as pd
import sys
import os


def find_longest_stay_period(csv_path):
    """
    CSVファイルを分析し、最も長かった連続滞在期間の
    開始フレームと終了フレームを特定する。
    """
    try:
        if not os.path.exists(csv_path):
            print(f"エラー: ファイルが見つかりません - {csv_path}", file=sys.stderr)
            return

        df = pd.read_csv(csv_path)

        if "stay_duration" not in df.columns or "timestamp" not in df.columns or "frame_number" not in df.columns:
            print(
                f"エラー: 必要な列 ('stay_duration', 'timestamp', 'frame_number') が見つかりません in {csv_path}",
                file=sys.stderr,
            )
            return

        # 最も長い滞在時間とその時点（期間の終わり）のインデックスを取得
        longest_stay_duration = df["stay_duration"].max()
        end_idx = df["stay_duration"].idxmax()

        # 期間の終わりの情報を取得
        end_row = df.loc[end_idx]
        end_frame = int(end_row["frame_number"])
        end_timestamp = end_row["timestamp"]

        # 期間の始まりのタイムスタンプを計算
        start_timestamp = end_timestamp - longest_stay_duration

        # 始まりのタイムスタンプに最も近いフレームを見つける
        # `iloc[0]` を使って、複数の候補がある場合は最初のものを採用
        start_row_idx = (df["timestamp"] - start_timestamp).abs().idxmin()
        start_row = df.loc[start_row_idx]
        start_frame = int(start_row["frame_number"])

        print(f"--- 分析結果: {os.path.basename(csv_path)} ---")
        print(f"最も長い連続滞在時間: {longest_stay_duration:.2f} 秒")
        print(f"期間: フレーム {start_frame} から フレーム {end_frame} まで")
        print(f"(タイムスタンプ: {start_row['timestamp']:.2f}秒 から {end_timestamp:.2f}秒 まで)")
        print("-" * 40)

    except Exception as e:
        print(f"{csv_path} の分析中にエラーが発生しました: {e}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使い方: python analysis/find_longest_stay.py <CSVファイル>", file=sys.stderr)
        sys.exit(1)

    file_path = sys.argv[1]
    find_longest_stay_period(file_path)



