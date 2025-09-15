import pandas as pd
import sys
import os


def analyze_stay_duration(csv_path):
    """CSVファイルを分析し、滞在時間に関する統計情報を出力する。"""
    try:
        if not os.path.exists(csv_path):
            print(f"エラー: ファイルが見つかりません - {csv_path}", file=sys.stderr)
            return

        df = pd.read_csv(csv_path)

        if "stay_duration" not in df.columns or "timestamp" not in df.columns:
            print(
                f"エラー: 必要な列 ('stay_duration', 'timestamp') が見つかりません in {csv_path}",
                file=sys.stderr,
            )
            return

        # 動画の総時間
        total_duration = df["timestamp"].max()
        if total_duration == 0:
            print(f"警告: 動画の総時間がゼロです in {csv_path}", file=sys.stderr)
            return

        # 最も長かった連続滞在時間
        longest_stay = df["stay_duration"].max()

        # 滞在がリセットされた箇所（＝移動が確定した箇所）を特定する
        # stay_durationが前のフレームより1秒以上急に短くなったらリセットと見なす
        resets = df["stay_duration"].diff() < -1
        # リセット直前のフレームが、各滞在期間の終わり
        ended_stay_periods = df["stay_duration"][resets.shift(1).fillna(False)]

        # 最後の滞在期間はリセットで終わらないため、最終行の値を別途追加する
        last_stay_period = df["stay_duration"].iloc[-1]
        all_stay_durations = pd.concat([ended_stay_periods, pd.Series([last_stay_period])])

        # 合計滞在時間
        total_stay_time = all_stay_durations.sum()

        # 滞在期間の数
        num_stays = len(all_stay_durations)

        # 平均滞在時間
        avg_stay = total_stay_time / num_stays if num_stays > 0 else 0

        # 滞在時間の割合
        stay_percentage = (total_stay_time / total_duration) * 100

        print(f"--- 分析結果: {os.path.basename(csv_path)} ---")
        print(f"動画の総時間: {total_duration:.2f} 秒")
        print(f"最も長かった連続滞在時間: {longest_stay:.2f} 秒")
        print(f"合計滞在時間 (推定): {total_stay_time:.2f} 秒")
        print(f"滞在時間の割合: {stay_percentage:.1f}%")
        print(f"滞在期間の数: {num_stays} 回")
        print(f"平均滞在時間: {avg_stay:.2f} 秒")
        print("-" * 40)

    except Exception as e:
        print(f"{csv_path} の分析中にエラーが発生しました: {e}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python analysis/analyze_stay_time.py <CSVファイル1> [<CSVファイル2> ...]", file=sys.stderr)
        sys.exit(1)

    for file_path in sys.argv[1:]:
        analyze_stay_duration(file_path)
