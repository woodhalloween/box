import argparse
import pandas as pd
import numpy as np
import sys

def analyze_ideas(csv_file: str):
    """
    指定されたCSVファイルを分析し、5つの異なるアイデアに基づいて動作フレームを検出します。

    この関数は以下の処理を実行します：
    1. CSVファイルをpandas DataFrameとして読み込みます。
    2. 'person_scale'がほぼゼロの行（信頼性が低いデータ）を除外します。
    3. 各フレーム間の腰中心（'hip_center_x', 'hip_center_y'）の移動距離をピクセル単位で計算します（'distance_px'）。
    4. 移動距離を'person_scale'で正規化し、体格に依存しない移動量（'distance_normalized'）を算出します。
    5. 以下の5つのアイデアを適用し、それぞれで移動が検知されたフレーム数をカウントします。
        - アイデア1（単純な閾値）: 'distance_normalized'が0.1以上のフレーム。
        - アイデア2（スパイク検知）: 1秒間（30フレーム）の'distance_normalized'の最大値が1.5以上のフレーム。
        - アイデア3（累積移動量）: 3秒間（90フレーム）の'distance_normalized'の合計が5.0を超えるフレーム。
        - アイデア4（安定性チェック）: 2秒間（60フレーム）の'person_scale'の標準偏差が50.0を超えるフレーム。
        - アイデア5（複合条件）: アイデア2またはアイデア4のいずれかの条件を満たすフレーム。
    6. 総フレーム数と、各アイデアで検知されたフレーム数を格納した辞書を返します。

    Args:
        csv_file (str): 分析対象のCSVファイルへのパス。

    Returns:
        dict or None:
            成功した場合、キーとして 'total', 'idea1'〜'idea5' を持ち、
            値としてフレーム数を持つ辞書。
            処理中にエラーが発生した場合はNoneを返します。
    """
    try:
        df = pd.read_csv(csv_file)
        
        # --- 事前計算 ---
        threshold = 1e-6
        df_filtered = df[(df["person_scale"].abs() > threshold)].copy()
        
        df_filtered.loc[:, "prev_x"] = df_filtered["hip_center_x"].shift(1)
        df_filtered.loc[:, "prev_y"] = df_filtered["hip_center_y"].shift(1)
        df_filtered.dropna(subset=["prev_x", "prev_y"], inplace=True)
        
        if df_filtered.empty: return None

        df_filtered.loc[:, "distance_px"] = np.sqrt(
            (df_filtered["hip_center_x"] - df_filtered["prev_x"])**2 +
            (df_filtered["hip_center_y"] - df_filtered["prev_y"])**2
        )
        df_filtered.loc[:, "distance_normalized"] = df_filtered["distance_px"] / df_filtered["person_scale"]
        
        total_frames = len(df_filtered)
        if total_frames == 0: return None

        # --- 5つのアイデアを適用 ---

        # アイデア1：単純な閾値 (0.1)
        idea1_frames = (df_filtered['distance_normalized'] >= 0.1).sum()

        # アイデア2：スパイク検知 (1秒/30フレームで1.5以上)
        is_spike = df_filtered['distance_normalized'].rolling(window=30, min_periods=1).max() >= 1.5
        idea2_frames = is_spike.sum()
        
        # アイデア3：累積移動量 (3秒/90フレームの合計が5.0を超える)
        is_cumulative_move = df_filtered['distance_normalized'].rolling(window=90, min_periods=1).sum() > 5.0
        idea3_frames = is_cumulative_move.sum()

        # アイデア4：安定性 (2秒/60フレームでの体幹長の標準偏差が50pxを超える)
        is_unstable = df_filtered['person_scale'].rolling(window=60, min_periods=1).std().fillna(0) > 50.0
        idea4_frames = is_unstable.sum()
        
        # アイデア5：複合条件 (アイデア2またはアイデア4)
        is_composite_move = is_spike | is_unstable
        idea5_frames = is_composite_move.sum()
        
        return {
            "total": total_frames,
            "idea1": idea1_frames,
            "idea2": idea2_frames,
            "idea3": idea3_frames,
            "idea4": idea4_frames,
            "idea5": idea5_frames,
        }

    except Exception as e:
        print(f"{csv_file} の処理中にエラーが発生しました: {e}", file=sys.stderr)
        return None

def main(file1: str, file2: str):
    """2つのCSVファイルに対して5つのアイデアの分析結果を比較します。"""
    results1 = analyze_ideas(file1)
    results2 = analyze_ideas(file2)

    if not results1 or not results2:
        sys.exit(1)

    # --- 比較表の出力 ---
    header = f"| {'アイデアと条件':<45} | {'動画1':<20} | {'動画2':<20} |"
    separator = "-" * len(header)
    
    print(separator)
    print(header)
    print(separator)

    def print_row(name, desc, res1, res2):
        total1, total2 = results1['total'], results2['total']
        val1 = f"{res1} ({res1/total1:.1%})"
        val2 = f"{res2} ({res2/total2:.1%})"
        print(f"| {name:<45} | {val1:<20} | {val2:<20} |")
        print(f"|  - {desc:<42} | {'':<20} | {'':<20} |")

    print_row("アイデア1：単純な閾値", "正規化移動量 >= 0.1", results1['idea1'], results2['idea1'])
    print(separator)
    print_row("アイデア2：スパイク検知", "1秒間の最大正規化移動量 >= 1.5", results1['idea2'], results2['idea2'])
    print(separator)
    print_row("アイデア3：累積移動量", "3秒間の正規化移動量の合計 > 5.0", results1['idea3'], results2['idea3'])
    print(separator)
    print_row("アイデア4：安定性チェック", "2秒間の体幹長の標準偏差 > 50px", results1['idea4'], results2['idea4'])
    print(separator)
    print_row("アイデア5：複合条件 (2または4)", "スパイク検知または安定性チェックが真", results1['idea5'], results2['idea5'])
    print(separator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5つの動作検知アイデアをシミュレートし、比較します。")
    parser.add_argument("file1", type=str, help="1つ目のCSVファイルへのパス。")
    parser.add_argument("file2", type=str, help="2つ目のCSVファイルへのパス。")
    args = parser.parse_args()
    
    main(args.file1, args.file2)
