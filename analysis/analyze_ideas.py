import argparse
import pandas as pd
import numpy as np
import sys
import os
from scipy import stats as scipy_stats

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
    6. 総フレーム数、各アイデアで検知されたフレーム数、正規化移動量の記述統計量を格納した辞書を返します。

    Args:
        csv_file (str): 分析対象のCSVファイルへのパス。

    Returns:
        dict or None:
            成功した場合、キーとして 'total', 'idea1'〜'idea5', 'stats', 'series', 'output_path' を持ち、
            値としてフレーム数、統計情報、移動量データ、出力ファイルパスを持つ辞書。
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
        
        total_frames = len(df_filtered)
        if total_frames == 0:
            stats = {"mean": np.nan, "median": np.nan, "std": np.nan, "max": np.nan, "min": np.nan,
                     "q1": np.nan, "q3": np.nan, "skew": np.nan, "kurtosis": np.nan}
            return {"total": 0, "idea1": 0, "idea2": 0, "idea3": 0, "idea4": 0, "idea5": 0, "stats": stats, "series": pd.Series(), "output_path": None}

        df_filtered.loc[:, "distance_px"] = np.sqrt(
            (df_filtered["hip_center_x"] - df_filtered["prev_x"])**2 +
            (df_filtered["hip_center_y"] - df_filtered["prev_y"])**2
        )
        df_filtered.loc[:, "distance_normalized"] = df_filtered["distance_px"] / df_filtered["person_scale"]
        
        # --- 5つのアイデアを適用 ---

        # アイデア1：単純な閾値 (0.1)
        idea1_frames = (df_filtered['distance_normalized'] >= 0.1).sum()

        # アイデア2：スパイク検知 (1秒間（30フレーム）の'distance_normalized'の最大値が1.5以上)
        is_spike = df_filtered['distance_normalized'].rolling(window=30, min_periods=1).max() >= 1.5
        idea2_frames = is_spike.sum()
        df_filtered.loc[:,"idea2_is_spike"] = is_spike
        
        # アイデア3：累積移動量 (3秒間（90フレーム）の'distance_normalized'の合計が5.0を超える)
        is_cumulative_move = df_filtered['distance_normalized'].rolling(window=90, min_periods=1).sum() > 5.0
        idea3_frames = is_cumulative_move.sum()
        df_filtered.loc[:,"idea3_is_cumulative_move"] = is_cumulative_move

        # アイデア4：安定性 (2秒間（60フレーム）の'person_scale'の標準偏差が50pxを超える)
        is_unstable = df_filtered['person_scale'].rolling(window=60, min_periods=1).std().fillna(0) > 50.0
        idea4_frames = is_unstable.sum()
        df_filtered.loc[:,"idea4_is_unstable"] = is_unstable
        
        # アイデア5：複合条件 (アイデア2またはアイデア4)
        is_composite_move = is_spike | is_unstable
        idea5_frames = is_composite_move.sum()
        df_filtered.loc[:,"idea5_is_composite_move"] = is_composite_move
        
        # --- 首振り検知の分析 ---
        head_shake_h_frames = df_filtered['head_shake_horizontal_detected'].sum()
        head_shake_v_frames = df_filtered['head_shake_vertical_detected'].sum()

        # --- 膝角度アラートの分析 ---
        KNEE_ANGLE_THRESHOLD = 90.0
        MOVING_WINDOW_SECONDS = 5
        
        knee_alert_seconds = 0
        total_seconds = 0
        if 'timestamp' in df_filtered.columns and not df_filtered.empty:
            total_seconds = df_filtered['timestamp'].max()
            df_filtered['seconds'] = df_filtered['timestamp'].astype(int)
            
            if not df_filtered.empty:
                # 秒ごとの中央値を計算
                knee_angles_sec = df_filtered.groupby('seconds').agg(
                    left_knee_median=('left_knee_angle', 'median'),
                    right_knee_median=('right_knee_angle', 'median')
                ).reset_index()

                # 5秒移動平均を計算
                knee_angles_sec['left_knee_ma5'] = knee_angles_sec['left_knee_median'].rolling(window=MOVING_WINDOW_SECONDS, min_periods=1).mean()
                knee_angles_sec['right_knee_ma5'] = knee_angles_sec['right_knee_median'].rolling(window=MOVING_WINDOW_SECONDS, min_periods=1).mean()

                # アラート条件を判定
                is_median_low = (knee_angles_sec['left_knee_median'] < KNEE_ANGLE_THRESHOLD) | (knee_angles_sec['right_knee_median'] < KNEE_ANGLE_THRESHOLD)
                is_ma5_low = (knee_angles_sec['left_knee_ma5'] < KNEE_ANGLE_THRESHOLD) | (knee_angles_sec['right_knee_ma5'] < KNEE_ANGLE_THRESHOLD)
                
                knee_alert_seconds = (is_median_low | is_ma5_low).sum()

        # --- 記述統計量の計算 ---
        stats = {
            "mean": df_filtered["distance_normalized"].mean(),
            "median": df_filtered["distance_normalized"].median(),
            "std": df_filtered["distance_normalized"].std(),
            "max": df_filtered["distance_normalized"].max(),
            "min": df_filtered["distance_normalized"].min(),
            "q1": df_filtered["distance_normalized"].quantile(0.25),
            "q3": df_filtered["distance_normalized"].quantile(0.75),
            "skew": df_filtered["distance_normalized"].skew(),
            "kurtosis": df_filtered["distance_normalized"].kurtosis(),
        }

        # ---膝角度の統計量 ---
        left_knee_stats = {
            "mean": df_filtered["left_knee_angle"].mean(),
            "median": df_filtered["left_knee_angle"].median(),
            "std": df_filtered["left_knee_angle"].std(),
            "max": df_filtered["left_knee_angle"].max(),
            "min": df_filtered["left_knee_angle"].min(),
        }
        right_knee_stats = {
            "mean": df_filtered["right_knee_angle"].mean(),
            "median": df_filtered["right_knee_angle"].median(),
            "std": df_filtered["right_knee_angle"].std(),
            "max": df_filtered["right_knee_angle"].max(),
            "min": df_filtered["right_knee_angle"].min(),
        }
        
        # --- 首振り角度の統計量 ---
        head_h_stats = {
            "mean": df_filtered["head_horizontal_rotation_angle"].mean(),
            "median": df_filtered["head_horizontal_rotation_angle"].median(),
            "std": df_filtered["head_horizontal_rotation_angle"].std(),
            "max": df_filtered["head_horizontal_rotation_angle"].max(),
            "min": df_filtered["head_horizontal_rotation_angle"].min(),
        }
        head_v_stats = {
            "mean": df_filtered["head_vertical_nod_angle"].mean(),
            "median": df_filtered["head_vertical_nod_angle"].median(),
            "std": df_filtered["head_vertical_nod_angle"].std(),
            "max": df_filtered["head_vertical_nod_angle"].max(),
            "min": df_filtered["head_vertical_nod_angle"].min(),
        }
        
        # --- 計算結果をCSVに出力 ---
        output_dir = os.path.dirname(csv_file)
        base_name = os.path.basename(csv_file)
        file_name, file_ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{file_name}_with_calc.csv")
        df_filtered.to_csv(output_path, index=False)

        return {
            "total": total_frames,
            "idea1": idea1_frames,
            "idea2": idea2_frames,
            "idea3": idea3_frames,
            "idea4": idea4_frames,
            "idea5": idea5_frames,
            "head_shake_h": head_shake_h_frames,
            "head_shake_v": head_shake_v_frames,
            "knee_alert_seconds": knee_alert_seconds,
            "total_seconds": total_seconds,
            "stats": stats,
            "left_knee_stats": left_knee_stats,
            "right_knee_stats": right_knee_stats,
            "head_h_stats": head_h_stats,
            "head_v_stats": head_v_stats,
            "series": df_filtered["distance_normalized"],
            "output_path": output_path,
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

    # --- アイデア比較のDataFrameを作成 ---
    total1, total2 = results1['total'], results2['total']
    
    def format_val(res, total):
        return f"{res} ({res/total:.1%})" if total > 0 else "0 (0.0%)"

    ideas_data = {
        'アイデア': [
            'アイデア1：単純な閾値',
            'アイデア2：スパイク検知',
            'アイデア3：累積移動量',
            'アイデア4：安定性チェック',
            'アイデア5：複合条件 (2または4)',
        ],
        '条件': [
            '正規化移動量 >= 0.1',
            '1秒間の最大正規化移動量 >= 1.5',
            '3秒間の正規化移動量の合計 > 5.0',
            '2秒間の体幹長の標準偏差 > 50px',
            'スパイク検知または安定性チェックが真',
        ],
        '動画1': [format_val(results1[f'idea{i}'], total1) for i in range(1, 6)],
        '動画2': [format_val(results2[f'idea{i}'], total2) for i in range(1, 6)],
    }
    ideas_df = pd.DataFrame(ideas_data).set_index('アイデア')

    # --- 統計量比較のDataFrameを作成 ---
    stats_data = {
        '動画1': results1['stats'],
        '動画2': results2['stats'],
    }
    index_labels = [
        '平均', '中央値', '標準偏差', '最大値', '最小値',
        '第1四分位数 (25%)', '第3四分位数 (75%)', '歪度 (Skewness)', '尖度 (Kurtosis)'
    ]
    stats_df = pd.DataFrame(stats_data, index=index_labels)

    # --- 結果の出力 ---
    print("--- 動作検知アイデアの比較 ---")
    print(ideas_df.to_string())
    
    # --- その他の分析比較 ---
    total_seconds1 = results1['total_seconds']
    total_seconds2 = results2['total_seconds']

    def format_seconds(res, total):
        return f"{res} ({res/total:.1%})" if total > 0 else "0 (0.0%)"

    other_analysis_data = {
        '分析項目': [
            '水平方向の首振り (フレーム数)',
            '垂直方向の首振り (フレーム数)',
            '膝角度低下 (秒数)',
        ],
        '動画1': [
            format_val(results1['head_shake_h'], total1),
            format_val(results1['head_shake_v'], total1),
            format_seconds(results1['knee_alert_seconds'], total_seconds1),
        ],
        '動画2': [
            format_val(results2['head_shake_h'], total2),
            format_val(results2['head_shake_v'], total2),
            format_seconds(results2['knee_alert_seconds'], total_seconds2),
        ],
    }
    other_df = pd.DataFrame(other_analysis_data).set_index('分析項目')

    print("\n" + "-"*80)
    print("--- その他の分析比較 ---")
    print(other_df.to_string())

    print("\n" + "-"*80)
    print("--- 統計量の比較 ---")
    print(stats_df.to_string(float_format="{:.4f}".format))

    # --- 膝角度の統計量比較 ---
    knee_stats_data = {
        '左膝 (動画1)': results1['left_knee_stats'],
        '右膝 (動画1)': results1['right_knee_stats'],
        '左膝 (動画2)': results2['left_knee_stats'],
        '右膝 (動画2)': results2['right_knee_stats'],
    }
    knee_stats_df = pd.DataFrame(knee_stats_data)
    knee_stats_df.index = ['平均', '中央値', '標準偏差', '最大値', '最小値']

    print("\n" + "-"*80)
    print("--- 膝角度の統計量比較 (度) ---")
    print(knee_stats_df.to_string(float_format="{:.2f}".format))

    # --- 首振り角度の統計量比較 ---
    head_stats_data = {
        '水平 (動画1)': results1['head_h_stats'],
        '垂直 (動画1)': results1['head_v_stats'],
        '水平 (動画2)': results2['head_h_stats'],
        '垂直 (動画2)': results2['head_v_stats'],
    }
    head_stats_df = pd.DataFrame(head_stats_data)
    head_stats_df.index = ['平均', '中央値', '標準偏差', '最大値', '最小値']

    print("\n" + "-"*80)
    print("--- 首振り角度の統計量比較 (度) ---")
    print(head_stats_df.to_string(float_format="{:.2f}".format))

    # --- 仮説検定 ---
    print("\n" + "-"*80)
    print("--- マン・ホイットニーのU検定 ---")
    series1 = results1['series'].dropna()
    series2 = results2['series'].dropna()

    if len(series1) > 0 and len(series2) > 0:
        u_stat, p_value = scipy_stats.mannwhitneyu(series1, series2, alternative='two-sided')
        print(f"U統計量: {u_stat:.2f}")
        print(f"P値: {p_value:.4f}")
        if p_value < 0.05:
            print("結果: P値が0.05未満であるため、2つの動画の移動量分布には統計的に有意な差があると言えます。")
        else:
            print("結果: P値が0.05以上であるため、2つの動画の移動量分布に統計的に有意な差があるとは言えません。")
    else:
        print("データが不足しているため、検定を実行できませんでした。")
    
    # --- 出力ファイルパスの表示 ---
    print("\n" + "-"*80)
    print("--- 計算結果CSVファイル ---")
    if results1['output_path']:
        print(f"動画1: {results1['output_path']}")
    if results2['output_path']:
        print(f"動画2: {results2['output_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5つの動作検知アイデアをシミュレートし、比較します。")
    parser.add_argument("file1", type=str, help="1つ目のCSVファイルへのパス。")
    parser.add_argument("file2", type=str, help="2つ目のCSVファイルへのパス。")
    args = parser.parse_args()
    
    main(args.file1, args.file2)
