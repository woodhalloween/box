import pandas as pd
import numpy as np
import glob
import os

def analyze_log_file(file_path):
    print(f"\n=== {os.path.basename(file_path)} の分析結果 ===")
    
    try:
        # CSVファイルを読み込む
        df = pd.read_csv(file_path, comment='#')  # '#'で始まる行はヘッダーとしてスキップ
        
        # データが空かどうかを確認
        if df.empty:
            print(f"警告: ログファイル '{os.path.basename(file_path)}' にデータがありません。")
            print(f"ヘッダーは存在しますが、実際のデータ行がありません。")
            print("\n" + "="*50)
            return
            
        # 必要なカラムが存在するかチェック
        required_columns = ['Detection_Time_ms', 'Tracking_Time_ms', 'Total_Time_ms', 'FPS', 'Objects_Detected']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"警告: ログファイル '{os.path.basename(file_path)}' に以下のカラムがありません: {', '.join(missing_columns)}")
            print("\n" + "="*50)
            return

        # 基本的な統計情報を計算
        stats = {
            '検出時間 (ms)': {
                '平均': df['Detection_Time_ms'].mean(),
                '標準偏差': df['Detection_Time_ms'].std(),
                '最小': df['Detection_Time_ms'].min(),
                '最大': df['Detection_Time_ms'].max(),
            },
            'トラッキング時間 (ms)': {
                '平均': df['Tracking_Time_ms'].mean(),
                '標準偏差': df['Tracking_Time_ms'].std(),
                '最小': df['Tracking_Time_ms'].min(),
                '最大': df['Tracking_Time_ms'].max(),
            },
            '合計処理時間 (ms)': {
                '平均': df['Total_Time_ms'].mean(),
                '標準偏差': df['Total_Time_ms'].std(),
                '最小': df['Total_Time_ms'].min(),
                '最大': df['Total_Time_ms'].max(),
            },
            'FPS': {
                '平均': df['FPS'].mean(),
                '標準偏差': df['FPS'].std(),
                '最小': df['FPS'].min(),
                '最大': df['FPS'].max(),
            }
        }

        # 検出されたオブジェクトの統計
        total_frames = len(df)
        
        # 0除算を防ぐ
        if total_frames == 0:
            print(f"警告: ログファイル '{os.path.basename(file_path)}' にフレームデータがありません。")
            print("\n" + "="*50)
            return
            
        frames_with_objects = len(df[df['Objects_Detected'] > 0])
        detection_rate = frames_with_objects / total_frames * 100

        print("\n=== 処理性能の統計情報 ===")
        for metric, values in stats.items():
            print(f"\n{metric}:")
            for stat, value in values.items():
                print(f"{stat}: {value:.2f}")

        print(f"\n=== 検出率 ===")
        print(f"総フレーム数: {total_frames}")
        print(f"オブジェクト検出フレーム数: {frames_with_objects}")
        print(f"検出率: {detection_rate:.2f}%")
        print("\n" + "="*50)
        
    except Exception as e:
        print(f"エラー: ファイル '{os.path.basename(file_path)}' の分析中に問題が発生しました: {e}")
        print("\n" + "="*50)

# log_WINで始まるすべてのCSVファイルを検索
log_files = glob.glob('log_WIN*.csv')

if not log_files:
    print("log_WINで始まるCSVファイルが見つかりませんでした。")
else:
    print(f"合計{len(log_files)}個のログファイルを処理します。\n")
    for file_path in log_files:
        analyze_log_file(file_path) 