import argparse
import glob
import os

import pandas as pd


def find_latest_csv(video_path):
    """outputディレクトリから最新の分析CSVファイルを見つける"""
    output_dir = "output"
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    search_pattern = os.path.join(
        output_dir, f"{video_basename}_integrated_analysis_*.csv"
    )
    files = glob.glob(search_pattern)
    print(f"検索パターン: {search_pattern}")
    print(f"見つかったファイル: {files}")
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    print(f"最新ファイルとして選択: {latest_file}")
    return latest_file


def analyze_hip_detector_state(csv_path):
    """CSVファイルを読み込み、hip_detector_stateの統計情報を表示する"""
    if not os.path.exists(csv_path):
        print(f"エラー: ファイルが見つかりません {csv_path}")
        return

    print(f"分析中のファイル: {csv_path}")
    df = pd.read_csv(csv_path)

    if "hip_detector_state" not in df.columns:
        print("エラー: 'hip_detector_state' カラムがCSVファイルにありません。")
        return

    # 状態ごとのフレーム数をカウント
    state_counts = df["hip_detector_state"].value_counts()
    total_frames = len(df)

    print("\n--- 移動検知アルゴリズムの状態分析 ---")
    print("各状態のフレーム数:")
    print(state_counts)

    print("\n各状態の割合:")
    state_percentages = state_counts / total_frames * 100
    print(state_percentages.round(2).to_string())

    # Movement Confirmedの割合を抽出
    confirmed_key = "MOVEMENT_CONFIRMED"
    if confirmed_key in state_counts:
        confirmed_frames = state_counts[confirmed_key]
        confirmed_percentage = (confirmed_frames / total_frames) * 100
        print(f"\n移動が確定されたフレームの割合: {confirmed_percentage:.2f}%")
    else:
        print("\n移動が確定されたフレームはありませんでした。")


def main():
    parser = argparse.ArgumentParser(description="動画処理結果のCSVを分析します。")
    parser.add_argument(
        "--video-path",
        type=str,
        help="分析対象の元となった動画ファイルのパス（任意）。省略した場合は定義済みの動画を分析します。",
    )
    args = parser.parse_args()

    if args.video_path:
        video_paths = [args.video_path]
    else:
        # 引数が省略された場合は、主要な動画ファイルを直接指定
        print("動画パスが指定されなかったため、定義済みの動画ファイルを分析します。")
        video_paths = [
            "output/車椅子おばあちゃん1.mp4",
            "output/車椅子おばあちゃん2_rotated_270.mp4",
        ]

    for video_path in video_paths:
        print(f"\n======== {os.path.basename(video_path)} の分析結果 ========")
        latest_csv = find_latest_csv(video_path)

        if latest_csv:
            analyze_hip_detector_state(latest_csv)
        else:
            print(
                f"エラー: {video_path} に対応する分析CSVファイルが見つかりませんでした。"
            )


if __name__ == "__main__":
    main()
