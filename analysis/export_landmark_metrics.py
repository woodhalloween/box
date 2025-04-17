"""export_landmark_metrics.py
Skeleton CSVログから各ランドマークの座標と速度(dx/dt, dy/dt)を計算し、CSV形式で出力します。
Usage:
    python analysis/export_landmark_metrics.py \ 
        logs/skeleton_data_20250413_231131.csv \ 
        --output output/landmark_metrics.csv
"""
import argparse
from pathlib import Path
import pandas as pd

def compute_velocity(series: pd.Series, timestamps: pd.Series) -> pd.Series:
    """時刻差分から速度を計算(座標/秒)"""
    dt = timestamps.diff()
    return series.diff() / dt


def main():
    parser = argparse.ArgumentParser(description='Export landmark metrics to CSV')
    parser.add_argument('csv_path', help='Path to input skeleton CSV log')
    parser.add_argument('--output', required=True, help='Path to output metrics CSV')
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # タイムスタンプ列
    if 'timestamp' not in df.columns:
        raise KeyError('timestamp column not found in CSV')
    timestamps = df['timestamp']

    # 出力用DataFrame: フレームインデックスとタイムスタンプ
    metrics = pd.DataFrame({
        'frame_index': df.index,
        'timestamp': timestamps
    })

    # ランドマークID一覧を抽出
    landmark_ids = sorted({int(c.split('_')[1]) for c in df.columns if c.startswith('landmark_') and c.endswith('_x')})

    for idx in landmark_ids:
        # 座標
        x_col = f'landmark_{idx}_x'
        y_col = f'landmark_{idx}_y'
        vis_col = f'landmark_{idx}_visibility'
        if x_col in df.columns and y_col in df.columns and vis_col in df.columns:
            metrics[x_col] = df[x_col]
            metrics[y_col] = df[y_col]
            metrics[vis_col] = df[vis_col]
            # 速度計算
            metrics[f'landmark_{idx}_dx'] = compute_velocity(df[x_col], timestamps)
            metrics[f'landmark_{idx}_dy'] = compute_velocity(df[y_col], timestamps)

    # 出力先ディレクトリ作成
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)
    print(f'Metrics saved to {output_path}')

if __name__ == '__main__':
    main() 