from ultralytics import YOLO
import inspect

# YOLOv8モデル
try:
    model_v8 = YOLO('yolov8n.pt')
    print('==== YOLOv8 ====')
    print('初期化引数:')
    print(inspect.signature(YOLO.__init__))
    print('\npredict引数:')
    print(inspect.signature(model_v8.predict))
    print('\nYOLOv8の結果の例:')
    results = model_v8.predict('./line_fortuna_demo_multipersons.mp4', stream=True, verbose=False)
    result = next(results)
    print(f'結果タイプ: {type(result)}')
    print(f'結果属性: {dir(result)}')
    print(f'boxes属性: {dir(result.boxes) if hasattr(result, "boxes") else "なし"}')
except Exception as e:
    print(f'YOLOv8エラー: {e}')

# YOLOv10モデル（v11がない場合の代替）
try:
    model_v10 = YOLO('yolov10n.pt')
    print('\n==== YOLOv10 ====')
    print('初期化引数:')
    print(inspect.signature(YOLO.__init__))
    print('\npredict引数:')
    print(inspect.signature(model_v10.predict))
    print('\nYOLOv10の結果の例:')
    results = model_v10.predict('./line_fortuna_demo_multipersons.mp4', stream=True, verbose=False)
    result = next(results)
    print(f'結果タイプ: {type(result)}')
    print(f'結果属性: {dir(result)}')
    print(f'boxes属性: {dir(result.boxes) if hasattr(result, "boxes") else "なし"}')
except Exception as e:
    print(f'YOLOv10エラー: {e}')

# YOLOv11モデル
try:
    model_v11 = YOLO('yolov11n.pt')
    print('\n==== YOLOv11 ====')
    print('初期化引数:')
    print(inspect.signature(YOLO.__init__))
    print('\npredict引数:')
    print(inspect.signature(model_v11.predict))
    print('\nYOLOv11の結果の例:')
    results = model_v11.predict('./line_fortuna_demo_multipersons.mp4', stream=True, verbose=False)
    result = next(results)
    print(f'結果タイプ: {type(result)}')
    print(f'結果属性: {dir(result)}')
    print(f'boxes属性: {dir(result.boxes) if hasattr(result, "boxes") else "なし"}')
except Exception as e:
    print(f'YOLOv11エラー: {e}') 