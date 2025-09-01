print("--- モジュールのインポート処理をデバッグします ---")
print("対象: src.detect_joint_movement_with_hip_stay")

try:
    # 問題のモジュールをインポートしてみる
    import src.detect_joint_movement_with_hip_stay
    print("\n--- 結果 ---")
    print("インポートは成功しました。モジュール自体に構文エラー等はありません。")

except Exception as e:
    print("\n--- エラー ---")
    print("モジュールのインポート中にエラーが発生しました。")
    print(f"エラーの種類: {type(e).__name__}")
    print(f"エラーメッセージ: {e}")
    # スタックトレースをインポートして詳細情報を表示
    import traceback
    print("\n--- スタックトレース ---")
    traceback.print_exc()

print("\n--- デバッグ終了 ---")


