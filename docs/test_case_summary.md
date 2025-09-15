# テストケース一覧

本ドキュメントは、プロジェクトに実装されている自動テストの全テストケースを一覧にしたものです。

## `src` ディレクトリ (コアロジック)

### `tests/unit/src/test_detect_joint_movement_with_hip_stay.py`
| テストケース (関数名) | 検証内容 |
| :--- | :--- |
| `test_posture_snapshot_dataclass` | `PostureSnapshot` データクラスが期待されるフィールドを持って正しく定義されているか検証する。 |
| `test_hip_stay_info_dataclass` | `HipStayInfo` データクラスが期待されるフィールドを持って正しく定義されているか検証する。 |
| `test_hip_stay_state_enum` | `HipStayState` Enumが期待されるメンバー (`STAYING`, `POTENTIAL_MOVE` など) を持っているか検証する。 |
| `TestPostureMonitor.test_initialization` | `PostureMonitor` クラスが期待される初期状態でインスタンス化されるか検証する。 |
| `TestPostureMonitor.test_update_and_history_management` | `update` メソッド呼び出し時に、内部の姿勢データ履歴 (`posture_history`) が正しく追加・削除されるか検証する。 |
| `TestPostureMonitor.test_is_forward_leaning_posture` | 前傾姿勢と直立姿勢のダミーデータを渡し、前傾判定ロジックが正しく機能するか検証する。 |
| `TestPostureMonitor.test_alert_triggering` | 前傾姿勢が一定の割合を超えた場合に、正しくアラートが生成されるか検証する。 |
| `TestPostureMonitor.test_alert_cooldown` | 一度アラートが発火した後、クールダウン期間中は新たなアラートが抑制されることを検証する。 |
| `TestHipBasedStayDetector.test_initialization` | `HipBasedStayDetector` クラスが期待される初期状態でインスタンス化されるか検証する。 |
| `TestHipBasedStayDetector.test_extract_hip_center` | ランドマークデータから、腰の中心座標が正しく計算・抽出されるか検証する。 |
| `TestHipBasedStayDetector.test_state_transition_stay_to_move` | 静止状態 (`STAYING`) から、閾値を超える動きがあった場合に `POTENTIAL_MOVE` へ正しく遷移するか検証する。 |
| `TestHipBasedStayDetector.test_state_transition_move_to_stay` | `POTENTIAL_MOVE` 状態になった後、猶予期間内に動きが収まった場合に `STAYING` 状態へ復帰するか検証する。 |
| `TestHipBasedStayDetector.test_state_transition_move_confirmed` | `POTENTIAL_MOVE` 状態の後、猶予期間中も動きが続いた場合に移動が確定し、滞在時間がリセットされるか検証する。 |
| `TestHipBasedStayDetector.test_long_stay_alert` | 滞在時間が設定した閾値を超えた場合に、長期滞在アラートが正しく生成されるか検証する。 |
| `TestKneeAngleMonitor.test_angle_accumulation_and_finalization` | `update` メソッド呼び出し時に、膝の角度が秒単位で正しく集計され、中央値が計算されるか検証する。 |
| `TestKneeAngleMonitor.test_confidence_threshold` | 関節点の信頼度が設定した閾値未満の場合、そのデータが計算から除外されることを検証する。 |
| `TestKneeAngleMonitor.test_alert_triggering_by_median` | 秒単位の膝角度の中央値が閾値を下回った場合に、アラートが正しく生成されるか検証する。 |
| `TestKneeAngleMonitor.test_alert_triggering_by_moving_average` | 膝角度の移動平均が閾値を下回った場合に、アラートが正しく生成されるか検証する。 |
| `TestKneeAngleMonitor.test_no_alert` | 膝の角度が安全な範囲にある場合に、アラートが生成されないことを検証する。 |

### `tests/unit/src/test_head_shake_detector.py`
| テストケース (関数名) | 検証内容 |
| :--- | :--- |
| `test_detect_oscillation_pattern_positive` | 内部関数 `_detect_oscillation_pattern` が、人工的な振動データ (sin波) を正しく「振動あり」と判定するか検証する。 |
| `test_detect_oscillation_pattern_negative` | 内部関数 `_detect_oscillation_pattern` が、変動のないデータを「振動なし」と判定するか検証する。 |
| `test_update_static` | 頭が動いていないダミーデータを連続して入力し、状態が `HEAD_STATIC` のままであることを検証する。 |
| `test_head_turn` | 頭を左右に向けたダミーデータを入力し、状態が `HEAD_RIGHT_TURN`, `HEAD_LEFT_TURN` に正しく遷移するか検証する。 |
| `test_horizontal_shake` | 左右に首を振る動きをシミュレートしたダミーデータを連続入力し、`HORIZONTAL_SHAKE` が検出されるか検証する。 |
| `test_vertical_nod` | 上下に頷く動きをシミュレートしたダミーデータを連続入力し、`VERTICAL_NOD` が検出されるか検証する。 |

### `tests/unit/src/test_movement_analyzer.py`
| テストケース (関数名) | 検証内容 |
| :--- | :--- |
| `test_movement_analyzer_initial_state` | `analyze` メソッドの初回呼び出し時に、全ての関節の動作状態が `STATIC` として初期化されることを検証する。 |
| `test_movement_analyzer_state_change` | 関節を「曲げる」「伸ばす」動きをシミュレートし、状態が `FLEXION`, `EXTENSION` へ正しく遷移するか検証する。 |
| `test_body_tilt_forward` | 体を前傾させたダミーデータを入力し、`FORWARD_TILT` (前傾) が検出されるか検証する。 |
| `test_hunch_detection` | 猫背やうつむき姿勢をシミュレートし、`HUNCH` (猫背) が検出されるか検証する。 |
| `test_lateral_tilt_detection` | 体を左右に傾ける動きをシミュレートし、`RIGHT_TILT`, `LEFT_TILT` (側屈) が検出されるか検証する。 |

### `tests/unit/src/test_pose_estimator.py`
| テストケース (関数名) | 検証内容 |
| :--- | :--- |
| `test_pose_estimator_init` | `PoseEstimator` の初期化時に、内部のMediaPipeモデルが期待される引数で初期化されるか検証する (モック使用)。 |
| `test_pose_estimator_estimate_landmarks_found` | MediaPipeモデルがランドマークを検出した場合に、それが正しい形式 (numpy配列) で返されることを検証する (モック使用)。 |
| `test_pose_estimator_estimate_no_landmarks` | MediaPipeモデルがランドマークを検出しなかった場合に、`None` が返されることを検証する (モック使用)。 |
| `test_pose_estimator_close` | `close` メソッドが、内部のMediaPipeモデルの `close` メソッドを正しく呼び出すことを検証する (モック使用)。 |

### `tests/unit/src/test_drawing_utils.py`
| テストケース (関数名) | 検証内容 |
| :--- | :--- |
| `test_draw_landmarks` | ランドマークを描画する際に、内部のOpenCV関数 (`cv2.circle`, `cv2.line`) が期待される回数だけ呼び出されるか検証する (モック使用)。 |
| `test_draw_japanese_text` | 日本語を描画する際に、内部のPillow (PIL) 関数が期待される引数で呼び出されるか検証する (モック使用)。 |
| `test_draw_japanese_text_font_not_found` | 指定された日本語フォントが見つからない場合に、フォールバックしてデフォルトフォントを読み込もうとするか検証する (モック使用)。 |
| `test_draw_analysis_results` | 分析結果を描画する際に、テキスト描画関数が期待される回数呼び出されるか検証する (モック使用)。 |
| `test_draw_analysis_results_english` | 英語モードでの描画時に、`cv2.putText` が正しく呼び出されるか検証する (モック使用)。 |

### `tests/unit/src/pose/test_utils.py`
| テストケース (関数名) | 検証内容 |
| :--- | :--- |
| `test_calculate_angle` | 3点の座標から角度を計算する `calculate_angle` が、直角・直線・45度などのケースで正しい値を返すか検証する。 |
| `test_calculate_midpoint` | 2点の座標から中点を計算する `calculate_midpoint` が、2D・3D・負の座標を含むケースで正しい値を返すか検証する。 |

## `scripts` ディレクトリ (ユーティリティスクリプト)

### `tests/scripts/test_clip_video_by_frame.py`
| テストケース (関数名) | 検証内容 |
| :--- | :--- |
| `test_clip_successfully` | 指定した開始・終了フレームで、動画が正しく切り抜かれ、期待されるフレーム数のファイルが生成されるか検証する。 |
| `test_invalid_input_file` | 存在しない動画ファイルを入力として指定した場合に、スクリプトがエラー終了することを検証する。 |
| `test_invalid_frame_range` | 終了フレームが開始フレームより小さいなど、不正なフレーム範囲を指定した場合に、スクリプトがエラー終了することを検証する。 |

### `tests/scripts/test_clip_segments_by_idea.py`
| テストケース (関数名) | 検証内容 |
| :--- | :--- |
| `test_clip_successfully` | CSVファイルで定義された複数の区間に基づいて、動画が正しく複数のセグメントに切り抜かれるか検証する。 |
| `test_missing_csv_column` | 入力CSVファイルに必須のカラム (`idea`など) が欠けている場合に、スクリプトがエラー終了することを検証する。 |
| `test_non_existent_input` | 存在しない動画ファイルやCSVファイルを入力として指定した場合に、スクリプトがエラー終了することを検証する。 |
