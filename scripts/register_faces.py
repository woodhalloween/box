"""
顔登録スクリプト
新しい人物の顔をデータベースに登録するためのツール
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

# パスの追加
sys.path.append(str(Path(__file__).parent.parent))

from src.tracking.hybrid_tracker import HybridTracker


def setup_logging():
    """ログ設定"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_arguments():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description="Face Registration Tool")

    parser.add_argument("--name", type=str, required=True, help="Person name to register")
    parser.add_argument("--image", type=str, required=True, help="Face image file path")
    parser.add_argument(
        "--config",
        type=str,
        default="config/face_recognition_config.yaml",
        help="Configuration file path",
    )
    parser.add_argument(
        "--show", action="store_true", help="Show the face image before registration"
    )

    return parser.parse_args()


def main():
    """メイン処理"""
    args = parse_arguments()
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # ハイブリッド追跡システムの初期化
        hybrid_tracker = HybridTracker(args.config)

        # 画像の読み込み
        image = cv2.imread(args.image)
        if image is None:
            raise ValueError(f"Cannot load image: {args.image}")

        logger.info(f"Loaded image: {args.image}")

        # 画像表示（オプション）
        if args.show:
            cv2.imshow("Face to Register", image)
            print("Press any key to continue with registration, or 'q' to quit...")
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()

            if key == ord("q"):
                logger.info("Registration cancelled by user")
                return

        # 顔の登録
        logger.info(f"Registering person: {args.name}")
        person_id = hybrid_tracker.register_new_person(args.name, image)

        if person_id:
            logger.info(f"Successfully registered {args.name} with ID: {person_id}")
        else:
            logger.error("Failed to register person. No face detected or encoding failed.")

    except Exception as e:
        logger.error(f"Error during registration: {e}")
        raise


if __name__ == "__main__":
    main()
