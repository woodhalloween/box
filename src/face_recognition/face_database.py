"""
顔データベース管理モジュール
SQLiteを使用した顔認識データの永続化
"""

import logging
import os
import pickle
import sqlite3
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FaceDatabase:
    """
    顔認識データベース管理クラス
    """

    def __init__(self, config: dict[str, Any]):
        """
        データベース管理器の初期化

        Args:
            config: 設定辞書
        """
        self.config = config
        db_config = config.get("database", {})

        self.db_path = db_config.get("path", "data/face_database/face_recognition.db")
        self.encodings_path = db_config.get("face_encodings_path", "data/face_encodings/")
        self.max_stored_faces = db_config.get("max_stored_faces", 1000)

        # データベース初期化
        self._init_database()

        logger.info(f"FaceDatabase initialized with path: {self.db_path}")

    def _init_database(self):
        """データベースとテーブルの初期化"""
        try:
            # ディレクトリ作成
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            os.makedirs(self.encodings_path, exist_ok=True)

            # データベース接続とテーブル作成
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # personsテーブル
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS persons (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        total_appearances INTEGER DEFAULT 1
                    )
                """)

                # face_encodingsテーブル
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS face_encodings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_id INTEGER,
                        encoding BLOB NOT NULL,
                        quality_score REAL DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (person_id) REFERENCES persons (id)
                    )
                """)

                # tracking_sessionsテーブル
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tracking_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_id INTEGER,
                        track_id INTEGER,
                        start_time TIMESTAMP,
                        end_time TIMESTAMP,
                        confidence REAL DEFAULT 0.0,
                        FOREIGN KEY (person_id) REFERENCES persons (id)
                    )
                """)

                # performance_logsテーブル
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        log_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        fps REAL,
                        face_detection_time REAL,
                        face_recognition_time REAL,
                        active_tracks INTEGER
                    )
                """)

                conn.commit()
                logger.info("Database tables initialized successfully")

        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise

    def add_person(
        self, name: str, face_encoding: np.ndarray, quality_score: float = 0.0
    ) -> int | None:
        """
        新しい人物を追加

        Args:
            name: 人物名
            face_encoding: 顔エンコーディング
            quality_score: 品質スコア

        Returns:
            人物ID or None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 人物を追加
                cursor.execute(
                    """
                    INSERT INTO persons (name, first_seen, last_seen, total_appearances)
                    VALUES (?, ?, ?, ?)
                """,
                    (name, datetime.now(), datetime.now(), 1),
                )

                person_id = cursor.lastrowid

                # 顔エンコーディングを追加
                encoding_blob = pickle.dumps(face_encoding)
                cursor.execute(
                    """
                    INSERT INTO face_encodings (person_id, encoding, quality_score)
                    VALUES (?, ?, ?)
                """,
                    (person_id, encoding_blob, quality_score),
                )

                conn.commit()
                logger.info(f"Added new person {person_id}: {name}")
                return person_id

        except Exception as e:
            logger.error(f"Error adding person: {e}")
            return None

    def update_person_appearance(self, person_id: int) -> bool:
        """
        人物の出現情報を更新

        Args:
            person_id: 人物ID

        Returns:
            更新成功フラグ
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    UPDATE persons 
                    SET last_seen = ?, total_appearances = total_appearances + 1
                    WHERE id = ?
                """,
                    (datetime.now(), person_id),
                )

                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Error updating person appearance: {e}")
            return False

    def get_person(self, person_id: int) -> dict[str, Any] | None:
        """
        人物情報を取得

        Args:
            person_id: 人物ID

        Returns:
            人物情報辞書 or None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT id, name, first_seen, last_seen, total_appearances
                    FROM persons WHERE id = ?
                """,
                    (person_id,),
                )

                result = cursor.fetchone()
                if result:
                    return {
                        "id": result[0],
                        "name": result[1],
                        "first_seen": result[2],
                        "last_seen": result[3],
                        "total_appearances": result[4],
                    }

        except Exception as e:
            logger.error(f"Error getting person: {e}")

        return None

    def get_all_face_encodings(self) -> list[tuple[int, str, np.ndarray]]:
        """
        すべての顔エンコーディングを取得

        Returns:
            [(person_id, name, encoding), ...] のリスト
        """
        encodings = []

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT p.id, p.name, fe.encoding
                    FROM persons p
                    JOIN face_encodings fe ON p.id = fe.person_id
                    ORDER BY fe.quality_score DESC
                """)

                results = cursor.fetchall()
                for result in results:
                    person_id, name, encoding_blob = result
                    encoding = pickle.loads(encoding_blob)
                    encodings.append((person_id, name, encoding))

        except Exception as e:
            logger.error(f"Error getting face encodings: {e}")

        return encodings

    def add_face_encoding(
        self, person_id: int, face_encoding: np.ndarray, quality_score: float = 0.0
    ) -> bool:
        """
        新しい顔エンコーディングを追加

        Args:
            person_id: 人物ID
            face_encoding: 顔エンコーディング
            quality_score: 品質スコア

        Returns:
            追加成功フラグ
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                encoding_blob = pickle.dumps(face_encoding)
                cursor.execute(
                    """
                    INSERT INTO face_encodings (person_id, encoding, quality_score)
                    VALUES (?, ?, ?)
                """,
                    (person_id, encoding_blob, quality_score),
                )

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Error adding face encoding: {e}")
            return False

    def start_tracking_session(
        self, person_id: int, track_id: int, confidence: float = 0.0
    ) -> int | None:
        """
        追跡セッションを開始

        Args:
            person_id: 人物ID
            track_id: 追跡ID
            confidence: 信頼度

        Returns:
            セッションID or None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO tracking_sessions (person_id, track_id, start_time, confidence)
                    VALUES (?, ?, ?, ?)
                """,
                    (person_id, track_id, datetime.now(), confidence),
                )

                session_id = cursor.lastrowid
                conn.commit()
                return session_id

        except Exception as e:
            logger.error(f"Error starting tracking session: {e}")
            return None

    def end_tracking_session(self, session_id: int) -> bool:
        """
        追跡セッションを終了

        Args:
            session_id: セッションID

        Returns:
            終了成功フラグ
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    UPDATE tracking_sessions 
                    SET end_time = ?
                    WHERE id = ?
                """,
                    (datetime.now(), session_id),
                )

                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Error ending tracking session: {e}")
            return False

    def log_performance(
        self,
        fps: float,
        face_detection_time: float,
        face_recognition_time: float,
        active_tracks: int,
    ) -> bool:
        """
        パフォーマンスログを記録

        Args:
            fps: FPS
            face_detection_time: 顔検出時間
            face_recognition_time: 顔認識時間
            active_tracks: アクティブ追跡数

        Returns:
            記録成功フラグ
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO performance_logs 
                    (fps, face_detection_time, face_recognition_time, active_tracks)
                    VALUES (?, ?, ?, ?)
                """,
                    (fps, face_detection_time, face_recognition_time, active_tracks),
                )

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Error logging performance: {e}")
            return False

    def cleanup_old_data(self, days_to_keep: int = 30) -> bool:
        """
        古いデータをクリーンアップ

        Args:
            days_to_keep: 保持する日数

        Returns:
            クリーンアップ成功フラグ
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 古いパフォーマンスログを削除
                cursor.execute(f"""
                    DELETE FROM performance_logs 
                    WHERE log_time < datetime('now', '-{days_to_keep} days')
                """)

                # 古い追跡セッションを削除
                cursor.execute(f"""
                    DELETE FROM tracking_sessions 
                    WHERE start_time < datetime('now', '-{days_to_keep} days')
                """)

                conn.commit()
                logger.info(f"Cleaned up data older than {days_to_keep} days")
                return True

        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return False

    def get_statistics(self) -> dict[str, Any]:
        """
        データベース統計を取得

        Returns:
            統計辞書
        """
        stats = {}

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 人物数
                cursor.execute("SELECT COUNT(*) FROM persons")
                stats["total_persons"] = cursor.fetchone()[0]

                # 顔エンコーディング数
                cursor.execute("SELECT COUNT(*) FROM face_encodings")
                stats["total_face_encodings"] = cursor.fetchone()[0]

                # 追跡セッション数
                cursor.execute("SELECT COUNT(*) FROM tracking_sessions")
                stats["total_tracking_sessions"] = cursor.fetchone()[0]

                # 最近の平均FPS
                cursor.execute("""
                    SELECT AVG(fps) FROM performance_logs 
                    WHERE log_time > datetime('now', '-1 hour')
                """)
                result = cursor.fetchone()
                stats["avg_fps_last_hour"] = result[0] if result[0] else 0.0

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")

        return stats

    def backup_database(self, backup_path: str) -> bool:
        """
        データベースをバックアップ

        Args:
            backup_path: バックアップ先パス

        Returns:
            バックアップ成功フラグ
        """
        try:
            import shutil

            # ディレクトリ作成
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)

            # ファイルコピー
            shutil.copy2(self.db_path, backup_path)

            logger.info(f"Database backed up to {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Error backing up database: {e}")
            return False
