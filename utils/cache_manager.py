"""Cache manager for storing intermediate results."""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    """SQLite-based cache for embeddings and OCR outputs."""

    def __init__(self, cache_dir: Path, ttl_days: int = 90):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for cache storage
            ttl_days: Cache entry time-to-live in days
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_days = ttl_days

        self.db_path = cache_dir / "cache.db"
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                paper_id TEXT PRIMARY KEY,
                metadata TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS text_extracts (
                paper_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS processing_state (
                paper_id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def get_embedding(self, key: str) -> Optional[List[float]]:
        """Get cached embedding."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT embedding FROM embeddings WHERE key = ?", (key,))
        result = cursor.fetchone()
        conn.close()

        if result:
            embedding: List[float] = json.loads(result[0])
            return embedding
        return None

    def set_embedding(self, key: str, embedding: List[float]):
        """Cache embedding."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT OR REPLACE INTO embeddings (key, embedding) VALUES (?, ?)",
            (key, json.dumps(embedding)),
        )

        conn.commit()
        conn.close()

    def get_metadata(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get cached metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT metadata FROM metadata WHERE paper_id = ?", (paper_id,))
        result = cursor.fetchone()
        conn.close()

        if result:
            metadata: Dict[str, Any] = json.loads(result[0])
            return metadata
        return None

    def set_metadata(self, paper_id: str, metadata: Dict[str, Any]):
        """Cache metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT OR REPLACE INTO metadata (paper_id, metadata) VALUES (?, ?)",
            (paper_id, json.dumps(metadata)),
        )

        conn.commit()
        conn.close()

    def get_text(self, paper_id: str) -> Optional[tuple[str, str]]:
        """Get cached text extract and hash."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT text, text_hash FROM text_extracts WHERE paper_id = ?", (paper_id,))
        result = cursor.fetchone()
        conn.close()

        return result if result else None

    def set_text(self, paper_id: str, text: str, text_hash: str):
        """Cache text extract."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT OR REPLACE INTO text_extracts (paper_id, text, text_hash) VALUES (?, ?, ?)",
            (paper_id, text, text_hash),
        )

        conn.commit()
        conn.close()

    def get_processing_state(self, paper_id: str, stage: str) -> Optional[Dict[str, Any]]:
        """Get processing state for a paper at a specific stage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT state FROM processing_state WHERE paper_id = ? AND stage = ?", (paper_id, stage)
        )
        result = cursor.fetchone()
        conn.close()

        if result:
            state: Dict[str, Any] = json.loads(result[0])
            return state
        return None

    def set_processing_state(self, paper_id: str, stage: str, state: Dict[str, Any]):
        """Set processing state for a paper at a specific stage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO processing_state (paper_id, stage, state, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (paper_id, stage, json.dumps(state)),
        )

        conn.commit()
        conn.close()

    def clear_expired(self):
        """Clear expired cache entries."""
        cutoff = datetime.now() - timedelta(days=self.ttl_days)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Fixed table names (not user input) - safe from SQL injection
        tables = ["embeddings", "metadata", "text_extracts", "processing_state"]
        for table in tables:
            # Table names cannot be parameterized in SQLite, but these are hardcoded constants
            cursor.execute(f"DELETE FROM {table} WHERE created_at < ?", (cutoff.isoformat(),))  # nosec B608

        conn.commit()
        conn.close()
        logger.info("Cleared expired cache entries")
