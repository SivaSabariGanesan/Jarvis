"""
SQLite memory and conversation persistence layer for JARVIS.
Stores chat sessions, message turns, user preferences, and metadata.
"""

import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from agent.config import settings

logger = logging.getLogger("jarvis.memory")


class JarvisDatabase:
    """
    SQLite database managing conversation logs and persistent memory.
    Designed with a modular schema so vector storage can be attached in future milestones.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else settings.database_file
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    room_name TEXT,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ended_at TIMESTAMP,
                    metadata TEXT
                )
            """)

            # Messages / Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    latency_ms REAL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # Key-Value Memory / Preferences
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_kv (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()
            logger.info(f"Initialized SQLite database at {self.db_path}")

    def create_session(self, session_id: str, room_name: str = "") -> None:
        """Record the start of a conversation session."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO sessions (session_id, room_name) VALUES (?, ?)",
                    (session_id, room_name),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error creating session {session_id}: {e}")

    def log_message(
        self,
        session_id: str,
        role: str,
        content: str,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Store a single turn message."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT INTO messages (session_id, role, content, latency_ms) VALUES (?, ?, ?, ?)",
                    (session_id, role, content, latency_ms),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging message: {e}")

    def get_recent_history(
        self, session_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Retrieve recent conversation history for context injection."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                    (session_id, limit),
                )
                rows = cursor.fetchall()
                # Return in chronological order
                return [dict(r) for r in reversed(rows)]
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []

    def set_memory(self, key: str, value: str) -> None:
        """Store or update a persistent memory item."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO memory_kv (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                    (key, value),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing memory '{key}': {e}")

    def get_memory(self, key: str) -> Optional[str]:
        """Retrieve a stored persistent memory item."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT value FROM memory_kv WHERE key = ?",
                    (key,),
                )
                row = cursor.fetchone()
                return row["value"] if row else None
        except Exception as e:
            logger.error(f"Error reading memory '{key}': {e}")
            return None
