import aiosqlite
import json
import os
from datetime import datetime
from typing import Optional, Dict
import logging
from contextlib import asynccontextmanager

try:
    from .schema_validator import SchemaValidator
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from schema_validator import SchemaValidator

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv('DATABASE_PATH', 'meeting_minutes.db')
        self.schema_validator = SchemaValidator(self.db_path)
        self._init_db()

    def _init_db(self):
        """Initialize the database with legacy approach"""
        try:
            logger.info("Initializing database tables...")
            self._legacy_init_db()
            logger.info("Validating schema integrity...")
            self.schema_validator.validate_schema()
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise

    def _legacy_init_db(self):
        """Legacy database initialization"""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS meetings (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcripts (
                    id TEXT PRIMARY KEY,
                    meeting_id TEXT NOT NULL,
                    transcript TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    summary TEXT,
                    action_items TEXT,
                    key_points TEXT,
                    FOREIGN KEY(meeting_id) REFERENCES meetings(id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summary_processes (
                    meeting_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT,
                    result TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    processing_time REAL DEFAULT 0.0,
                    metadata TEXT,
                    FOREIGN KEY(meeting_id) REFERENCES meetings(id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcript_chunks (
                    meeting_id TEXT PRIMARY KEY,
                    meeting_name TEXT,
                    transcript_text TEXT NOT NULL,
                    model TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    chunk_size INTEGER,
                    overlap INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(meeting_id) REFERENCES meetings(id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    whisperModel TEXT NOT NULL,
                    groqApiKey TEXT,
                    openaiApiKey TEXT,
                    anthropicApiKey TEXT,
                    ollamaApiKey TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcript_settings (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    whisperApiKey TEXT,
                    deepgramApiKey TEXT,
                    elevenLabsApiKey TEXT,
                    groqApiKey TEXT,
                    openaiApiKey TEXT
                )
            """)
            conn.commit()

    @asynccontextmanager
    async def _get_connection(self):
        conn = await aiosqlite.connect(self.db_path)
        try:
            yield conn
        finally:
            await conn.close()

    # --- Async helper for transactions ---
    async def _execute_transaction(self, queries):
        async with self._get_connection() as conn:
            try:
                await conn.execute("BEGIN")
                for query, params in queries:
                    await conn.execute(query, params)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

    # --- Other async methods like create_process, update_process, save_transcript ---
    # Convert synchronous sqlite3 calls to async aiosqlite
    # Consolidate repetitive logic in API key methods
