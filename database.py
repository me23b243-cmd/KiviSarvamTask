"""
SQLite persistence layer.

This module owns every raw SQL statement in the application. Nothing
outside this file should open a sqlite3 connection directly.

Schema
------
memory (
    representative              TEXT PRIMARY KEY,   -- "<og_word> : <pref_word>"
    og_word                     TEXT NOT NULL,
    pref_word                   TEXT NOT NULL,
    occurrences                 INTEGER NOT NULL,    -- MemoryElement.occurrence
    context                     TEXT NOT NULL,       -- JSON list[str] (raw, full history)
    summary                     TEXT,                -- cached compact rule (nullable)
    summarized_context_count    INTEGER NOT NULL,    -- len(context) when `summary` was
                                                       -- last computed, used to detect
                                                       -- staleness cheaply (no LLM call
                                                       -- needed to check)
    updated_at                  TEXT                 -- informational only
)

`summary` / `summarized_context_count` are the extra columns beyond the
minimum the task described. They exist to support section 9 of the brief
("do NOT unnecessarily summarize every memory element every time the
application runs") without losing the raw context history that
`update_memory_element` keeps appending to. See memory_store.py for how
staleness is decided and used.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Iterable, Optional

import config

# A single process-wide lock is enough here: sqlite itself serializes
# writers, but wrapping our own transactions in a lock avoids "database is
# locked" errors under Streamlit's occasional concurrent reruns, and keeps
# each read-modify-write sequence atomic from the application's point of view.
_lock = threading.RLock()


class DatabaseError(RuntimeError):
    """Raised when a database operation fails after retries/rollback."""


def _ensure_data_dir() -> None:
    directory = os.path.dirname(config.DB_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)


@contextmanager
def get_connection():
    """Context-managed SQLite connection with row access by column name."""
    _ensure_data_dir()
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """
    Create the data directory / database file / table if they do not exist.
    Safe to call on every app startup — never destroys existing data.
    """
    with _lock:
        try:
            with get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory (
                        representative TEXT PRIMARY KEY,
                        og_word TEXT NOT NULL,
                        pref_word TEXT NOT NULL,
                        occurrences INTEGER NOT NULL DEFAULT 0,
                        context TEXT NOT NULL DEFAULT '[]',
                        summary TEXT,
                        summarized_context_count INTEGER NOT NULL DEFAULT 0,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to initialize the database: {e}") from e


def fetch_all_rows() -> list[dict[str, Any]]:
    with _lock:
        try:
            with get_connection() as conn:
                cur = conn.execute(
                    "SELECT representative, og_word, pref_word, occurrences, "
                    "context, summary, summarized_context_count FROM memory"
                )
                return [dict(row) for row in cur.fetchall()]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to read memory from the database: {e}") from e


def fetch_row(representative: str) -> Optional[dict[str, Any]]:
    with _lock:
        try:
            with get_connection() as conn:
                cur = conn.execute(
                    "SELECT representative, og_word, pref_word, occurrences, "
                    "context, summary, summarized_context_count FROM memory "
                    "WHERE representative = ?",
                    (representative,),
                )
                row = cur.fetchone()
                return dict(row) if row is not None else None
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to read '{representative}' from the database: {e}") from e


def upsert_memory_rows(rows: Iterable[dict[str, Any]]) -> None:
    """
    Insert-or-update a batch of memory rows in a single transaction.

    Each `row` dict must contain: representative, og_word, pref_word,
    occurrences (int), context (list[str] — will be JSON-encoded here).

    Existing `summary` / `summarized_context_count` values are preserved as
    the raw context grows (they are managed separately by update_summary,
    since re-summarizing is an LLM call and must stay lazy).

    Either every row in the batch is committed, or none are (rollback on
    any failure) — the caller should never be told a save succeeded if it
    did not.
    """
    rows = list(rows)
    if not rows:
        return

    with _lock:
        try:
            with get_connection() as conn:
                for row in rows:
                    context_json = json.dumps(row["context"], ensure_ascii=False)
                    conn.execute(
                        """
                        INSERT INTO memory (
                            representative, og_word, pref_word, occurrences, context, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        ON CONFLICT(representative) DO UPDATE SET
                            og_word = excluded.og_word,
                            pref_word = excluded.pref_word,
                            occurrences = excluded.occurrences,
                            context = excluded.context,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (
                            row["representative"],
                            row["og_word"],
                            row["pref_word"],
                            int(row["occurrences"]),
                            context_json,
                        ),
                    )
                conn.commit()
        except (sqlite3.Error, TypeError, ValueError) as e:
            raise DatabaseError(f"Failed to save memory changes: {e}") from e


def update_summary(representative: str, summary_text: str, summarized_context_count: int) -> None:
    """Persist a freshly computed summary for one representative."""
    with _lock:
        try:
            with get_connection() as conn:
                cur = conn.execute(
                    """
                    UPDATE memory
                    SET summary = ?, summarized_context_count = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE representative = ?
                    """,
                    (summary_text, int(summarized_context_count), representative),
                )
                conn.commit()
                if cur.rowcount == 0:
                    raise DatabaseError(
                        f"Could not persist summary: '{representative}' no longer exists in the database."
                    )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to save summary for '{representative}': {e}") from e


def count_rows() -> int:
    with _lock:
        try:
            with get_connection() as conn:
                cur = conn.execute("SELECT COUNT(*) FROM memory")
                return int(cur.fetchone()[0])
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to count memory rows: {e}") from e
