"""
Lightweight, purely-additive metrics recording.

Tracks the three things that can actually be *measured* on a local,
CPU-only, single-model setup — nothing invented or estimated beyond what's
clearly flagged:

- latency          wall-clock seconds per LLM call, grouped by purpose
                    ('mapping' | 'summarize' | 'reconstruct')
- model usage       call counts per purpose, plus input/output token counts
                    (exact, via the pipeline's own tokenizer when available;
                    falls back to a word count if not, and is flagged as
                    such — never presented as an exact token count)
- cost              there is no metered API cost for local inference, so
                    "cost" is reported as total compute-time spent — the
                    real resource being consumed — rather than a fabricated
                    dollar figure
- database growth   row count + on-disk file size of the SQLite database,
                    snapshotted over time so growth is visible across runs

This module does not modify mapping.py / memory.py / the underlying
algorithms in any way. It only wraps timing around the existing call sites
in app.py and stores results in two new tables inside the same SQLite
database file (via database.get_connection()).
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Optional

import config
import database


def init_metrics_tables() -> None:
    """Create the metrics tables if absent. Safe to call every startup."""
    with database.get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                purpose TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                tokens_are_estimated INTEGER NOT NULL DEFAULT 0,
                success INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS db_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                row_count INTEGER NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def _token_count(text: str, tokenizer=None) -> tuple[int, bool]:
    """Return (count, was_estimated)."""
    if not text:
        return 0, False
    if tokenizer is not None:
        try:
            return len(tokenizer(text)["input_ids"]), False
        except Exception:
            pass
    return len(text.split()), True


def record_llm_call(
    purpose: str,
    duration_seconds: float,
    input_text: str = "",
    output_text: str = "",
    tokenizer=None,
    success: bool = True,
) -> None:
    in_count, in_est = _token_count(input_text, tokenizer)
    out_count, out_est = _token_count(output_text, tokenizer)
    estimated = 1 if (in_est or out_est) else 0
    try:
        with database.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO llm_calls
                    (purpose, duration_seconds, input_tokens, output_tokens, tokens_are_estimated, success)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (purpose, duration_seconds, in_count, out_count, estimated, int(success)),
            )
            conn.commit()
    except Exception:
        # Metrics must never break the actual feature they're measuring.
        pass


@contextmanager
def timed_call(purpose: str, tokenizer=None, input_text: str = ""):
    """
    Times a block of code and logs it as one LLM call.

    Usage:
        with metrics.timed_call("reconstruct", tokenizer=big_llm.tokenizer, input_text=sentence) as m:
            result = reconstruct_preferred_sentence(sentence, memory, big_llm)
            m["output_text"] = result
    """
    start = time.perf_counter()
    holder = {"output_text": "", "success": True}
    try:
        yield holder
    except Exception:
        holder["success"] = False
        raise
    finally:
        duration = time.perf_counter() - start
        record_llm_call(
            purpose=purpose,
            duration_seconds=duration,
            input_text=input_text,
            output_text=holder.get("output_text", ""),
            tokenizer=tokenizer,
            success=holder.get("success", True),
        )


def llm_usage_summary() -> list[dict]:
    """Per-purpose call count, latency stats, and token totals."""
    with database.get_connection() as conn:
        cur = conn.execute(
            """
            SELECT
                purpose,
                COUNT(*) AS calls,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS successes,
                AVG(duration_seconds) AS avg_seconds,
                MIN(duration_seconds) AS min_seconds,
                MAX(duration_seconds) AS max_seconds,
                SUM(duration_seconds) AS total_seconds,
                SUM(input_tokens) AS total_input_tokens,
                SUM(output_tokens) AS total_output_tokens,
                MAX(tokens_are_estimated) AS any_estimated
            FROM llm_calls
            GROUP BY purpose
            ORDER BY purpose
            """
        )
        return [dict(row) for row in cur.fetchall()]


def record_db_snapshot() -> dict:
    """Persist a snapshot of the current row count + on-disk file size."""
    row_count = database.count_rows()
    file_size = os.path.getsize(config.DB_PATH) if os.path.exists(config.DB_PATH) else 0
    try:
        with database.get_connection() as conn:
            conn.execute(
                "INSERT INTO db_snapshots (row_count, file_size_bytes) VALUES (?, ?)",
                (row_count, file_size),
            )
            conn.commit()
    except Exception:
        pass
    return {"row_count": row_count, "file_size_bytes": file_size}


def database_growth() -> dict:
    """Compare the earliest recorded snapshot to right now (also snapshots now)."""
    current = record_db_snapshot()
    with database.get_connection() as conn:
        cur = conn.execute(
            "SELECT row_count, file_size_bytes FROM db_snapshots ORDER BY id ASC LIMIT 1"
        )
        first = cur.fetchone()
    first = dict(first) if first is not None else current
    return {
        "current_rows": current["row_count"],
        "current_size_bytes": current["file_size_bytes"],
        "first_rows": first["row_count"],
        "first_size_bytes": first["file_size_bytes"],
        "rows_added": current["row_count"] - first["row_count"],
        "size_added_bytes": current["file_size_bytes"] - first["file_size_bytes"],
    }


def format_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
