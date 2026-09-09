"""
The compatibility layer between SQLite (the persistent source of truth) and
the temporary `memory: dict[str, MemoryElement]` that the existing
notebook functions (edit_memory, give_elements, reconstruct_preferred_sentence,
...) expect to operate on.

    SQLite row  <-- row_to_memory_element / memory_element_to_row -->  MemoryElement
    SQLite table <-- load_memory_from_db / persist_memory_elements --> memory dict

Nothing in here talks to an LLM. Summarization staleness is tracked here
(via the `summary` / `summarized_context_count` columns) but the actual
summarization call is triggered by app.py, which owns the decision of
*when* it's worth spending inference time.
"""

from __future__ import annotations

import json
from typing import Optional

import database
from memory import MemoryElement


def _safe_load_context(raw: Optional[str]) -> list[str]:
    """
    Parse the JSON-encoded context column defensively. Never raises — a
    malformed or missing value degrades to an empty (or best-effort
    single-entry) list rather than crashing the whole memory load.
    """
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Not valid JSON at all (e.g. hand-edited row, legacy plain string).
        # Treat the raw text as a single historical context entry instead
        # of discarding it.
        return [raw] if isinstance(raw, str) and raw.strip() else []

    if isinstance(data, list):
        return [str(item) for item in data]

    # JSON parsed but wasn't a list (e.g. a bare string or number) — wrap it
    # rather than silently losing the data.
    return [str(data)]


def row_to_memory_element(row: dict) -> MemoryElement:
    """REVERSE direction: SQLite row -> MemoryElement."""
    context = _safe_load_context(row.get("context"))
    return MemoryElement(
        og_word=row["og_word"],
        pref_word=row["pref_word"],
        context=context,
        occurrence=int(row.get("occurrences") or 0),
        representative=row["representative"],
    )


def memory_element_to_row(element: MemoryElement) -> dict:
    """FORWARD direction: MemoryElement -> SQLite row fields."""
    context = element.context
    if not isinstance(context, list):
        # summarize_memory_element replaces .context with a plain string.
        # If a caller ever tries to persist a summarized element as if it
        # were raw context, preserve the text rather than losing it.
        context = [str(context)] if context else []

    representative = element.representative or f"{element.og_word} : {element.pref_word}"

    return {
        "representative": representative,
        "og_word": element.og_word,
        "pref_word": element.pref_word,
        "occurrences": element.occurrence,
        "context": context,
    }


def load_memory_from_db() -> dict[str, MemoryElement]:
    """
    Load the full persistent memory table into the temporary in-memory
    dict shape the existing functions (edit_memory, give_elements, ...)
    expect. `MemoryElement.context` is always the raw list[str] here —
    summaries are applied separately/lazily (see get_summary_state).
    """
    database.init_db()
    rows = database.fetch_all_rows()

    memory: dict[str, MemoryElement] = {}
    for row in rows:
        try:
            element = row_to_memory_element(row)
        except (KeyError, TypeError):
            # Skip an individual malformed row rather than failing the
            # entire load.
            continue
        memory[element.representative] = element

    return memory


def persist_memory_elements(elements: dict[str, MemoryElement]) -> None:
    """
    FORWARD persistence: write the given MemoryElements' raw fields
    (og_word, pref_word, occurrence, context) back to SQLite in one
    transaction. Existing `summary` / `summarized_context_count` values for
    those rows are left untouched (they naturally become "stale" once the
    raw context grows past them — see is_summary_stale below).
    """
    if not elements:
        return
    rows = [memory_element_to_row(element) for element in elements.values()]
    database.upsert_memory_rows(rows)


def get_summary_state(representative: str) -> tuple[Optional[str], int]:
    """Return (summary_text_or_None, summarized_context_count) for a representative."""
    row = database.fetch_row(representative)
    if row is None:
        return None, 0
    return row.get("summary"), int(row.get("summarized_context_count") or 0)


def is_summary_stale(representative: str, current_context_len: int) -> bool:
    """
    A summary is stale (needs recomputation) if it has never been computed,
    or if new context entries have been appended since it was last computed.
    """
    summary, summarized_count = get_summary_state(representative)
    if summary is None:
        return True
    return current_context_len != summarized_count


def persist_summary(representative: str, summary_text: str, context_len_at_summary_time: int) -> None:
    """Cache a freshly computed summary alongside the context length it covers."""
    database.update_summary(representative, summary_text, context_len_at_summary_time)
