"""
Centralized configuration for the Kivi application.

Everything that another module needs to know about *which* models to use,
*where* the database lives, and the handful of generation knobs that were
missing from the original notebook lives here — nowhere else in the
codebase should hardcode a model name or a database path.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# These two identifiers came directly from the notebook (the cell that built
# `llm` and `llm_big`). They are the two values you would change if you want
# to swap in different Hugging Face models.
#
#   SMALL_MODEL_NAME -> used for: mapping-fallback (get_mappings) and
#                        memory summarization (summarize_memory_element), sentence reconstruction
#
# Both can be overridden with environment variables without touching code.
SMALL_MODEL_NAME = os.environ.get("KIVI_SMALL_MODEL", "Qwen/Qwen2.5-3B-Instruct")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("KIVI_DB_PATH", os.path.join(_PROJECT_ROOT, "data", "memory.db"))

# ---------------------------------------------------------------------------
# Generation limits
# ---------------------------------------------------------------------------
# The mapping-fallback call already hardcodes max_new_tokens=1000 and the
# summarizer already hardcodes max_new_tokens=100 inside the notebook's own
# functions (mapping.py / memory.py) — those are left untouched.
#
# The one call that had NO max_new_tokens at all in the notebook was
# reconstruct_preferred_sentence's `llm(messages)` call. Without an explicit
# limit, a text-generation pipeline falls back to a very small default
# token budget, which truncates the reconstructed sentence and makes the
# strict token-count validation fail (silently falling back to the original
# sentence every time). This constant is the fix for that gap — see
# "What I changed from your notebook" in the final summary.
MAX_NEW_TOKENS_RECONSTRUCT = int(os.environ.get("KIVI_RECONSTRUCT_MAX_NEW_TOKENS", "300"))
