"""
Word-mapping module.

- Same token count  -> existing deterministic 1->1 logic (unchanged).
- Different token count -> LLM fallback that can produce 1->1, 1->2, or 2->1
  mappings.

The public entry point is get_mappings(), which always returns list[Mapping]
regardless of which path was used internally.

This module is preserved from the source notebook essentially unchanged.
The only integration fix applied here is documented right above
`_HF_MODEL_NAME` below: the notebook cell referenced a module-level `llm`
variable that did not exist yet at that point in the notebook (it was only
created three cells later). That has been replaced with the centralized
config value so this module is importable/runnable stand-alone.
"""

import json
import re
from dataclasses import dataclass
from typing import Callable, List, Optional

try:
    from transformers import pipeline  # pip install transformers torch
except ImportError:  # pragma: no cover - only needed for the real LLM path
    pipeline = None

import config

# Any text-generation/instruct model on the Hugging Face Hub works here.
# --- Integration fix -------------------------------------------------------
# Original notebook cell had: `_HF_MODEL_NAME = llm`, which referred to a
# pipeline object defined in a *later* cell (and even if it had been defined
# earlier, assigning a live pipeline object here rather than a model-name
# string would break get_llm_client(), which passes this value as the
# `model=` argument to `pipeline(...)`). This mapping LLM is the "small"
# model per the two-LLM split described in the task, so it now points at the
# centralized small-model name.
_HF_MODEL_NAME = config.SMALL_MODEL_NAME
# ---------------------------------------------------------------------------

# _TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


def _tokenize(text: str) -> list[str]:
    """Tokenize text into words, preserving internal apostrophes (e.g. don't, it's)."""
    return _TOKEN_RE.findall(text)


@dataclass
class Mapping:
    og_word: str
    pref_word: str
    pref_set: str


# ---------------------------------------------------------------------------
# Existing deterministic logic -- UNCHANGED. Used only when token counts match.
# ---------------------------------------------------------------------------
def build_mappings(original: str, preferred: str) -> list[Mapping]:
    """
    Build strictly positional 1 -> 1 word mappings between the original and
    preferred sentences, emitting a Mapping ONLY where the words at a given
    position differ. Identical words at a position produce no mapping.
    """
    og_tokens = _tokenize(original)
    pref_tokens = _tokenize(preferred)

    if len(og_tokens) != len(pref_tokens):
        raise ValueError(
            "The simplified mapper only supports equal-length sentences "
            f"(1 -> 1 mappings). Got {len(og_tokens)} original token(s) "
            f"({og_tokens}) but {len(pref_tokens)} preferred token(s) "
            f"({pref_tokens})."
        )

    mappings = [
        Mapping(og_word=og, pref_word=pref, pref_set=preferred)
        for og, pref in zip(og_tokens, pref_tokens)
        if og != pref
    ]

    for mapping in mappings:
        assert len(_tokenize(mapping.og_word)) == 1
        assert len(_tokenize(mapping.pref_word)) == 1
        assert mapping.pref_set == preferred

    num_mismatches = sum(1 for og, pref in zip(og_tokens, pref_tokens) if og != pref)
    assert len(mappings) == num_mismatches

    return mappings


# ---------------------------------------------------------------------------
# LLM fallback -- only triggered when token counts differ.
# ---------------------------------------------------------------------------
_ALLOWED_MAPPING_TYPES = {"1_to_1", "1_to_2", "2_to_1"}

# expected (og_token_count, pref_token_count) for each allowed mapping type
_EXPECTED_TOKEN_COUNTS = {
    "1_to_1": (1, 1),
    "1_to_2": (1, 2),
    "2_to_1": (2, 1),
}

_LLM_SYSTEM_PROMPT = """You align two sentences: an original/ASR sentence and a preferred sentence.

Rules:
- Preserve the exact words from both sentences. Never invent words. Never rewrite either sentence. Never perform spelling correction.
- Only identify mappings where the corresponding word/group differs between the two sentences. Unchanged words must NOT be included.
- Only use these mapping types:
  - "1_to_1": one original word -> one preferred word
  - "1_to_2": one original word -> two consecutive preferred words
  - "2_to_1": two consecutive original words -> one preferred word
- Do NOT produce any other mapping shape (no 2_to_2, 1_to_3, 3_to_1, etc.).
- Preserve the original left-to-right order of the words.
- Handle multiple mappings within the same sentence pair.
- Handle repeated words correctly based on their position/context.
- If there are no differences, return an empty mappings list.
- Never return explanations, comments, or Markdown fences.

Return ONLY valid JSON, exactly in this shape, and nothing else:
{
  "mappings": [
    {"og_word": "New York", "pref_word": "NY", "mapping_type": "2_to_1"},
    {"og_word": "Apple", "pref_word": "Apple Inc.", "mapping_type": "1_to_2"}
  ]
}
"""


class LLMMappingError(Exception):
    """Raised when the LLM fallback fails to produce a valid, usable mapping set."""


def get_llm_client():
    """
    Isolated factory for the LLM client so callers/tests can swap it out easily.
    Requires the `transformers` (and `torch`) packages for real use. Loads the
    model/tokenizer once and returns a reusable text-generation pipeline.

    NOTE: the Streamlit app never calls this directly — it always passes an
    already-loaded, `st.cache_resource`-cached pipeline in as `client=...`
    (see llm.py). This factory exists so the module remains usable/testable
    standalone, and as the fallback for any caller that does not manage its
    own cached client.
    """
    if pipeline is None:
        raise RuntimeError(
            "The 'transformers' package is not installed. Run `pip install transformers torch`."
        )
    return pipeline(
        "text-generation",
        model=_HF_MODEL_NAME,
        device_map="auto",
    )


def _build_llm_user_prompt(original: str, preferred: str) -> str:
    return (
        f'Original sentence: "{original}"\n'
        f'Preferred sentence: "{preferred}"\n\n'
        "Return the JSON mapping object described in the system prompt."
    )


def _call_llm(client, original: str, preferred: str, strict: bool = False) -> str:
    """Isolated helper that performs the actual LLM call and returns raw text."""
    system_prompt = _LLM_SYSTEM_PROMPT
    if strict:
        system_prompt += (
            "\n\nIMPORTANT: Your previous response could not be parsed as valid JSON "
            "matching the required schema. Return ONLY the raw JSON object. "
            "No Markdown code fences, no prose, no explanations."
        )

    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_llm_user_prompt(original, preferred)},
    ]

    outputs = client(
        chat,
        max_new_tokens=1000,
        do_sample=False,
        return_full_text=False,
    )

    # transformers chat-pipeline output shape: [{"generated_text": "..."}]
    return outputs[0]["generated_text"].strip()


def _parse_llm_json(raw_text: str) -> list[dict]:
    cleaned = raw_text.strip()
    # defensive cleanup in case the model wraps the JSON in fences anyway
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LLMMappingError(f"LLM did not return valid JSON: {e}") from e

    if not isinstance(data, dict) or "mappings" not in data or not isinstance(data["mappings"], list):
        raise LLMMappingError("LLM JSON is missing a top-level 'mappings' list.")

    return data["mappings"]


def _validate_and_convert(raw_mappings: list[dict], original: str, preferred: str) -> list[Mapping]:
    mappings: list[Mapping] = []

    for i, m in enumerate(raw_mappings):
        if not isinstance(m, dict):
            raise LLMMappingError(f"Mapping {i} is not an object: {m!r}")

        for key in ("og_word", "pref_word", "mapping_type"):
            if key not in m:
                raise LLMMappingError(f"Mapping {i} is missing required field '{key}': {m!r}")

        og_word = m["og_word"]
        pref_word = m["pref_word"]
        mapping_type = m["mapping_type"]

        if mapping_type not in _ALLOWED_MAPPING_TYPES:
            raise LLMMappingError(
                f"Mapping {i} has unsupported mapping_type {mapping_type!r}. "
                f"Allowed: {sorted(_ALLOWED_MAPPING_TYPES)}"
            )

        if og_word not in original:
            raise LLMMappingError(
                f"Mapping {i}: og_word {og_word!r} was not found verbatim in the "
                f"original sentence {original!r}."
            )

        if pref_word not in preferred:
            raise LLMMappingError(
                f"Mapping {i}: pref_word {pref_word!r} was not found verbatim in the "
                f"preferred sentence {preferred!r}."
            )

        og_token_count = len(_tokenize(og_word))
        pref_token_count = len(_tokenize(pref_word))
        expected_counts = _EXPECTED_TOKEN_COUNTS[mapping_type]

        if (og_token_count, pref_token_count) != expected_counts:
            raise LLMMappingError(
                f"Mapping {i}: mapping_type {mapping_type!r} expects "
                f"{expected_counts} tokens but got ({og_token_count}, {pref_token_count}) "
                f"for og_word={og_word!r}, pref_word={pref_word!r}."
            )

        # pref_set follows the same convention as the deterministic path: the
        # full preferred sentence.
        mappings.append(Mapping(og_word=og_word, pref_word=pref_word, pref_set=preferred))

    return mappings


def _llm_fallback_mappings(
    client,
    original: str,
    preferred: str,
    call_llm_fn: Callable = _call_llm,
) -> list[Mapping]:
    og_tokens = _tokenize(original)
    pref_tokens = _tokenize(preferred)

    print(
        f"[LLM FALLBACK] Sentence lengths differ (ASR: {len(og_tokens)}, "
        f"Preferred: {len(pref_tokens)}). Calling LLM for mapping..."
    )

    last_error: Optional[Exception] = None
    max_attempts = 2  # 1 initial attempt + 1 stricter retry

    for attempt in range(max_attempts):
        try:
            raw_text = call_llm_fn(client, original, preferred, strict=(attempt > 0))
            raw_mappings = _parse_llm_json(raw_text)
            return _validate_and_convert(raw_mappings, original, preferred)
        except LLMMappingError as e:
            last_error = e
            print(f"[LLM FALLBACK] Attempt {attempt + 1} failed: {e}")

    raise LLMMappingError(
        f"LLM fallback failed after {max_attempts} attempt(s) for "
        f"original={original!r}, preferred={preferred!r}. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def get_mappings(
    original: str,
    preferred: str,
    client=None,
    call_llm_fn: Callable = _call_llm,
) -> list[Mapping]:
    """
    Return list[Mapping] describing how `original` differs from `preferred`.

    - If both sentences have the same token count, the existing deterministic
      1->1 logic is used (no LLM call).
    - If token counts differ, an LLM fallback determines 1->1 / 1->2 / 2->1
      mappings.

    `client` and `call_llm_fn` are injectable for testing; callers normally
    don't need to pass them. The Streamlit app always passes its cached
    small-model pipeline in as `client` so that a fresh model is never loaded
    on this path.
    """
    og_tokens = _tokenize(original)
    pref_tokens = _tokenize(preferred)

    if len(og_tokens) == len(pref_tokens):
        return build_mappings(original, preferred)

    if client is None:
        client = get_llm_client()

    return _llm_fallback_mappings(client, original, preferred, call_llm_fn=call_llm_fn)
