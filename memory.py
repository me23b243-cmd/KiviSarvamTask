"""
MemoryElement + memory manipulation.

Preserved from the notebook: add_memory_element, update_memory_element,
edit_memory, give_elements, summarize_memory_element,
reconstruct_preferred_sentence.

Two small, clearly-flagged integration fixes were made inside
`reconstruct_preferred_sentence` — search for "Integration fix" below for
both. Everything else is unchanged in behavior.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import config
from mapping import Mapping


@dataclass
class MemoryElement:
    og_word: str
    pref_word: str
    context: list[str] = field(default_factory=list)
    occurrence: int = 0
    representative: str = ""


def add_memory_element(
    memory: dict[str, MemoryElement],
    mapping: Mapping
) -> dict[str, MemoryElement]:
    """
    Add a new MemoryElement to memory.

    The context is taken directly from mapping.pref_set.
    """

    representative = f"{mapping.og_word} : {mapping.pref_word}"

    element = MemoryElement(
        og_word=mapping.og_word,
        pref_word=mapping.pref_word,
        context=[mapping.pref_set],
        occurrence=1,
        representative=representative
    )

    memory[representative] = element

    return memory


def update_memory_element(
    memory: dict[str, MemoryElement],
    mapping
) -> dict[str, MemoryElement]:
    """
    Update an existing MemoryElement using a Mapping.

    The mapping's pref_set is added as the context.
    """

    representative = f"{mapping.og_word} : {mapping.pref_word}"

    element = memory[representative]

    element.occurrence += 1
    element.context.append(mapping.pref_set)

    return memory


def edit_memory(
    mappings: list[Mapping],
    memory: dict[str, MemoryElement]
) -> dict[str, MemoryElement]:

    for mapping in mappings:

        representative = f"{mapping.og_word} : {mapping.pref_word}"

        if representative in memory:
            memory = update_memory_element(memory, mapping)
        else:
            memory = add_memory_element(memory, mapping)

    return memory


def give_elements(sentence: str, memory):
    """
    Find all memory mappings applicable to words in a sentence.

    Returns:
        {
            "og_word": [
                "og_word : pref_word",
                ...
            ]
        }
    """
    mapping_dict = {}

    # Split on whitespace
    words = sentence.split()

    for word in words:

        # Remove possessive 's / 's first
        normalized_word = re.sub(r"['\u2019]s$", "", word, flags=re.IGNORECASE)
        normalized_word = normalized_word.strip(".,!?;:\"'()[]{}@#%^&*()")
        normalized_word = normalized_word.split("-", 1)[0]

        if not normalized_word:
            continue

        matched_mappings = []

        for memory_element in memory.values():
            og_word = memory_element.og_word
            pref_word = memory_element.pref_word

            if og_word == normalized_word:
                matched_mappings.append(
                    f"{og_word} : {pref_word}"
                )

        if matched_mappings:
            mapping_dict[normalized_word] = matched_mappings

    return mapping_dict


def summarize_memory_element(memory_element, llm):
    og_word = memory_element.og_word
    pref_word = memory_element.pref_word
    previous_context = memory_element.context

    examples = "\n".join(f"- {s}" for s in previous_context) if previous_context else "(no examples provided)"

    system_prompt = (
        "You are a linguistic analyst that writes compact correction rules for a "
        "word-correction system. You infer, from example sentences, why an observed "
        "word form should (or should not) be replaced by a preferred form, and you "
        "state that as a single reusable rule. You never just define the words."
    )

    prompt = f"""Observed form (og_word): "{og_word}"
Preferred form (pref_word): "{pref_word}"

Example sentences where this mapping was observed:
{examples}

Task:
Write ONE compact rule (1-2 sentences) that a downstream word-correction model can use
to decide whether to replace "{og_word}" with "{pref_word}" in a NEW, unseen sentence.

The rule must:
- Explain what "{og_word}" represents in these examples, what "{pref_word}" represents,
  and the contextual condition under which "{pref_word}" is the correct/preferred choice.
- Generalize beyond the exact example sentences (describe the reusable pattern, not the
  specific sentences).
- Preserve the direction of correction: "{og_word}" -> "{pref_word}". Do not reverse it.
- If og_word and pref_word are identical, describe the useful contextual role/identity
  instead of inventing a correction (e.g. "X is used as a person's proper name.").
- Focus only on what is actually supported by the examples: usage, role, entity identity,
  spelling, capitalization, ASR error, or disambiguating context — whichever applies.
- NOT define either word independently, NOT copy the example sentences, and NOT invent
  facts not supported by the examples.

Output ONLY the rule text. No labels, no quotes, no JSON, no explanation, no preamble.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    response = llm(
        messages,
        max_new_tokens=100,
        do_sample=False,
        return_full_text=False
    )[0]["generated_text"]

    memory_element.context = response.strip()
    return memory_element


# def reconstruct_preferred_sentence(sentence, memory, llm, max_new_tokens: Optional[int] = None):
#     """
#     Reconstruct the Preferred-form version of an ASR sentence using learned
#     word-level corrections stored in `memory`.

#     Pipeline:
#         1. Find candidate corrections for this sentence via get_mappings().
#         2. Pull each candidate's pref_word + summarized context from memory.
#         3. Build one sentence-level prompt containing ALL candidates + evidence.
#         4. Ask the LLM to pick the best coherent combination of substitutions
#            (or none), under strict "don't invent/reorder/rewrite" constraints.
#         5. Validate the LLM output against the allowed candidate set before
#            trusting it; fall back to the original sentence on any doubt.
#     """

#     if max_new_tokens is None:
#             max_new_tokens = config.MAX_NEW_TOKENS_RECONSTRUCT

#     # 1. Get word-level candidate mappings for this sentence.
#     mappings = give_elements(sentence, memory)
#     if not mappings:
#         return sentence

#     # 2. Resolve each mapping into (og_word -> [(pref_word, context), ...]),
#     #    skipping missing/broken memory entries and de-duplicating candidates.
#     candidates_by_word = {}

#     # allowed_replacements: whole-token authorization, e.g. {"oracle": {"Oracle"}}
#     allowed_replacements = {}

#     # CHANGED (new): allowed_components: component-level authorization, keyed
#     # by the memory's own canonical og_word (e.g. "oracle" -> {"Oracle"}).
#     # This is what lets hyphenated/compound sentence tokens such as
#     # "oracle-based" be validated even though the whole token itself was
#     # never registered as a key (only its lexical component was).
#     allowed_components = {}

#     for og_word, representatives in mappings.items():
#         if not representatives:
#             continue

#         seen_pref_words = set()
#         resolved_candidates = []
#         for representative in representatives:
#             memory_element = memory.get(representative)
#             if memory_element is None:
#                 continue

#             pref_word = memory_element.pref_word
#             if not pref_word or pref_word in seen_pref_words:
#                 continue
#             seen_pref_words.add(pref_word)

#             context = memory_element.context
#             context_text = context if isinstance(context, str) else " ".join(context or [])
#             # FIXED (critical bug): the previous code did
#             #     context_text = context_text[0].strip() or "..."
#             # which indexes the FIRST CHARACTER of the (already-built)
#             # context string instead of using the string itself. That threw
#             # away virtually all contextual evidence produced by
#             # summarize_memory_element(), leaving the reconstruction LLM
#             # with almost nothing to judge KEEP-vs-REPLACE decisions on.
#             context_text = context_text.strip() or "No specific usage evidence recorded."

#             resolved_candidates.append((pref_word, context_text))

#             base_og = getattr(memory_element, "og_word", None)
#             if base_og:
#                 allowed_components.setdefault(base_og, set()).add(pref_word)
#                 allowed_components.setdefault(base_og.lower(), set()).add(pref_word)

#         if resolved_candidates:
#             candidates_by_word[og_word] = resolved_candidates
#             allowed_replacements[og_word] = {p for p, _ in resolved_candidates}

#     # If nothing usable survived resolution, skip the LLM entirely.
#     if not candidates_by_word:
#         return sentence

#     # 3. Build the candidate-mapping block of the prompt.
#     mapping_lines = []
#     for og_word, candidates in candidates_by_word.items():
#         mapping_lines.append(f'"{og_word}":')
#         for pref_word, context_text in candidates:
#             mapping_lines.append(f'  -> "{pref_word}" | evidence: {context_text}')
#     mapping_block = "\n".join(mapping_lines)

#     system_prompt = (
#         "You are a constrained sentence reconstruction system. You are given an ASR "
#         "sentence and a set of learned candidate word substitutions, each backed by "
#         "contextual evidence describing how that candidate has previously been used. "
#         "You decide which substitutions, if any, make the sentence most contextually "
#         "and semantically correct. You are NOT a general grammar corrector or "
#         "paraphraser: you may only apply the exact substitutions supplied to you, and "
#         "everything else in the sentence must remain untouched."
#     )

#     prompt = f"""ASR sentence:
# {sentence}

# Candidate substitutions (original word -> possible replacement | supporting evidence):
# {mapping_block}

# Instructions:
# - For each original word above, use its evidence to judge whether a replacement fits THIS sentence.
# - Consider the full sentence's meaning, not each word in isolation — choices may interact.
# - A word may be left unchanged even if candidates exist for it; evidence is guidance, not a command.
# - Do not replace a word just because a mapping exists for it. Only replace it if the ORIGINAL
#   word does NOT make sense in this sentence, or if the evidence clearly shows the preferred
#   form is required here. If the original word already reads naturally and correctly, KEEP it.
#   Example: if a candidate is "week" -> "weak", but the sentence says "posting photos all week",
#   "week" is already a correct, natural time expression here, so it must be KEPT, not replaced.
# - If an original word is part of a hyphenated compound (e.g. "oracle-based"), and the candidate
#   substitution applies to just the lexical part of it, apply the substitution to only that part
#   and keep the rest of the compound attached and unchanged (e.g. "oracle-based" -> "Oracle-based").
#   Never replace the whole compound with the bare replacement word alone.
# - Apply ONLY the exact replacement words listed above. Never invent, add, remove, reorder, or paraphrase.
# - Do not alter spelling, punctuation, or grammar of any word that has no listed candidate or isn't chosen.
# - Never copy the evidence text into your answer.
# - Output ONLY the final reconstructed sentence — no quotes, labels, or explanation.
# """

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": prompt},
#     ]

#     raw_response = llm(messages)

#     # Normalize whatever shape the llm callable returns into plain text.
#     def _extract_text(resp):
#         if isinstance(resp, list) and resp:
#             resp = resp[0]
#         if isinstance(resp, dict):
#             resp = resp.get("generated_text", "")
#         if isinstance(resp, list):
#             if resp and isinstance(resp[-1], dict):
#                 return str(resp[-1].get("content", ""))
#             return " ".join(str(item) for item in resp)
#         return str(resp) if resp is not None else ""

#     candidate_text = _extract_text(raw_response)
#     candidate_text = candidate_text.strip().strip('"').strip("'").strip()

#     # 4. Validate: same token count, and every changed token is an
#     #    explicitly authorized replacement for its original token.
#     original_tokens = sentence.split()
#     candidate_tokens = candidate_text.split()

#     if not candidate_text or len(candidate_tokens) != len(original_tokens):
#         return sentence

#     # Splits a token into (leading non-word chars, core lexical word,
#     # trailing non-word chars). Internal characters (e.g. the apostrophe in
#     # "don't", or a hyphen in "oracle-based") are NOT treated as boundaries,
#     # since \W* only anchors at the very start/end of the token via ^...$.
#     _BOUNDARY_RE = re.compile(r"^(\W*)(.*?)(\W*)$", re.UNICODE)

#     # Trailing possessive marker ('s or 's) attached directly to the core,
#     # e.g. "oracle's" -> core "oracle", suffix "'s". Only this specific
#     # case is special-cased; other apostrophes (don't, y'all) are left
#     # untouched inside the core as before.
#     _POSSESSIVE_RE = re.compile(r"^(.*)('s|\u2019s)$", re.UNICODE)

#     def _split_word(tok):
#         m = _BOUNDARY_RE.match(tok)
#         if not m:
#             return "", tok, ""
#         prefix, core, suffix = m.group(1), m.group(2), m.group(3)

#         poss_m = _POSSESSIVE_RE.match(core)
#         if poss_m and poss_m.group(1):
#             core, possessive = poss_m.group(1), poss_m.group(2)
#             suffix = possessive + suffix

#         return prefix, core, suffix

#     # CHANGED (new): whole-token check first (unchanged behaviour), then a
#     # hyphen-aware fallback that authorizes changing exactly one lexical
#     # component of a hyphenated compound, provided that specific component
#     # change is an authorized mapping and every other component of the
#     # compound is preserved character-for-character. This is what allows
#     # "oracle-based" -> "Oracle-based" to validate correctly.
#     def _core_change_is_authorized(orig_core, new_core):
#         if new_core in allowed_replacements.get(orig_core, set()):
#             return True

#         if "-" in orig_core and "-" in new_core:
#             orig_parts = orig_core.split("-")
#             new_parts = new_core.split("-")
#             if len(orig_parts) == len(new_parts):
#                 changed_parts = [(op, np) for op, np in zip(orig_parts, new_parts) if op != np]
#                 if len(changed_parts) == 1:
#                     op, np = changed_parts[0]
#                     allowed = allowed_components.get(op) or allowed_components.get(op.lower(), set())
#                     if allowed and np in allowed:
#                         return True

#         return False

#     for orig_tok, new_tok in zip(original_tokens, candidate_tokens):
#         if orig_tok == new_tok:
#             continue

#         orig_prefix, orig_core, orig_suffix = _split_word(orig_tok)
#         new_prefix, new_core, new_suffix = _split_word(new_tok)

#         # Any attached non-word characters (punctuation, quotes, dashes, etc.)
#         # must be preserved exactly. Only the core word itself may change.
#         if orig_prefix != new_prefix or orig_suffix != new_suffix:
#             return sentence

#         if orig_core == new_core:
#             continue

#         if not _core_change_is_authorized(orig_core, new_core):
#             # Either an unlisted word changed, or an invented replacement was used.
#             return sentence

#     return candidate_text

# @title
import re

# ---------------------------------------------------------------------------
# Shared helpers (extracted from the original reconstruct_preferred_sentence
# so both the Stage-1 and the new Stage-2 function can reuse the exact same
# mapping-building, LLM-response parsing, and strict validation logic
# instead of duplicating/forking it.)
# ---------------------------------------------------------------------------

def _extract_llm_text(resp):
    """Normalize whatever shape the llm callable returns into plain text."""
    if isinstance(resp, list) and resp:
        resp = resp[0]
    if isinstance(resp, dict):
        resp = resp.get("generated_text", "")
    if isinstance(resp, list):
        if resp and isinstance(resp[-1], dict):
            return str(resp[-1].get("content", ""))
        return " ".join(str(item) for item in resp)
    return str(resp) if resp is not None else ""


def _build_candidate_data(sentence, memory):
    """
    Resolve get_mappings() output into everything a reconstruction prompt
    needs: per-word candidate lists with evidence, whole-token authorization,
    component-level (hyphen-aware) authorization, and a rendered prompt block.

    Returns None if there is nothing usable to work with.
    """
    mappings = give_elements(sentence, memory)
    if not mappings:
        return None

    candidates_by_word = {}

    # allowed_replacements: whole-token authorization, e.g. {"oracle": {"Oracle"}}
    allowed_replacements = {}

    # allowed_components: component-level authorization, keyed by the
    # memory's own canonical og_word (e.g. "oracle" -> {"Oracle"}). This is
    # what lets hyphenated/compound sentence tokens such as "oracle-based"
    # be validated even though the whole token itself was never registered
    # as a key (only its lexical component was).
    allowed_components = {}

    for og_word, representatives in mappings.items():
        if not representatives:
            continue

        seen_pref_words = set()
        resolved_candidates = []
        for representative in representatives:
            memory_element = memory.get(representative)
            if memory_element is None:
                continue

            pref_word = memory_element.pref_word
            if not pref_word or pref_word in seen_pref_words:
                continue
            seen_pref_words.add(pref_word)

            context = memory_element.context
            context_text = context if isinstance(context, str) else " ".join(context or [])
            context_text = context_text.strip() or "No specific usage evidence recorded."

            resolved_candidates.append((pref_word, context_text))

            base_og = getattr(memory_element, "og_word", None)
            if base_og:
                allowed_components.setdefault(base_og, set()).add(pref_word)
                allowed_components.setdefault(base_og.lower(), set()).add(pref_word)

        if resolved_candidates:
            candidates_by_word[og_word] = resolved_candidates
            allowed_replacements[og_word] = {p for p, _ in resolved_candidates}

    if not candidates_by_word:
        return None

    mapping_lines = []
    for og_word, candidates in candidates_by_word.items():
        mapping_lines.append(f'"{og_word}":')
        for pref_word, context_text in candidates:
            mapping_lines.append(f'  -> "{pref_word}" | evidence: {context_text}')
    mapping_block = "\n".join(mapping_lines)

    return {
        "candidates_by_word": candidates_by_word,
        "allowed_replacements": allowed_replacements,
        "allowed_components": allowed_components,
        "mapping_block": mapping_block,
    }


# Splits a token into (leading non-word chars, core lexical word, trailing
# non-word chars). Internal characters (e.g. the apostrophe in "don't", or a
# hyphen in "oracle-based") are NOT treated as boundaries, since \W* only
# anchors at the very start/end of the token via ^...$.
_BOUNDARY_RE = re.compile(r"^(\W*)(.*?)(\W*)$", re.UNICODE)

# Trailing possessive marker ('s or 's) attached directly to the core, e.g.
# "oracle's" -> core "oracle", suffix "'s". Only this specific case is
# special-cased; other apostrophes (don't, y'all) are left untouched inside
# the core as before.
_POSSESSIVE_RE = re.compile(r"^(.*)('s|\u2019s)$", re.UNICODE)


def _split_word(tok):
    m = _BOUNDARY_RE.match(tok)
    if not m:
        return "", tok, ""
    prefix, core, suffix = m.group(1), m.group(2), m.group(3)

    poss_m = _POSSESSIVE_RE.match(core)
    if poss_m and poss_m.group(1):
        core, possessive = poss_m.group(1), poss_m.group(2)
        suffix = possessive + suffix

    return prefix, core, suffix


def _core_change_is_authorized(orig_core, new_core, allowed_replacements, allowed_components):
    """
    Whole-token check first (unchanged behaviour), then a hyphen-aware
    fallback that authorizes changing exactly one lexical component of a
    hyphenated compound, provided that specific component change is an
    authorized mapping and every other component of the compound is
    preserved character-for-character.
    """
    if new_core in allowed_replacements.get(orig_core, set()):
        return True

    if "-" in orig_core and "-" in new_core:
        orig_parts = orig_core.split("-")
        new_parts = new_core.split("-")
        if len(orig_parts) == len(new_parts):
            changed_parts = [(op, np) for op, np in zip(orig_parts, new_parts) if op != np]
            if len(changed_parts) == 1:
                op, np = changed_parts[0]
                allowed = allowed_components.get(op) or allowed_components.get(op.lower(), set())
                if allowed and np in allowed:
                    return True

    return False


def _validate_reconstruction(base_sentence, candidate_text, allowed_replacements, allowed_components):
    """
    Validate `candidate_text` against `base_sentence`: same token count, and
    every changed token is an explicitly authorized replacement (whole-token
    or hyphen-component) for its original token. Returns candidate_text if
    valid, or None if it must be rejected.
    """
    original_tokens = base_sentence.split()
    candidate_tokens = candidate_text.split()

    if not candidate_text or len(candidate_tokens) != len(original_tokens):
        return None

    for orig_tok, new_tok in zip(original_tokens, candidate_tokens):
        if orig_tok == new_tok:
            continue

        orig_prefix, orig_core, orig_suffix = _split_word(orig_tok)
        new_prefix, new_core, new_suffix = _split_word(new_tok)

        # Any attached non-word characters (punctuation, quotes, dashes, etc.)
        # must be preserved exactly. Only the core word itself may change.
        if orig_prefix != new_prefix or orig_suffix != new_suffix:
            return None

        if orig_core == new_core:
            continue

        if not _core_change_is_authorized(orig_core, new_core, allowed_replacements, allowed_components):
            return None

    return candidate_text


# ---------------------------------------------------------------------------
# 2. first_stage()  — STAGE 1, behaviour unchanged
# ---------------------------------------------------------------------------
def first_stage(sentence, memory, llm):
    """
    Reconstruct the Preferred-form version of an ASR sentence using learned
    word-level corrections stored in `memory`.

    Pipeline:
        1. Find candidate corrections for this sentence via get_mappings().
        2. Pull each candidate's pref_word + summarized context from memory.
        3. Build one sentence-level prompt containing ALL candidates + evidence.
        4. Ask the LLM to pick the best coherent combination of substitutions
           (or none), under strict "don't invent/reorder/rewrite" constraints.
        5. Validate the LLM output against the allowed candidate set before
           trusting it; fall back to the original sentence on any doubt.

    External input/output behaviour is unchanged from before; the internal
    mapping-building/parsing/validation logic has only been moved into
    shared helper functions (_build_candidate_data / _extract_llm_text /
    _validate_reconstruction) so Stage 2 can reuse it identically.
    """
    data = _build_candidate_data(sentence, memory)
    if data is None:
        return sentence

    system_prompt = (
        "You are a constrained sentence reconstruction system. You are given an ASR "
        "sentence and a set of learned candidate word substitutions, each backed by "
        "contextual evidence describing how that candidate has previously been used. "
        "You decide which substitutions, if any, make the sentence most contextually "
        "and semantically correct. You are NOT a general grammar corrector or "
        "paraphraser: you may only apply the exact substitutions supplied to you, and "
        "everything else in the sentence must remain untouched."
    )

    prompt = f"""ASR sentence:
{sentence}

Candidate substitutions (original word -> possible replacement | supporting evidence):
{data['mapping_block']}

Instructions:
- For each original word above, use its evidence to judge whether a replacement fits THIS sentence.
- Consider the full sentence's meaning, not each word in isolation — choices may interact.
- A word may be left unchanged even if candidates exist for it; evidence is guidance, not a command.
- Do not replace a word just because a mapping exists for it. Only replace it if the ORIGINAL
  word does NOT make sense in this sentence, or if the evidence clearly shows the preferred
  form is required here. If the original word already reads naturally and correctly, KEEP it.
  Example: if a candidate is "week" -> "weak", but the sentence says "posting photos all week",
  "week" is already a correct, natural time expression here, so it must be KEPT, not replaced.
- If an original word is part of a hyphenated compound (e.g. "oracle-based"), and the candidate
  substitution applies to just the lexical part of it, apply the substitution to only that part
  and keep the rest of the compound attached and unchanged (e.g. "oracle-based" -> "Oracle-based").
  Never replace the whole compound with the bare replacement word alone.
- Apply ONLY the exact replacement words listed above. Never invent, add, remove, reorder, or paraphrase.
- Do not alter spelling, punctuation, or grammar of any word that has no listed candidate or isn't chosen.
- Never copy the evidence text into your answer.
- Output ONLY the final reconstructed sentence — no quotes, labels, or explanation.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    raw_response = llm(messages)
    candidate_text = _extract_llm_text(raw_response).strip().strip('"').strip("'").strip()

    validated = _validate_reconstruction(
        sentence, candidate_text, data["allowed_replacements"], data["allowed_components"]
    )
    return validated if validated is not None else sentence


# ---------------------------------------------------------------------------
# 3. semantic_review_correction()  — NEW STAGE 2
# ---------------------------------------------------------------------------
def reconstruct_preferred_sentence(sentence, memory, llm):
    """
    Two-stage correction pipeline:

        ASR sentence -> first_stage() [Stage 1]
                      -> semantic review by reconstruct_preferred_sentence [Stage 2]
                      -> final sentence

    Stage 1 is called completely unchanged and produces `upstream_sentence`.

    Stage 2 is deliberately NOT a second independent correction pass. It:
      1. Is shown the ORIGINAL sentence, the Stage-1 output, and the same
         candidate mappings + evidence Stage 1 had available (rebuilt from
         the ORIGINAL sentence, not from upstream_sentence, so the
         authorized substitution set can never drift or go stale).
      2. Is explicitly instructed to first judge whether the Stage-1
         sentence, as a whole, already makes semantic/contextual/logical
         sense — and to only touch it if it doesn't. This is what stops it
         from habitually "fixing" sentences that are already fine.
      3. May keep all, some, or none of Stage 1's substitutions, and may
         also apply a mapping Stage 1 didn't use, or restore a mapping
         Stage 1 wrongly applied — but only from the same authorized set.
      4. Its output is run through the exact same strict validator used in
         Stage 1 (same token count as the ORIGINAL sentence, every changed
         word an authorized whole-token or hyphen-component replacement).
         Any invalid/malformed output falls back to `upstream_sentence`
         (never all the way back to the raw ASR sentence), since Stage 1 is
         already a reasonably trustworthy baseline and Stage 2's only job is
         to improve on it, never to make it worse.

    If there are no candidate mappings for this sentence at all, Stage 2 has
    nothing to review and the Stage-1 result is returned as-is.
    """
    upstream_sentence = first_stage(sentence, memory, llm)

    # Rebuilt from the ORIGINAL sentence (not upstream_sentence) so the
    # authorization set used for validation is identical to the one Stage 1
    # itself was constrained by.
    data = _build_candidate_data(sentence, memory)
    if data is None:
        return upstream_sentence

    system_prompt = (
        "You are a semantic-review system for an ASR correction pipeline. You are "
        "given the original ASR sentence, a proposed corrected sentence produced by "
        "an upstream correction model, and the learned candidate substitutions (with "
        "supporting evidence) that were available to that upstream model. Your job is "
        "NOT to perform a fresh, independent correction. Your job is to critically "
        "judge whether the upstream sentence, taken as a whole, is already semantically, "
        "contextually and logically sound. Only if it is not should you selectively "
        "adjust it, and only using the exact substitutions supplied. A mapping is a "
        "candidate, not an instruction: it is valid to keep all of them, change all of "
        "them, or change only some, whichever combination produces the most natural "
        "and meaningful sentence while preserving the original meaning."
    )

    prompt = f"""Original ASR sentence:
{sentence}

Upstream proposed sentence:
{upstream_sentence}

Candidate substitutions available to the upstream model (original word -> possible replacement | supporting evidence):
{data['mapping_block']}

Task:
1. First decide: does the upstream sentence, as a complete sentence, already make full
   semantic, contextual and logical sense? Judge the sentence as a whole — some word
   choices are only correct in combination with each other.
2. If YES, the final sentence is the upstream sentence, completely unchanged.
3. If NO, construct the final sentence by starting from the original ASR sentence and
   applying only whichever of the candidate substitutions above (individually or in
   combination) are needed to make it correct. Preserve every upstream correction that
   is still valid; only undo or add a substitution where the evidence and the full
   sentence's meaning clearly justify it.
   Example: if a candidate is "week" -> "weak" but the sentence reads "posting photos
   all week", "week" is already correct there and must be kept even if the upstream
   model changed it.

Rules:
- Never invent a replacement word that is not listed above.
- Never add, remove, or reorder words.
- Never change punctuation or any word that has no listed candidate.
- The output must always have exactly the same number of words as the original ASR sentence.
- Output ONLY the final sentence — no labels, no explanation, no quotes.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    raw_response = llm(messages)
    candidate_text = _extract_llm_text(raw_response).strip().strip('"').strip("'").strip()

    validated = _validate_reconstruction(
        sentence, candidate_text, data["allowed_replacements"], data["allowed_components"]
    )

    return validated if validated is not None else upstream_sentence