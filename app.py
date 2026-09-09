"""
Kivi — Adaptive Sentence Correction

Streamlit application. Run with:

    streamlit run app.py

This file only contains UI/orchestration code. All core logic lives in
mapping.py, memory.py, database.py, memory_store.py, and llm.py.
"""

from __future__ import annotations

import os

import streamlit as st

import database
import llm
import memory_store
import metrics
import validation
from mapping import LLMMappingError, _tokenize, get_mappings
from memory import (
    MemoryElement,
    edit_memory,
    give_elements,
    reconstruct_preferred_sentence,
    summarize_memory_element,
)

st.set_page_config(page_title="Kivi — Adaptive Sentence Correction", page_icon="🧠", layout="centered")


# ---------------------------------------------------------------------------
# Startup: make sure the database exists. Models are NOT loaded here — they
# are loaded lazily, only when a code path actually needs them.
# ---------------------------------------------------------------------------
try:
    database.init_db()
    metrics.init_metrics_tables()
except database.DatabaseError as e:
    st.error(f"Could not initialize the database: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _needs_mapping_llm(original: str, preferred: str) -> bool:
    """Mirrors get_mappings' own deterministic/LLM-fallback decision, using
    the exact same tokenizer, so we only load the small model when it will
    actually be used."""
    return len(_tokenize(original)) != len(_tokenize(preferred))


def _apply_mappings_with_status(mappings, memory: dict[str, MemoryElement]):
    """
    Applies each mapping via edit_memory (unchanged logic) one at a time so
    we can report, per mapping, whether it created a new memory element or
    updated an existing one — without altering edit_memory's own behavior.
    """
    statuses = []
    for mapping in mappings:
        representative = f"{mapping.og_word} : {mapping.pref_word}"
        existed_before = representative in memory
        memory = edit_memory([mapping], memory)
        statuses.append(
            {
                "Original": mapping.og_word,
                "Preferred": mapping.pref_word,
                "Status": "Updated" if existed_before else "New",
            }
        )
    return memory, statuses


def _relevant_elements(memory: dict[str, MemoryElement], mapping_dict: dict) -> dict[str, MemoryElement]:
    """Collect the unique MemoryElements referenced by a give_elements() result."""
    elements: dict[str, MemoryElement] = {}
    for representatives in mapping_dict.values():
        for representative in representatives:
            element = memory.get(representative)
            if element is not None:
                elements[representative] = element
    return elements


def _display_context(context) -> str:
    """Human-readable rendering of a MemoryElement's context for the UI table."""
    if isinstance(context, str):
        return context
    if isinstance(context, list):
        return " | ".join(context) if context else "(no examples yet)"
    return str(context)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🧠 Kivi — Adaptive Sentence Correction")
st.caption(
    "Teach Kivi how your ASR output should be corrected, then let it apply what "
    "it has learned to new sentences."
)

tab_learn, tab_correct, tab_validate = st.tabs(
    ["📚 Learn a Correction", "✏️ Correct a Sentence", "🧪 Validate System"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Learn a Correction
# ---------------------------------------------------------------------------
with tab_learn:
    st.subheader("Learn a Correction")
    st.write(
        "Enter the raw ASR sentence and the sentence you actually meant. "
        "Kivi will figure out which words changed and remember the correction."
    )

    asr_sentence = st.text_area("ASR Sentence", key="asr_sentence", height=90)
    preferred_sentence = st.text_area("Preferred Sentence", key="preferred_sentence", height=90)

    if st.button("Save Correction", type="primary"):
        asr_clean = (asr_sentence or "").strip()
        pref_clean = (preferred_sentence or "").strip()

        if not asr_clean or not pref_clean:
            st.warning("Please fill in both the ASR sentence and the preferred sentence.")
        else:
            try:
                # Only load the mapping LLM if this sentence pair actually
                # needs the fallback path (unequal token counts).
                small_llm = None
                if _needs_mapping_llm(asr_clean, pref_clean):
                    with st.spinner("Aligning sentences with the mapping model..."):
                        small_llm = llm.load_small_llm()

                if small_llm is not None:
                    with metrics.timed_call(
                        "mapping", tokenizer=getattr(small_llm, "tokenizer", None),
                        input_text=f"{asr_clean} || {pref_clean}",
                    ) as m:
                        mappings = get_mappings(asr_clean, pref_clean, client=small_llm)
                        m["output_text"] = str(mappings)
                else:
                    mappings = get_mappings(asr_clean, pref_clean, client=None)

            except llm.ModelLoadError as e:
                st.error(str(e))
            except LLMMappingError as e:
                st.error(
                    "Kivi couldn't confidently align these two sentences, so nothing was saved. "
                    f"You can try rephrasing and saving again. (Details: {e})"
                )
            except ValueError as e:
                st.error(f"Couldn't process that pair: {e}")
            else:
                if not mappings:
                    st.info("No word-level differences were found between the two sentences — nothing to learn.")
                else:
                    try:
                        memory = memory_store.load_memory_from_db()
                        memory, statuses = _apply_mappings_with_status(mappings, memory)

                        touched_reps = {
                            f"{m.og_word} : {m.pref_word}" for m in mappings
                        }
                        touched_elements = {
                            rep: el for rep, el in memory.items() if rep in touched_reps
                        }
                        memory_store.persist_memory_elements(touched_elements)
                        metrics.record_db_snapshot()

                    except database.DatabaseError as e:
                        st.error(
                            "The correction could NOT be saved — a database error occurred, "
                            f"so nothing was persisted. (Details: {e})"
                        )
                    else:
                        new_count = sum(1 for s in statuses if s["Status"] == "New")
                        updated_count = sum(1 for s in statuses if s["Status"] == "Updated")

                        st.success(
                            f"✓ Correction saved — {len(statuses)} mapping(s) learned "
                            f"({new_count} new, {updated_count} updated)."
                        )
                        st.table(statuses)

    with st.expander("View learned memory"):
        try:
            current_memory = memory_store.load_memory_from_db()
        except database.DatabaseError as e:
            st.error(f"Could not load memory: {e}")
        else:
            if not current_memory:
                st.write("No corrections learned yet.")
            else:
                rows = [
                    {
                        "Original": el.og_word,
                        "Preferred": el.pref_word,
                        "Occurrences": el.occurrence,
                        "Representative": el.representative,
                        "Context / Rule": _display_context(el.context),
                    }
                    for el in sorted(current_memory.values(), key=lambda e: -e.occurrence)
                ]
                st.dataframe(rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 2 — Correct a Sentence
# ---------------------------------------------------------------------------
with tab_correct:
    st.subheader("Correct a Sentence")
    st.write("Enter a sentence and Kivi will apply any corrections it has learned so far.")

    input_sentence = st.text_area("Enter ASR Sentence", key="input_sentence", height=110)

    if st.button("Reconstruct Preferred Sentence", type="primary"):
        sentence_clean = (input_sentence or "").strip()

        if not sentence_clean:
            st.warning("Please enter a sentence.")
        else:
            try:
                memory = memory_store.load_memory_from_db()
                mapping_dict = give_elements(sentence_clean, memory)

                if not mapping_dict:
                    st.info("No applicable learned correction was found for this sentence.")
                    st.markdown("**ASR Sentence**")
                    st.write(sentence_clean)
                else:
                    relevant_elements = _relevant_elements(memory, mapping_dict)

                    # Summarize only the candidate elements whose context has
                    # changed since they were last summarized — never the
                    # whole memory table, and never on every run.
                    stale_reps = [
                        rep
                        for rep, el in relevant_elements.items()
                        if memory_store.is_summary_stale(rep, len(el.context) if isinstance(el.context, list) else 0)
                    ]

                    if stale_reps:
                        with st.spinner("Refreshing learned rules for the relevant corrections..."):
                            small_llm = llm.load_small_llm()
                            for rep in stale_reps:
                                element = relevant_elements[rep]
                                raw_len = len(element.context) if isinstance(element.context, list) else 0
                                try:
                                    input_text = (
                                        "\n".join(element.context)
                                        if isinstance(element.context, list)
                                        else str(element.context)
                                    )
                                    with metrics.timed_call(
                                        "summarize", tokenizer=getattr(small_llm, "tokenizer", None),
                                        input_text=input_text,
                                    ) as m:
                                        summarize_memory_element(element, small_llm)
                                        m["output_text"] = element.context
                                    memory_store.persist_summary(rep, element.context, raw_len)
                                except Exception:
                                    # Summarization failure -> keep the raw
                                    # context as-is (already the case, since
                                    # summarize_memory_element only mutates
                                    # element.context on success) and move on.
                                    pass

                    # For any relevant element that already has a cached,
                    # up-to-date summary but wasn't just recomputed, use that
                    # cached summary as its context (matches the notebook's
                    # own intent: summarize_memory_element permanently
                    # replaces .context with the compact rule).
                    for rep, element in relevant_elements.items():
                        if rep in stale_reps:
                            continue
                        cached_summary, _ = memory_store.get_summary_state(rep)
                        if cached_summary:
                            element.context = cached_summary

                    with st.spinner("Reconstructing the preferred sentence..."):
                        big_llm = llm.load_small_llm()
                        with metrics.timed_call(
                            "reconstruct", tokenizer=getattr(big_llm, "tokenizer", None),
                            input_text=sentence_clean,
                        ) as m:
                            result = reconstruct_preferred_sentence(sentence_clean, memory, big_llm)
                            m["output_text"] = result

                    st.markdown("**ASR Sentence**")
                    st.write(sentence_clean)
                    st.markdown("**Preferred Sentence**")
                    st.success(result)

                    if result == sentence_clean:
                        st.caption(
                            "Kivi found relevant learned corrections but decided none of them "
                            "fit this sentence, or the reconstruction could not be validated — "
                            "the original sentence was kept as a safe fallback."
                        )
                    else:
                        st.caption(f"Relevant memory entries considered: {len(relevant_elements)}")

            except llm.ModelLoadError as e:
                st.error(str(e))
            except database.DatabaseError as e:
                st.error(f"Could not read memory from the database: {e}")


# ---------------------------------------------------------------------------
# Tab 3 — Validate System
# ---------------------------------------------------------------------------
with tab_validate:
    st.subheader("Validate System")
    st.write(
        "Run the correction pipeline over a labeled test set "
        "(`asr_output`, `expected_output`, `expected_action`) and score it."
    )

    uploaded_csv = st.file_uploader("Test CSV", type=["csv"])
    if uploaded_csv is None and os.path.exists(validation.DEFAULT_TEST_CSV_PATH):
        st.caption(f"No file uploaded — will use `{os.path.basename(validation.DEFAULT_TEST_CSV_PATH)}` found in the project folder.")

    if st.button("Validate System", type="primary"):
        source = uploaded_csv if uploaded_csv is not None else (
            validation.DEFAULT_TEST_CSV_PATH if os.path.exists(validation.DEFAULT_TEST_CSV_PATH) else None
        )

        if source is None:
            st.warning(
                "No test CSV found. Upload one, or place a `df_combined_test.csv` "
                "file in the project folder."
            )
        else:
            try:
                df_test = validation.load_test_csv(source)
            except (ValueError, Exception) as e:
                st.error(f"Could not read the test CSV: {e}")
            else:
                try:
                    memory = memory_store.load_memory_from_db()

                    # Refresh only the summaries this test set actually needs,
                    # once, before the row-by-row loop (not per row).
                    with st.spinner("Preparing memory (refreshing stale summaries)..."):
                        small_llm = llm.load_small_llm()
                        validation.refresh_stale_summaries(df_test, memory, small_llm)
                        memory = memory_store.load_memory_from_db()  # reload with fresh summaries applied

                    big_llm = llm.load_small_llm()

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def _on_progress(done, total):
                        progress_bar.progress(done / total if total else 1.0)
                        status_text.text(f"Processing row {done}/{total}...")

                    df_result = validation.run_validation(df_test, memory, big_llm, progress_callback=_on_progress)
                    status_text.text(f"Done — processed {len(df_result)} rows.")

                    saved_path = validation.save_results(df_result)
                    metrics.record_db_snapshot()

                    results = validation.compute_metrics(df_result)
                    counts = results.pop("confusion_counts")

                    st.success(f"Validation complete. Results saved to `{saved_path}`.")

                    metric_rows = [{"Metric": k, "Value": f"{v:.3f}"} for k, v in results.items()]
                    st.table(metric_rows)
                    st.caption(
                        f"Confusion counts — TP: {counts['TP']}, FP: {counts['FP']}, "
                        f"FN: {counts['FN']}, TN: {counts['TN']} (n={counts['n']})"
                    )

                    st.download_button(
                        "Download df_test_result.csv",
                        data=df_result.to_csv(index=False).encode("utf-8"),
                        file_name="df_test_result.csv",
                        mime="text/csv",
                    )

                    with st.expander("View result rows"):
                        st.dataframe(df_result, use_container_width=True)

                except llm.ModelLoadError as e:
                    st.error(str(e))
                except database.DatabaseError as e:
                    st.error(f"Database error during validation: {e}")


# ---------------------------------------------------------------------------
# Metrics dashboard — latency, model usage, cost proxy, database growth.
# ---------------------------------------------------------------------------
with st.expander("📊 Usage & performance metrics"):
    try:
        usage = metrics.llm_usage_summary()
        growth = metrics.database_growth()
    except database.DatabaseError as e:
        st.error(f"Could not read metrics: {e}")
    else:
        st.markdown("**Model latency & usage** (per call purpose)")
        if not usage:
            st.write("No LLM calls recorded yet.")
        else:
            rows = [
                {
                    "Purpose": u["purpose"],
                    "Calls": u["calls"],
                    "Succeeded": u["successes"],
                    "Avg latency (s)": round(u["avg_seconds"], 2),
                    "Min / Max (s)": f"{u['min_seconds']:.2f} / {u['max_seconds']:.2f}",
                    "Total time (s)": round(u["total_seconds"], 2),
                    "Tokens in / out": f"{u['total_input_tokens']} / {u['total_output_tokens']}"
                    + (" (est.)" if u["any_estimated"] else ""),
                }
                for u in usage
            ]
            st.table(rows)

            total_seconds = sum(u["total_seconds"] for u in usage)
            st.markdown(
                f"**Cost:** $0 — running locally, no metered API. "
                f"Total compute time spent across all calls so far: **{total_seconds:.1f}s**, "
                f"the closest measurable resource-cost proxy on this setup."
            )

        st.markdown("**Database growth**")
        st.write(
            f"Current: {growth['current_rows']} rows, "
            f"{metrics.format_bytes(growth['current_size_bytes'])} on disk."
        )
        st.write(
            f"Since first recorded snapshot: "
            f"{'+' if growth['rows_added'] >= 0 else ''}{growth['rows_added']} rows, "
            f"{'+' if growth['size_added_bytes'] >= 0 else ''}{metrics.format_bytes(abs(growth['size_added_bytes']))}."
        )
        st.caption(
            "Token counts use each model's real tokenizer when available; 'est.' means a "
            "word-count fallback was used instead of exact tokenization."
        )
