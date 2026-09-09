"""
Batch validation against a labeled test set.

Mirrors this Colab logic exactly (same column derivations), just wired
through the app's own memory (SQLite-backed) and cached big model instead
of Colab globals:

    df_test["system_answer"] = df_test["asr_output"].apply(
        lambda sentence: semantic_review_correction(sentence, memory, llm_big)
    )
    df_test["is_same"] = ...
    df_test["mismatched_words"] = ...
    df_test["total_mismatched_words"] = ...
    df_test["mismatch_count"] = ...

`semantic_review_correction` in the Colab snippet is this app's
`reconstruct_preferred_sentence`. One extra column, `system_action`, is
derived (system intervened iff system_answer != asr_output) since it's
needed to compare against `expected_action` for the intervention metrics.

Required input columns: asr_output, expected_output, expected_action
(values "intervene" / "do_nothing").
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

import pandas as pd

import config
import memory_store
import metrics as metrics_module
from memory import give_elements, reconstruct_preferred_sentence, summarize_memory_element

REQUIRED_COLUMNS = {"asr_output", "expected_output", "expected_action"}
RESULTS_FILENAME = "df_test_result.csv"

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEST_CSV_PATH = os.path.join(_PROJECT_ROOT, "df_combined_test.csv")


def results_path() -> str:
    return os.path.join(_PROJECT_ROOT, RESULTS_FILENAME)


def load_test_csv(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Test CSV is missing required column(s): {sorted(missing)}")
    return df


def refresh_stale_summaries(df: pd.DataFrame, memory: dict, small_llm) -> int:
    """
    One pass over the WHOLE test set up front: find every distinct memory
    representative referenced by any row, and summarize (SMALL_LLM) only
    the ones whose context is stale. Mirrors Tab 2's own staleness trigger,
    just applied once for the whole set instead of once per row, so the
    same representative is never re-summarized multiple times during a run.

    Returns the number of representatives that were (re)summarized.
    """
    all_reps: set[str] = set()
    for sentence in df["asr_output"].astype(str):
        mapping_dict = give_elements(sentence, memory)
        for reps in mapping_dict.values():
            all_reps.update(reps)

    stale = [
        rep
        for rep in all_reps
        if rep in memory
        and memory_store.is_summary_stale(
            rep, len(memory[rep].context) if isinstance(memory[rep].context, list) else 0
        )
    ]

    for rep in stale:
        element = memory[rep]
        raw_len = len(element.context) if isinstance(element.context, list) else 0
        input_text = "\n".join(element.context) if isinstance(element.context, list) else str(element.context)
        try:
            with metrics_module.timed_call(
                "summarize", tokenizer=getattr(small_llm, "tokenizer", None), input_text=input_text
            ) as m:
                summarize_memory_element(element, small_llm)
                m["output_text"] = element.context
            memory_store.persist_summary(rep, element.context, raw_len)
        except Exception:
            # Leave this element's context as-is (raw list); reconstruction
            # still works, just without a cached compact rule for it.
            pass

    return len(stale)


def run_validation(
    df: pd.DataFrame,
    memory: dict,
    big_llm,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """
    Runs reconstruct_preferred_sentence over every row's asr_output.
    Returns a NEW dataframe (input df is not mutated) with the columns:
        system_answer, system_action, is_same,
        mismatched_words, total_mismatched_words, mismatch_count
    """
    total = len(df)
    system_answers = []

    for i, asr_sentence in enumerate(df["asr_output"].astype(str)):
        start = time.perf_counter()
        try:
            answer = reconstruct_preferred_sentence(asr_sentence, memory, big_llm)
            success = True
        except Exception:
            # Same conservative contract as the rest of the app: never let a
            # broken/failed reconstruction propagate — fall back to the
            # original sentence and keep going.
            answer = asr_sentence
            success = False
        duration = time.perf_counter() - start

        metrics_module.record_llm_call(
            "reconstruct",
            duration,
            input_text=asr_sentence,
            output_text=answer,
            tokenizer=getattr(big_llm, "tokenizer", None),
            success=success,
        )

        system_answers.append(answer)
        if progress_callback is not None:
            progress_callback(i + 1, total)

    df_result = df.copy()
    df_result["system_answer"] = system_answers

    df_result["is_same"] = [
        1 if a == b else 0
        for a, b in zip(df_result["expected_output"], df_result["system_answer"])
    ]

    df_result["mismatched_words"] = [
        [word for word in system.split() if word not in preferred.split()]
        for preferred, system in zip(df_result["expected_output"], df_result["system_answer"])
    ]

    df_result["total_mismatched_words"] = [
        [word for word in system.split() if word not in preferred.split()]
        for preferred, system in zip(df_result["expected_output"], df_result["asr_output"])
    ]

    df_result["mismatch_count"] = [len(word_list) for word_list in df_result["mismatched_words"]]

    # Needed to compare against expected_action for the intervention metrics.
    df_result["system_action"] = [
        "do_nothing" if system == asr else "intervene"
        for asr, system in zip(df_result["asr_output"], df_result["system_answer"])
    ]

    return df_result


def compute_metrics(df_result: pd.DataFrame) -> dict:
    """
    'intervene' = positive class, 'do_nothing' = negative class.
    """
    expected = df_result["expected_action"]
    predicted = df_result["system_action"]
    n = len(df_result)

    exact_output_accuracy = float(df_result["is_same"].mean()) if n else 0.0

    intervention_decision_accuracy = float((expected == predicted).mean()) if n else 0.0

    tp = int(((predicted == "intervene") & (expected == "intervene")).sum())
    fp = int(((predicted == "intervene") & (expected == "do_nothing")).sum())
    fn = int(((predicted == "do_nothing") & (expected == "intervene")).sum())
    tn = int(((predicted == "do_nothing") & (expected == "do_nothing")).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    unnecessary_intervention_rate = fp / (fp + tn) if (fp + tn) else 0.0  # false positive rate
    miss_rate = fn / (tp + fn) if (tp + fn) else 0.0  # false negative rate = 1 - recall

    return {
        "Exact Output Accuracy": exact_output_accuracy,
        "Intervention Decision Accuracy": intervention_decision_accuracy,
        "Intervention Precision": precision,
        "Intervention Recall": recall,
        "Intervention F1": f1,
        "Unnecessary Intervention Rate": unnecessary_intervention_rate,
        "Miss / Under-intervention Rate": miss_rate,
        "confusion_counts": {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "n": n},
    }


def save_results(df_result: pd.DataFrame, path: Optional[str] = None) -> str:
    out_path = path or results_path()
    df_result.to_csv(out_path, index=False)
    return out_path
