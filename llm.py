"""
Cached loading of the two LLMs used by the application.

    SMALL_LLM (~3B-class, config.SMALL_MODEL_NAME)
        - mapping fallback (mapping.get_mappings)
        - memory summarization (memory.summarize_memory_element)

Loaded at most once per Streamlit process via st.cache_resource.
Caller should always fetch the model through load_small_llm() 
never construct a pipeline directly elsewhere in the app,
or it will be reloaded on every rerun/button click.
"""

from __future__ import annotations

import streamlit as st

import config

try:
    from transformers import pipeline
except ImportError:  # pragma: no cover
    pipeline = None


class ModelLoadError(RuntimeError):
    """Raised with a clear, user-facing message when a model fails to load."""


def _build_pipeline(model_name: str, friendly_name: str):
    if pipeline is None:
        raise ModelLoadError(
            "The 'transformers' package is not installed. "
            "Install the requirements with `pip install -r requirements.txt`."
        )
    try:
        return pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            torch_dtype="auto",
        )
    except Exception as e:  # noqa: BLE001 - surfaced to the UI, not swallowed
        raise ModelLoadError(
            f"Failed to load the {friendly_name} model ('{model_name}'). "
            f"Check your internet connection (first run downloads the model), "
            f"available disk space, and that the model name in config.py is correct. "
            f"Underlying error: {e}"
        ) from e


@st.cache_resource(show_spinner=False)
def load_small_llm():
    """Cached small model (~1.5B-class): mapping fallback + summarization."""
    return _build_pipeline(config.SMALL_MODEL_NAME, "small")
