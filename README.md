# Kivi — Adaptive Sentence Correction

A Streamlit app that learns word-level ASR corrections from examples you give it,
stores them persistently in SQLite, and later applies the learned corrections to
new sentences.

## Architecture

```
SQLite (memory table, persistent source of truth)
        ↕  memory_store.py  (forward/reverse mapping layer)
memory: dict[str, MemoryElement]  (temporary, in-process)
        ↕
mapping.py / memory.py  (existing core logic, preserved from the notebook)
        ↕
app.py  (Streamlit UI)
```

- **SQLite is always the source of truth.** The `memory` dict is rebuilt from
  the database at the start of every relevant action and never trusted across
  reruns.
- **LLM loaded once**, via `st.cache_resource` in `llm.py`:
  - `SMALL_LLM` (default `Qwen/Qwen2.5-3B-Instruct`) — mapping fallback + memory summarization + sentence reconstruction.

## File structure

```
app.py            Streamlit UI only
database.py       SQLite connection/schema/query/persistence
memory_store.py   Forward/reverse mapping layer: SQLite rows <-> MemoryElement
mapping.py        get_mappings + deterministic/LLM-fallback mapping logic
memory.py         MemoryElement, edit_memory, give_elements, summarize_memory_element,
                   reconstruct_preferred_sentence
llm.py            Cached loading of the two LLMs
config.py         Model names, DB path, generation limits (single config location)
requirements.txt  Dependencies
tests/            Non-LLM-dependent tests (LLM calls are stubbed/mocked)
data/             SQLite database file lives here (memory.db), created on first run
memory_export.json consists of all the memory elements needed to populate the database
populate_memory.py file for populating with initial data (for testing it is required)
reset_db.py        used for resetting the database by deleting everything stored
validation.py      responsible for validating against the test csv
results_df_test.csv the results expected after validation are in this file, which was obtained by testing the same approach
                    on colab
metrics.py           is there to give the latency etc. displayed on Front End
```

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
streamlit run app.py
```

The app opens in your browser. The database and its containing `data/`
directory are created automatically on first run if they don't exist —
you don't need to do anything manually.

## Models

Model is downloaded from the Hugging Face Hub the first time it's
needed (not at app startup) and cached locally by `transformers`/`huggingface_hub`
afterward. First use of each tab may therefore take a while and requires an
internet connection the first time.

- Change the model names in **`config.py`** (`SMALL_MODEL_NAME`, `BIG_MODEL_NAME`),
  or override them via environment variables `KIVI_SMALL_MODEL` / `KIVI_BIG_MODEL`
  without touching code.
- `device_map="auto"` is used, so that it can run on GPU if available and fall
  back to CPU automatically. The  model (~3B) will be considerably slower on CPU.

## Database

- SQLite file: `data/memory.db` (path configurable via `config.DB_PATH` /
  `KIVI_DB_PATH`).
- Table `memory`: one row per `"<original> : <preferred>"` mapping, with
  `occurrences`, the full raw `context` history (JSON list of preferred
  sentences), and a lazily-computed `summary` (+ the context length it covers,
  used to detect when it's gone stale).
- Existing data survives app restarts; the table is only ever created if
  missing, never dropped.

## Basic workflow

1. **Learn a Correction** tab: enter an ASR sentence and the sentence you meant.
   Kivi diffs them (deterministically if word counts match, via the small LLM
   otherwise), and remembers each word-level change. Repeating the same
   correction increases its `occurrence` count and appends another example to
   its context rather than creating a duplicate entry.
2. **Correct a Sentence** tab: enter a new sentence. Kivi looks up which of its
   learned corrections apply, refreshes their summarized "rule" only if new
   examples were learned since it was last computed, and asks the big model to
   decide which substitutions actually fit — falling back to your original
   sentence whenever the model's answer can't be strictly validated (wrong
   word count, an unlisted substitution, altered punctuation, etc.).
3. Expand **"View learned memory"** at any time to see everything stored so far.

## Troubleshooting

- **"Failed to load the small/large model"** — usually a missing internet
  connection on first run, insufficient disk space, or a typo'd model name in
  `config.py`. The error message names which model failed.
- **Reconstruction always returns the original sentence** — this is the
  intended conservative fallback whenever the model's output can't be
  strictly validated against the learned candidates. Check "View learned
  memory" to confirm the correction was actually learned first.
- **`transformers`/`torch` import errors** — re-run
  `pip install -r requirements.txt` inside your virtual environment.
- **Slow on first correction** — the relevant model is being downloaded;
  subsequent runs reuse the local cache.

## Validation

- Just need to run the app and then click the validate button it will automatically
  start validating against the df_combined_test.csv

