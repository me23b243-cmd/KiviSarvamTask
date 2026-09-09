# RUN.md — Reproduction Guide

Exact steps to set up, run, and evaluate Kivi. Every command below is meant
to be copy-pasted as-is from the project root.

---

## Requirements / dependency setup

- **Python:** 3.10+ recommended (3.9 minimum).
- **OS:** any (CPU-only is fully supported; GPU is used automatically if present).

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

No other runtime is required — everything (app, DB, LLMs) runs in this one
Python environment.

---

## Environment variables

**None are required.** Kivi runs entirely on local, self-hosted Hugging Face
models — there is no API key to configure.

Optional overrides (model names, DB path, generation length) are listed in
[`.env.example`](.env.example) and read by `config.py` via `os.environ`. To
use them: copy `.env.example` to `.env`, edit as needed, and `export`
the values in your shell before running the app (this project does not
auto-load `.env` files — plain `export`/`set` works fine for a local demo).

---

## Database setup commands

The SQLite database (`data/memory.db`) is created automatically, empty, the
first time you run the app or any script below — **no manual creation step
is required.**

To seed it with example learned corrections (recommended for the demo, so
Tab 2 / Tab 3 have something to correct) run:

```bash
python populate_memory.py
```

This reads [`memory_export.json`](memory_export.json) (in the project root)
and writes each entry into `data/memory.db` via the app's own persistence
layer (`memory_store.py` / `database.py`) — the same code path the running
app itself uses, so nothing about the seeded data is special-cased.

There is no separate "migrate" step — the single `memory` table's schema is
created idempotently by `database.init_db()` on every startup and is never
dropped except via `reset_db.py` (see the Reset procedure section below).
Re-running `python populate_memory.py` is safe/idempotent — it upserts by
representative, so it won't create duplicates.

---

## Start commands

```bash
streamlit run app.py
```

(Run `python populate_memory.py` first if you want the demo pre-seeded —
see §13.)

---

## Interface access instructions

Streamlit prints a local URL on startup — by default:

```
http://localhost:8501
```

Open that in a browser. No login/auth is required.

---

## Primary interactions to try

1. **📚 Learn a Correction** — enter an ASR sentence and a preferred sentence
   **with the same number of words** (equal word count triggers the fast,
   deterministic word-alignment path and needs no model download), e.g.:
   - ASR: `I like swiggi`
   - Preferred: `I like swiggy`

   Click **Save Correction**. You'll see which word-level mapping(s) were
   learned and saved. Expand **"View learned memory"** to see everything
   stored so far, including the 5 entries from `memory_export.json` if you
   ran §13's seed step.

2. **✏️ Correct a Sentence** — enter a new sentence containing a word you've
   already taught it (or one from the seed data, e.g. `Please tell suman
   about the meeting.`), click **Reconstruct Preferred Sentence**, and
   confirm the learned correction is applied (`suman` → `Sumanth`).

3. **🧪 Validate System** — see §17 below.

---

## Evaluation command

**Option A — inside the running app:** open the **🧪 Validate System** tab,
confirm `df_combined_test.csv` (already in the project root) is detected,
and click **Validate System**. A progress bar advances row-by-row; metrics
are displayed when it finishes.

**Option B — command line, no browser needed:**

```bash
python evaluate.py
```

or with explicit paths:

```bash
python evaluate.py --input df_combined_test.csv --output df_test_result.csv
```

Both options run the identical pipeline (`validation.py`) and print/display
the same seven metrics: Exact Output Accuracy, Intervention Decision
Accuracy, Intervention Precision/Recall/F1, Unnecessary Intervention Rate,
Miss/Under-intervention Rate.

> Note: `evaluate.py` imports the same `@st.cache_resource`-decorated model
> loaders the app uses (`llm.py`). Running it outside of `streamlit run`
> prints a harmless one-line "missing ScriptRunContext" warning to the
> console from Streamlit — it does not affect the result and can be
> ignored.

---

## Evaluation output location

Both evaluation paths write the full per-row results to:

```
df_test_result.csv     (project root, next to app.py)
```

Override the path with `python evaluate.py --output <path>`, or via the
**Download df_test_result.csv** button shown after validation in the UI.
The seven aggregate metrics themselves are printed to the terminal
(`evaluate.py`) or displayed as a table in the app (Tab 3) — they are not
additionally written to a separate metrics file.

---

## Reset procedure — clear the database and start fresh

```bash
python reset_db.py            # asks for confirmation, then deletes data/memory.db
python reset_db.py --yes      # same, no confirmation prompt (for scripts/CI)
```

This deletes the SQLite file entirely and recreates an empty schema (via
`database.init_db()`) — all learned corrections, summaries, and metrics
history are wiped. Re-run `python populate_memory.py` afterward if you want
the demo re-seeded (§13).

Equivalent manual alternative (no script): just delete the file —
`rm data/memory.db` — it's auto-recreated empty on the next app/script run.

---

## Reference: full command sequence (copy-paste)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python populate_memory.py
streamlit run app.py
# in a second terminal, once dependencies are installed:
python evaluate.py
```
