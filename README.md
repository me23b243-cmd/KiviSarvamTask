## Phonetic Memory based Correction Generic Approach & Architecture Overview

The system has two independent parts: a memory that accumulates corrections over time, and a correction engine that applies that memory to new sentences. They only talk to each other through the stored mappings — nothing else is shared.

## 1) Memory Storage

What it stores: every time a user corrects an ASR sentence into a preferred one, the system diffs the two and records each word-level change as a mapping — original word → preferred word. Each mapping is a durable record with:

- the two words themselves,

- how many times this exact correction has been made (occurrence),

- the actual sentences it was seen in (context — real evidence, not a rule someone wrote by hand).

How a mapping is learned: if the two sentences have the same number of words, the diff is purely positional — word 3 differs from word 3, that's a mapping, done, no model involved. If the word counts differ (e.g. one word becomes two), a small model is asked to align them instead, since position alone can't tell you what maps to what anymore. Though the latter part isn't tested and is less reliable, so I stuck with the 1 to 1 mappings case only.

Repetition strengthens memory, it doesn't duplicate it. Correcting "suman" → "Sumanth" five times doesn't create five entries — it's one entry with occurrence 5 and five pieces of evidence. This means confidence in a mapping grows naturally from real usage, and the system starts distinguishing "seen once, might be a fluke" from "seen repeatedly, this is a real pattern."

Persistence is boring on purpose. Everything above lives in a plain database. It's the single source of truth — nothing is trusted from memory alone unless it's actually been written to disk. Restarting the system loses nothing.

Evidence gets compressed, not discarded. Raw context (a growing list of example sentences) isn't directly handed to the sentence generation directly every time. So each mapping's evidence is periodically distilled into one compact rule/ summary ("X is corrected to Y when it refers to a person's name," etc.) — but only when there's new evidence to fold in, and only for mappings that are actually about to be used. The raw evidence is never deleted; the compact version is a cache derived from it.


## 2) Intelligent Correction

Given a new sentence, correction happens in three stages:

Stage 0 — find candidates. Look at each word in the sentence and check: does memory have any known confusion for this word? This is a cheap, deterministic lookup — no model involved yet. If nothing matches, the sentence is returned untouched; no model is ever called for a sentence with no relevant history.

Stage 1 — propose a correction. If there are candidates, an LLM is shown the sentence, the candidate word-swaps, and the evidence behind each one, and asked to produce a corrected sentence — using only those exact swaps, nothing invented. This is a judgment call, not a lookup: evidence is guidance, not a command, so the model can legitimately decide a candidate doesn't fit this particular sentence and leave the word alone.

Now quite a few tests were run on using the stage 1 alone, but then there were many cases when it wasn't able to judge whether to intervene or not and then was just giving a nonsensical sentence, so a natural idea was to try out if stacking another LLM could improve performance, guess what it did. (99/151 to 113/151 correct, for smaller part of the current test dataset )

Stage 2 — review the proposal. The first pass isn't trusted blindly, because it can overcorrect, undercorrect, or get it half right. A second LLM call re-examines the original sentence, the proposed correction, and the same candidate evidence, and asks: does this actually make sense? If yes, it's accepted as-is. If not, it revises — which might mean undoing an unwarranted swap, applying one that was missed, or fixing only part of it. Critically, this second pass is a reviewer checking one specific proposal, not a second independent guess — it only ever picks from the same allowed candidates the first pass had.

Safety check : Whatever comes out of stages 1–2, it is checked mechanically: same number of words as the original, and every word that changed is one of the explicitly authorized swaps — nothing invented, nothing changed that wasn't on the list. Failing means it's discarded and the system falls back to the safer earlier version.

The decision of using LLM doesn't come right away. I first studied a few materials on the net from which Beam search based correction stood out for me, it used embedding similarity for giving beam scores for a particular sentence combination. Embedding-based similarity could only provide a relatively loose measure of semantic conformity. It could indicate that two sentences were similar in meaning, but it was not sufficiently reliable for determining whether a particular word-level correction was actually appropriate in the given context. A stronger context-aware mechanism was therefore needed. It was also tested on the dataset, giving a poor performance of 53/ 151 right corrections (on the smaller part of the current test dataset of size 251).


## Validation, Metrics and some important declarations -

- 1. The repo consists of a validation file and even the UI has a button clicking which we can already do the validation.

- 2. Now the validation metrics which I am gonna present are obtained by testing the approach on Colab server with LLM having 3B parametres, this was because my laptop is 8gb, 4 core, i3 so it was very difficult to test on it, hence colab was used, the zip file with dataset (for storage and sentence correction tests) and .ipynb file is also attached for validation to confirm that the validation metrics are consistent with the following -

Total Samples : 251

Confusion counts: TP=137, FP=25, FN=36, TN=53

| Exact Output Accuracy | 0.741 |
| --- | --- |
| Intervention Decision Accuracy | 0.757 |
| Intervention Precision | 0.846 |
| Intervention Recall | 0.792 |
| Intervention F1 | 0.818 |
| Unnecessary Intervention Rate | 0.321 |
| Miss / Under-intervention Rate | 0.208 |

Note :- The metrics might differ because the LLM is a bit undeterministic, but yeah wont differ wildly.

The zip file's ipynb can simply be uploaded to colab with the files then uploaded and "run all" gives the desired file.

## Limitations -

- 1. Even after having the 2 staged approach the system still continues to struggle sometimes in deciding whether to intervene or not.

- 2. Currently the system for memory creation and correcting only considers 1 to 1 word (Aditya -> Aaditya and not SarvamAI -> Sarvam AI) mapping case while chances of 1 -> 2 or 2->1 are slim in case of phonetic corrections but there should still be use of provision for such cases.

- 3. If the sentence length grows a lot, LLM might struggle, in that case a separate mechanism for chunking the input before correcting has to be introduced.

- 4. A single correction used 2 LLM calls but when tested on colab the numbers weren't as bad.

- a. Testing over the test set of 251 took roughly 8- 10 minutes for correction decisions on all of them.

- b. Memory summarisation took roughly 5 minutes.

- c. 2 calls per generation, 3B LLM was used from HuggingFace (Qwen)

## Honest Admission of AI Usage -

- 1. The whole architecture of both the systems were designed purely by me.

- 2. Getting to the decision of correction system included going through some literature, where GPT was used for explaining the paper’s approach in simple words.

- 3. GPT was clearly prompted for construction of each and every function in detail and only the syntax generation was on it’s part.

- 4. Claude was used for intelligent prompting the LLM used for summarisation of memory_element’s context and Correction mechanism.

- 5. The streamlit was created using claude but every detail about the function usage and the architecture was told for reliable generation.

## Steamlit Specific Architecture -

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
Alternate_testing_colab.zip Zip file with .ipynb and the csv files for validation
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
  or override them via environment variables `KIVI_SMALL_MODEL`
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
