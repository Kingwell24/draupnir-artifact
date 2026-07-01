# Draupnir Artifact

This artifact supports the syzbot evaluation in the paper. It provides the
released syzbot crash dataset, the crash-card and cause-card generation code,
and the embedding-based clustering pipeline.

## Layout

- `data/`: syzbot benchmark data and crash-to-label mapping.
  - `README.md`: dataset layout and naming notes.
  - `ground_truth.csv`: mapping from each benchmark crash to its final
    root-cause label.
  - `crashes_by_bug/`: raw syzbot crash artifacts grouped by syzbot bug id.
    This directory is distributed as a separate archive linked from HotCRP.
- `scripts/`: command-line entry points for reproducing the pipeline.
  - `generate_crash_cards.py`: raw crash artifacts to representative crash
    cards.
  - `generate_cause_cards.py`: representative crash cards to source-grounded
    cause cards through an LLM and local Linux source-search harness.
  - `cluster_cause_cards.py`: cause cards to six-field embeddings and final
    duplicate clusters.
- `src/draupnir/`: implementation modules.
  - `crash_cards/`: syzbot crash-card parser and representative selection.
  - `cause_cards/`: LLM harness, prompt, and Linux source-search backend.
  - `clustering/`: cause-card feature extraction, embedding, metrics, and
    clustering outputs.

Run the commands below from the repository root. Linux source code is not
included; Stage 2 needs a local Linux checkout path, passed with `--linux-src`.

## Setup

The scripts use Python 3.10+ syntax.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Dataset Archive

The code repository includes `data/ground_truth.csv` and `data/README.md`.
The raw syzbot crash artifacts are distributed as a separate archive:

```text
https://drive.google.com/file/d/1MSvdNj_8zBPOZTWnvo5NyCqcTEK9rNsJ/view
```

Download `draupnir-syzbot-crashes.tar.gz` and unpack it at the repository root:

```powershell
tar -xzf draupnir-syzbot-crashes.tar.gz
```

After extraction, the dataset layout should contain:

```text
data/crashes_by_bug/
data/ground_truth.csv
```

## Pipeline

Generate crash cards from the released syzbot crash artifacts and select
representatives. This stage is deterministic and does not require model APIs:

```powershell
python scripts\generate_crash_cards.py `
  --input data\crashes_by_bug `
  --ground-truth data\ground_truth.csv `
  --out outputs\stage1_crash_cards `
  --resume
```

Generate cause cards. Provide a local Linux source tree and an OpenAI-compatible
chat endpoint. The source-search backend resolves file paths relative to the
checkout supplied by `--linux-src`. API credentials can be supplied through
`OPENAI_API_KEY`/`OPENAI_BASE_URL`, a `.env` file, or an `--api-file` containing
a `/v1` base URL and `Authorization: Bearer ...` token:

```powershell
$env:OPENAI_MODEL = "<chat model>"

python scripts\generate_cause_cards.py `
  --cards-file outputs\stage1_crash_cards\cards_to_process.txt `
  --linux-src external\linux `
  --api-file path\to\api.txt `
  --out outputs\stage2_cause_cards
```

Embed generated cause cards and cluster them. This stage assumes Stage 2 has
produced cause-card JSON files under `outputs\stage2_cause_cards`:

```powershell
python scripts\cluster_cause_cards.py `
  --cause-cards outputs\stage2_cause_cards `
  --representative-weights outputs\stage1_crash_cards\representative_weights.csv `
  --api-file path\to\api.txt `
  --out outputs\stage3_clustering
```

If `--cached-vectors` is omitted, Stage 3 uses `--api-file` or
`OPENAI_API_KEY`/`OPENAI_BASE_URL` to call the embedding model specified by
`--model`; the default embedding model is `text-embedding-3-large`.

## Smoke Tests

```powershell
python scripts\generate_crash_cards.py --limit-bugs 1 --out outputs\smoke_stage1
python scripts\generate_cause_cards.py --cards-file outputs\smoke_stage1\cards_to_process.txt --limit 2 --dry-run --out outputs\smoke_stage2
```

## Field Names

The paper describes six cause-card fields as object, invariant, propagation,
patch, core context, and auxiliary context. In the implementation, the
intermediate cause-card representation uses `must_match_tokens` and
`should_match_tokens`. During feature extraction, tokens already assigned to
the four typed fields are removed from these context fields; the remaining
`must` field corresponds to core context, and the remaining `should` field
corresponds to auxiliary context.
