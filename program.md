# Autonomous Anomaly Benchmark

## Setup

1. Work on a dedicated git branch for this autoresearch run.
2. Read `README.md`, `prepare.py`, `train.py`, and this file before changing anything.
3. Verify data exists under `data/ESA-Mission1/` and `data/preprocessed/`.
4. Use the project virtualenv and invoke Python as `.venv/bin/python`.
5. Do not modify `prepare.py`. Keep changes focused unless the user explicitly asks for broader work.

## In-Scope Files

- `train.py`: primary experiment file. This is the main file to edit.
- `reading_materials/*.md`: optional literature notes for additional context.
- `program.md`: human-authored operating instructions for the agent.

## Do Not Edit

- `prepare.py`: fixed harness for data loading, ESA metrics, memory bank logic, and result I/O.

## Objective

Maximize the scalar printed at the end of each successful run as `primary_f05=...`.

- Primary metric key: `memory.anomaly_only.Anomaly.EW_F_0.50`
- Direction: `maximize`
- Meaning: event-wise anomaly F0.5 after memory gating on the anomaly-only labels

## Default Experiment Contract

- Use the default benchmark command unless there is a strong reason not to:
  `.venv/bin/python train.py`
- The default experiment scope is `--detectors tcn --splits 10_months`.
- The default TCN training budget is fixed at `900` seconds.
- Keep the same split, seed, and data layout for fair comparison unless the human changes the contract.

## Output Contract

Successful runs should leave behind:

- stdout line `primary_f05=...`
- stdout line `reading_materials_count=...`
- stdout line `run_status=success run_summary_json=...`
- `summary.csv`
- `reading_materials_snapshot.json`
- `run_summary.json`
- append-only `experiment_log.jsonl`

Crash runs should still leave:

- stdout line `run_status=crash run_summary_json=...`
- `run_summary.json`
- append-only `experiment_log.jsonl`

## Experiment Loop

1. Establish a baseline by running the current `train.py` once before making changes.
2. Make a focused change in `train.py`.
3. Commit the change.
4. Run the benchmark with a short description, for example:
   `.venv/bin/python train.py --experiment-description "increase hidden dim"`
5. Read `primary_f05` from stdout or `run_summary.json`.
6. Keep the commit only if the result is better or meaningfully simpler at equal quality.
7. Discard regressions and move to the next idea.

## Crash Handling

- Treat missing `primary_f05` as a failed run.
- Read `run_summary.json` first; it should contain the error type and message.
- If the problem is a simple bug, fix it and rerun once.
- If the idea is fundamentally bad or unstable, discard it and move on.

## Reading Materials

- Folder: `reading_materials/`
- Format: one `.md` file per work with YAML front matter and Markdown body
- At the start of every run, the literature snapshot is embedded into `reading_materials_snapshot.json` and `experiment_log.jsonl`

## Dependencies

Use the existing environment from `requirements.txt`. The `timeeval` metrics package is vendored locally from `timeeval/`.
