# autoresearch anomaly benchmark

This repository is designed to support an autonomous experiment loop.

## Setup

To set up a new autonomous research run, work with the user to:

1. Agree on a run tag based on today's date, for example `mar26` or `mar26-gpu0`. The branch `autoresearch/<tag>` must not already exist.
2. Create the branch from the current best base branch:
   `git checkout -b autoresearch/<tag>`
3. Read the in-scope files for full context:
   - `README.md` for repository context and operational notes
   - `prepare.py` for the fixed harness and evaluation logic
   - `train.py` for the model, scoring path, and experiment surface
   - `program_2.md` for the operating instructions in this file
4. Verify data exists under `data/ESA-Mission1/`, `data/ESA-Mission2/`, and `data/preprocessed/`. If the split CSVs are missing, tell the human to run `.venv/bin/python ingest.py`.
5. Verify the local virtual environment exists and use `.venv/bin/python` for all benchmark commands.
6. Initialize `results.tsv` in the repository root if it does not exist, with the header:
   `commit	primary_f05	status	description`
7. Confirm setup looks good, then begin experimentation.

Once setup is confirmed, do not stop for further confirmation. Start the loop.

## In Scope

You may edit:

- `train.py`: this is the primary experiment file and the main place to work
- `reading_materials/*.md`: optional literature notes when useful

You may read:

- `README.md`
- `prepare.py`
- `program_2.md`
- `results/mission1_subset/*`

## Out Of Scope

You must not:

- modify `prepare.py`
- install new packages or add dependencies
- change the evaluation ground truth in `prepare.py`
- broaden the project structure unless the human explicitly asks

## Objective

The goal is to maximize the scalar printed at the end of a successful run as:

`primary_f05=...`

Ground truth:

- primary metric key: `memory.anomaly_only.Anomaly.EW_F_0.50`
- direction: `maximize`
- meaning: anomaly-only event-wise F0.5 after memory gating

Higher is better.

The research goal is to push this metric as high as possible while progressively using less training data. Treat smaller-data success as the main optimization target, not the full-channel longest split.

## Default Experiment Contract

Each experiment runs with the default command:

`.venv/bin/python train.py`

The default benchmark contract is:

- detector scope: `tcn`
- starting split scope: `10_months`
- promotion threshold: once the active split reaches `primary_f05 >= 0.95`, switch to the next smaller split
- current progression ladder: `10_months` -> `9_months` -> `8_months` -> `7_months` -> `6_months` -> `5_months` -> `4_months` -> `3_months` -> `2_months` -> `1_months`
- TCN wall-clock budget: `900` seconds
- fixed seeds and comparable data layout unless the human changes the contract

Split naming:

- The small-data Mission1 search ladder should explicitly use these files under `data/preprocessed/multivariate/ESA-Mission1-semi-supervised/`:
- `1_months.train.csv`
- `2_months.train.csv`
- `3_months.train.csv`
- `3_months.val.csv`
- `4_months.train.csv`
- `5_months.train.csv`
- `6_months.train.csv`
- `7_months.train.csv`
- `8_months.train.csv`
- `9_months.train.csv`
- `10_months.train.csv`
- The intended search policy is to train on `10_months.train.csv` until the threshold is reached, then move to `9_months.train.csv`, then `8_months.train.csv`, and continue downward one month at a time until `1_months.train.csv`.
- `84_months` is the logical Mission1 paper split and resolves to `81_months.train.csv`, `3_months.val.csv`, and `84_months.test.csv`.
- `21_months` is the logical Mission2 paper split and resolves to `18_months.train.csv`, `3_months.val.csv`, and `21_months.test.csv`.

Because the training budget is fixed, you should focus on improving the metric, not on making runs longer.

Do not switch splits based on subjective judgment. Use the threshold rule above. Stay on the current split until the best recorded score on that split is at least `0.95`, then move down exactly one step in the month ladder and continue the search there.

VRAM is a soft constraint. Some increase is acceptable if it leads to meaningful gains, but avoid gratuitous growth.

Simplicity matters. If two approaches perform similarly, prefer the simpler one. Small improvements that add messy complexity are often not worth keeping.

The first run must always be the baseline with the current code as-is.

## Output Contract

Every successful run should produce:

- stdout line `primary_f05=...`
- stdout line `reading_materials_count=...`
- stdout line `run_status=success run_summary_json=...`
- `results/mission1_subset/summary.csv`
- `results/mission1_subset/reading_materials_snapshot.json`
- `results/mission1_subset/run_summary.json`
- `results/mission1_subset/experiment_log.jsonl`

After each successful run, also execute:

`.venv/bin/python eval.py --results-root results/mission1_subset`

This should produce:

- stdout line `eval_best_primary=...`
- stdout line `eval_summary_json=...`
- stdout line `metrics_long_csv=...`
- `results/mission1_subset/compact_summary.csv`
- `results/mission1_subset/compact_summary.json`
- `results/mission1_subset/leaderboard.csv`
- `results/mission1_subset/metrics_long.csv`
- `results/mission1_subset/metrics_long.jsonl`
- `results/mission1_subset/eval_summary.json`

Crash runs should still produce:

- stdout line `run_status=crash run_summary_json=...`
- `results/mission1_subset/run_summary.json`
- append-only `results/mission1_subset/experiment_log.jsonl`

## Logging Results

Record every experiment in `results.tsv` as tab-separated values. Do not commit this file.

The TSV columns are:

`commit	primary_f05	status	description`

Where:

1. `commit`: short git hash for the experiment commit
2. `primary_f05`: achieved metric value, or `0.000000` for crashes
3. `status`: `keep`, `discard`, or `crash`
4. `description`: short description of the experiment idea

Example:

```text
commit	primary_f05	status	description
a1b2c3d	0.165957	keep	fix memory query features
b2c3d4e	0.150926	discard	raise memory threshold to 0.94
c3d4e5f	0.000000	crash	break event scoring path
```

## Search Strategy

Do not reduce the search to hyperparameter tuning only.

You should explore across multiple levels of the system:

- training hyperparameters such as learning rate, dropout, weight decay, stride, horizon, and batch behavior
- score construction and aggregation
- threshold calibration and thresholding behavior
- alert persistence, event formation, and fragmentation control
- memory query and prototype representation
- memory gating logic and suppression criteria
- interactions between the learned detector and the memory stage

When one family of ideas has been tried several times in a row, deliberately switch to a different axis. Do not pigeonhole yourself into only threshold tweaks or only optimizer tweaks.

Prefer experiments motivated by a concrete hypothesis about why the current best run is failing. Use `summary.csv`, `run_summary.json`, `metrics_long.csv`, `leaderboard.csv`, and `results.tsv` to identify the next bottleneck.

When comparing experiments, compare runs against the current active split first. Do not let a worse score on a smaller split send you back to a larger split unless the human explicitly changes the contract.

## The Experiment Loop

The experiment runs on a dedicated branch such as `autoresearch/mar26`.

LOOP FOREVER:

1. Look at the git state and identify the current best commit for the current active split.
2. Tune `train.py` with one focused experimental idea.
3. Commit the change.
4. Run the experiment:
   `.venv/bin/python train.py --experiment-description "<short description>" > run.log 2>&1`
5. If the run succeeds, run:
   `.venv/bin/python eval.py --results-root results/mission1_subset >> run.log 2>&1`
6. Read the results from `run.log`, `results/mission1_subset/run_summary.json`, and `results/mission1_subset/eval_summary.json`.
7. Record the result in `results.tsv`.
8. If `primary_f05` improved on the current active split, keep the commit and continue from there.
9. If `primary_f05` is equal or worse, revert to the previous best commit and continue from there.
10. If the best recorded score for the current active split reaches at least `0.95`, switch the active split to the next smaller split in the progression ladder.
11. Immediately choose the next experiment and continue the loop in the same session.

The idea is simple: if a change works, keep it and advance the branch. If it does not work, discard it and keep searching.

Do not treat a local summary, a new best result, a revert, or a clean worktree as a stopping point. Those are normal loop states. Continue immediately.

## Timeout Policy

Each experiment should fit comfortably within the 15-minute training budget plus overhead. If a run becomes unexpectedly long or appears hung, kill it and treat it as a failure.

## Crash Policy

If `primary_f05` is missing, treat the run as a crash.

When a run crashes:

1. Read `results/mission1_subset/run_summary.json` first.
2. If needed, inspect the tail of `run.log`.
3. If the bug is simple and local, fix it and retry once.
4. If the idea is fundamentally bad or unstable, log `crash`, revert, and move on.

## Reading Materials

The folder `reading_materials/` contains optional literature notes.

- Format: one Markdown file per work with YAML front matter and Markdown body
- At the start of each run, the literature snapshot is written to `reading_materials_snapshot.json`
- The same snapshot is embedded into `experiment_log.jsonl`

Read these materials when you need better ideas. Do not let reading block the experiment loop.

## Never Stop

Once the loop begins, do not pause to ask the human whether to continue. Do not ask whether this is a good stopping point. Keep iterating until the human explicitly interrupts you.

Do not stop after one experiment, a few experiments, a new best result, or a written progress summary. A summary is not an endpoint. After any summary, immediately resume the next experiment unless the human has explicitly told you to stop.

The default behavior is continuation, not handoff.

If the same unfinished candidate is resumed and there is no live train.py/eval.py process and run.log only contains the startup line, treat it as an infrastructure crash after one retry, record it as crash, restore the previous best state, and move on to a different experiment. Do not retry the same stuck candidate indefinitely.

Only stop for one of these reasons:

- the human explicitly says `stop`
- the repository becomes unsafe to modify
- the runtime environment is broken and cannot execute further experiments
- repeated crashes make further autonomous progress impossible

If you run out of ideas:

- re-read `train.py`
- re-read recent results
- inspect failure cases in `metrics_long.csv`
- inspect `results.tsv` to avoid repeating the same family of experiments
- revisit thresholding, calibration, event formation, scoring aggregation, query/prototype representation, and memory gating
- inspect detector-vs-memory tradeoffs instead of only scalar ranking
- combine near-miss ideas
- shift to a different search axis if recent experiments are too similar
- try bolder but still focused changes

The loop is the job.
