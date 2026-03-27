# ESA Mission 1 Subset Project

This repo is centered around a few entrypoints plus optional search:

- `ingest.py`
- `prepare.py` — fixed harness (data, ESA metrics, memory bank, **reading-material index** for runs)
- `train.py` — benchmark + TCN training with fixed-budget autoresearch defaults and machine-readable run logs
- `eval.py`
- `optuna_search.py`
- `program.md` — autoresearch-style agent brief; `reading_materials/` — one markdown file per paper (see `reading_materials/README.md`)

The dataset folders under `data/` were kept intact. The large benchmark framework and paper-oriented extras were removed.

## Main Files

- `ingest.py`: builds the six-channel Mission 1 subset CSVs
- `train.py`: default autoresearch entrypoint; writes `summary.csv`, `reading_materials_snapshot.json`, `run_summary.json`, and append-only `experiment_log.jsonl`
- `eval.py`: generates human-readable and machine-readable summaries from `summary.csv`
- `optuna_search.py`: runs Optuna/TPE search over the strongest TCN and memory hyperparameters
- `timeeval/`: vendored TimeEval **metrics** code (lazy imports; do not `pip install timeeval` into the same env)
- `results/mission1_subset/`: generated benchmark outputs

## Expected Data Layout

The raw ESA download should already be present under:

```text
data/
├── ESA-Mission1/
│   ├── labels.csv
│   ├── anomaly_types.csv
│   ├── telecommands.csv
│   ├── channels/
│   └── telecommands/
└── preprocessed/
```

## Environment

Use the local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Preprocess

Generate the project subset:

```bash
.venv/bin/python ingest.py
```

This writes:

```text
data/preprocessed/multivariate/ESA-Mission1-subset-semi-supervised/
├── 3_months.train.csv
├── 10_months.train.csv
└── 84_months.test.csv
```

## Run Benchmark

Run the reduced benchmark:

```bash
.venv/bin/python train.py
```

The default autoresearch run is the TCN on `10_months` with a fixed `900` second TCN training budget. Override the CLI only when you intentionally want a different experiment contract.

To rerun the proven `10_months` TCN finalist config from the search results:

```bash
.venv/bin/python train.py --detectors tcn --splits 10_months --tcn-preset best_10m
```

The random search now scores configurations on both `3_months` and `10_months` by default, instead of promoting `3_months` winners first.

## Run Optuna Search

To run Bayesian optimization with Optuna on the `10_months` split:

```bash
.venv/bin/python optuna_search.py --splits 10_months --num-trials 50
```

This writes:

```text
results/optuna_search/
├── best_params.json
├── leaderboard.csv
└── trial_0000/
```

The default objective is `memory.anomaly_only.Anomaly.EW_F_0.50`, with penalties for very low event precision or recall so the search does not overfit to noisy alerting.

Generate the summary report and plots:

```bash
.venv/bin/python eval.py --results-root results/mission1_subset
```

## Outputs

Results are written to:

```text
results/mission1_subset/
├── tcn_baseline/
├── tcn_memory/
├── summary.csv
├── run_summary.json
├── experiment_log.jsonl
├── compact_summary.csv
├── leaderboard.csv
└── metrics_long.csv
```

`summary.csv` contains the raw detector metrics, while `run_summary.json`, `compact_summary.csv`, `leaderboard.csv`, and `metrics_long.csv` are the quickest machine-friendly artifacts for automated comparison.

## Notes

- The current baseline is the local `std` path, not `Telemanom-ESA`.
- Docker is no longer part of the active project path.
- If you later want a stronger baseline, it is better to add it back deliberately rather than keep the full old framework around.
