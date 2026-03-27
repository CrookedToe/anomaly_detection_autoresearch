#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare import PRIMARY_METRIC_DIRECTION, PRIMARY_METRIC_KEY


REPORT_METRICS = {
    "event_precision": "anomaly_only.Anomaly.EW_precision",
    "event_recall": "anomaly_only.Anomaly.EW_recall",
    "event_f05": "anomaly_only.Anomaly.EW_F_0.50",
    "channel_f05": "anomaly_only.PC_Anomaly.channel_F0.50",
    "timing_total": "anomaly_only.ADTQC_Anomaly.Total",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate human-readable and machine-readable benchmark summaries.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/mission1_subset"),
        help="Benchmark results directory containing summary.csv",
    )
    return parser.parse_args()


def load_summary(results_root: Path) -> pd.DataFrame:
    summary_path = results_root / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find summary file: {summary_path}")
    return pd.read_csv(summary_path)


def metric_column(kind: str, metric_key: str) -> str:
    return f"{kind}.{REPORT_METRICS[metric_key]}"


def _safe_float(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_ready(inner) for inner in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def build_compact_frame(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    has_detector = "detector" in summary.columns

    for _, row in summary.iterrows():
        detector = row["detector"] if has_detector else "std"
        compact: dict[str, float | str | int] = {
            "detector": detector,
            "split": row["split"],
            "memory_size": int(row.get("memory_size", 0)),
            "suppressed_total": int(row.get("suppressed_total", 0)),
            "suppressed_overlapping_anomalies": int(row.get("suppressed_overlapping_anomalies", 0)),
            "suppressed_overlapping_rare_events": int(row.get("suppressed_overlapping_rare_events", 0)),
            "suppressed_nominal_only": int(row.get("suppressed_nominal_only", 0)),
        }
        for metric_key in REPORT_METRICS:
            baseline_value = _safe_float(row.get(metric_column("baseline", metric_key), np.nan))
            memory_value = _safe_float(row.get(metric_column("memory", metric_key), np.nan))
            compact[f"baseline_{metric_key}"] = baseline_value
            compact[f"memory_{metric_key}"] = memory_value
            compact[f"delta_{metric_key}"] = memory_value - baseline_value
        rows.append(compact)

    compact = pd.DataFrame(rows)
    if compact.empty:
        return compact
    return compact.sort_values(["memory_event_f05", "delta_event_f05"], ascending=[False, False]).reset_index(drop=True)


def build_metrics_long_frame(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metadata_columns = {
        "detector",
        "split",
        "memory_size",
        "suppressed_total",
        "suppressed_overlapping_anomalies",
        "suppressed_overlapping_rare_events",
        "suppressed_nominal_only",
    }
    metric_columns = [column for column in summary.columns if column.startswith("baseline.") or column.startswith("memory.")]

    for _, row in summary.iterrows():
        metadata = {column: row[column] for column in metadata_columns if column in summary.columns}
        for column in metric_columns:
            kind, metric_path = column.split(".", 1)
            parts = metric_path.split(".")
            if len(parts) >= 3:
                metric_scope = parts[0]
                metric_entity = parts[1]
                metric_name = ".".join(parts[2:])
            else:
                metric_scope = ""
                metric_entity = ""
                metric_name = metric_path
            rows.append(
                {
                    **metadata,
                    "kind": kind,
                    "metric_path": metric_path,
                    "metric_scope": metric_scope,
                    "metric_entity": metric_entity,
                    "metric_name": metric_name,
                    "value": _safe_float(row[column]),
                }
            )
    return pd.DataFrame(rows)


def build_leaderboard(compact: pd.DataFrame) -> pd.DataFrame:
    if compact.empty:
        return compact.copy()
    leaderboard = compact.copy()
    leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1))
    return leaderboard.sort_values(
        ["memory_event_f05", "delta_event_f05", "memory_event_precision"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _best_row(compact: pd.DataFrame) -> dict[str, Any]:
    if compact.empty:
        return {}
    best = compact.sort_values(
        ["memory_event_f05", "delta_event_f05", "memory_event_precision"],
        ascending=[False, False, False],
    ).iloc[0]
    return _json_ready(best.to_dict())


def write_machine_artifacts(
    results_root: Path,
    compact: pd.DataFrame,
    metrics_long: pd.DataFrame,
    leaderboard: pd.DataFrame,
) -> dict[str, Path]:
    artifact_paths = {
        "compact_csv": results_root / "compact_summary.csv",
        "compact_json": results_root / "compact_summary.json",
        "leaderboard_csv": results_root / "leaderboard.csv",
        "metrics_long_csv": results_root / "metrics_long.csv",
        "metrics_long_jsonl": results_root / "metrics_long.jsonl",
        "eval_summary_json": results_root / "eval_summary.json",
    }

    compact.to_csv(artifact_paths["compact_csv"], index=False)
    artifact_paths["compact_json"].write_text(
        json.dumps(_json_ready(compact.to_dict(orient="records")), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    leaderboard.to_csv(artifact_paths["leaderboard_csv"], index=False)
    metrics_long.to_csv(artifact_paths["metrics_long_csv"], index=False)
    with artifact_paths["metrics_long_jsonl"].open("w", encoding="utf-8") as handle:
        for row in metrics_long.to_dict(orient="records"):
            handle.write(json.dumps(_json_ready(row), sort_keys=True) + "\n")

    eval_summary = {
        "primary_metric_key": PRIMARY_METRIC_KEY,
        "primary_metric_direction": PRIMARY_METRIC_DIRECTION,
        "row_count": int(len(compact)),
        "best_run": _best_row(compact),
        "artifacts": artifact_paths,
    }
    artifact_paths["eval_summary_json"].write_text(
        json.dumps(_json_ready(eval_summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return artifact_paths


def write_markdown_report(results_root: Path, compact: pd.DataFrame) -> None:
    report_lines: list[str] = []
    report_lines.append("# Mission 1 Subset Report")
    report_lines.append("")
    report_lines.append(f"Results directory: `{results_root}`")
    report_lines.append("")

    best_rows = compact.sort_values("memory_event_f05", ascending=False).groupby("split", as_index=False).first()
    report_lines.append("## Quick Takeaways")
    report_lines.append("")
    for row in best_rows.itertuples(index=False):
        report_lines.append(
            f"- Best memory-gated primary score on `{row.split}`: `{row.detector}` with `{row.memory_event_f05:.4f}`."
        )
    if (compact["suppressed_total"] == 0).all():
        report_lines.append("- Memory gating did not suppress any alerts in these runs.")
    else:
        report_lines.append(
            f"- Total suppressed alerts across all rows: `{int(compact['suppressed_total'].sum())}`."
        )
    report_lines.append("")

    report_lines.append("## Metrics")
    report_lines.append("")
    report_lines.append(
        "| Detector | Split | Base EW Precision | Base EW Recall | Base EW F0.5 | Mem EW F0.5 | Delta EW F0.5 | Suppressed |"
    )
    report_lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in compact.itertuples(index=False):
        report_lines.append(
            f"| {row.detector} | {row.split} | {row.baseline_event_precision:.4f} | "
            f"{row.baseline_event_recall:.4f} | {row.baseline_event_f05:.4f} | "
            f"{row.memory_event_f05:.4f} | {row.delta_event_f05:.4f} | {row.suppressed_total} |"
        )
    report_lines.append("")

    report_lines.append("## Machine Artifacts")
    report_lines.append("")
    report_lines.append("- `compact_summary.csv`: row-wise benchmark snapshot for agents.")
    report_lines.append("- `compact_summary.json`: JSON version of the compact snapshot.")
    report_lines.append("- `leaderboard.csv`: primary-metric ranking of all rows.")
    report_lines.append("- `metrics_long.csv`: long-form export of every baseline/memory metric in `summary.csv`.")
    report_lines.append("- `metrics_long.jsonl`: JSONL version of the long-form metric export.")
    report_lines.append("- `eval_summary.json`: top-level summary with the best run and artifact paths.")
    report_lines.append("")

    report_lines.append("## Interpretation Guide")
    report_lines.append("")
    report_lines.append("- `EW Precision`: fewer false alerts is better.")
    report_lines.append("- `EW Recall`: more true anomalies found is better.")
    report_lines.append("- `EW F0.5`: precision-weighted event score.")
    report_lines.append("- `Channel F0.5`: whether the right channels are being flagged.")
    report_lines.append("- `ADTQC Total`: whether detections happen at a useful time.")
    report_lines.append("- `Suppressed`: how many baseline alerts the memory bank removed.")
    report_lines.append("")

    (results_root / "report.md").write_text("\n".join(report_lines), encoding="utf-8")


def _plot_grouped_bars(
    compact: pd.DataFrame,
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    labels = [f"{row.detector}\n{row.split}" for row in compact.itertuples(index=False)]
    baseline_values = compact[f"baseline_{metric_key}"].to_numpy(dtype=float)
    memory_values = compact[f"memory_{metric_key}"].to_numpy(dtype=float)

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 5))
    ax.bar(x - width / 2, baseline_values, width=width, label="baseline")
    ax.bar(x + width / 2, memory_values, width=width, label="memory")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_suppression_counts(compact: pd.DataFrame, output_path: Path) -> None:
    labels = [f"{row.detector}\n{row.split}" for row in compact.itertuples(index=False)]
    total = compact["suppressed_total"].to_numpy(dtype=float)
    nominal = compact["suppressed_nominal_only"].to_numpy(dtype=float)
    anomalies = compact["suppressed_overlapping_anomalies"].to_numpy(dtype=float)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 5))
    ax.bar(x, total, label="suppressed_total")
    ax.bar(x, nominal, label="nominal_only")
    ax.bar(x, anomalies, label="overlapping_anomalies")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title("Memory Suppression Counts")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_plots(results_root: Path, compact: pd.DataFrame) -> None:
    plots_dir = results_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _plot_grouped_bars(
        compact,
        metric_key="event_f05",
        ylabel="Score",
        title="Event F0.5: Baseline vs Memory",
        output_path=plots_dir / "event_f05.png",
    )
    _plot_grouped_bars(
        compact,
        metric_key="event_precision",
        ylabel="Score",
        title="Event Precision: Baseline vs Memory",
        output_path=plots_dir / "event_precision.png",
    )
    _plot_grouped_bars(
        compact,
        metric_key="timing_total",
        ylabel="Score",
        title="Timing Quality (ADTQC Total): Baseline vs Memory",
        output_path=plots_dir / "timing_total.png",
    )
    _plot_suppression_counts(compact, plots_dir / "suppression_counts.png")


def main() -> None:
    args = parse_args()
    summary = load_summary(args.results_root)
    compact = build_compact_frame(summary)
    metrics_long = build_metrics_long_frame(summary)
    leaderboard = build_leaderboard(compact)
    artifact_paths = write_machine_artifacts(args.results_root, compact, metrics_long, leaderboard)
    write_markdown_report(args.results_root, compact)
    generate_plots(args.results_root, compact)

    best_run = _best_row(compact)
    if best_run:
        print(
            f"eval_best_primary={best_run['memory_event_f05']:.6f} "
            f"direction={PRIMARY_METRIC_DIRECTION} "
            f"key={PRIMARY_METRIC_KEY} "
            f"detector={best_run['detector']} "
            f"split={best_run['split']}",
            flush=True,
        )
    print(f"eval_summary_json={artifact_paths['eval_summary_json']}", flush=True)
    print(f"metrics_long_csv={artifact_paths['metrics_long_csv']}", flush=True)


if __name__ == "__main__":
    main()
