#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingest import mission_channels
from prepare import reading_materials_snapshot, write_json
from train import PRIMARY_METRIC_DIRECTION, PRIMARY_METRIC_KEY, _collect_git_metadata, _mean_primary_metric


FORWARDED_FLAG_BLOCKLIST = {
    "--data-root",
    "--results-root",
    "--splits",
    "--target-channels",
    "--detectors",
    "--experiment-description",
    "--experiment-decision",
    "--experiment-tag",
}

MISSION_RUNS = (
    ("mission1", "ESA-Mission1", "84_months"),
    ("mission2", "ESA-Mission2", "21_months"),
)

CATEGORY_COLORS = {
    "Anomaly": "red",
    "Communication Gap": "green",
    "Rare Event": "blue",
    "Model Trigger": "darkorange",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run full-channel human-oriented training for Mission1 and Mission2, then merge normal train.py outputs."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-root", type=Path, default=Path("results/human"))
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--detectors", nargs="+", default=["tcn"])
    parser.add_argument("--experiment-description", type=str, default="full-channel human training")
    parser.add_argument("--experiment-decision", choices=["candidate", "keep", "discard"], default="candidate")
    parser.add_argument("--experiment-tag", type=str, default="human")
    parser.add_argument("--mission1-split", type=str, default="84_months")
    parser.add_argument("--mission2-split", type=str, default="21_months")
    parser.add_argument("--skip-mission1", action="store_true")
    parser.add_argument("--skip-mission2", action="store_true")
    parser.add_argument(
        "--no-auto-memory-safe",
        action="store_true",
        help="Do not inject conservative full-channel TCN settings.",
    )
    parser.add_argument(
        "--tcn-runtime-profile",
        choices=["auto-safe", "2c9bdc4"],
        default="auto-safe",
        help="Choose wrapper-injected TCN runtime settings for full-channel runs.",
    )
    return parser.parse_known_args()


def validate_forwarded_args(forwarded_args: list[str]) -> None:
    blocked = [arg for arg in forwarded_args if arg in FORWARDED_FLAG_BLOCKLIST]
    if blocked:
        blocked_text = ", ".join(sorted(set(blocked)))
        raise ValueError(f"Do not pass {blocked_text} directly to human_train.py; use the human_train-specific options instead.")


def _has_flag(forwarded_args: list[str], flag: str) -> bool:
    return flag in forwarded_args


def build_2c9bdc4_overrides(forwarded_args: list[str]) -> list[str]:
    overrides: list[str] = []
    if not _has_flag(forwarded_args, "--tcn-dataloader-workers"):
        overrides.extend(["--tcn-dataloader-workers", "8"])
    if not _has_flag(forwarded_args, "--tcn-batch-size"):
        overrides.extend(["--tcn-batch-size", "2048"])
    if not _has_flag(forwarded_args, "--tcn-train-stride"):
        overrides.extend(["--tcn-train-stride", "8"])
    if not _has_flag(forwarded_args, "--tcn-inference-stride"):
        overrides.extend(["--tcn-inference-stride", "16"])
    return overrides


def build_resource_safe_overrides(
    args: argparse.Namespace,
    target_channels: list[str],
    forwarded_args: list[str],
) -> list[str]:
    if args.tcn_runtime_profile == "2c9bdc4":
        return build_2c9bdc4_overrides(forwarded_args)

    if args.no_auto_memory_safe:
        return []

    channel_count = len(target_channels)
    if channel_count <= 16:
        return []

    overrides: list[str] = []
    if not _has_flag(forwarded_args, "--tcn-no-preload"):
        overrides.append("--tcn-no-preload")
    if not _has_flag(forwarded_args, "--tcn-dataloader-workers"):
        overrides.extend(["--tcn-dataloader-workers", "0"])
    if not _has_flag(forwarded_args, "--tcn-batch-size"):
        overrides.extend(["--tcn-batch-size", "256"])
    if not _has_flag(forwarded_args, "--tcn-train-stride"):
        overrides.extend(["--tcn-train-stride", "64"])
    if not _has_flag(forwarded_args, "--tcn-inference-stride"):
        overrides.extend(["--tcn-inference-stride", "32"])
    return overrides


def build_child_command(
    args: argparse.Namespace,
    mission_name: str,
    split: str,
    target_channels: list[str],
    mission_results_root: Path,
    forwarded_args: list[str],
) -> list[str]:
    description = f"{args.experiment_description} | {mission_name} full-channel"
    auto_overrides = build_resource_safe_overrides(args, target_channels, forwarded_args)
    command = [
        args.python_executable,
        str(REPO_ROOT / "train.py"),
        "--data-root",
        str(args.data_root),
        "--results-root",
        str(mission_results_root),
        "--splits",
        split,
        "--detectors",
        *args.detectors,
        "--target-channels",
        *target_channels,
        "--experiment-description",
        description,
        "--experiment-decision",
        args.experiment_decision,
        "--experiment-tag",
        args.experiment_tag,
        *auto_overrides,
        *forwarded_args,
    ]
    return command


def run_eval(results_root: Path, python_executable: str) -> None:
    command = [python_executable, str(REPO_ROOT / "eval.py"), "--results-root", str(results_root)]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def parse_class_number(class_name: str) -> int:
    match = re.fullmatch(r"class_(\d+)", str(class_name))
    if not match:
        return 0
    return int(match.group(1))


def load_label_events(mission_root: Path, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
    labels = pd.read_csv(mission_root / "labels.csv", parse_dates=["StartTime", "EndTime"])
    anomaly_types = pd.read_csv(mission_root / "anomaly_types.csv")
    merged = labels.merge(anomaly_types[["ID", "Class", "Category"]], on="ID", how="left")
    merged["StartTime"] = pd.to_datetime(merged["StartTime"]).dt.tz_localize(None)
    merged["EndTime"] = pd.to_datetime(merged["EndTime"]).dt.tz_localize(None)
    windowed = merged[(merged["EndTime"] >= start_time) & (merged["StartTime"] <= end_time)].copy()
    windowed["class_number"] = windowed["Class"].map(parse_class_number)
    return windowed.sort_values(["class_number", "StartTime", "EndTime", "Channel"])


def read_trigger_intervals(predictions_path: Path, chunksize: int = 200_000) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp]], pd.Timestamp, pd.Timestamp]:
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    active_start: pd.Timestamp | None = None
    previous_timestamp: pd.Timestamp | None = None
    first_timestamp: pd.Timestamp | None = None

    for chunk in pd.read_csv(predictions_path, parse_dates=["timestamp"], chunksize=chunksize):
        timestamps = pd.to_datetime(chunk["timestamp"])
        if len(timestamps) == 0:
            continue
        if first_timestamp is None:
            first_timestamp = pd.Timestamp(timestamps.iloc[0]).to_pydatetime()
            first_timestamp = pd.Timestamp(first_timestamp)
        values = chunk.drop(columns=["timestamp"]).to_numpy(dtype=np.uint8, copy=False)
        any_positive = values.max(axis=1) > 0 if values.size else np.zeros(len(chunk), dtype=bool)
        for timestamp, is_positive in zip(timestamps, any_positive):
            current_timestamp = pd.Timestamp(timestamp).to_pydatetime()
            current_timestamp = pd.Timestamp(current_timestamp)
            if is_positive and active_start is None:
                active_start = current_timestamp
            elif not is_positive and active_start is not None and previous_timestamp is not None:
                intervals.append((active_start, previous_timestamp))
                active_start = None
            previous_timestamp = current_timestamp

    if first_timestamp is None or previous_timestamp is None:
        raise ValueError(f"Prediction file is empty: {predictions_path}")
    if active_start is not None:
        intervals.append((active_start, previous_timestamp))
    return intervals, first_timestamp, previous_timestamp


def plot_mission_timeline(
    ax: plt.Axes,
    mission_label: str,
    mission_root: Path,
    predictions_path: Path,
) -> None:
    trigger_intervals, start_time, end_time = read_trigger_intervals(predictions_path)
    label_events = load_label_events(mission_root, start_time=start_time, end_time=end_time)

    seen_categories: set[str] = set()
    for row in label_events.itertuples(index=False):
        category = str(row.Category)
        color = CATEGORY_COLORS.get(category, "gray")
        label = category if category not in seen_categories else None
        ax.hlines(row.class_number, row.StartTime, row.EndTime, colors=color, linewidth=1.5, alpha=0.9, label=label)
        seen_categories.add(category)

    trigger_label = "Model Trigger"
    for index, (interval_start, interval_end) in enumerate(trigger_intervals):
        ax.hlines(
            0,
            interval_start,
            interval_end,
            colors=CATEGORY_COLORS["Model Trigger"],
            linewidth=2.6,
            alpha=0.95,
            label=trigger_label if index == 0 else None,
        )

    class_numbers = sorted(number for number in label_events["class_number"].unique() if number > 0)
    max_class = max(class_numbers, default=1)
    ax.set_ylim(-0.75, max_class + 0.75)
    ax.set_xlim(start_time, end_time)
    ax.set_title(mission_label)
    ax.set_ylabel("Event class")
    ax.grid(True, alpha=0.35)
    ax.set_yticks([0, *class_numbers])
    ax.set_yticklabels(["trigger", *[str(number) for number in class_numbers]])
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def locate_predictions_path(results_root: Path, split: str, detectors: list[str]) -> Path:
    preferred_detectors = detectors + ["tcn", "std"]
    seen: set[str] = set()
    for detector in preferred_detectors:
        if detector in seen:
            continue
        seen.add(detector)
        memory_path = results_root / f"{detector}_memory" / split / "predictions.csv"
        baseline_path = results_root / f"{detector}_baseline" / split / "predictions.csv"
        if memory_path.exists():
            return memory_path
        if baseline_path.exists():
            return baseline_path
    raise FileNotFoundError(f"Could not find predictions.csv under {results_root} for split {split}")


def generate_timeline_plots(
    results_root: Path,
    completed_runs: list[dict[str, Any]],
    detectors: list[str],
) -> list[Path]:
    plots_dir = results_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(len(completed_runs), 1, figsize=(14, max(5, len(completed_runs) * 4.5)), squeeze=False)
    output_paths: list[Path] = []
    legend_handles = [
        Line2D([0], [0], color=CATEGORY_COLORS["Anomaly"], lw=3, label="Anomaly"),
        Line2D([0], [0], color=CATEGORY_COLORS["Communication Gap"], lw=3, label="Communication Gap"),
        Line2D([0], [0], color=CATEGORY_COLORS["Rare Event"], lw=3, label="Rare Event"),
        Line2D([0], [0], color=CATEGORY_COLORS["Model Trigger"], lw=3, label="Model Trigger"),
    ]

    for row_index, run_info in enumerate(completed_runs):
        axis = axes[row_index][0]
        predictions_path = locate_predictions_path(run_info["results_root"], run_info["split"], detectors)
        plot_mission_timeline(
            ax=axis,
            mission_label=run_info["mission_label"].replace("ESA-", ""),
            mission_root=run_info["mission_root"],
            predictions_path=predictions_path,
        )
        individual_output = plots_dir / f"{run_info['mission_key']}_event_timeline.png"
        axis.figure.tight_layout()
        single_figure = plt.figure(figsize=(14, 4.8))
        single_axis = single_figure.add_subplot(111)
        plot_mission_timeline(
            ax=single_axis,
            mission_label=run_info["mission_label"].replace("ESA-", ""),
            mission_root=run_info["mission_root"],
            predictions_path=predictions_path,
        )
        single_axis.legend(handles=legend_handles, loc="upper right", ncol=4)
        single_axis.set_xlabel("Time")
        single_figure.tight_layout()
        single_figure.savefig(individual_output, dpi=180)
        plt.close(single_figure)
        output_paths.append(individual_output)

    axes[-1][0].set_xlabel("Time")
    axes[0][0].legend(handles=legend_handles, loc="upper right", ncol=4)
    figure.tight_layout()
    combined_output = plots_dir / "event_timelines.png"
    figure.savefig(combined_output, dpi=180)
    plt.close(figure)
    output_paths.insert(0, combined_output)
    return output_paths


def append_experiment_log(results_root: Path, payload: dict[str, Any]) -> None:
    path = results_root / "experiment_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def build_run_payload(
    args: argparse.Namespace,
    forwarded_args: list[str],
    rm_snap: dict[str, Any],
    mean_primary: float | None,
    completed_runs: list[dict[str, Any]],
    run_status: str,
    started_at: str,
    elapsed_seconds: float,
    timeline_plots: list[Path] | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "elapsed_seconds": float(elapsed_seconds),
        "run_status": run_status,
        "description": args.experiment_description,
        "decision": args.experiment_decision,
        "tag": args.experiment_tag,
        "detectors": list(args.detectors),
        "splits": [run["split"] for run in completed_runs],
        "direction": PRIMARY_METRIC_DIRECTION,
        "primary_metric_key": PRIMARY_METRIC_KEY,
        "primary_metric_mean": mean_primary,
        "reading_materials": rm_snap,
        "artifacts": {
            "summary_csv": args.results_root / "summary.csv",
            "reading_materials_snapshot": args.results_root / "reading_materials_snapshot.json",
            "run_summary_json": args.results_root / "run_summary.json",
            "plots_dir": args.results_root / "plots",
            "timeline_plots": timeline_plots or [],
        },
        "children": completed_runs,
        "cli": {
            "argv": sys.argv,
            "python_executable": sys.executable,
            "args": vars(args),
            "forwarded_args": forwarded_args,
        },
        "git": _collect_git_metadata(REPO_ROOT),
    }
    if error is not None:
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    return payload


def main() -> None:
    args, forwarded_args = parse_args()
    validate_forwarded_args(forwarded_args)
    args.results_root.mkdir(parents=True, exist_ok=True)

    run_started_at = datetime.now(timezone.utc).isoformat()
    run_started_perf = time.perf_counter()
    rm_snap = reading_materials_snapshot()
    write_json(args.results_root / "reading_materials_snapshot.json", rm_snap)

    mission_plan: list[tuple[str, str, str]] = []
    for mission_key, mission_label, default_split in MISSION_RUNS:
        if mission_key == "mission1" and args.skip_mission1:
            continue
        if mission_key == "mission2" and args.skip_mission2:
            continue
        split = args.mission1_split if mission_key == "mission1" else args.mission2_split
        mission_plan.append((mission_key, mission_label, split or default_split))

    if not mission_plan:
        raise ValueError("At least one mission must be enabled.")

    completed_runs: list[dict[str, Any]] = []
    timeline_plots: list[Path] = []
    try:
        summary_frames: list[pd.DataFrame] = []
        for mission_key, mission_label, split in mission_plan:
            mission_root = args.data_root / mission_label
            target_channels = mission_channels(mission_root)
            if not target_channels:
                raise RuntimeError(f"No channels found under {mission_root}")

            mission_results_root = args.results_root / mission_key
            mission_results_root.mkdir(parents=True, exist_ok=True)
            child_command = build_child_command(
                args=args,
                mission_name=mission_label,
                split=split,
                target_channels=target_channels,
                mission_results_root=mission_results_root,
                forwarded_args=forwarded_args,
            )
            subprocess.run(child_command, cwd=REPO_ROOT, check=True)
            run_eval(mission_results_root, python_executable=args.python_executable)

            mission_summary = pd.read_csv(mission_results_root / "summary.csv")
            mission_summary.insert(0, "mission", mission_label)
            summary_frames.append(mission_summary)

            mission_run_summary = json.loads((mission_results_root / "run_summary.json").read_text(encoding="utf-8"))
            completed_runs.append(
                {
                    "mission_key": mission_key,
                    "mission_label": mission_label,
                    "mission_root": str(mission_root),
                    "split": split,
                    "results_root": str(mission_results_root),
                    "run_summary_json": str(mission_results_root / "run_summary.json"),
                    "primary_metric_mean": mission_run_summary.get("primary_metric_mean"),
                    "detectors": mission_run_summary.get("detectors", []),
                }
            )

        combined_summary = pd.concat(summary_frames, ignore_index=True)
        combined_summary.to_csv(args.results_root / "summary.csv", index=False)
        run_eval(args.results_root, python_executable=args.python_executable)

        completed_runs_for_payload = [
            {
                **run,
                "mission_root": run["mission_root"],
                "results_root": run["results_root"],
                "run_summary_json": run["run_summary_json"],
            }
            for run in completed_runs
        ]
        completed_runs_for_plot = [
            {
                **run,
                "mission_root": Path(run["mission_root"]),
                "results_root": Path(run["results_root"]),
            }
            for run in completed_runs
        ]
        timeline_plots = generate_timeline_plots(
            results_root=args.results_root,
            completed_runs=completed_runs_for_plot,
            detectors=args.detectors,
        )

        mean_primary = _mean_primary_metric(combined_summary.to_dict(orient="records"))
        payload = build_run_payload(
            args=args,
            forwarded_args=forwarded_args,
            rm_snap=rm_snap,
            mean_primary=mean_primary,
            completed_runs=completed_runs_for_payload,
            run_status="success",
            started_at=run_started_at,
            elapsed_seconds=time.perf_counter() - run_started_perf,
            timeline_plots=timeline_plots,
        )
        write_json(args.results_root / "run_summary.json", payload)
        append_experiment_log(args.results_root, payload)

        if mean_primary is not None:
            print(
                f"primary_f05={mean_primary:.6f} direction={PRIMARY_METRIC_DIRECTION} key={PRIMARY_METRIC_KEY}",
                flush=True,
            )
        print(
            f"reading_materials_count={rm_snap['count']} snapshot={args.results_root / 'reading_materials_snapshot.json'}",
            flush=True,
        )
        print(f"run_status=success run_summary_json={args.results_root / 'run_summary.json'}", flush=True)
    except Exception as exc:
        payload = build_run_payload(
            args=args,
            forwarded_args=forwarded_args,
            rm_snap=rm_snap,
            mean_primary=None,
            completed_runs=completed_runs,
            run_status="crash",
            started_at=run_started_at,
            elapsed_seconds=time.perf_counter() - run_started_perf,
            timeline_plots=timeline_plots,
            error=exc,
        )
        write_json(args.results_root / "run_summary.json", payload)
        append_experiment_log(args.results_root, payload)
        print(f"run_status=crash run_summary_json={args.results_root / 'run_summary.json'}", flush=True)
        raise


if __name__ == "__main__":
    main()
