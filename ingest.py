#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.parser import parse as parse_date


VALIDATION_MONTHS = 3


@dataclass(frozen=True)
class MissionConfig:
    name: str
    half_months: int
    quick_subset_months: int

    @property
    def train_months(self) -> int:
        return self.half_months - VALIDATION_MONTHS


MISSION_CONFIGS = (
    MissionConfig(name="ESA-Mission1", half_months=84, quick_subset_months=10),
    MissionConfig(name="ESA-Mission2", half_months=21, quick_subset_months=10),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess the ESA missions using the paper splits.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory containing the raw ESA mission folders and the preprocessed output folder.",
    )
    parser.add_argument(
        "--missions",
        nargs="+",
        choices=[config.name for config in MISSION_CONFIGS],
        default=[config.name for config in MISSION_CONFIGS],
        help="Mission folders to preprocess.",
    )
    parser.add_argument(
        "--resample-seconds",
        type=int,
        default=30,
        help="Resampling frequency in seconds.",
    )
    parser.add_argument(
        "--rebuild-from-raw",
        action="store_true",
        help="Ignore existing preprocessed full-train/full-test files and rebuild the mission outputs from raw channel archives.",
    )
    return parser.parse_args()


def label_value(category: str) -> np.uint8:
    if category == "Anomaly":
        return np.uint8(1)
    if category == "Rare Event":
        return np.uint8(2)
    if category == "Communication Gap":
        return np.uint8(3)
    return np.uint8(0)


def build_channel_labels(labels_df: pd.DataFrame, anomaly_types_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    label_lookup = anomaly_types_df.assign(encoded=anomaly_types_df["Category"].map(label_value)).set_index("ID")["encoded"]
    annotated = labels_df.copy()
    annotated["label"] = annotated["ID"].map(label_lookup).fillna(0).astype(np.uint8)
    grouped: dict[str, pd.DataFrame] = {}
    for channel, frame in annotated.groupby("Channel", sort=False):
        grouped[channel] = frame[["StartTime", "EndTime", "label"]].sort_values(["StartTime", "EndTime"]).reset_index(drop=True)
    return grouped


def load_channel_frame(
    source_folder: Path,
    channel: str,
    channel_labels: pd.DataFrame | None,
    resampling_rule: pd.Timedelta,
) -> pd.DataFrame:
    channel_df = pd.read_pickle(source_folder / "channels" / f"{channel}.zip")
    channel_df = channel_df.rename(columns={channel: "value"})
    labels = np.zeros(len(channel_df), dtype=np.uint8)
    if channel_labels is not None and not channel_labels.empty:
        raw_index_ns = channel_df.index.view("i8")
        for row in channel_labels.itertuples(index=False):
            left = raw_index_ns.searchsorted(pd.Timestamp(row.StartTime).value, side="left")
            right = raw_index_ns.searchsorted(pd.Timestamp(row.EndTime).value, side="right")
            if left < right:
                labels[left:right] = row.label
    channel_df["label"] = labels

    first_index_resampled = pd.Timestamp(channel_df.index[0]).floor(freq=resampling_rule)
    last_index_resampled = pd.Timestamp(channel_df.index[-1]).ceil(freq=resampling_rule)
    resampled_range = pd.date_range(first_index_resampled, last_index_resampled, freq=resampling_rule)
    resampled = channel_df.reindex(resampled_range, method="ffill")
    resampled.iloc[0] = channel_df.iloc[0]

    annotated = channel_df.loc[channel_df["label"].isin([1, 2]), ["value", "label"]]
    if not annotated.empty:
        annotated = annotated.copy()
        annotated.index = annotated.index.floor(freq=resampling_rule) + resampling_rule
        annotated = annotated[~annotated.index.duplicated(keep="last")]
        annotated = annotated[annotated.index.isin(resampled.index)]
        if not annotated.empty:
            resampled.loc[annotated.index, ["value", "label"]] = annotated.to_numpy()

    return resampled.sort_index()


def mission_channels(source_folder: Path) -> list[str]:
    return sorted(path.stem for path in (source_folder / "channels").glob("channel_*.zip"))


def build_dataset(
    source_folder: Path,
    channel_labels_map: dict[str, pd.DataFrame],
    target_channels: list[str] | None,
    resampling_rule: pd.Timedelta,
) -> pd.DataFrame:
    selected_channels = mission_channels(source_folder) if target_channels is None else target_channels
    channels: list[pd.DataFrame] = []

    for channel in selected_channels:
        channel_df = load_channel_frame(source_folder, channel, channel_labels_map.get(channel), resampling_rule)
        if channel_df.empty:
            continue

        renamed = channel_df.rename(columns={"value": channel, "label": f"is_anomaly_{channel}"})
        channels.append(renamed)

    if not channels:
        raise RuntimeError("No channel data was prepared for the requested split.")

    dataset = pd.concat(channels, axis=1, sort=True).sort_index()
    full_index = pd.date_range(dataset.index[0], dataset.index[-1], freq=resampling_rule)
    dataset = dataset.reindex(full_index)
    value_columns = selected_channels
    label_columns = [f"is_anomaly_{channel}" for channel in selected_channels]
    dataset[value_columns] = dataset[value_columns].ffill().bfill()
    dataset[label_columns] = dataset[label_columns].ffill().bfill().fillna(0).astype(np.uint8)

    return dataset.sort_index()


def split_boundaries(
    index: pd.DatetimeIndex,
    half_months: int,
    validation_months: int = VALIDATION_MONTHS,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    if len(index) == 0:
        raise RuntimeError("Cannot infer dataset splits from an empty dataset.")
    if half_months <= validation_months:
        raise RuntimeError(
            f"Training half spans only {half_months} months, which is not enough to reserve {validation_months} months for validation."
        )

    mission_start = pd.Timestamp(index.min()).to_period("M").to_timestamp()
    train_end = mission_start + pd.DateOffset(months=half_months)
    val_start = train_end - pd.DateOffset(months=validation_months)
    mission_end = mission_start + pd.DateOffset(months=2 * half_months)
    return val_start, train_end, mission_end


def finalize_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    finalized = dataset.copy()
    finalized.insert(0, "timestamp", finalized.index.strftime("%Y-%m-%d %H:%M:%S"))
    return finalized.reset_index(drop=True)


def load_labels(input_path: Path) -> pd.DataFrame:
    labels_df = pd.read_csv(input_path / "labels.csv")
    for column in ("StartTime", "EndTime"):
        labels_df[column] = labels_df[column].map(lambda value: parse_date(value, ignoretz=True))
    return labels_df


def write_split(output_root: Path, split_name: str, dataset: pd.DataFrame) -> None:
    finalize_dataset(dataset).to_csv(output_root / split_name, index=False, lineterminator="\n")


def mission_output_path(data_root: Path, mission: MissionConfig) -> Path:
    return data_root / "preprocessed" / "multivariate" / f"{mission.name}-semi-supervised"


def load_preprocessed_frame(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, parse_dates=["timestamp"])
    return frame.set_index("timestamp").sort_index()


def has_nonempty_csv(csv_path: Path) -> bool:
    return csv_path.exists() and csv_path.stat().st_size > 0


def move_if_needed(source: Path, target: Path) -> None:
    if target.exists() or not source.exists():
        return
    source.replace(target)


def migrate_legacy_split_names(output_path: Path, mission: MissionConfig) -> None:
    move_if_needed(output_path / f"{mission.half_months}_months.train.csv", output_path / f"{mission.train_months}_months.train.csv")
    move_if_needed(output_path / f"{mission.half_months}_months.val.csv", output_path / f"{VALIDATION_MONTHS}_months.val.csv")


def derive_and_write_auxiliary_splits(output_path: Path, mission: MissionConfig) -> bool:
    migrate_legacy_split_names(output_path, mission)

    full_train_path = output_path / f"{mission.train_months}_months.train.csv"
    full_val_path = output_path / f"{VALIDATION_MONTHS}_months.val.csv"
    full_test_path = output_path / f"{mission.half_months}_months.test.csv"
    quick_subset_path = output_path / f"{mission.quick_subset_months}_months.train.csv"
    if not has_nonempty_csv(full_train_path) or not has_nonempty_csv(full_test_path):
        return False
    if has_nonempty_csv(full_val_path) and has_nonempty_csv(quick_subset_path):
        return True

    full_train_df = load_preprocessed_frame(full_train_path)
    mission_start = pd.Timestamp(full_train_df.index.min()).to_period("M").to_timestamp()
    quick_subset_end = mission_start + pd.DateOffset(months=mission.quick_subset_months)

    quick_subset_df = full_train_df[full_train_df.index < quick_subset_end].copy()
    if quick_subset_df.empty:
        raise RuntimeError(f"Could not derive quick-subset rows for mission '{mission.name}'.")

    if not has_nonempty_csv(full_val_path):
        raise RuntimeError(f"Missing validation split for mission '{mission.name}': {full_val_path}")

    if not has_nonempty_csv(quick_subset_path):
        write_split(output_path, f"{mission.quick_subset_months}_months.train.csv", quick_subset_df)
    return True


def preprocess_mission(
    data_root: Path,
    mission: MissionConfig,
    resampling_rule: pd.Timedelta,
    rebuild_from_raw: bool,
) -> None:
    output_path = mission_output_path(data_root, mission)
    output_path.mkdir(parents=True, exist_ok=True)

    if not rebuild_from_raw and resampling_rule == pd.Timedelta(seconds=30) and derive_and_write_auxiliary_splits(output_path, mission):
        return

    input_path = data_root / mission.name
    labels_df = load_labels(input_path)
    anomaly_types_df = pd.read_csv(input_path / "anomaly_types.csv")
    channel_labels_map = build_channel_labels(labels_df, anomaly_types_df)
    dataset = build_dataset(
        source_folder=input_path,
        channel_labels_map=channel_labels_map,
        target_channels=None,
        resampling_rule=resampling_rule,
    )

    val_start, test_start, mission_end = split_boundaries(dataset.index, half_months=mission.half_months)
    bounded = dataset[(dataset.index >= dataset.index.min()) & (dataset.index < mission_end)].copy()
    train_df = bounded[bounded.index < val_start].copy()
    val_df = bounded[(bounded.index >= val_start) & (bounded.index < test_start)].copy()
    test_df = bounded[(bounded.index >= test_start) & (bounded.index < mission_end)].copy()
    quick_subset_end = pd.Timestamp(bounded.index.min()).to_period("M").to_timestamp() + pd.DateOffset(
        months=mission.quick_subset_months
    )
    quick_subset_df = train_df[train_df.index < quick_subset_end].copy()

    if train_df.empty or val_df.empty or test_df.empty or quick_subset_df.empty:
        raise RuntimeError("The inferred train/validation/test split produced an empty partition.")

    full_split_prefix = f"{mission.half_months}_months"
    train_split_prefix = f"{mission.train_months}_months"
    write_split(output_path, f"{train_split_prefix}.train.csv", train_df)
    write_split(output_path, f"{VALIDATION_MONTHS}_months.val.csv", val_df)
    write_split(output_path, f"{full_split_prefix}.test.csv", test_df)
    write_split(output_path, f"{mission.quick_subset_months}_months.train.csv", quick_subset_df)


def main() -> None:
    args = parse_args()
    resampling_rule = pd.Timedelta(seconds=args.resample_seconds)
    selected_missions = [config for config in MISSION_CONFIGS if config.name in set(args.missions)]
    for mission in selected_missions:
        preprocess_mission(args.data_root, mission, resampling_rule, rebuild_from_raw=args.rebuild_from_raw)


if __name__ == "__main__":
    main()
