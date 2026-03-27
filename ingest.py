#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.parser import parse as parse_date


DEFAULT_TARGET_CHANNELS = [
    "channel_41",
    "channel_42",
    "channel_43",
    "channel_44",
    "channel_45",
    "channel_46",
]

DATASET_SPLITS = {
    "3_months": "2000-04-01",
    "10_months": "2000-11-01",
}
TEST_SPLIT = "2007-01-01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess the Mission 1 lightweight subset.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/ESA-Mission1"),
        help="Path to the raw ESA Mission 1 folder.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/preprocessed/multivariate/ESA-Mission1-subset-semi-supervised"),
        help="Path where the subset CSVs should be written.",
    )
    parser.add_argument(
        "--target-channels",
        nargs="+",
        default=DEFAULT_TARGET_CHANNELS,
        help="Target channels to keep in the lightweight subset.",
    )
    parser.add_argument(
        "--resample-seconds",
        type=int,
        default=30,
        help="Resampling frequency in seconds.",
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


def load_channel_frame(
    source_folder: Path,
    channel: str,
    labels_df: pd.DataFrame,
    anomaly_types_df: pd.DataFrame,
    resampling_rule: pd.Timedelta,
) -> pd.DataFrame:
    channel_df = pd.read_pickle(source_folder / "channels" / f"{channel}.zip")
    channel_df["label"] = np.uint8(0)
    channel_df = channel_df.rename(columns={channel: "value"})

    channel_labels = labels_df.loc[labels_df["Channel"] == channel]
    for row in channel_labels.itertuples(index=False):
        category = anomaly_types_df.loc[anomaly_types_df["ID"] == row.ID, "Category"].values[0]
        channel_df.loc[row.StartTime : row.EndTime, "label"] = label_value(category)

    first_index_resampled = pd.Timestamp(channel_df.index[0]).floor(freq=resampling_rule)
    last_index_resampled = pd.Timestamp(channel_df.index[-1]).ceil(freq=resampling_rule)
    resampled_range = pd.date_range(first_index_resampled, last_index_resampled, freq=resampling_rule)
    resampled = channel_df.reindex(resampled_range, method="ffill")
    resampled.iloc[0] = channel_df.iloc[0]

    grouped = channel_df.groupby(pd.Grouper(freq=resampling_rule))
    for timestamp, group_indices in grouped.indices.items():
        if len(group_indices) <= 1:
            continue
        original = channel_df.iloc[group_indices]
        annotated = original[original["label"].isin([1, 2])]
        if annotated.empty:
            continue
        resampled.loc[timestamp + resampling_rule] = annotated.iloc[-1]

    return resampled.sort_index()


def build_dataset(
    source_folder: Path,
    labels_df: pd.DataFrame,
    anomaly_types_df: pd.DataFrame,
    target_channels: list[str],
    split_at: str | None,
    resampling_rule: pd.Timedelta,
) -> pd.DataFrame:
    channels: list[pd.DataFrame] = []
    dataset_start = None
    dataset_end = None

    for channel in target_channels:
        channel_df = load_channel_frame(source_folder, channel, labels_df, anomaly_types_df, resampling_rule)
        if split_at is not None:
            split_time = parse_date(split_at)
            if split_at == TEST_SPLIT:
                channel_df = channel_df[channel_df.index > split_time].copy()
            else:
                channel_df = channel_df[channel_df.index <= split_time].copy()

        if channel_df.empty:
            continue

        dataset_start = channel_df.index[0] if dataset_start is None else min(dataset_start, channel_df.index[0])
        dataset_end = channel_df.index[-1] if dataset_end is None else max(dataset_end, channel_df.index[-1])
        renamed = channel_df.rename(columns={"value": channel, "label": f"is_anomaly_{channel}"})
        channels.append(renamed)

    if not channels or dataset_start is None or dataset_end is None:
        raise RuntimeError("No channel data was prepared for the requested split.")

    index = pd.date_range(dataset_start, dataset_end, freq=resampling_rule)
    dataset = pd.DataFrame(index=index)

    for frame in channels:
        dataset = dataset.join(frame, how="left")

    for channel in target_channels:
        dataset[channel] = dataset[channel].ffill().bfill()
        dataset[f"is_anomaly_{channel}"] = dataset[f"is_anomaly_{channel}"].ffill().bfill().fillna(0).astype(np.uint8)

    dataset.insert(0, "timestamp", dataset.index.strftime("%Y-%m-%d %H:%M:%S"))
    return dataset.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.output_path.mkdir(parents=True, exist_ok=True)
    resampling_rule = pd.Timedelta(seconds=args.resample_seconds)

    labels_df = pd.read_csv(
        args.input_path / "labels.csv",
        parse_dates=["StartTime", "EndTime"],
        date_parser=lambda value: parse_date(value, ignoretz=True),
    )
    anomaly_types_df = pd.read_csv(args.input_path / "anomaly_types.csv")

    test_df = build_dataset(
        source_folder=args.input_path,
        labels_df=labels_df,
        anomaly_types_df=anomaly_types_df,
        target_channels=args.target_channels,
        split_at=TEST_SPLIT,
        resampling_rule=resampling_rule,
    )
    test_df.to_csv(args.output_path / "84_months.test.csv", index=False, lineterminator="\n")

    for split_name, split_at in DATASET_SPLITS.items():
        train_df = build_dataset(
            source_folder=args.input_path,
            labels_df=labels_df,
            anomaly_types_df=anomaly_types_df,
            target_channels=args.target_channels,
            split_at=split_at,
            resampling_rule=resampling_rule,
        )
        train_df.to_csv(args.output_path / f"{split_name}.train.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
