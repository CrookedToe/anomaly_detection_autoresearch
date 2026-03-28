#!/usr/bin/env python3
"""Fixed harness: data, labels, ESA metrics, memory bank, STD detector, artifacts.

Do not modify this file during autonomous experiments; edit train.py only.
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent
_rp = str(_repo_root)
if _rp not in sys.path:
    sys.path.insert(0, _rp)

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

PRIMARY_METRIC_KEY = "memory.anomaly_only.Anomaly.EW_F_0.50"
PRIMARY_METRIC_DIRECTION = "maximize"

DEFAULT_TARGET_CHANNELS = [
    "channel_41",
    "channel_42",
    "channel_43",
    "channel_44",
    "channel_45",
    "channel_46",
]

READING_MATERIALS_DIR: Path = Path(__file__).resolve().parent / "reading_materials"
_READING_MATERIALS_SKIP_NAMES = frozenset(
    {name.lower() for name in ("README.md", "_TEMPLATE.md", "_template.md")}
)


def reading_materials_markdown_paths() -> list[Path]:
    """Sorted paper files under reading_materials/ (excludes README and _TEMPLATE)."""
    if not READING_MATERIALS_DIR.is_dir():
        return []
    paths: list[Path] = []
    for path in READING_MATERIALS_DIR.glob("*.md"):
        if path.name.lower() in _READING_MATERIALS_SKIP_NAMES:
            continue
        paths.append(path)
    return sorted(paths, key=lambda p: p.name)


def _parse_reading_material_frontmatter(text: str) -> dict[str, str]:
    """Extract scalar YAML-like keys from the first --- ... --- block (no PyYAML dependency)."""
    if not text.startswith("---"):
        return {}
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        return {}
    block = match.group(1)
    line_key = re.compile(r"^([A-Za-z0-9_]+):\s*(.*)\s*$")
    out: dict[str, str] = {}
    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- ") or (raw.startswith("  ") and ":" not in line):
            continue
        m = line_key.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if not val:
            continue
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        out[key] = val
    return out


def reading_materials_snapshot() -> dict[str, Any]:
    """Index for autoresearch logs: paths, count, and basic metadata per paper file."""
    papers: list[dict[str, str]] = []
    for path in reading_materials_markdown_paths():
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        meta = _parse_reading_material_frontmatter(text)
        papers.append(
            {
                "filename": path.name,
                "id": meta.get("id", ""),
                "title": meta.get("title", ""),
                "year": meta.get("year", ""),
                "url": meta.get("url", ""),
            }
        )
    return {
        "count": len(papers),
        "papers": papers,
        "relative_root": "reading_materials",
        "root": str(READING_MATERIALS_DIR),
    }


def _normalize_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp


def log_debug(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_filtered_labels(
    labels_path: str,
    anomaly_types_path: str,
    target_channels: list[str],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    labels = pd.read_csv(labels_path, parse_dates=["StartTime", "EndTime"])
    labels["StartTime"] = pd.to_datetime(labels["StartTime"]).map(_normalize_timestamp)
    labels["EndTime"] = pd.to_datetime(labels["EndTime"]).map(_normalize_timestamp)
    anomaly_types = pd.read_csv(anomaly_types_path)
    merged = labels.merge(anomaly_types, on="ID", how="left")
    start_time = _normalize_timestamp(start_time)
    end_time = _normalize_timestamp(end_time)
    in_channels = merged["Channel"].isin(target_channels)
    overlaps_window = (merged["EndTime"] >= start_time) & (merged["StartTime"] <= end_time)
    return merged.loc[in_channels & overlaps_window].copy()


def to_global_prediction_series(predictions: pd.DataFrame) -> np.ndarray:
    timestamps = [pd.Timestamp(timestamp) for timestamp in predictions.index]
    any_positive = predictions.max(axis=1).astype(int).to_numpy()
    return np.array(list(zip(timestamps, any_positive)), dtype=object)


def to_channel_prediction_dict(predictions: pd.DataFrame) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    timestamps = [pd.Timestamp(timestamp) for timestamp in predictions.index]
    for channel in predictions.columns:
        result[channel] = np.array(list(zip(timestamps, predictions[channel].astype(int).to_numpy())), dtype=object)
    return result


def compute_esa_metrics(labels: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
    # Submodules only: avoids loading the full TimeEval stack (Docker, dask, etc.).
    from timeeval.metrics.ESA_ADB_metrics import ESAScores
    from timeeval.metrics.latency_metrics import ADTQC
    from timeeval.metrics.ranking_metrics import ChannelAwareFScore

    event_metrics = [
        ("rare_event_and_anomaly", ESAScores(betas=0.5, select_labels={"Category": ["Rare Event", "Anomaly"]})),
        ("anomaly_only", ESAScores(betas=0.5, select_labels={"Category": ["Anomaly"]})),
    ]
    ranking_metrics = [
        ("rare_event_and_anomaly", ChannelAwareFScore(beta=0.5, select_labels={"Category": ["Rare Event", "Anomaly"]})),
        ("anomaly_only", ChannelAwareFScore(beta=0.5, select_labels={"Category": ["Anomaly"]})),
        ("rare_event_and_anomaly", ADTQC(select_labels={"Category": ["Rare Event", "Anomaly"]})),
        ("anomaly_only", ADTQC(select_labels={"Category": ["Anomaly"]})),
    ]

    results: dict[str, Any] = {}
    global_predictions = to_global_prediction_series(predictions)
    channel_predictions = to_channel_prediction_dict(predictions)

    for prefix, metric in event_metrics:
        for name, value in metric.score(labels, global_predictions).items():
            results[f"{prefix}.{metric.name}.{name}"] = _to_builtin(value)

    for prefix, metric in ranking_metrics:
        for name, value in metric.score(labels, channel_predictions).items():
            results[f"{prefix}.{metric.name}.{name}"] = _to_builtin(value)

    return results


def summarize_suppressions(labels: pd.DataFrame, suppressed_events: pd.DataFrame) -> dict[str, Any]:
    if suppressed_events.empty:
        return {
            "suppressed_total": 0,
            "suppressed_overlapping_rare_events": 0,
            "suppressed_overlapping_anomalies": 0,
            "suppressed_nominal_only": 0,
        }

    summary = {
        "suppressed_total": int(len(suppressed_events)),
        "suppressed_overlapping_rare_events": 0,
        "suppressed_overlapping_anomalies": 0,
        "suppressed_nominal_only": 0,
    }

    labels = labels.copy()
    labels["StartTime"] = pd.to_datetime(labels["StartTime"]).map(_normalize_timestamp)
    labels["EndTime"] = pd.to_datetime(labels["EndTime"]).map(_normalize_timestamp)

    for row in suppressed_events.itertuples(index=False):
        start_time = _normalize_timestamp(pd.Timestamp(row.start_time))
        end_time = _normalize_timestamp(pd.Timestamp(row.end_time))
        overlaps = labels[
            (labels["Channel"] == row.channel)
            & (labels["EndTime"] >= start_time)
            & (labels["StartTime"] <= end_time)
        ]
        if (overlaps["Category"] == "Anomaly").any():
            summary["suppressed_overlapping_anomalies"] += 1
        elif (overlaps["Category"] == "Rare Event").any():
            summary["suppressed_overlapping_rare_events"] += 1
        else:
            summary["suppressed_nominal_only"] += 1

    return summary


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_builtin(val) for key, val in value.items()}
    return value


def cosine_similarity(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float:
    left_vec = np.asarray(left, dtype=np.float32)
    right_vec = np.asarray(right, dtype=np.float32)
    left_norm = np.linalg.norm(left_vec)
    right_norm = np.linalg.norm(right_vec)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left_vec, right_vec) / (left_norm * right_norm))


def euclidean_distance(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float:
    left_vec = np.asarray(left, dtype=np.float32)
    right_vec = np.asarray(right, dtype=np.float32)
    return float(np.linalg.norm(left_vec - right_vec))


def compute_similarity(left: list[float] | np.ndarray, right: list[float] | np.ndarray, metric: str) -> float:
    if metric == "cosine":
        return cosine_similarity(left, right)
    if metric == "euclidean":
        return euclidean_distance(left, right)
    raise ValueError(f"Unsupported similarity metric: {metric}")


def is_match(score: float, threshold: float, metric: str) -> bool:
    if metric == "cosine":
        return score >= threshold
    if metric == "euclidean":
        return score <= threshold
    raise ValueError(f"Unsupported similarity metric: {metric}")


def best_match(
    query_vector: list[float] | np.ndarray,
    prototypes: list[list[float] | np.ndarray],
    metric: str,
) -> tuple[int | None, float | None]:
    best_index = None
    best_score = None

    for index, prototype in enumerate(prototypes):
        score = compute_similarity(query_vector, prototype, metric)
        if best_score is None:
            best_index = index
            best_score = score
            continue

        if metric == "cosine" and score > best_score:
            best_index = index
            best_score = score
        elif metric == "euclidean" and score < best_score:
            best_index = index
            best_score = score

    return best_index, best_score


def _ensure_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "timestamp" in result.columns:
        result["timestamp"] = pd.to_datetime(result["timestamp"])
        result = result.set_index("timestamp")
    if not isinstance(result.index, pd.DatetimeIndex):
        raise ValueError("Expected a datetime index or a timestamp column.")
    return result.sort_index()


def extract_centered_window(
    frame: pd.DataFrame,
    center_time: pd.Timestamp,
    target_channels: list[str],
    half_window: int,
) -> pd.DataFrame:
    if half_window < 1:
        raise ValueError("half_window must be at least 1.")

    indexed = _ensure_datetime_index(frame)
    center_idx = indexed.index.get_indexer([pd.Timestamp(center_time)], method="nearest")[0]
    start_idx = max(center_idx - half_window, 0)
    end_idx = min(center_idx + half_window + 1, len(indexed))

    window = indexed.iloc[start_idx:end_idx][target_channels].copy()
    expected_length = (2 * half_window) + 1
    if len(window) < expected_length:
        missing = expected_length - len(window)
        pad_before = missing // 2
        pad_after = missing - pad_before
        data = window.to_numpy(dtype=np.float32)
        if len(data) == 0:
            data = np.zeros((expected_length, len(target_channels)), dtype=np.float32)
        if pad_before > 0:
            data = np.vstack([np.repeat(data[:1], pad_before, axis=0), data])
        if pad_after > 0:
            data = np.vstack([data, np.repeat(data[-1:], pad_after, axis=0)])
        window = pd.DataFrame(data, columns=target_channels)
    return window


def extract_centered_windows_array(
    frame: pd.DataFrame,
    center_times: list[pd.Timestamp] | np.ndarray,
    target_channels: list[str],
    half_window: int,
) -> np.ndarray:
    if half_window < 1:
        raise ValueError("half_window must be at least 1.")

    indexed = _ensure_datetime_index(frame)
    expected_length = (2 * half_window) + 1
    if len(center_times) == 0:
        return np.zeros((0, expected_length, len(target_channels)), dtype=np.float32)

    centers = pd.to_datetime(center_times)
    center_indices = indexed.index.get_indexer(centers, method="nearest")
    offsets = np.arange(-half_window, half_window + 1, dtype=np.int64)
    gather_indices = center_indices[:, None] + offsets[None, :]
    gather_indices = np.clip(gather_indices, 0, max(len(indexed) - 1, 0))

    values = indexed[target_channels].to_numpy(dtype=np.float32, copy=False)
    if len(values) == 0:
        return np.zeros((len(center_times), expected_length, len(target_channels)), dtype=np.float32)
    return np.ascontiguousarray(values[gather_indices])


def window_to_vector(window: pd.DataFrame) -> np.ndarray:
    values = window.to_numpy(dtype=np.float32)
    channel_means = values.mean(axis=0, keepdims=True)
    channel_stds = values.std(axis=0, keepdims=True)
    channel_stds = np.where(channel_stds == 0.0, 1.0, channel_stds)

    normalized = (values - channel_means) / channel_stds
    first_diff = np.diff(normalized, axis=0, prepend=normalized[:1])
    summary = np.concatenate(
        [
            normalized.mean(axis=0),
            normalized.std(axis=0),
            normalized.min(axis=0),
            normalized.max(axis=0),
        ]
    )
    return np.concatenate([normalized.reshape(-1), first_diff.reshape(-1), summary]).astype(np.float32)


def windows_to_vectors(windows: np.ndarray) -> np.ndarray:
    if len(windows) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    values = np.asarray(windows, dtype=np.float32)
    channel_means = values.mean(axis=1, keepdims=True)
    channel_stds = values.std(axis=1, keepdims=True)
    channel_stds = np.where(channel_stds == 0.0, 1.0, channel_stds)

    normalized = (values - channel_means) / channel_stds
    first_diff = np.diff(normalized, axis=1, prepend=normalized[:, :1, :])
    summary = np.concatenate(
        [
            normalized.mean(axis=1),
            normalized.std(axis=1),
            normalized.min(axis=1),
            normalized.max(axis=1),
        ],
        axis=1,
    )
    return np.concatenate(
        [
            normalized.reshape(len(values), -1),
            first_diff.reshape(len(values), -1),
            summary,
        ],
        axis=1,
    ).astype(np.float32)


@dataclass
class MemoryPrototype:
    prototype_id: str
    event_id: str
    channel: str
    start_time: str
    end_time: str
    vector: np.ndarray

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["vector"] = self.vector.tolist()
        return record


@dataclass
class MemoryMatch:
    prototype: MemoryPrototype
    score: float
    metric: str


class RareNominalMemoryBank:
    def __init__(self, prototypes: list[MemoryPrototype] | None = None):
        self.prototypes = prototypes or []
        if self.prototypes:
            self.prototype_matrix = np.stack([prototype.vector for prototype in self.prototypes]).astype(np.float32)
            self.prototype_norms = np.linalg.norm(self.prototype_matrix, axis=1).astype(np.float32)
            self.prototype_squared_norms = np.sum(self.prototype_matrix * self.prototype_matrix, axis=1).astype(np.float32)
        else:
            self.prototype_matrix = np.zeros((0, 0), dtype=np.float32)
            self.prototype_norms = np.zeros(0, dtype=np.float32)
            self.prototype_squared_norms = np.zeros(0, dtype=np.float32)

    @classmethod
    def from_labeled_rare_events(
        cls,
        frame: pd.DataFrame,
        labels: pd.DataFrame,
        target_channels: list[str],
        half_window: int,
        vectorizer: Any | None = None,
    ) -> RareNominalMemoryBank:
        indexed = _ensure_datetime_index(frame)
        rare_events = (
            labels.loc[labels["Category"] == "Rare Event", ["ID", "Channel", "StartTime", "EndTime"]]
            .drop_duplicates()
            .sort_values(["StartTime", "Channel"])
        )

        event_rows: list[Any] = []
        windows: list[pd.DataFrame] = []
        for row in rare_events.itertuples(index=False):
            if row.Channel not in target_channels:
                continue
            center_time = row.StartTime + (row.EndTime - row.StartTime) / 2
            event_rows.append(row)
            windows.append(extract_centered_window(indexed, center_time, target_channels, half_window))

        if vectorizer is not None:
            vectors = vectorizer(windows)
        else:
            vectors = np.asarray([window_to_vector(window.reset_index(drop=True)) for window in windows], dtype=np.float32)

        prototypes: list[MemoryPrototype] = []
        for row, vector in zip(event_rows, vectors):
            prototypes.append(
                MemoryPrototype(
                    prototype_id=f"{row.ID}:{row.Channel}",
                    event_id=str(row.ID),
                    channel=str(row.Channel),
                    start_time=pd.Timestamp(row.StartTime).isoformat(),
                    end_time=pd.Timestamp(row.EndTime).isoformat(),
                    vector=vector,
                )
            )

        return cls(prototypes)

    def query(self, query_vector: np.ndarray, metric: str, threshold: float) -> MemoryMatch | None:
        return self.query_many(np.asarray([query_vector], dtype=np.float32), metric=metric, threshold=threshold)[0]

    def query_many(self, query_vectors: np.ndarray, metric: str, threshold: float) -> list[MemoryMatch | None]:
        if not self.prototypes:
            return [None] * len(query_vectors)

        queries = np.asarray(query_vectors, dtype=np.float32)
        if len(queries) == 0:
            return []

        if metric == "cosine":
            query_norms = np.linalg.norm(queries, axis=1, keepdims=True).astype(np.float32)
            safe_query_norms = np.where(query_norms > 0.0, query_norms, 1.0)
            safe_prototype_norms = np.where(self.prototype_norms > 0.0, self.prototype_norms, 1.0)[None, :]
            scores = (queries @ self.prototype_matrix.T) / (safe_query_norms * safe_prototype_norms)
            scores = np.where(query_norms > 0.0, scores, 0.0)
            best_indices = np.argmax(scores, axis=1)
            best_scores = scores[np.arange(len(queries)), best_indices]
            matched = best_scores >= threshold
        elif metric == "euclidean":
            query_squared_norms = np.sum(queries * queries, axis=1, keepdims=True)
            distances_squared = np.maximum(
                query_squared_norms + self.prototype_squared_norms[None, :] - (2.0 * (queries @ self.prototype_matrix.T)),
                0.0,
            )
            scores = np.sqrt(distances_squared, dtype=np.float32)
            best_indices = np.argmin(scores, axis=1)
            best_scores = scores[np.arange(len(queries)), best_indices]
            matched = best_scores <= threshold
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")

        matches: list[MemoryMatch | None] = []
        for is_valid, best_index, best_score in zip(matched, best_indices, best_scores):
            if not is_valid:
                matches.append(None)
                continue
            matches.append(
                MemoryMatch(
                    prototype=self.prototypes[int(best_index)],
                    score=float(best_score),
                    metric=metric,
                )
            )
        return matches

    def to_frame(self) -> pd.DataFrame:
        if not self.prototypes:
            return pd.DataFrame(columns=["prototype_id", "event_id", "channel", "start_time", "end_time", "vector"])
        return pd.DataFrame([prototype.to_record() for prototype in self.prototypes])


@dataclass
class SuppressedEvent:
    channel: str
    start_time: str
    end_time: str
    prototype_id: str
    score: float
    metric: str


def _iter_positive_runs(series: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    runs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    active_start: pd.Timestamp | None = None
    previous_index: pd.Timestamp | None = None

    for timestamp, value in series.items():
        if int(value) == 1 and active_start is None:
            active_start = pd.Timestamp(timestamp)
        elif int(value) == 0 and active_start is not None:
            runs.append((active_start, pd.Timestamp(previous_index)))
            active_start = None
        previous_index = pd.Timestamp(timestamp)

    if active_start is not None and previous_index is not None:
        runs.append((active_start, previous_index))

    return runs


def apply_memory_gating(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    target_channels: list[str],
    memory_bank: RareNominalMemoryBank,
    half_window: int,
    metric: str,
    threshold: float,
    vectorizer: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed_frame = frame.copy()
    if "timestamp" in indexed_frame.columns:
        indexed_frame["timestamp"] = pd.to_datetime(indexed_frame["timestamp"])
        indexed_frame = indexed_frame.set_index("timestamp")

    gated = predictions.copy()
    suppressed_events: list[SuppressedEvent] = []
    candidate_events: list[tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []

    for channel in target_channels:
        for start_time, end_time in _iter_positive_runs(predictions[channel]):
            center_time = start_time + (end_time - start_time) / 2
            candidate_events.append((channel, start_time, end_time, center_time))

    total_candidates = len(candidate_events)
    if total_candidates == 0:
        return gated, pd.DataFrame(columns=["channel", "start_time", "end_time", "prototype_id", "score", "metric"])

    chunk_size = 2048
    log_debug(f"[memory] gating candidates={total_candidates} chunk_size={chunk_size}")

    for chunk_start in range(0, total_candidates, chunk_size):
        chunk_events = candidate_events[chunk_start : chunk_start + chunk_size]
        if chunk_start == 0 or (chunk_start // chunk_size) % 10 == 0 or (chunk_start + chunk_size) >= total_candidates:
            log_debug(
                f"[memory] processing gating chunk {chunk_start + 1}-{min(chunk_start + chunk_size, total_candidates)} of {total_candidates}"
            )
        center_times = [center_time for _, _, _, center_time in chunk_events]
        chunk_windows = extract_centered_windows_array(indexed_frame, center_times, target_channels, half_window)
        if vectorizer is not None:
            query_vectors = vectorizer(chunk_windows)
        else:
            query_vectors = windows_to_vectors(chunk_windows)
        matches = memory_bank.query_many(query_vectors=query_vectors, metric=metric, threshold=threshold)

        for (channel, start_time, end_time, _), match in zip(chunk_events, matches):
            if match is None:
                continue
            gated.loc[start_time:end_time, channel] = 0
            suppressed_events.append(
                SuppressedEvent(
                    channel=channel,
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    prototype_id=match.prototype.prototype_id,
                    score=match.score,
                    metric=metric,
                )
            )

    return gated, pd.DataFrame([event.__dict__ for event in suppressed_events])


def load_dataset_record(preprocessed_root: Path, split: str) -> pd.Series:
    datasets = pd.read_csv(preprocessed_root / "datasets.csv")
    row = datasets[
        (datasets["collection_name"] == "ESA-Mission1")
        & (datasets["dataset_name"] == split)
    ]
    if row.empty:
        raise FileNotFoundError(f"Could not find dataset record for split '{split}'.")
    return row.iloc[0]


def infer_mission_from_split(split: str) -> str | None:
    if split in {"84_months", "81_months", "10_months"}:
        return "ESA-Mission1"
    if split in {"21_months", "18_months"}:
        return "ESA-Mission2"
    return None


def resolve_dataset_paths(preprocessed_root: Path, split: str) -> tuple[Path, Path | None, Path]:
    subset_root = preprocessed_root / "multivariate" / "ESA-Mission1-subset-semi-supervised"
    subset_train_path = subset_root / f"{split}.train.csv"
    subset_val_path = subset_root / f"{split}.val.csv"
    subset_test_path = subset_root / f"{split}.test.csv"
    if subset_train_path.exists() and subset_test_path.exists():
        return subset_train_path, subset_val_path if subset_val_path.exists() else None, subset_test_path

    legacy_subset_test_path = subset_root / "84_months.test.csv"
    if subset_train_path.exists() and legacy_subset_test_path.exists():
        return subset_train_path, None, legacy_subset_test_path

    mission = infer_mission_from_split(split)
    if mission is not None:
        mission_root = preprocessed_root / "multivariate" / f"{mission}-semi-supervised"
        mission_train_candidates = [mission_root / f"{split}.train.csv"]
        if split == "84_months":
            mission_train_candidates.insert(0, mission_root / "81_months.train.csv")
        elif split == "21_months":
            mission_train_candidates.insert(0, mission_root / "18_months.train.csv")
        mission_val_path = mission_root / "3_months.val.csv"
        mission_test_path = mission_root / f"{split}.test.csv"
        for mission_train_path in mission_train_candidates:
            if mission_train_path.exists() and mission_test_path.exists():
                return mission_train_path, mission_val_path if mission_val_path.exists() else None, mission_test_path

    record = load_dataset_record(preprocessed_root, split)
    train_path = preprocessed_root / Path(record["train_path"])
    test_path = preprocessed_root / Path(record["test_path"])
    precise_match = re.fullmatch(r"(\d+)_months\.train\.csv", train_path.name)
    precise_val_path = train_path.with_name("3_months.val.csv")
    if precise_match and precise_val_path.exists():
        precise_train_months = int(precise_match.group(1)) - 3
        precise_train_path = train_path.with_name(f"{precise_train_months}_months.train.csv")
        if precise_train_path.exists():
            return precise_train_path, precise_val_path, test_path

    val_path = train_path.with_name(train_path.name.replace(".train.csv", ".val.csv"))
    return train_path, val_path if val_path.exists() else None, test_path


def load_subset_frame(csv_path: Path, target_channels: list[str]) -> pd.DataFrame:
    anomaly_columns = [f"is_anomaly_{channel}" for channel in target_channels]
    usecols = ["timestamp", *target_channels, *anomaly_columns]
    frame = pd.read_csv(csv_path, usecols=usecols, parse_dates=["timestamp"])
    return frame.set_index("timestamp").sort_index()


def train_std_baseline(train_df: pd.DataFrame, target_channels: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for channel in target_channels:
        label_column = f"is_anomaly_{channel}"
        nominal_values = train_df.loc[train_df[label_column] == 0, channel].to_numpy(dtype=np.float32)
        if len(nominal_values) == 0:
            nominal_values = train_df[channel].to_numpy(dtype=np.float32)
        stats[channel] = {
            "mean": float(nominal_values.mean()),
            "std": float(max(nominal_values.std(), 1.0)),
        }
    return stats


def score_std_baseline(
    test_df: pd.DataFrame,
    target_channels: list[str],
    stats: dict[str, dict[str, float]],
    tol: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.DataFrame(index=test_df.index)
    predictions = pd.DataFrame(index=test_df.index)
    for channel in target_channels:
        centered = np.abs(test_df[channel].to_numpy(dtype=np.float32) - stats[channel]["mean"])
        z_scores = centered / stats[channel]["std"]
        anomaly_scores = np.maximum(z_scores - tol, 0.0)
        scores[channel] = anomaly_scores.astype(np.float32)
        predictions[channel] = (anomaly_scores > 0.0).astype(np.uint8)
    return scores, predictions


def ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected file was not found: {path}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_split_data(
    args: argparse.Namespace, split: str
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    preprocessed_root = args.data_root / "preprocessed"
    train_path, val_path, test_path = resolve_dataset_paths(preprocessed_root, split)
    ensure_file(train_path)
    ensure_file(test_path)

    train_df = load_subset_frame(train_path, args.target_channels)
    val_df: pd.DataFrame | None = None
    if val_path is not None:
        ensure_file(val_path)
        val_df = load_subset_frame(val_path, args.target_channels)
        train_df = train_df[train_df.index < val_df.index.min()].copy()
        if train_df.empty:
            raise RuntimeError(f"Training split '{split}' becomes empty after removing the explicit validation tail.")
    test_df = load_subset_frame(test_path, args.target_channels)

    mission = infer_mission_from_split(split) or "ESA-Mission1"
    raw_labels_path = args.data_root / mission / "labels.csv"
    anomaly_types_path = args.data_root / mission / "anomaly_types.csv"
    ensure_file(raw_labels_path)
    ensure_file(anomaly_types_path)

    train_labels = load_filtered_labels(
        labels_path=str(raw_labels_path),
        anomaly_types_path=str(anomaly_types_path),
        target_channels=args.target_channels,
        start_time=train_df.index.min(),
        end_time=train_df.index.max(),
    )
    test_labels = load_filtered_labels(
        labels_path=str(raw_labels_path),
        anomaly_types_path=str(anomaly_types_path),
        target_channels=args.target_channels,
        start_time=test_df.index.min(),
        end_time=test_df.index.max(),
    )
    return train_df, val_df, test_df, train_labels, test_labels


def save_detector_results(
    detector_name: str,
    split: str,
    results_root: Path,
    baseline_scores: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    baseline_metrics: dict[str, Any],
    baseline_parameters: dict[str, Any],
    memory_bank: RareNominalMemoryBank,
    gated_predictions: pd.DataFrame,
    memory_metrics: dict[str, Any],
    memory_parameters: dict[str, Any],
    suppression_summary: dict[str, Any],
    suppressed_events: pd.DataFrame,
) -> None:
    baseline_dir = results_root / f"{detector_name}_baseline" / split
    memory_dir = results_root / f"{detector_name}_memory" / split
    baseline_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)

    log_debug(f"[{detector_name}] writing baseline scores for '{split}'")
    baseline_scores.to_csv(baseline_dir / "scores.csv")
    log_debug(f"[{detector_name}] writing baseline predictions for '{split}'")
    baseline_predictions.to_csv(baseline_dir / "predictions.csv")
    log_debug(f"[{detector_name}] writing baseline metrics for '{split}'")
    write_json(
        baseline_dir / "metrics.json",
        {
            "split": split,
            "detector": detector_name,
            "parameters": baseline_parameters,
            "metrics": baseline_metrics,
        },
    )

    log_debug(f"[{detector_name}] writing memory bank for '{split}'")
    memory_bank.to_frame().to_csv(memory_dir / "memory_bank.csv", index=False)
    log_debug(f"[{detector_name}] writing memory predictions for '{split}'")
    gated_predictions.to_csv(memory_dir / "predictions.csv")
    log_debug(f"[{detector_name}] writing suppressed events for '{split}'")
    suppressed_events.to_csv(memory_dir / "suppressed_events.csv", index=False)
    log_debug(f"[{detector_name}] writing memory metrics for '{split}'")
    write_json(
        memory_dir / "metrics.json",
        {
            "split": split,
            "detector": f"{detector_name}+memory",
            "parameters": memory_parameters,
            "metrics": memory_metrics,
            "suppression_summary": suppression_summary,
        },
    )


def summarize_detector_run(
    detector_name: str,
    split: str,
    memory_bank: RareNominalMemoryBank,
    suppression_summary: dict[str, Any],
    baseline_metrics: dict[str, Any],
    memory_metrics: dict[str, Any],
) -> dict[str, Any]:
    summary_row: dict[str, Any] = {
        "detector": detector_name,
        "split": split,
        "memory_size": len(memory_bank.prototypes),
        "suppressed_total": suppression_summary["suppressed_total"],
        "suppressed_overlapping_anomalies": suppression_summary["suppressed_overlapping_anomalies"],
        "suppressed_overlapping_rare_events": suppression_summary["suppressed_overlapping_rare_events"],
        "suppressed_nominal_only": suppression_summary["suppressed_nominal_only"],
    }
    for key, value in baseline_metrics.items():
        summary_row[f"baseline.{key}"] = value
    for key, value in memory_metrics.items():
        summary_row[f"memory.{key}"] = value
    return summary_row


def run_std_split(
    args: argparse.Namespace,
    split: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_labels: pd.DataFrame,
    test_labels: pd.DataFrame,
) -> dict[str, Any]:
    stats = train_std_baseline(train_df, args.target_channels)
    baseline_scores, baseline_predictions = score_std_baseline(test_df, args.target_channels, stats, args.tol)

    memory_bank = RareNominalMemoryBank.from_labeled_rare_events(
        frame=train_df,
        labels=train_labels,
        target_channels=args.target_channels,
        half_window=args.half_window,
    )
    gated_predictions, suppressed_events = apply_memory_gating(
        frame=test_df,
        predictions=baseline_predictions,
        target_channels=args.target_channels,
        memory_bank=memory_bank,
        half_window=args.half_window,
        metric=args.metric,
        threshold=args.memory_threshold,
    )

    baseline_metrics = compute_esa_metrics(test_labels, baseline_predictions)
    memory_metrics = compute_esa_metrics(test_labels, gated_predictions)
    suppression_summary = summarize_suppressions(test_labels, suppressed_events)

    save_detector_results(
        detector_name="std",
        split=split,
        results_root=args.results_root,
        baseline_scores=baseline_scores,
        baseline_predictions=baseline_predictions,
        baseline_metrics=baseline_metrics,
        baseline_parameters={"tol": args.tol, "target_channels": args.target_channels},
        memory_bank=memory_bank,
        gated_predictions=gated_predictions,
        memory_metrics=memory_metrics,
        memory_parameters={
            "tol": args.tol,
            "target_channels": args.target_channels,
            "half_window": args.half_window,
            "metric": args.metric,
            "memory_threshold": args.memory_threshold,
            "memory_size": len(memory_bank.prototypes),
        },
        suppression_summary=suppression_summary,
        suppressed_events=suppressed_events,
    )
    return summarize_detector_run("std", split, memory_bank, suppression_summary, baseline_metrics, memory_metrics)
