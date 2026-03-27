#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import site
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from prepare import (
    DEFAULT_TARGET_CHANNELS,
    PRIMARY_METRIC_DIRECTION,
    PRIMARY_METRIC_KEY,
    READING_MATERIALS_DIR,
    apply_memory_gating,
    compute_esa_metrics,
    load_split_data,
    log_debug,
    reading_materials_snapshot,
    RareNominalMemoryBank,
    run_std_split,
    save_detector_results,
    summarize_detector_run,
    summarize_suppressions,
    write_json,
)


@dataclass
class TcnModelConfig:
    input_dim: int
    target_dim: int
    horizon: int = 8
    hidden_dim: int = 64
    embedding_dim: int = 64
    kernel_size: int = 3
    num_blocks: int = 5
    dropout: float = 0.05
    use_depthwise_separable: bool = True


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        padded = F.pad(inputs, (self.left_padding, 0))
        return self.conv(padded)


class SeparableCausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.depthwise = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        padded = F.pad(inputs, (self.left_padding, 0))
        return self.pointwise(self.depthwise(padded))


class ResidualTemporalBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_depthwise_separable: bool,
    ) -> None:
        super().__init__()
        conv_layer: nn.Module
        if use_depthwise_separable:
            conv_layer = SeparableCausalConv1d(channels, kernel_size, dilation)
        else:
            conv_layer = CausalConv1d(channels, channels, kernel_size, dilation)

        self.block = nn.Sequential(
            conv_layer,
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size=1, dilation=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class SelfSupervisedTcnForecaster(nn.Module):
    def __init__(self, config: TcnModelConfig) -> None:
        super().__init__()
        self.config = config

        self.input_projection = nn.Sequential(
            nn.Conv1d(config.input_dim, config.hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        self.backbone = nn.ModuleList(
            [
                ResidualTemporalBlock(
                    channels=config.hidden_dim,
                    kernel_size=config.kernel_size,
                    dilation=2**block_index,
                    dropout=config.dropout,
                    use_depthwise_separable=config.use_depthwise_separable,
                )
                for block_index in range(config.num_blocks)
            ]
        )
        self.forecast_head = nn.Linear(config.hidden_dim, config.horizon * config.target_dim)
        self.reconstruction_head = nn.Sequential(
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(config.hidden_dim, config.target_dim, kernel_size=1),
        )
        self.embedding_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
            nn.GELU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        features = inputs.transpose(1, 2)
        hidden = self.input_projection(features)
        for block in self.backbone:
            hidden = block(hidden)
        return hidden

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encode(inputs)
        pooled_state = hidden[:, :, -1]
        forecast = self.forecast_head(pooled_state).view(
            inputs.shape[0],
            self.config.horizon,
            self.config.target_dim,
        )
        reconstruction = self.reconstruction_head(hidden).transpose(1, 2)
        embedding = F.normalize(self.embedding_head(hidden), dim=-1)
        return forecast, reconstruction, embedding


EPSILON = 1e-6


@dataclass
class FeatureScaler:
    raw_mean: np.ndarray
    raw_std: np.ndarray
    dt_mean: float
    dt_std: float
    nominal_dt_seconds: float


@dataclass
class TcnTrainingConfig:
    sequence_length: int = 128
    horizon: int = 8
    hidden_dim: int = 64
    embedding_dim: int = 64
    num_blocks: int = 5
    kernel_size: int = 3
    dropout: float = 0.05
    batch_size: int = 512
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 6
    mask_ratio: float = 0.15
    train_stride: int = 8
    inference_stride: int = 4
    forecast_loss_weight: float = 1.0
    reconstruction_loss_weight: float = 0.5
    forecast_score_weight: float = 1.0
    reconstruction_score_weight: float = 0.35
    threshold_window: int = 288
    threshold_std_factor: float = 4.0
    calibration_quantile: float = 0.995
    score_smoothing_window: int = 5
    min_anomaly_run_length: int = 5
    max_gap_fill: int = 2
    validation_fraction: float = 0.1
    random_seed: int = 42
    device: str = "cuda"
    use_depthwise_separable: bool = True
    mixed_precision: bool = True
    allow_tf32: bool = True
    dataloader_workers: int = 4
    pin_memory: bool = True
    use_cupy: bool = True
    preload_dataset: bool = True
    preload_max_gb: float = 8.0
    training_wall_seconds: float | None = None


class SelfSupervisedWindowDataset(Dataset):
    def __init__(
        self,
        base_features: np.ndarray,
        raw_targets: np.ndarray,
        window_starts: np.ndarray,
        sequence_length: int,
        horizon: int,
        mask_ratio: float,
        random_seed: int,
    ) -> None:
        self.window_starts = window_starts
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.mask_ratio = mask_ratio
        self.random_seed = random_seed
        self.target_dim = raw_targets.shape[1]
        (
            self.inputs,
            self.forecast_targets,
            self.reconstruction_targets,
            self.reconstruction_mask,
        ) = self._build_tensors(base_features=base_features, raw_targets=raw_targets)
        self.total_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.inputs,
                self.forecast_targets,
                self.reconstruction_targets,
                self.reconstruction_mask,
            )
        )

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "inputs": self.inputs[index],
            "forecast_targets": self.forecast_targets[index],
            "reconstruction_targets": self.reconstruction_targets[index],
            "reconstruction_mask": self.reconstruction_mask[index],
        }

    def move_to_device(self, device: torch.device) -> None:
        self.inputs = self.inputs.to(device, non_blocking=True)
        self.forecast_targets = self.forecast_targets.to(device, non_blocking=True)
        self.reconstruction_targets = self.reconstruction_targets.to(device, non_blocking=True)
        self.reconstruction_mask = self.reconstruction_mask.to(device, non_blocking=True)

    def _build_tensors(
        self,
        base_features: np.ndarray,
        raw_targets: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_windows = len(self.window_starts)
        if n_windows == 0:
            empty_inputs = torch.zeros((0, self.sequence_length, base_features.shape[1] + 1), dtype=torch.float32)
            empty_future = torch.zeros((0, self.horizon, self.target_dim), dtype=torch.float32)
            empty_recon = torch.zeros((0, self.sequence_length, self.target_dim), dtype=torch.float32)
            return empty_inputs, empty_future, empty_recon, empty_recon.clone()

        history_indices = self.window_starts[:, None] + np.arange(self.sequence_length, dtype=np.int64)[None, :]
        future_indices = self.window_starts[:, None] + self.sequence_length + np.arange(self.horizon, dtype=np.int64)[None, :]

        inputs = np.ascontiguousarray(base_features[history_indices].copy())
        current_targets = np.ascontiguousarray(raw_targets[history_indices])
        future_targets = np.ascontiguousarray(raw_targets[future_indices])

        rng = np.random.default_rng(self.random_seed)
        mask = rng.random((n_windows, self.sequence_length, self.target_dim)) < self.mask_ratio
        empty_mask_rows = ~mask.reshape(n_windows, -1).any(axis=1)
        if empty_mask_rows.any():
            empty_indices = np.flatnonzero(empty_mask_rows)
            random_steps = rng.integers(0, self.sequence_length, size=len(empty_indices))
            random_channels = rng.integers(0, self.target_dim, size=len(empty_indices))
            mask[empty_indices, random_steps, random_channels] = True

        inputs[:, :, : self.target_dim][mask] = 0.0
        mask_indicator = mask.any(axis=2, keepdims=True).astype(np.float32)
        model_inputs = np.ascontiguousarray(np.concatenate([inputs, mask_indicator], axis=2))

        return (
            torch.from_numpy(model_inputs).float(),
            torch.from_numpy(future_targets).float(),
            torch.from_numpy(current_targets).float(),
            torch.from_numpy(mask.astype(np.float32)).float(),
        )


class TcnAnomalyPipeline:
    def __init__(self, target_channels: list[str], config: TcnTrainingConfig):
        self.target_channels = target_channels
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.scaler: FeatureScaler | None = None
        self.model: SelfSupervisedTcnForecaster | None = None
        self.global_thresholds: np.ndarray | None = None
        self.use_amp = self.device.type == "cuda" and config.mixed_precision
        self.cp = self._try_load_cupy()

        if self.device.type == "cuda" and config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    def fit(self, train_df: pd.DataFrame) -> dict[str, Any]:
        self._set_random_seed()
        self.scaler = self._fit_scaler(train_df)

        base_features, raw_targets = self._transform_frame(train_df)
        window_starts = self._build_window_starts(train_df, self.config.train_stride, training=True)
        if len(window_starts) == 0:
            raise RuntimeError("No valid training windows were found for the TCN model.")

        split_index = max(1, int(len(window_starts) * (1.0 - self.config.validation_fraction)))
        train_starts = window_starts[:split_index]
        val_starts = window_starts[split_index:] if split_index < len(window_starts) else window_starts[-1:]

        train_dataset = SelfSupervisedWindowDataset(
            base_features=base_features,
            raw_targets=raw_targets,
            window_starts=train_starts,
            sequence_length=self.config.sequence_length,
            horizon=self.config.horizon,
            mask_ratio=self.config.mask_ratio,
            random_seed=self.config.random_seed,
        )
        val_dataset = SelfSupervisedWindowDataset(
            base_features=base_features,
            raw_targets=raw_targets,
            window_starts=val_starts,
            sequence_length=self.config.sequence_length,
            horizon=self.config.horizon,
            mask_ratio=self.config.mask_ratio,
            random_seed=self.config.random_seed + 10_000,
        )
        preloaded_dataset = False
        dataset_bytes = train_dataset.total_bytes + val_dataset.total_bytes

        input_dim = base_features.shape[1] + 1
        self.model = SelfSupervisedTcnForecaster(
            TcnModelConfig(
                input_dim=input_dim,
                target_dim=len(self.target_channels),
                horizon=self.config.horizon,
                hidden_dim=self.config.hidden_dim,
                embedding_dim=self.config.embedding_dim,
                kernel_size=self.config.kernel_size,
                num_blocks=self.config.num_blocks,
                dropout=self.config.dropout,
                use_depthwise_separable=self.config.use_depthwise_separable,
            )
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        grad_scaler = torch.amp.GradScaler(device="cuda", enabled=self.use_amp)

        loader_kwargs = {
            "batch_size": self.config.batch_size,
            "drop_last": False,
            "num_workers": self.config.dataloader_workers,
            "pin_memory": self.config.pin_memory and self.device.type == "cuda",
        }
        preload_limit_bytes = int(self.config.preload_max_gb * (1024**3))
        if self.config.preload_dataset and self.device.type == "cuda" and dataset_bytes <= preload_limit_bytes:
            train_dataset.move_to_device(self.device)
            val_dataset.move_to_device(self.device)
            loader_kwargs["num_workers"] = 0
            loader_kwargs["pin_memory"] = False
            preloaded_dataset = True
        if loader_kwargs["num_workers"] > 0:
            loader_kwargs["persistent_workers"] = True

        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

        best_state: dict[str, torch.Tensor] | None = None
        best_val_loss = float("inf")
        history: list[dict[str, float]] = []

        wall = self.config.training_wall_seconds
        log_debug(
            (
                f"[tcn] training on {self.device} | epochs_max={self.config.epochs} | "
                f"wall_s={wall} | "
                f"train_windows={len(train_starts)} | val_windows={len(val_starts)} | "
                f"batch_size={self.config.batch_size} | preloaded={preloaded_dataset}"
            )
        )

        train_start = time.perf_counter()
        epoch = 0
        while epoch < self.config.epochs:
            if wall is not None and time.perf_counter() - train_start >= wall:
                log_debug(f"[tcn] stopping training: wall clock {wall}s reached after {epoch} epoch(s)")
                break
            train_loss = self._run_epoch(train_loader, optimizer, grad_scaler)
            val_loss = self._evaluate_epoch(val_loader)
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                }
            )
            log_debug(f"[tcn] epoch {epoch + 1}/{self.config.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()}
            epoch += 1

        training_elapsed = time.perf_counter() - train_start

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.global_thresholds = self._calibrate_thresholds(train_df, train_starts)

        return {
            "device": str(self.device),
            "epochs": int(epoch),
            "epochs_requested": self.config.epochs,
            "training_wall_seconds": wall,
            "training_elapsed_seconds": float(training_elapsed),
            "mixed_precision": self.use_amp,
            "dataset_preloaded": preloaded_dataset,
            "dataset_gb": float(dataset_bytes / (1024**3)),
            "num_train_windows": int(len(train_starts)),
            "num_val_windows": int(len(val_starts)),
            "history": history,
            "global_thresholds": self.global_thresholds.tolist(),
        }

    def score_sequence(self, frame: pd.DataFrame, stride: int | None = None) -> pd.DataFrame:
        combined, covered = self._aggregate_scores(frame=frame, stride=stride)
        scores = np.where(covered, combined, 0.0).astype(np.float32)
        return pd.DataFrame(scores, index=frame.index, columns=self.target_channels)

    def _aggregate_scores(
        self,
        frame: pd.DataFrame,
        stride: int | None = None,
        window_starts: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None or self.scaler is None:
            raise RuntimeError("The TCN pipeline must be fitted before scoring.")

        self.model.eval()
        stride = stride or self.config.inference_stride
        if window_starts is None:
            effective_stride = max(1, min(stride, self.config.horizon))
            starts = self._build_window_starts(frame, stride=effective_stride, training=False)
        else:
            starts = np.asarray(window_starts, dtype=np.int64)
        base_features, raw_targets = self._transform_frame(frame)
        n_rows = len(frame)
        target_dim = len(self.target_channels)

        if self.cp is not None:
            xp = self.cp
            forecast_sum = xp.zeros((n_rows, target_dim), dtype=xp.float32)
            forecast_count = xp.zeros((n_rows, target_dim), dtype=xp.float32)
            recon_sum = xp.zeros((n_rows, target_dim), dtype=xp.float32)
            recon_count = xp.zeros((n_rows, target_dim), dtype=xp.float32)
        else:
            xp = np
            forecast_sum = xp.zeros((n_rows, target_dim), dtype=xp.float32)
            forecast_count = xp.zeros((n_rows, target_dim), dtype=xp.float32)
            recon_sum = xp.zeros((n_rows, target_dim), dtype=xp.float32)
            recon_count = xp.zeros((n_rows, target_dim), dtype=xp.float32)

        batch_size = self.config.batch_size
        with torch.no_grad():
            for batch_start in range(0, len(starts), batch_size):
                batch_indices = starts[batch_start : batch_start + batch_size]
                model_inputs, batch_targets, batch_histories = self._build_inference_batch(
                    base_features=base_features,
                    raw_targets=raw_targets,
                    window_starts=batch_indices,
                )
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    forecast, reconstruction, _ = self.model(model_inputs.to(self.device, non_blocking=True))
                    forecast_error_t = torch.abs(forecast - batch_targets.to(self.device, non_blocking=True)).float()
                    reconstruction_error_t = torch.abs(
                        reconstruction[:, -1, :] - batch_histories[:, -1, :].to(self.device, non_blocking=True)
                    ).float()

                if self.cp is not None:
                    forecast_error = self.cp.from_dlpack(forecast_error_t)
                    reconstruction_error = self.cp.from_dlpack(reconstruction_error_t)
                    anchor_indices = self.cp.asarray(batch_indices + self.config.sequence_length - 1, dtype=self.cp.int64)
                    future_indices = (
                        self.cp.asarray(batch_indices, dtype=self.cp.int64)[:, None]
                        + self.config.sequence_length
                        + self.cp.arange(self.config.horizon, dtype=self.cp.int64)[None, :]
                    )
                    valid_mask = future_indices < n_rows
                    for channel_index in range(target_dim):
                        self.cp.add.at(recon_sum[:, channel_index], anchor_indices, reconstruction_error[:, channel_index])
                        self.cp.add.at(recon_count[:, channel_index], anchor_indices, 1.0)
                        self.cp.add.at(
                            forecast_sum[:, channel_index],
                            future_indices[valid_mask],
                            forecast_error[:, :, channel_index][valid_mask],
                        )
                        self.cp.add.at(
                            forecast_count[:, channel_index],
                            future_indices[valid_mask],
                            1.0,
                        )
                else:
                    forecast_error = forecast_error_t.cpu().numpy()
                    reconstruction_error = reconstruction_error_t.cpu().numpy()
                    for row_index, start in enumerate(batch_indices):
                        anchor_index = int(start + self.config.sequence_length - 1)
                        recon_sum[anchor_index] += reconstruction_error[row_index]
                        recon_count[anchor_index] += 1.0

                        future_indices = range(
                            int(start + self.config.sequence_length),
                            int(start + self.config.sequence_length + self.config.horizon),
                        )
                        for horizon_offset, future_index in enumerate(future_indices):
                            if future_index >= n_rows:
                                break
                            forecast_sum[future_index] += forecast_error[row_index, horizon_offset]
                            forecast_count[future_index] += 1.0

        if self.cp is not None:
            forecast_scores = xp.where(forecast_count > 0, forecast_sum / xp.maximum(forecast_count, 1.0), 0.0)
            recon_scores = xp.where(recon_count > 0, recon_sum / xp.maximum(recon_count, 1.0), 0.0)
        else:
            forecast_scores = np.divide(
                forecast_sum,
                np.maximum(forecast_count, 1.0),
                out=np.zeros_like(forecast_sum),
                where=forecast_count > 0,
            )
            recon_scores = np.divide(
                recon_sum,
                np.maximum(recon_count, 1.0),
                out=np.zeros_like(recon_sum),
                where=recon_count > 0,
            )
        combined = (
            self.config.forecast_score_weight * forecast_scores
            + self.config.reconstruction_score_weight * recon_scores
        )
        covered = (forecast_count > 0) | (recon_count > 0)

        if self.cp is not None:
            combined = self.cp.asnumpy(combined)
            covered = self.cp.asnumpy(covered)
        return combined.astype(np.float32), covered.astype(bool)

    def predict(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        scores = self.score_sequence(frame)
        predictions = self._dynamic_threshold(scores)
        return scores, predictions

    def vectorize_window(self, window: pd.DataFrame) -> np.ndarray:
        return self.vectorize_windows([window])[0]

    def vectorize_windows(self, windows: list[pd.DataFrame] | np.ndarray) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("The TCN pipeline must be fitted before vectorizing windows.")
        if len(windows) == 0:
            embedding_dim = self.config.embedding_dim
            return np.zeros((0, embedding_dim), dtype=np.float32)

        self.model.eval()
        batch_size = max(1, self.config.batch_size)
        embeddings: list[np.ndarray] = []

        with torch.no_grad():
            for batch_start in range(0, len(windows), batch_size):
                batch_windows = windows[batch_start : batch_start + batch_size]
                if isinstance(batch_windows, np.ndarray):
                    window_values = np.asarray(batch_windows, dtype=np.float32)
                    if np.isnan(window_values).any():
                        fill_values = self.scaler.raw_mean.reshape(1, 1, -1)
                        window_values = np.where(np.isnan(window_values), fill_values, window_values)
                    flattened = window_values.reshape(-1, window_values.shape[-1])
                    raw_normalized = ((flattened - self.scaler.raw_mean) / self.scaler.raw_std).reshape(window_values.shape)
                    range_dt_seconds = np.float32(self.scaler.nominal_dt_seconds)
                    normalized_dt_value = np.float32((range_dt_seconds - self.scaler.dt_mean) / self.scaler.dt_std)
                    dt_template = np.full(
                        (len(window_values), window_values.shape[1], 1),
                        normalized_dt_value,
                        dtype=np.float32,
                    )
                    gap_template = np.zeros((len(window_values), window_values.shape[1], 1), dtype=np.float32)
                    base_features = np.concatenate([raw_normalized, dt_template, gap_template], axis=2).astype(np.float32)
                    mask_indicator = np.zeros((len(window_values), window_values.shape[1], 1), dtype=np.float32)
                    model_inputs = np.concatenate([base_features, mask_indicator], axis=2)
                else:
                    model_inputs_list: list[np.ndarray] = []
                    for window in batch_windows:
                        padded = self._pad_or_trim_window(window)
                        base_features, _ = self._transform_frame(padded)
                        mask_indicator = np.zeros((len(base_features), 1), dtype=np.float32)
                        model_inputs_list.append(np.concatenate([base_features, mask_indicator], axis=1))
                    model_inputs = np.asarray(model_inputs_list, dtype=np.float32)

                tensor = torch.from_numpy(np.asarray(model_inputs, dtype=np.float32)).float().to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    _, _, embedding = self.model(tensor)
                embeddings.append(embedding.cpu().numpy().astype(np.float32))

        return np.concatenate(embeddings, axis=0)

    def save(self, path: Path, metadata: dict[str, Any]) -> None:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Cannot save an unfitted TCN pipeline.")

        payload = {
            "config": asdict(self.config),
            "model_state": self.model.state_dict(),
            "scaler": {
                "raw_mean": self.scaler.raw_mean.tolist(),
                "raw_std": self.scaler.raw_std.tolist(),
                "dt_mean": self.scaler.dt_mean,
                "dt_std": self.scaler.dt_std,
                "nominal_dt_seconds": self.scaler.nominal_dt_seconds,
            },
            "global_thresholds": None if self.global_thresholds is None else self.global_thresholds.tolist(),
            "metadata": metadata,
        }
        torch.save(payload, path)

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        grad_scaler: torch.cuda.amp.GradScaler,
    ) -> float:
        assert self.model is not None
        self.model.train()
        losses: list[float] = []

        for batch in loader:
            inputs = batch["inputs"].to(self.device, non_blocking=True)
            forecast_targets = batch["forecast_targets"].to(self.device, non_blocking=True)
            reconstruction_targets = batch["reconstruction_targets"].to(self.device, non_blocking=True)
            reconstruction_mask = batch["reconstruction_mask"].to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                forecast, reconstruction, _ = self.model(inputs)
                loss = self._compute_loss(
                    forecast,
                    forecast_targets,
                    reconstruction,
                    reconstruction_targets,
                    reconstruction_mask,
                )
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            losses.append(float(loss.detach().cpu().item()))

        return float(np.mean(losses)) if losses else 0.0

    def _evaluate_epoch(self, loader: DataLoader) -> float:
        assert self.model is not None
        self.model.eval()
        losses: list[float] = []

        with torch.no_grad():
            for batch in loader:
                inputs = batch["inputs"].to(self.device, non_blocking=True)
                forecast_targets = batch["forecast_targets"].to(self.device, non_blocking=True)
                reconstruction_targets = batch["reconstruction_targets"].to(self.device, non_blocking=True)
                reconstruction_mask = batch["reconstruction_mask"].to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    forecast, reconstruction, _ = self.model(inputs)
                    loss = self._compute_loss(
                        forecast,
                        forecast_targets,
                        reconstruction,
                        reconstruction_targets,
                        reconstruction_mask,
                    )
                losses.append(float(loss.detach().cpu().item()))

        return float(np.mean(losses)) if losses else 0.0

    def _compute_loss(
        self,
        forecast: torch.Tensor,
        forecast_targets: torch.Tensor,
        reconstruction: torch.Tensor,
        reconstruction_targets: torch.Tensor,
        reconstruction_mask: torch.Tensor,
    ) -> torch.Tensor:
        forecast_loss = torch.mean((forecast - forecast_targets) ** 2)

        squared_error = (reconstruction - reconstruction_targets) ** 2
        masked_error = squared_error * reconstruction_mask
        if reconstruction_mask.sum() > 0:
            reconstruction_loss = masked_error.sum() / reconstruction_mask.sum()
        else:
            reconstruction_loss = squared_error.mean()

        return (
            self.config.forecast_loss_weight * forecast_loss
            + self.config.reconstruction_loss_weight * reconstruction_loss
        )

    def _fit_scaler(self, frame: pd.DataFrame) -> FeatureScaler:
        raw = frame[self.target_channels].to_numpy(dtype=np.float32)
        dt_seconds = self._delta_seconds(frame.index)
        positive_deltas = dt_seconds[dt_seconds > 0]
        return FeatureScaler(
            raw_mean=raw.mean(axis=0),
            raw_std=np.where(raw.std(axis=0) < EPSILON, 1.0, raw.std(axis=0)),
            dt_mean=float(dt_seconds.mean()),
            dt_std=float(max(dt_seconds.std(), EPSILON)),
            nominal_dt_seconds=float(np.median(positive_deltas)) if len(positive_deltas) > 0 else 0.0,
        )

    def _transform_frame(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        assert self.scaler is not None
        raw = frame[self.target_channels].to_numpy(dtype=np.float32)
        raw_normalized = (raw - self.scaler.raw_mean) / self.scaler.raw_std
        dt_seconds = self._delta_seconds(frame.index)
        dt_normalized = ((dt_seconds - self.scaler.dt_mean) / self.scaler.dt_std).reshape(-1, 1)
        gap_mask = self._gap_mask(frame.index, self.scaler.nominal_dt_seconds).reshape(-1, 1)
        base_features = np.concatenate([raw_normalized, dt_normalized, gap_mask], axis=1).astype(np.float32)
        return base_features, raw_normalized.astype(np.float32)

    def _gap_mask(self, index: pd.Index, nominal_dt_seconds: float) -> np.ndarray:
        if nominal_dt_seconds <= EPSILON:
            return np.zeros(len(index), dtype=np.float32)
        dt_seconds = self._delta_seconds(index)
        return (dt_seconds > (nominal_dt_seconds * 1.5)).astype(np.float32)

    def _build_window_starts(self, frame: pd.DataFrame, stride: int, training: bool) -> np.ndarray:
        max_start = len(frame) - self.config.sequence_length - self.config.horizon + 1
        if max_start <= 0:
            return np.empty(0, dtype=np.int64)

        starts = np.arange(0, max_start, stride, dtype=np.int64)
        if not training:
            return starts

        label_columns = [f"is_anomaly_{channel}" for channel in self.target_channels if f"is_anomaly_{channel}" in frame.columns]
        if not label_columns:
            return starts

        labels = frame[label_columns].to_numpy()
        valid_starts: list[int] = []
        for start in starts:
            stop = start + self.config.sequence_length + self.config.horizon
            window_labels = labels[start:stop]
            if ((window_labels == 1) | (window_labels == 3)).any():
                continue
            valid_starts.append(int(start))
        return np.asarray(valid_starts, dtype=np.int64)

    def _build_inference_batch(
        self,
        base_features: np.ndarray,
        raw_targets: np.ndarray,
        window_starts: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_indices = window_starts[:, None] + np.arange(self.config.sequence_length, dtype=np.int64)[None, :]
        future_indices = (
            window_starts[:, None] + self.config.sequence_length + np.arange(self.config.horizon, dtype=np.int64)[None, :]
        )
        histories = np.ascontiguousarray(raw_targets[history_indices])
        future_targets = np.ascontiguousarray(raw_targets[future_indices])
        mask_indicator = np.zeros((len(window_starts), self.config.sequence_length, 1), dtype=np.float32)
        model_inputs = np.ascontiguousarray(np.concatenate([base_features[history_indices], mask_indicator], axis=2))

        return (
            torch.from_numpy(model_inputs).float(),
            torch.from_numpy(future_targets).float(),
            torch.from_numpy(histories).float(),
        )

    def _dynamic_threshold(self, scores: pd.DataFrame) -> pd.DataFrame:
        thresholded = pd.DataFrame(index=scores.index)
        global_thresholds = self.global_thresholds if self.global_thresholds is not None else np.zeros(len(self.target_channels))

        for channel_index, channel in enumerate(self.target_channels):
            series = scores[channel].astype(np.float32)
            rolling_mean = series.rolling(
                self.config.score_smoothing_window,
                min_periods=1,
            ).mean()
            rolling_max = series.rolling(
                self.config.score_smoothing_window,
                min_periods=1,
            ).max()
            smoothed_series = (0.5 * (rolling_mean + rolling_max)).astype(np.float32)
            threshold_source = smoothed_series.shift(1)
            rolling_median = threshold_source.rolling(
                self.config.threshold_window,
                min_periods=max(16, self.config.threshold_window // 8),
            ).median()
            rolling_mad = (threshold_source - rolling_median).abs().rolling(
                self.config.threshold_window,
                min_periods=max(16, self.config.threshold_window // 8),
            ).median()
            dynamic_threshold = rolling_median + (self.config.threshold_std_factor * 1.4826 * rolling_mad.fillna(0.0))
            threshold = np.maximum(dynamic_threshold.fillna(global_thresholds[channel_index]), global_thresholds[channel_index])
            raw_prediction = (smoothed_series > threshold).astype(np.uint8).to_numpy(copy=True)
            thresholded[channel] = self._postprocess_prediction_runs(raw_prediction)

        return thresholded

    def _postprocess_prediction_runs(self, predictions: np.ndarray) -> np.ndarray:
        processed = np.asarray(predictions, dtype=np.uint8).copy()
        if processed.size == 0:
            return processed

        if self.config.max_gap_fill > 0:
            zero_start: int | None = None
            for index, value in enumerate(processed):
                if value == 0:
                    if zero_start is None:
                        zero_start = index
                    continue
                if zero_start is not None:
                    gap_length = index - zero_start
                    if zero_start > 0 and gap_length <= self.config.max_gap_fill and processed[zero_start - 1] == 1:
                        processed[zero_start:index] = 1
                    zero_start = None

        min_run = max(1, self.config.min_anomaly_run_length)
        run_start: int | None = None
        for index, value in enumerate(processed):
            if value == 1:
                if run_start is None:
                    run_start = index
                continue
            if run_start is not None:
                if index - run_start < min_run:
                    processed[run_start:index] = 0
                run_start = None
        if run_start is not None and processed.size - run_start < min_run:
            processed[run_start:] = 0

        return processed

    def _calibrate_thresholds(self, train_df: pd.DataFrame, train_starts: np.ndarray) -> np.ndarray:
        combined, covered = self._aggregate_scores(
            frame=train_df,
            stride=self.config.train_stride,
            window_starts=train_starts,
        )
        thresholds = np.zeros(len(self.target_channels), dtype=np.float32)
        for channel_index in range(len(self.target_channels)):
            valid_scores = combined[covered[:, channel_index], channel_index]
            if len(valid_scores) == 0:
                thresholds[channel_index] = 0.0
                continue
            thresholds[channel_index] = float(np.quantile(valid_scores, self.config.calibration_quantile))
        return thresholds

    def _pad_or_trim_window(self, window: pd.DataFrame) -> pd.DataFrame:
        target_window = window.copy()
        if "timestamp" in target_window.columns:
            target_window["timestamp"] = pd.to_datetime(target_window["timestamp"])
            target_window = target_window.set_index("timestamp")
        if not isinstance(target_window.index, pd.DatetimeIndex):
            target_window.index = pd.RangeIndex(len(target_window))

        target_length = self.config.sequence_length
        if len(target_window) > target_length:
            start = max((len(target_window) - target_length) // 2, 0)
            target_window = target_window.iloc[start : start + target_length].copy()
        elif len(target_window) < target_length:
            missing = target_length - len(target_window)
            if len(target_window) == 0:
                zero_frame = pd.DataFrame(
                    np.zeros((target_length, len(self.target_channels)), dtype=np.float32),
                    columns=self.target_channels,
                )
                return zero_frame
            pad_before = missing // 2
            pad_after = missing - pad_before
            front = target_window.iloc[[0]].copy()
            back = target_window.iloc[[-1]].copy()
            pieces = [front] * pad_before + [target_window] + [back] * pad_after
            target_window = pd.concat(pieces, axis=0)

        return target_window

    def _delta_seconds(self, index: pd.Index) -> np.ndarray:
        timestamps = pd.to_datetime(index)
        delta = timestamps.to_series().diff().dt.total_seconds().to_numpy(dtype=np.float32)
        if len(delta) == 0:
            return np.zeros(0, dtype=np.float32)
        if np.isnan(delta[0]):
            fallback = np.nanmedian(delta[1:]) if len(delta) > 1 else 0.0
            delta[0] = 0.0 if np.isnan(fallback) else fallback
        delta = np.nan_to_num(delta, nan=0.0)
        return delta.astype(np.float32)

    def _set_random_seed(self) -> None:
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_seed)

    def _try_load_cupy(self):
        if not self.config.use_cupy or self.device.type != "cuda":
            return None
        self._preload_cuda_libraries()
        sys.modules.pop("cupy", None)
        try:
            return importlib.import_module("cupy")
        except Exception:
            return None

    def _preload_cuda_libraries(self) -> None:
        candidate_roots = []
        for site_path in site.getsitepackages():
            candidate = Path(site_path) / "nvidia"
            if candidate.exists():
                candidate_roots.append(candidate)

        for root in candidate_roots:
            for lib_dir in root.glob("*/lib"):
                for library in sorted(lib_dir.glob("*.so*")):
                    try:
                        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        continue


DEFAULT_TARGET_CHANNELS = [
    "channel_41",
    "channel_42",
    "channel_43",
    "channel_44",
    "channel_45",
    "channel_46",
]

DEFAULT_AUTORESEARCH_SPLITS = ["10_months"]
DEFAULT_AUTORESEARCH_DETECTORS = ["tcn"]

DEFAULT_TCN_ARGS: dict[str, Any] = {
    "tcn_sequence_length": 128,
    "tcn_horizon": 8,
    "tcn_hidden_dim": 64,
    "tcn_embedding_dim": 64,
    "tcn_num_blocks": 5,
    "tcn_kernel_size": 3,
    "tcn_dropout": 0.05,
    "tcn_batch_size": 2048,
    "tcn_learning_rate": 5e-4,
    "tcn_weight_decay": 1e-4,
    "tcn_epochs": 4,
    "tcn_mask_ratio": 0.15,
    "tcn_train_stride": 8,
    "tcn_inference_stride": 16,
    "tcn_threshold_window": 288,
    "tcn_threshold_std_factor": 4.0,
    "tcn_calibration_quantile": 0.995,
    "tcn_score_smoothing_window": 5,
    "tcn_min_anomaly_run_length": 5,
    "tcn_max_gap_fill": 2,
    "tcn_device": "cuda",
    "tcn_dataloader_workers": 8,
    "tcn_preload_dataset": True,
    "tcn_preload_max_gb": 8.0,
    "tcn_training_wall_seconds": 900.0,
}

DEFAULT_MEMORY_ARGS: dict[str, Any] = {
    "half_window": 64,
    "memory_threshold": 0.92,
}

TCN_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "best_10m": {
        "10_months": {
            "tcn_batch_size": 2048,
            "tcn_horizon": 4,
            "tcn_hidden_dim": 64,
            "tcn_dropout": 0.05,
            "tcn_learning_rate": 3e-4,
            "tcn_weight_decay": 1e-4,
            "tcn_train_stride": 16,
            "tcn_inference_stride": 64,
            "tcn_threshold_std_factor": 5.0,
            "tcn_calibration_quantile": 0.999,
            "half_window": 32,
            "memory_threshold": 0.92,
        }
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Mission 1 lightweight subset benchmark.")
    parser.add_argument("--detectors", nargs="+", choices=["std", "tcn"], default=DEFAULT_AUTORESEARCH_DETECTORS)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_AUTORESEARCH_SPLITS)
    parser.add_argument("--target-channels", nargs="+", default=DEFAULT_TARGET_CHANNELS)
    parser.add_argument("--tol", type=float, default=5.0, help="STD threshold multiplier.")
    parser.add_argument("--half-window", type=int, default=64, help="Half-window size for memory matching.")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--memory-threshold", type=float, default=DEFAULT_MEMORY_ARGS["memory_threshold"])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-root", type=Path, default=Path("results/mission1_subset"))
    parser.add_argument("--tcn-preset", choices=["none", *TCN_PRESETS.keys()], default="none")
    parser.add_argument("--tcn-sequence-length", type=int, default=DEFAULT_TCN_ARGS["tcn_sequence_length"])
    parser.add_argument("--tcn-horizon", type=int, default=DEFAULT_TCN_ARGS["tcn_horizon"])
    parser.add_argument("--tcn-hidden-dim", type=int, default=DEFAULT_TCN_ARGS["tcn_hidden_dim"])
    parser.add_argument("--tcn-embedding-dim", type=int, default=DEFAULT_TCN_ARGS["tcn_embedding_dim"])
    parser.add_argument("--tcn-num-blocks", type=int, default=DEFAULT_TCN_ARGS["tcn_num_blocks"])
    parser.add_argument("--tcn-kernel-size", type=int, default=DEFAULT_TCN_ARGS["tcn_kernel_size"])
    parser.add_argument("--tcn-dropout", type=float, default=DEFAULT_TCN_ARGS["tcn_dropout"])
    parser.add_argument("--tcn-batch-size", type=int, default=DEFAULT_TCN_ARGS["tcn_batch_size"])
    parser.add_argument("--tcn-learning-rate", type=float, default=DEFAULT_TCN_ARGS["tcn_learning_rate"])
    parser.add_argument("--tcn-weight-decay", type=float, default=DEFAULT_TCN_ARGS["tcn_weight_decay"])
    parser.add_argument("--tcn-epochs", type=int, default=DEFAULT_TCN_ARGS["tcn_epochs"])
    parser.add_argument("--tcn-mask-ratio", type=float, default=DEFAULT_TCN_ARGS["tcn_mask_ratio"])
    parser.add_argument("--tcn-train-stride", type=int, default=DEFAULT_TCN_ARGS["tcn_train_stride"])
    parser.add_argument("--tcn-inference-stride", type=int, default=DEFAULT_TCN_ARGS["tcn_inference_stride"])
    parser.add_argument("--tcn-threshold-window", type=int, default=DEFAULT_TCN_ARGS["tcn_threshold_window"])
    parser.add_argument(
        "--tcn-threshold-std-factor",
        type=float,
        default=DEFAULT_TCN_ARGS["tcn_threshold_std_factor"],
    )
    parser.add_argument(
        "--tcn-calibration-quantile",
        type=float,
        default=DEFAULT_TCN_ARGS["tcn_calibration_quantile"],
    )
    parser.add_argument(
        "--tcn-score-smoothing-window",
        type=int,
        default=DEFAULT_TCN_ARGS["tcn_score_smoothing_window"],
    )
    parser.add_argument(
        "--tcn-min-anomaly-run-length",
        type=int,
        default=DEFAULT_TCN_ARGS["tcn_min_anomaly_run_length"],
    )
    parser.add_argument(
        "--tcn-max-gap-fill",
        type=int,
        default=DEFAULT_TCN_ARGS["tcn_max_gap_fill"],
    )
    parser.add_argument("--tcn-device", type=str, default=DEFAULT_TCN_ARGS["tcn_device"])
    parser.add_argument("--tcn-dataloader-workers", type=int, default=DEFAULT_TCN_ARGS["tcn_dataloader_workers"])
    parser.add_argument(
        "--tcn-preload-max-gb",
        type=float,
        default=DEFAULT_TCN_ARGS["tcn_preload_max_gb"],
        help="Preload train/val tensors to GPU when they fit within this budget.",
    )
    parser.add_argument("--tcn-no-amp", action="store_true", help="Disable mixed precision for TCN training.")
    parser.add_argument("--tcn-no-cupy", action="store_true", help="Disable CuPy acceleration for TCN inference aggregation.")
    parser.add_argument("--tcn-no-preload", action="store_true", help="Keep training tensors on CPU even if they fit on the GPU.")
    parser.add_argument(
        "--tcn-training-wall-seconds",
        type=float,
        default=DEFAULT_TCN_ARGS["tcn_training_wall_seconds"],
        help="Wall-clock cap for TCN training (seconds). Stops before starting a new epoch when exceeded.",
    )
    parser.add_argument(
        "--experiment-description",
        type=str,
        default="",
        help="Short machine-readable description of the current experiment.",
    )
    parser.add_argument(
        "--experiment-decision",
        choices=["candidate", "keep", "discard"],
        default="candidate",
        help="Current experiment decision for machine-readable logs.",
    )
    parser.add_argument(
        "--experiment-tag",
        type=str,
        default="",
        help="Optional tag for grouping related autonomous runs.",
    )
    return parser.parse_args()


def resolve_split_parameters(args: argparse.Namespace, split: str) -> dict[str, Any]:
    resolved = vars(args).copy()
    if args.tcn_preset == "none":
        return resolved

    preset = TCN_PRESETS[args.tcn_preset].get(split)
    if preset is None:
        return resolved

    for key, value in preset.items():
        if key in DEFAULT_TCN_ARGS:
            if resolved[key] == DEFAULT_TCN_ARGS[key]:
                resolved[key] = value
            continue
        if key in DEFAULT_MEMORY_ARGS and resolved[key] == DEFAULT_MEMORY_ARGS[key]:
            resolved[key] = value
    return resolved


def build_tcn_config(args: argparse.Namespace, split: str) -> TcnTrainingConfig:
    resolved = resolve_split_parameters(args, split)
    return TcnTrainingConfig(
        sequence_length=resolved["tcn_sequence_length"],
        horizon=resolved["tcn_horizon"],
        hidden_dim=resolved["tcn_hidden_dim"],
        embedding_dim=resolved["tcn_embedding_dim"],
        num_blocks=resolved["tcn_num_blocks"],
        kernel_size=resolved["tcn_kernel_size"],
        dropout=resolved["tcn_dropout"],
        batch_size=resolved["tcn_batch_size"],
        learning_rate=resolved["tcn_learning_rate"],
        weight_decay=resolved["tcn_weight_decay"],
        epochs=resolved["tcn_epochs"],
        mask_ratio=resolved["tcn_mask_ratio"],
        train_stride=resolved["tcn_train_stride"],
        inference_stride=resolved["tcn_inference_stride"],
        threshold_window=resolved["tcn_threshold_window"],
        threshold_std_factor=resolved["tcn_threshold_std_factor"],
        calibration_quantile=resolved["tcn_calibration_quantile"],
        score_smoothing_window=resolved["tcn_score_smoothing_window"],
        min_anomaly_run_length=resolved["tcn_min_anomaly_run_length"],
        max_gap_fill=resolved["tcn_max_gap_fill"],
        device=resolved["tcn_device"],
        dataloader_workers=resolved["tcn_dataloader_workers"],
        mixed_precision=not args.tcn_no_amp,
        use_cupy=not args.tcn_no_cupy,
        preload_dataset=not args.tcn_no_preload,
        preload_max_gb=resolved["tcn_preload_max_gb"],
        training_wall_seconds=resolved["tcn_training_wall_seconds"],
    )


def prune_short_isolated_runs(
    predictions: pd.DataFrame,
    target_channels: list[str],
    min_run_points: int,
    support_padding: int,
) -> pd.DataFrame:
    pruned = predictions.copy()
    values = pruned[target_channels].to_numpy(dtype=np.uint8, copy=True)

    for channel_index, channel in enumerate(target_channels):
        series = values[:, channel_index].copy()
        run_start: int | None = None
        for index, value in enumerate(series):
            if value == 1 and run_start is None:
                run_start = index
                continue
            if value == 1:
                continue
            if run_start is None:
                continue
            run_stop = index
            run_length = run_stop - run_start
            if run_length < min_run_points:
                support_start = max(0, run_start - support_padding)
                support_stop = min(len(series), run_stop + support_padding)
                support = values[support_start:support_stop].copy()
                support[:, channel_index] = 0
                if not support.any():
                    series[run_start:run_stop] = 0
            run_start = None
        if run_start is not None:
            run_stop = len(series)
            run_length = run_stop - run_start
            if run_length < min_run_points:
                support_start = max(0, run_start - support_padding)
                support = values[support_start:run_stop].copy()
                support[:, channel_index] = 0
                if not support.any():
                    series[run_start:run_stop] = 0
        pruned[channel] = series

    return pruned


def merge_supported_close_runs(
    predictions: pd.DataFrame,
    target_channels: list[str],
    max_gap_points: int,
    support_padding: int,
) -> pd.DataFrame:
    merged = predictions.copy()
    values = merged[target_channels].to_numpy(dtype=np.uint8, copy=True)

    for channel_index, channel in enumerate(target_channels):
        series = values[:, channel_index].copy()
        zero_start: int | None = None
        for index, value in enumerate(series):
            if value == 0:
                if zero_start is None:
                    zero_start = index
                continue
            if zero_start is None:
                continue
            gap_length = index - zero_start
            if zero_start > 0 and gap_length <= max_gap_points and series[zero_start - 1] == 1:
                support_start = max(0, zero_start - support_padding)
                support_stop = min(len(series), index + support_padding)
                support = values[support_start:support_stop].copy()
                support[:, channel_index] = 0
                if support.any():
                    series[zero_start:index] = 1
            zero_start = None
        merged[channel] = series

    return merged


def prune_weak_isolated_runs(
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
    target_channels: list[str],
    max_run_points: int,
    support_padding: int,
    peak_quantile: float,
    density_quantile: float,
) -> pd.DataFrame:
    pruned = predictions.copy()
    prediction_values = pruned[target_channels].to_numpy(dtype=np.uint8, copy=True)
    score_values = scores[target_channels].to_numpy(dtype=np.float32, copy=False)

    for channel_index, channel in enumerate(target_channels):
        series = prediction_values[:, channel_index].copy()
        channel_scores = score_values[:, channel_index]
        runs: list[tuple[int, int, int, float, float]] = []
        run_start: int | None = None

        for index, value in enumerate(series):
            if value == 1:
                if run_start is None:
                    run_start = index
                continue
            if run_start is None:
                continue
            run_stop = index
            segment = channel_scores[run_start:run_stop]
            run_length = run_stop - run_start
            runs.append((run_start, run_stop, run_length, float(segment.max()), float(segment.mean())))
            run_start = None

        if run_start is not None:
            run_stop = len(series)
            segment = channel_scores[run_start:run_stop]
            run_length = run_stop - run_start
            runs.append((run_start, run_stop, run_length, float(segment.max()), float(segment.mean())))

        if not runs:
            continue

        peak_cutoff = float(np.quantile([run[3] for run in runs], peak_quantile))
        density_cutoff = float(np.quantile([run[4] for run in runs], density_quantile))

        for run_start, run_stop, run_length, peak_score, mean_score in runs:
            if run_length > max_run_points:
                continue
            if peak_score > peak_cutoff or mean_score > density_cutoff:
                continue
            support_start = max(0, run_start - support_padding)
            support_stop = min(len(series), run_stop + support_padding)
            support = prediction_values[support_start:support_stop].copy()
            support[:, channel_index] = 0
            if support.any():
                continue
            series[run_start:run_stop] = 0

        pruned[channel] = series

    return pruned


def extend_high_confidence_run_edges(
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
    target_channels: list[str],
    global_thresholds: np.ndarray,
    min_run_peak_ratio: float,
    extension_score_ratio: float,
    max_extension_points: int,
) -> pd.DataFrame:
    extended = predictions.copy()
    prediction_values = extended[target_channels].to_numpy(dtype=np.uint8, copy=True)
    score_values = scores[target_channels].to_numpy(dtype=np.float32, copy=False)
    thresholds = np.asarray(global_thresholds, dtype=np.float32)

    for channel_index, channel in enumerate(target_channels):
        series = prediction_values[:, channel_index].copy()
        channel_scores = score_values[:, channel_index]
        threshold = max(float(thresholds[channel_index]), EPSILON)
        index = 0

        while index < len(series):
            if series[index] != 1:
                index += 1
                continue

            run_start = index
            while index < len(series) and series[index] == 1:
                index += 1
            run_stop = index

            run_peak = float(channel_scores[run_start:run_stop].max())
            if run_peak < (threshold * min_run_peak_ratio):
                continue

            extension_floor = threshold * extension_score_ratio

            left = run_start
            while left > 0 and (run_start - left) < max_extension_points:
                if series[left - 1] == 1 or channel_scores[left - 1] < extension_floor:
                    break
                left -= 1

            right = run_stop
            while right < len(series) and (right - run_stop) < max_extension_points:
                if series[right] == 1 or channel_scores[right] < extension_floor:
                    break
                right += 1

            series[left:right] = 1
            index = max(index, right)

        extended[channel] = series

    return extended


def expand_prediction_run_boundaries(
    predictions: pd.DataFrame,
    target_channels: list[str],
    pre_points: int,
    post_points: int,
) -> pd.DataFrame:
    expanded = predictions.copy()
    prediction_values = expanded[target_channels].to_numpy(dtype=np.uint8, copy=True)

    for channel_index, channel in enumerate(target_channels):
        series = prediction_values[:, channel_index].copy()
        index = 0

        while index < len(series):
            if series[index] != 1:
                index += 1
                continue

            run_start = index
            while index < len(series) and series[index] == 1:
                index += 1
            run_stop = index

            expand_start = max(0, run_start - max(0, pre_points))
            expand_stop = min(len(series), run_stop + max(0, post_points))
            series[expand_start:expand_stop] = 1

        expanded[channel] = series

    return expanded


def prune_noisy_channel_short_runs(
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
    target_channels: list[str],
    global_thresholds: np.ndarray,
    support_padding: int,
    noisy_run_median_threshold: float,
    noisy_peak_ratio_median_threshold: float,
    min_run_points: int,
    max_short_run_peak_ratio: float,
) -> pd.DataFrame:
    pruned = predictions.copy()
    prediction_values = pruned[target_channels].to_numpy(dtype=np.uint8, copy=True)
    score_values = scores[target_channels].to_numpy(dtype=np.float32, copy=False)
    thresholds = np.asarray(global_thresholds, dtype=np.float32)

    for channel_index, channel in enumerate(target_channels):
        series = prediction_values[:, channel_index].copy()
        channel_scores = score_values[:, channel_index]
        threshold = max(float(thresholds[channel_index]), EPSILON)
        runs: list[tuple[int, int, int, float]] = []
        run_start: int | None = None

        for index, value in enumerate(series):
            if value == 1:
                if run_start is None:
                    run_start = index
                continue
            if run_start is None:
                continue
            run_stop = index
            segment = channel_scores[run_start:run_stop]
            runs.append((run_start, run_stop, run_stop - run_start, float(segment.max()) / threshold))
            run_start = None

        if run_start is not None:
            run_stop = len(series)
            segment = channel_scores[run_start:run_stop]
            runs.append((run_start, run_stop, run_stop - run_start, float(segment.max()) / threshold))

        if not runs:
            continue

        median_run_length = float(np.median([run[2] for run in runs]))
        median_peak_ratio = float(np.median([run[3] for run in runs]))
        if median_run_length > noisy_run_median_threshold or median_peak_ratio > noisy_peak_ratio_median_threshold:
            continue

        for run_start, run_stop, run_length, peak_ratio in runs:
            if run_length >= min_run_points or peak_ratio > max_short_run_peak_ratio:
                continue
            support_start = max(0, run_start - support_padding)
            support_stop = min(len(series), run_stop + support_padding)
            support = prediction_values[support_start:support_stop].copy()
            support[:, channel_index] = 0
            if support.any():
                continue
            series[run_start:run_stop] = 0

        pruned[channel] = series

    return pruned


def restore_consensus_score_segments(
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
    target_channels: list[str],
    global_thresholds: np.ndarray,
    min_consensus_channels: int,
    support_score_ratio: float,
    anchor_score_ratio: float,
    min_segment_points: int,
    max_gap_points: int,
    pre_points: int,
    post_points: int,
) -> pd.DataFrame:
    restored = predictions.copy()
    prediction_values = restored[target_channels].to_numpy(dtype=np.uint8, copy=True)
    score_values = scores[target_channels].to_numpy(dtype=np.float32, copy=False)
    thresholds = np.maximum(np.asarray(global_thresholds, dtype=np.float32), EPSILON)
    ratio_values = score_values / thresholds[None, :]

    consensus = (ratio_values >= support_score_ratio).sum(axis=1).astype(np.uint8)
    mask = consensus >= min_consensus_channels
    if max_gap_points > 0:
        zero_start: int | None = None
        for index, value in enumerate(mask):
            if value:
                if zero_start is not None and zero_start > 0 and (index - zero_start) <= max_gap_points and mask[zero_start - 1]:
                    mask[zero_start:index] = True
                zero_start = None
                continue
            if zero_start is None:
                zero_start = index

    segment_start: int | None = None
    for index, value in enumerate(mask):
        if value and segment_start is None:
            segment_start = index
            continue
        if value:
            continue
        if segment_start is None:
            continue
        segment_stop = index
        if (segment_stop - segment_start) >= min_segment_points:
            segment_ratios = ratio_values[segment_start:segment_stop]
            channel_peaks = segment_ratios.max(axis=0)
            if (channel_peaks >= anchor_score_ratio).any():
                restore_channels = channel_peaks >= support_score_ratio
                if int(restore_channels.sum()) >= min_consensus_channels:
                    start = max(0, segment_start - pre_points)
                    stop = min(len(mask), segment_stop + post_points)
                    prediction_values[start:stop, restore_channels] = 1
        segment_start = None

    if segment_start is not None and (len(mask) - segment_start) >= min_segment_points:
        segment_ratios = ratio_values[segment_start:]
        channel_peaks = segment_ratios.max(axis=0)
        if (channel_peaks >= anchor_score_ratio).any():
            restore_channels = channel_peaks >= support_score_ratio
            if int(restore_channels.sum()) >= min_consensus_channels:
                start = max(0, segment_start - pre_points)
                prediction_values[start:, restore_channels] = 1

    for channel_index, channel in enumerate(target_channels):
        restored[channel] = prediction_values[:, channel_index]
    return restored


def apply_same_channel_memory_gating(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    target_channels: list[str],
    memory_bank: RareNominalMemoryBank,
    half_window: int,
    metric: str,
    threshold: float,
    vectorizer: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gated_predictions = predictions.copy()
    suppressed_frames: list[pd.DataFrame] = []

    for channel in target_channels:
        channel_prototypes = [prototype for prototype in memory_bank.prototypes if prototype.channel == channel]
        if not channel_prototypes:
            continue
        channel_bank = RareNominalMemoryBank(channel_prototypes)
        channel_predictions = predictions.copy()
        for other_channel in target_channels:
            if other_channel != channel:
                channel_predictions[other_channel] = 0
        channel_gated, channel_suppressed = apply_memory_gating(
            frame=frame,
            predictions=channel_predictions,
            target_channels=target_channels,
            memory_bank=channel_bank,
            half_window=half_window,
            metric=metric,
            threshold=threshold,
            vectorizer=vectorizer,
        )
        gated_predictions[channel] = channel_gated[channel].astype(np.uint8)
        if not channel_suppressed.empty:
            suppressed_frames.append(channel_suppressed)

    if suppressed_frames:
        suppressed_events = pd.concat(suppressed_frames, ignore_index=True)
    else:
        suppressed_events = pd.DataFrame(columns=["channel", "start_time", "end_time", "prototype_id", "score", "metric"])
    return gated_predictions, suppressed_events


def run_tcn_split(
    args: argparse.Namespace,
    split: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_labels: pd.DataFrame,
    test_labels: pd.DataFrame,
) -> dict[str, Any]:
    resolved_args = resolve_split_parameters(args, split)
    tcn_config = build_tcn_config(args, split)
    pipeline = TcnAnomalyPipeline(target_channels=args.target_channels, config=tcn_config)
    training_summary = pipeline.fit(train_df)

    log_debug(f"[tcn] scoring test split '{split}'")
    baseline_scores, baseline_predictions = pipeline.predict(test_df)
    baseline_predictions = prune_short_isolated_runs(
        predictions=baseline_predictions,
        target_channels=args.target_channels,
        min_run_points=20,
        support_padding=8,
    )
    baseline_predictions = merge_supported_close_runs(
        predictions=baseline_predictions,
        target_channels=args.target_channels,
        max_gap_points=8,
        support_padding=8,
    )
    baseline_predictions = prune_weak_isolated_runs(
        predictions=baseline_predictions,
        scores=baseline_scores,
        target_channels=args.target_channels,
        max_run_points=12,
        support_padding=8,
        peak_quantile=0.35,
        density_quantile=0.35,
    )
    baseline_predictions = extend_high_confidence_run_edges(
        predictions=baseline_predictions,
        scores=baseline_scores,
        target_channels=args.target_channels,
        global_thresholds=pipeline.global_thresholds,
        min_run_peak_ratio=1.15,
        extension_score_ratio=0.8,
        max_extension_points=6,
    )
    baseline_predictions = expand_prediction_run_boundaries(
        predictions=baseline_predictions,
        target_channels=args.target_channels,
        pre_points=1,
        post_points=0,
    )
    baseline_predictions = prune_noisy_channel_short_runs(
        predictions=baseline_predictions,
        scores=baseline_scores,
        target_channels=args.target_channels,
        global_thresholds=pipeline.global_thresholds,
        support_padding=8,
        noisy_run_median_threshold=8.0,
        noisy_peak_ratio_median_threshold=1.2,
        min_run_points=6,
        max_short_run_peak_ratio=1.35,
    )
    baseline_predictions = restore_consensus_score_segments(
        predictions=baseline_predictions,
        scores=baseline_scores,
        target_channels=args.target_channels,
        global_thresholds=pipeline.global_thresholds,
        min_consensus_channels=3,
        support_score_ratio=1.15,
        anchor_score_ratio=1.6,
        min_segment_points=4,
        max_gap_points=2,
        pre_points=1,
        post_points=0,
    )
    baseline_dir = args.results_root / "tcn_baseline" / split
    baseline_dir.mkdir(parents=True, exist_ok=True)
    log_debug(f"[tcn] saving model for '{split}'")
    pipeline.save(baseline_dir / "model.pt", metadata={"split": split, "target_channels": args.target_channels})
    log_debug(f"[tcn] writing training summary for '{split}'")
    write_json(baseline_dir / "training.json", training_summary)

    log_debug(f"[tcn] building memory bank for '{split}'")
    memory_bank = RareNominalMemoryBank.from_labeled_rare_events(
        frame=train_df,
        labels=train_labels,
        target_channels=args.target_channels,
        half_window=resolved_args["half_window"],
        vectorizer=pipeline.vectorize_windows,
    )
    log_debug(f"[tcn] applying memory gating for '{split}'")
    gated_predictions, suppressed_events = apply_same_channel_memory_gating(
        frame=test_df,
        predictions=baseline_predictions,
        target_channels=args.target_channels,
        memory_bank=memory_bank,
        half_window=resolved_args["half_window"],
        metric=args.metric,
        threshold=resolved_args["memory_threshold"],
        vectorizer=pipeline.vectorize_windows,
    )

    log_debug(f"[tcn] computing baseline ESA metrics for '{split}'")
    baseline_metrics = compute_esa_metrics(test_labels, baseline_predictions)
    log_debug(f"[tcn] computing memory ESA metrics for '{split}'")
    memory_metrics = compute_esa_metrics(test_labels, gated_predictions)
    log_debug(f"[tcn] summarizing suppressions for '{split}'")
    suppression_summary = summarize_suppressions(test_labels, suppressed_events)

    log_debug(f"[tcn] writing final results for '{split}'")
    save_detector_results(
        detector_name="tcn",
        split=split,
        results_root=args.results_root,
        baseline_scores=baseline_scores,
        baseline_predictions=baseline_predictions,
        baseline_metrics=baseline_metrics,
        baseline_parameters={
            "target_channels": args.target_channels,
            "config": asdict(tcn_config),
            "training": training_summary,
            "preset": args.tcn_preset,
        },
        memory_bank=memory_bank,
        gated_predictions=gated_predictions,
        memory_metrics=memory_metrics,
        memory_parameters={
            "target_channels": args.target_channels,
            "half_window": resolved_args["half_window"],
            "metric": args.metric,
            "memory_threshold": resolved_args["memory_threshold"],
            "memory_size": len(memory_bank.prototypes),
            "config": asdict(tcn_config),
            "preset": args.tcn_preset,
        },
        suppression_summary=suppression_summary,
        suppressed_events=suppressed_events,
    )
    return summarize_detector_run("tcn", split, memory_bank, suppression_summary, baseline_metrics, memory_metrics)


def run_split(args: argparse.Namespace, split: str) -> list[dict[str, Any]]:
    train_df, test_df, train_labels, test_labels = load_split_data(args, split)
    rows: list[dict[str, Any]] = []
    if "std" in args.detectors:
        rows.append(run_std_split(args, split, train_df, test_df, train_labels, test_labels))
    if "tcn" in args.detectors:
        rows.append(run_tcn_split(args, split, train_df, test_df, train_labels, test_labels))
    return rows


def _mean_primary_metric(summary_rows: list[dict[str, Any]]) -> float | None:
    tcn_rows = [row for row in summary_rows if row.get("detector") == "tcn"]
    candidates = tcn_rows if tcn_rows else summary_rows
    values: list[float] = []
    for row in candidates:
        if PRIMARY_METRIC_KEY in row and row[PRIMARY_METRIC_KEY] is not None:
            values.append(float(row[PRIMARY_METRIC_KEY]))
    if not values:
        return None
    return float(sum(values) / len(values))


def _append_experiment_log(results_root: Path, payload: dict[str, Any]) -> None:
    path = results_root / "experiment_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(inner) for inner in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _collect_git_metadata(repo_root: Path) -> dict[str, Any]:
    def run_git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return ""
        return completed.stdout.strip()

    inside = run_git("rev-parse", "--is-inside-work-tree")
    if inside != "true":
        return {"is_repo": False}

    branch = run_git("rev-parse", "--abbrev-ref", "HEAD")
    commit = run_git("rev-parse", "--short", "HEAD")
    status = run_git("status", "--short")
    return {
        "is_repo": True,
        "branch": branch,
        "commit": commit,
        "is_dirty": bool(status),
    }


def _build_run_payload(
    args: argparse.Namespace,
    rm_snap: dict[str, Any],
    mean_primary: float | None,
    run_status: str,
    started_at: str,
    elapsed_seconds: float,
    error: Exception | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "elapsed_seconds": float(elapsed_seconds),
        "run_status": run_status,
        "decision": args.experiment_decision,
        "description": args.experiment_description,
        "tag": args.experiment_tag,
        "detectors": list(args.detectors),
        "splits": list(args.splits),
        "direction": PRIMARY_METRIC_DIRECTION,
        "primary_metric_key": PRIMARY_METRIC_KEY,
        "primary_metric_mean": mean_primary,
        "reading_materials": rm_snap,
        "artifacts": {
            "summary_csv": args.results_root / "summary.csv",
            "reading_materials_snapshot": args.results_root / "reading_materials_snapshot.json",
            "run_summary_json": args.results_root / "run_summary.json",
        },
        "cli": {
            "argv": sys.argv,
            "python_executable": sys.executable,
            "args": vars(args),
        },
        "git": _collect_git_metadata(Path(__file__).resolve().parent),
    }
    if error is not None:
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    return _to_jsonable(payload)


def main() -> None:
    args = parse_args()
    args.results_root.mkdir(parents=True, exist_ok=True)
    run_started_at = datetime.now(timezone.utc).isoformat()
    run_started_perf = time.perf_counter()

    rm_snap = reading_materials_snapshot()
    log_debug(
        f"[reading_materials] dir={READING_MATERIALS_DIR} indexed={rm_snap['count']} paper file(s)"
    )
    write_json(args.results_root / "reading_materials_snapshot.json", rm_snap)
    try:
        summary_rows: list[dict[str, Any]] = []
        for split in args.splits:
            summary_rows.extend(run_split(args, split))
        log_debug(f"[summary] writing summary.csv to '{args.results_root}'")
        pd.DataFrame(summary_rows).to_csv(args.results_root / "summary.csv", index=False)

        mean_primary = _mean_primary_metric(summary_rows)
        payload = _build_run_payload(
            args=args,
            rm_snap=rm_snap,
            mean_primary=mean_primary,
            run_status="success",
            started_at=run_started_at,
            elapsed_seconds=time.perf_counter() - run_started_perf,
        )
        write_json(args.results_root / "run_summary.json", payload)
        _append_experiment_log(args.results_root, payload)

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
        payload = _build_run_payload(
            args=args,
            rm_snap=rm_snap,
            mean_primary=None,
            run_status="crash",
            started_at=run_started_at,
            elapsed_seconds=time.perf_counter() - run_started_perf,
            error=exc,
        )
        write_json(args.results_root / "run_summary.json", payload)
        _append_experiment_log(args.results_root, payload)
        print(f"run_status=crash run_summary_json={args.results_root / 'run_summary.json'}", flush=True)
        raise


if __name__ == "__main__":
    main()
