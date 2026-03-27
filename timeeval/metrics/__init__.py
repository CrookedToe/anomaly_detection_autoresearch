"""TimeEval metrics: submodules load on demand (lean import path for ESA-ADB scoring)."""

from __future__ import annotations

import importlib
from typing import Any

from .metric import Metric

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "AucMetric": (".auc_metrics", "AucMetric"),
    "RocAUC": (".auc_metrics", "RocAUC"),
    "PrAUC": (".auc_metrics", "PrAUC"),
    "Precision": (".classification_metrics", "Precision"),
    "Recall": (".classification_metrics", "Recall"),
    "F1Score": (".classification_metrics", "F1Score"),
    "ADTQC": (".latency_metrics", "ADTQC"),
    "AveragePrecision": (".other_metrics", "AveragePrecision"),
    "PrecisionAtK": (".other_metrics", "PrecisionAtK"),
    "FScoreAtK": (".other_metrics", "FScoreAtK"),
    "RangePrecisionRangeRecallAUC": (".range_metrics", "RangePrecisionRangeRecallAUC"),
    "RangePrecision": (".range_metrics", "RangePrecision"),
    "RangeRecall": (".range_metrics", "RangeRecall"),
    "RangeFScore": (".range_metrics", "RangeFScore"),
    "eTaPR_Fscore": (".range_metrics", "eTaPR_Fscore"),
    "eTaPR_PR_AUC": (".range_metrics", "eTaPR_PR_AUC"),
    "point_adjust_PR_AUC": (".range_metrics", "point_adjust_PR_AUC"),
    "MultiChannelMetric": (".ranking_metrics", "MultiChannelMetric"),
    "ChannelAwareFScore": (".ranking_metrics", "ChannelAwareFScore"),
    "DcVaeAnomalyScoring": (".thresholding", "DcVaeAnomalyScoring"),
    "DcVaeThresholding": (".thresholding", "DcVaeThresholding"),
    "ThresholdingStrategy": (".thresholding", "ThresholdingStrategy"),
    "PercentileThresholding": (".thresholding", "PercentileThresholding"),
    "TelemanomThresholding": (".thresholding", "TelemanomThresholding"),
    "RangePrAUC": (".vus_metrics", "RangePrAUC"),
    "RangeRocAUC": (".vus_metrics", "RangeRocAUC"),
    "RangePrVUS": (".vus_metrics", "RangePrVUS"),
    "RangeRocVUS": (".vus_metrics", "RangeRocVUS"),
    "ESAScores": (".ESA_ADB_metrics", "ESAScores"),
}


def __getattr__(name: str) -> Any:
    if name == "DefaultMetrics":
        from .default_metrics_bundle import DefaultMetrics

        return DefaultMetrics
    if name in _LAZY_ATTRS:
        mod_path, attr = _LAZY_ATTRS[name]
        module = importlib.import_module(mod_path, __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__))


__all__ = [
    "Metric",
    "DefaultMetrics",
    *_LAZY_ATTRS.keys(),
]
