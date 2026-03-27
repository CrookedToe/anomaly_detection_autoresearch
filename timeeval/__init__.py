"""
TimeEval (vendored). This repository uses only the metrics subtree for ESA-ADB scoring.

The full evaluation framework (TimeEval class, Docker adapters, etc.) is not loaded from this
package root. Import metrics directly, e.g. ``from timeeval.metrics import ESAScores``, or use
submodules under ``timeeval.metrics.*``.
"""

from ._version import __version__

__all__ = ["__version__"]
