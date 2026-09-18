"""Cross-correlation utilities for H1/L1 strain and residuals."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def lag_correlation(x: np.ndarray, y: np.ndarray, fs: float,
                    max_lag_s: float = 0.020) -> Tuple[np.ndarray, np.ndarray]:
    """Normalized cross-correlation of x and y on +/- max_lag_s."""
    n = len(x)
    x = (x - x.mean()) / (x.std() + 1e-30)
    y = (y - y.mean()) / (y.std() + 1e-30)
    full = np.correlate(x, y, mode="full") / n
    lags = (np.arange(full.size) - (n - 1)) / fs
    mask = np.abs(lags) <= max_lag_s
    return lags[mask], full[mask]


def residual(strain: np.ndarray, template: np.ndarray) -> np.ndarray:
    n = min(len(strain), len(template))
    return strain[:n] - template[:n]

