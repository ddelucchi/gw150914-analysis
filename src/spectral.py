"""Spectral utilities: PSDs and Q-transforms via GWpy."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def qtransform(strain_ts, cfg: Dict[str, Any]):
    """Compute the GW150914 chirp Q-transform per the GWpy example."""
    qcfg = cfg["qtransform"]
    return strain_ts.q_transform(
        outseg=tuple(qcfg["outseg_gps"]),
        qrange=tuple(qcfg["q_range"]),
        frange=tuple(qcfg["frange_hz"]),
        whiten=True,
    )


def psd_no_window(seg: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Naive periodogram with no window — for the leakage demo."""
    from scipy.signal import periodogram
    return periodogram(seg, fs=fs, window="boxcar", scaling="density")


def psd_tukey(seg: np.ndarray, fs: float, alpha: float = 0.25):
    from scipy.signal import periodogram
    from scipy.signal.windows import tukey
    win = tukey(len(seg), alpha=alpha)
    return periodogram(seg, fs=fs, window=win, scaling="density")

