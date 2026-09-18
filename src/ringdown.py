"""Damped-sinusoid ringdown guide curve for the late-time inset."""
from __future__ import annotations

import numpy as np


def damped_sinusoid(t: np.ndarray, t0: float, f_hz: float, tau_s: float,
                    amp: float = 1.0, phi: float = 0.0) -> np.ndarray:
    """A0 * exp(-(t-t0)/tau) * cos(2 pi f (t-t0) + phi), zero before t0."""
    dt = t - t0
    out = amp * np.exp(-dt / tau_s) * np.cos(2 * np.pi * f_hz * dt + phi)
    out[dt < 0] = 0.0
    return out

