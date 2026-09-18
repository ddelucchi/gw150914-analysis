"""Maximum-likelihood waveform construction.

Preferred path: PESummary's documented GW150914 workflow that builds an
IMRPhenomPv2 waveform from the bilby PE samples.  Where PESummary /
lalsimulation are unavailable, fall back to a SciPy TaylorF2 inspiral
parameterized by the GWTC-1 max-L (or median) component masses read from
`GW150914_GWTC-1.hdf5`.  The fallback is clearly labelled in any caption
that consumes it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def load_pesummary_samples(path: Path):
    import pandas as pd
    return pd.read_csv(path, sep=r"\s+", engine="python")


def _max_l_row(df) -> Dict[str, float]:
    cand = [c for c in ("log_likelihood", "logL", "deltalogl") if c in df.columns]
    if not cand:
        raise KeyError("No log-likelihood column in PESummary samples")
    idx = df[cand[0]].idxmax()
    return df.loc[idx].to_dict()


def _try_pesummary(samples_path: Path, cfg, t_center_gps, ifo):
    df = load_pesummary_samples(samples_path)
    p = _max_l_row(df)
    from pesummary.gw.waveform import td_waveform
    fs = float(cfg["data"]["master_sample_rate"])
    h = td_waveform(p, cfg["maxl"]["approximant"],
                    delta_t=1.0 / fs, f_low=20.0, project=ifo)
    times = np.asarray(h.sample_times) + t_center_gps
    return times, np.asarray(h)


def _gwtc1_max_l_masses(cfg) -> Tuple[float, float]:
    from src.config import REPO_ROOT
    from src.posterior_plots import load_gwtc1_samples
    h5 = (REPO_ROOT / cfg["paths"]["posterior_dir"]
          / cfg["posterior"]["gwtc1_filename"])
    if not h5.exists():
        return 36.0, 29.0
    s = load_gwtc1_samples(h5)
    if ("log_likelihood" in s and "m1_detector_frame_Msun" in s
            and "m2_detector_frame_Msun" in s):
        i = int(np.argmax(s["log_likelihood"]))
        return (float(s["m1_detector_frame_Msun"][i]),
                float(s["m2_detector_frame_Msun"][i]))
    for k1, k2 in (("mass_1", "mass_2"),
                   ("m1_source", "m2_source"),
                   ("mass_1_source", "mass_2_source")):
        if k1 in s and k2 in s:
            return float(np.median(s[k1])), float(np.median(s[k2]))
    return 36.0, 29.0


def _fallback_synth(cfg, t_center_gps):
    from src.synth_waveform import imr_waveform
    m1, m2 = _gwtc1_max_l_masses(cfg)
    fs = float(cfg["data"]["master_sample_rate"])
    duration = 4.0
    n = int(round(fs * duration))
    t_rel = np.arange(n) / fs - 0.5 * duration   # ±2 s about merger
    h = imr_waveform(t_rel, tc=0.0, m1_msun=m1, m2_msun=m2,
                     f_low=cfg["maxl"]["bandpass_hz"][0])
    return t_rel + t_center_gps, h


def maximum_likelihood_waveform(samples_path: Path, cfg: Dict[str, Any],
                                t_center_gps: float,
                                ifo: Optional[str] = None
                                ) -> Tuple[np.ndarray, np.ndarray]:
    ifo = ifo or cfg["maxl"]["detector_for_projection"]
    if samples_path.exists():
        try:
            return _try_pesummary(samples_path, cfg, t_center_gps, ifo)
        except Exception as exc:
            print(f"[info] PESummary path unavailable "
                  f"({exc.__class__.__name__}); using GWTC-1 + TaylorF2 fallback.")
    return _fallback_synth(cfg, t_center_gps)

