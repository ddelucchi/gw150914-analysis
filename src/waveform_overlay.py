"""Reference waveform construction with explicit provenance.

Preferred path
--------------
Use PESummary's documented GW150914 workflow to build a detector-projected
IMRPhenomPv2 waveform from released parameter-estimation samples.

Fallback path
-------------
When PESummary/lalsimulation or the required sample columns are unavailable,
construct a lightweight inspiral + ringdown *diagnostic sketch*.  The fallback
is not a maximum-likelihood IMR waveform and is labelled accordingly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class WaveformProduct:
    times: np.ndarray
    strain: np.ndarray
    provenance: str
    label: str
    detector_projected: bool
    sample_rate_hz: float


def load_pesummary_samples(path: Path):
    import pandas as pd

    return pd.read_csv(path, sep=r"\s+", engine="python")


def _max_l_row(df) -> Dict[str, float]:
    candidates = [
        column
        for column in ("log_likelihood", "logL", "deltalogl")
        if column in df.columns
    ]
    if not candidates:
        raise KeyError("No log-likelihood column in PESummary samples")
    idx = df[candidates[0]].idxmax()
    return df.loc[idx].to_dict()


def _try_pesummary(
    samples_path: Path,
    cfg: Dict[str, Any],
    t_center_gps: float,
    ifo: str,
    sample_rate_hz: float,
) -> WaveformProduct:
    df = load_pesummary_samples(samples_path)
    params = _max_l_row(df)

    from pesummary.gw.waveform import td_waveform

    waveform = td_waveform(
        params,
        cfg["maxl"]["approximant"],
        delta_t=1.0 / sample_rate_hz,
        f_low=20.0,
        project=ifo,
    )
    times = np.asarray(waveform.sample_times) + t_center_gps
    return WaveformProduct(
        times=times,
        strain=np.asarray(waveform),
        provenance="pesummary_max_l_imrphenompv2",
        label="max-L IMRPhenomPv2 (PESummary)",
        detector_projected=True,
        sample_rate_hz=sample_rate_hz,
    )


def _gwtc1_mass_estimate(cfg: Dict[str, Any]) -> Tuple[float, float, str]:
    from src.config import REPO_ROOT
    from src.posterior_plots import load_gwtc1_samples

    h5 = (
        REPO_ROOT
        / cfg["paths"]["posterior_dir"]
        / cfg["posterior"]["gwtc1_filename"]
    )

    if not h5.exists():
        return 36.0, 29.0, "default_reference_masses"

    samples = load_gwtc1_samples(h5)
    if (
        "log_likelihood" in samples
        and "m1_detector_frame_Msun" in samples
        and "m2_detector_frame_Msun" in samples
    ):
        idx = int(np.argmax(samples["log_likelihood"]))
        return (
            float(samples["m1_detector_frame_Msun"][idx]),
            float(samples["m2_detector_frame_Msun"][idx]),
            "gwtc1_max_l_detector_frame_masses",
        )

    for m1_key, m2_key in (
        ("mass_1", "mass_2"),
        ("m1_source", "m2_source"),
        ("mass_1_source", "mass_2_source"),
    ):
        if m1_key in samples and m2_key in samples:
            return (
                float(np.median(samples[m1_key])),
                float(np.median(samples[m2_key])),
                "gwtc1_median_available_mass_columns",
            )

    return 36.0, 29.0, "default_reference_masses"


def _fallback_synth(
    cfg: Dict[str, Any],
    t_center_gps: float,
    sample_rate_hz: float,
) -> WaveformProduct:
    from src.synth_waveform import imr_waveform

    m1, m2, mass_source = _gwtc1_mass_estimate(cfg)
    duration = 4.0
    n = int(round(sample_rate_hz * duration))
    t_rel = np.arange(n) / sample_rate_hz - 0.5 * duration
    strain = imr_waveform(
        t_rel,
        tc=0.0,
        m1_msun=m1,
        m2_msun=m2,
        f_low=cfg["maxl"]["bandpass_hz"][0],
    )

    if mass_source == "gwtc1_max_l_detector_frame_masses":
        label = "synthetic PN + ringdown sketch using GWTC-1 max-L masses"
    elif mass_source == "gwtc1_median_available_mass_columns":
        label = "synthetic PN + ringdown sketch using GWTC-1 median masses"
    else:
        label = "synthetic PN + ringdown sketch using reference masses"

    return WaveformProduct(
        times=t_rel + t_center_gps,
        strain=strain,
        provenance=f"diagnostic_synthetic:{mass_source}",
        label=label,
        detector_projected=False,
        sample_rate_hz=sample_rate_hz,
    )


def build_waveform_product(
    samples_path: Path,
    cfg: Dict[str, Any],
    t_center_gps: float,
    ifo: Optional[str] = None,
    sample_rate_hz: float | None = None,
) -> WaveformProduct:
    """Return a waveform together with evidence about how it was produced."""
    detector = ifo or cfg["maxl"]["detector_for_projection"]
    fs = (
        float(sample_rate_hz)
        if sample_rate_hz is not None
        else float(cfg["data"]["master_sample_rate"])
    )
    if fs <= 0:
        raise ValueError("sample_rate_hz must be positive")

    if samples_path.exists():
        try:
            return _try_pesummary(samples_path, cfg, t_center_gps, detector, fs)
        except Exception as exc:
            print(
                "[info] PESummary waveform unavailable "
                f"({exc.__class__.__name__}); using labelled diagnostic fallback."
            )

    return _fallback_synth(cfg, t_center_gps, fs)


def maximum_likelihood_waveform(
    samples_path: Path,
    cfg: Dict[str, Any],
    t_center_gps: float,
    ifo: Optional[str] = None,
    sample_rate_hz: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible tuple API.

    Callers that display or interpret the waveform should prefer
    :func:`build_waveform_product` so fallback provenance cannot be lost.
    """
    product = build_waveform_product(
        samples_path,
        cfg,
        t_center_gps,
        ifo=ifo,
        sample_rate_hz=sample_rate_hz,
    )
    return product.times, product.strain
