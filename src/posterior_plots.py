"""GWTC-1 posterior loading and corner-plot helpers.

The official GW150914 sample-release file
(``GW150914_GWTC-1.hdf5``, LIGO-P1800370) stores datasets with the
fields::

    ('costheta_jn', 'luminosity_distance_Mpc', 'right_ascension',
     'declination', 'm1_detector_frame_Msun', 'm2_detector_frame_Msun',
     'spin1', 'spin2', 'costilt1', 'costilt2')

The helpers here load the chosen group, derive a few convenience
quantities (chirp mass, mass ratio, effective inspiral spin), and
return a dictionary with human-readable LaTeX labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np


GROUP_CANDIDATES = (
    "Overall_posterior",
    "IMRPhenomPv2_posterior",
    "SEOBNRv3_posterior",
)


def load_gwtc1_samples(path: Path,
                       group_candidates: Tuple[str, ...] = GROUP_CANDIDATES,
                       ) -> Dict[str, np.ndarray]:
    """Return a dict of 1-D arrays keyed by the raw GWTC-1 field names."""
    with h5py.File(path, "r") as f:
        group = next((g for g in group_candidates if g in f), None)
        if group is None:
            group = list(f.keys())[0]
        ds = f[group][()]
    if ds.dtype.names is None:
        raise RuntimeError(f"Unexpected dataset layout in {path}")
    return {n: np.asarray(ds[n]) for n in ds.dtype.names}


def derive_quantities(samples: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Add chirp-mass, mass-ratio, and chi_eff columns (detector frame)."""
    out = dict(samples)
    m1 = samples["m1_detector_frame_Msun"]
    m2 = samples["m2_detector_frame_Msun"]
    mtot = m1 + m2
    mc = (m1 * m2) ** (3.0 / 5.0) / mtot ** (1.0 / 5.0)
    q = m2 / m1
    a1 = samples["spin1"]
    a2 = samples["spin2"]
    cos1 = samples["costilt1"]
    cos2 = samples["costilt2"]
    chi_eff = (m1 * a1 * cos1 + m2 * a2 * cos2) / mtot
    out["chirp_mass_Msun"] = mc
    out["mass_ratio"] = q
    out["chi_eff"] = chi_eff
    return out


# Map config-friendly names -> (raw key in HDF5/derived dict, LaTeX label)
PARAMETER_REGISTRY: Dict[str, Tuple[str, str]] = {
    "mass_1_det":          ("m1_detector_frame_Msun", r"$m_1^{\rm det}\,[M_\odot]$"),
    "mass_2_det":          ("m2_detector_frame_Msun", r"$m_2^{\rm det}\,[M_\odot]$"),
    "chirp_mass_det":      ("chirp_mass_Msun",        r"$\mathcal{M}^{\rm det}\,[M_\odot]$"),
    "mass_ratio":          ("mass_ratio",             r"$q = m_2/m_1$"),
    "luminosity_distance": ("luminosity_distance_Mpc", r"$d_L\,[\mathrm{Mpc}]$"),
    "chi_eff":             ("chi_eff",                r"$\chi_{\rm eff}$"),
    "costheta_jn":         ("costheta_jn",            r"$\cos\theta_{JN}$"),
    "spin1":               ("spin1",                  r"$a_1$"),
    "spin2":               ("spin2",                  r"$a_2$"),
}


def select_parameters(samples: Dict[str, np.ndarray],
                      params: List[str]) -> Dict[str, np.ndarray]:
    """Return ``{label: array}`` for the requested config names.

    Unknown names are silently skipped.  Derived quantities are produced on
    demand via :func:`derive_quantities`.
    """
    enriched = derive_quantities(samples)
    out: Dict[str, np.ndarray] = {}
    for p in params:
        entry = PARAMETER_REGISTRY.get(p)
        if entry is None:
            if p in enriched:
                out[p] = enriched[p]
            continue
        key, label = entry
        if key in enriched:
            out[label] = enriched[key]
    return out

