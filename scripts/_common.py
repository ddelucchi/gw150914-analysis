"""Common boilerplate for figure scripts: bootstrap sys.path, load config,
load strain, build long Welch ASD."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

# Bootstrap import path so 'src.*' works when scripts are run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Paths, load_config, write_software_versions          # noqa: E402
from src.style import apply_style                                            # noqa: E402
from src.preprocess import (                                                 # noqa: E402
    load_strain, welch_asd, viz_condition, extract_segment, apply_bandpass,
)


def setup() -> Tuple[Dict[str, Any], Paths]:
    cfg = load_config()
    paths = Paths.from_cfg(cfg)
    write_software_versions(paths.manifest)
    apply_style(cfg)
    np.random.seed(cfg["seeds"]["numpy"])
    return cfg, paths


def load_event_strain(cfg: Dict[str, Any], paths: Paths, ifo: str):
    sr = cfg["data"]["master_sample_rate"]
    t0, t1 = cfg["data"]["fetch_start_gps"], cfg["data"]["fetch_end_gps"]
    cache = paths.raw / f"{ifo}_strain_{t0}_{t1}_{sr}.hdf5"
    if not cache.exists():
        sr2 = cfg["data"]["quicklook_sample_rate"]
        cache = paths.raw / f"{ifo}_strain_{t0}_{t1}_{sr2}.hdf5"
    if not cache.exists():
        raise FileNotFoundError(
            f"Missing strain cache for {ifo}; run `python -m src.download_data` first")
    return load_strain(cache)


def long_welch_asd(strain_ts, cfg: Dict[str, Any]):
    """1024 s Welch ASD around merger, 4 s chunks, 2 s overlap."""
    pp = cfg["processing_pipeline"]
    fs = float(strain_ts.sample_rate.value)
    half = 0.5 * pp["welch_total_s"]
    t_c = pp["segment_center_gps"]
    long_seg = strain_ts.crop(t_c - half, t_c + half).value
    return welch_asd(long_seg, fs, pp["welch_chunk_s"], pp["welch_step_s"])

