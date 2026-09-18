"""Build every figure in sequence; supports `--mode {paper,robustness}`.

Usage:
    python scripts/make_all_figures.py
    python scripts/make_all_figures.py --mode robustness
"""
from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from src.config import load_config, Paths                   # noqa: E402
from src.download_data import main as download_main         # noqa: E402

PAPER_FIGURES = [
    # --- Main text (6; per-critic final curated set) ---
    "make_figure_04_qtransform",           # Q-transform hero
    "make_figure_05_dual_detector_overlay",  # H1/L1 aligned overlay
    "make_figure_appendix_projected_waveform",  # PROMOTED: LVC max-L waveform overlay + residuals
    "make_figure_03_psd_windowing",        # PSD / windowing / leakage
    "make_figure_02_processing_pipeline",  # conditioning sequence
    "make_figure_11_posterior_corner",     # GWTC-1 posterior corner
    # --- Appendix (pedagogical / qualitative) ---
    "make_figure_06_chirp_mass_fit",       # Hilbert Mc estimate (fragility demo)
    "make_figure_10_ringdown_inset",       # ringdown scale guide
    "make_figure_09_correlation_residuals",  # optional: lag-space residual logic
    # CUT per critic: 01_event_atlas (superseded by stronger main figs),
    #                 07_maxl_overlay (PN+RD cartoon, redundant with ringdown
    #                 guide and projected-waveform fig).
    # NOTE: make_figure_08_snr_timeseries deliberately NOT built --
    # detector timing inconsistency vs GWpy/PyCBC benchmark; kept on
    # disk for diagnostic re-runs but excluded from the paper build.
]

ROBUSTNESS_FIGURES = [
    "make_figure_appendix_robustness",
]


def run_module(name: str) -> bool:
    print(f"\n=== {name} ===")
    try:
        mod = importlib.import_module(name)
        importlib.reload(mod)
        mod.main()
        return True
    except Exception:
        traceback.print_exc()
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["paper", "robustness", "all"],
                   default="all")
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args()

    cfg = load_config()
    Paths.from_cfg(cfg)
    if not args.skip_download:
        download_main()

    targets = []
    if args.mode in ("paper", "all"):
        targets += PAPER_FIGURES
    if args.mode in ("robustness", "all"):
        targets += ROBUSTNESS_FIGURES

    failures = [m for m in targets if not run_module(m)]
    if failures:
        print(f"\n[FAIL] {len(failures)} figure(s) failed: {failures}")
        sys.exit(1)
    print(f"\n[OK] {len(targets)} figure(s) built.")


if __name__ == "__main__":
    main()

