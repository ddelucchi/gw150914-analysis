"""Figure 8 (appendix) — H1+L1 matched-filter |SNR(t)|, pedagogical.

This is the single hardest LVC tutorial figure to reproduce without
PyCBC: the GWpy/PyCBC tutorial uses an IMRPhenomD time-domain template,
a 32 s median-PSD whitening, and reports peak |rho| > 17.  Our SciPy
fallback uses a TaylorF2 *inspiral-only* template (no merger or
ringdown contribution to the matched filter) and a 32 s median Welch
PSD; with that simpler template the achievable peak SNR is structurally
limited to ~6-8 per detector.

This is therefore a *qualitative* validation panel: the SNR time series
is dominated by Gaussian noise except in a narrow window around the
event GPS, where both detectors show coincident peaks.  Network SNR is
quoted as the quadrature sum, with the explicit caveat that the LVC
network SNR ~24 is achievable only with a full IMR template.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import setup, load_event_strain
from src.matched_filter import matched_filter_snr
from src.style import detector_color, save_figure


def _peak(rel, snr, lo=-0.5, hi=0.5):
    win = (rel >= lo) & (rel <= hi)
    if not np.any(win):
        return None, None
    k = int(np.flatnonzero(win)[np.argmax(snr[win])])
    return rel[k], float(snr[k])


def main() -> None:
    cfg, paths = setup()
    crop = (1126259460.0, 1126259464.0)
    t_c = cfg["event"]["gps_merger"]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.0), sharex=True)

    peaks = {}
    series = {}
    for ifo in ("H1", "L1"):
        try:
            ts = load_event_strain(cfg, paths, ifo)
            t, snr = matched_filter_snr(ts, cfg, crop)
            rel = t - t_c
            tp, sp = _peak(rel, snr)
            peaks[ifo] = (tp, sp)
            series[ifo] = (rel, snr)
        except Exception as exc:
            print(f"[warn] matched filter failed for {ifo}: {exc}")

    if not series:
        raise RuntimeError("Matched filter failed for all detectors")

    for ax, ifo in zip(axes, ("H1", "L1")):
        if ifo not in series:
            ax.text(0.5, 0.5, f"{ifo} matched filter failed",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        rel, snr = series[ifo]
        ax.plot(rel, snr, color=detector_color(cfg, ifo), lw=0.9,
                label=f"{ifo}  TaylorF2 (36, 29) $M_\\odot$")
        tp, sp = peaks[ifo]
        ax.axvline(0, color="0.4", ls=":", lw=0.7)
        ax.plot([tp], [sp], "o", color="k", ms=4, zorder=5)
        ax.text(tp + 0.05, sp,
                fr"peak $|\rho|={sp:.1f}$ at $t={tp*1e3:+.0f}$ ms",
                fontsize=9, ha="left", va="bottom",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                      "alpha": 0.9, "edgecolor": "0.7"})
        ax.axhline(17.0, color="k", ls="--", lw=0.7, alpha=0.7)
        ax.text(-1.45, 17.3, "GWpy/PyCBC IMRPhenomD reference $|\\rho|\\sim17$",
                fontsize=8, color="k", alpha=0.8)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(0, 20)
        ax.set_ylabel(r"$|\rho(t)|$")
        ax.legend(loc="upper left", fontsize=8)

    if len(peaks) >= 2:
        rho_net = float(np.sqrt(sum(p[1] ** 2 for p in peaks.values())))
        net_msg = (f"Network $|\\rho|_{{\\rm net}} = "
                   f"\\sqrt{{\\sum_i |\\rho_i|^2}} = {rho_net:.1f}$  "
                   f"(LVC IMR network SNR $\\sim$24)")
    else:
        net_msg = ""

    axes[-1].set_xlabel("time from merger (s)")
    fig.suptitle(
        "H1 + L1 matched-filter SNR — TaylorF2 inspiral template + 32 s median Welch PSD\n"
        "SciPy/NumPy fallback (PyCBC unavailable on this platform).  " + net_msg,
        fontsize=10)
    fig.tight_layout()
    save_figure(fig, paths.fig_appendix, "figure_appendix_snr_timeseries", cfg)


if __name__ == "__main__":
    main()

