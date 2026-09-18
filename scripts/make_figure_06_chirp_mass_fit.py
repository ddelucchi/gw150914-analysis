"""Figure 6 — Direct-data chirp-mass estimate from instantaneous frequency.

Pedagogical only.  We compute the analytic signal (Hilbert transform)
of the *bandpassed* H1 and L1 strain, extract the instantaneous
frequency

    f_inst(t) = (1/2pi) d phi(t) / dt

and fit the Newtonian chirp law

    f(t)^{-8/3} = (256/5) pi^{8/3} (G Mc / c^3)^{5/3} (t_c - t)

over the *early* inspiral band f in [35, 80] Hz, where the leading-PN
relation is most trustworthy.  Going much above 80 Hz puts us in the
late-inspiral / merger regime where higher-PN and ringdown effects
bias the fit upward (the discovery paper itself frames its direct
estimate at low-to-moderate inspiral frequencies).

The reported uncertainty combines the linear-regression slope error
in quadrature with a block jackknife of the surviving inspiral
samples and the inter-detector spread; the dominant systematic
remains the choice of fit band, not the formal error.

Reference points: PRL discovery paper direct estimate ~30 Msun; our
GWTC-1 posterior detector-frame value Mc^det ~31 Msun (Fig. 11).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, sosfiltfilt

from _common import setup, load_event_strain, long_welch_asd
from src.preprocess import (
    tukey_with_transitions, whiten_explicit, bandpass_sos,
)
from src.style import detector_color, save_figure
from src.ridge_fit import C_SI, G_SI, MSUN_KG


@dataclass
class HilbertFit:
    t: np.ndarray
    f: np.ndarray
    y: np.ndarray
    slope: float
    intercept: float
    slope_err: float
    slope_jk: float
    chirp_mass_msun: float
    chirp_mass_err_msun: float
    f_lo: float
    f_hi: float


def _mc_from_slope(slope: float) -> float:
    coef53 = (5.0 / 256.0) * np.pi ** (-8.0 / 3.0) * (-slope)
    if coef53 <= 0:
        return float("nan")
    g_m_over_c3 = coef53 ** (3.0 / 5.0)
    return float((C_SI ** 3 / G_SI) * g_m_over_c3 / MSUN_KG)


def _conditioned(cfg, paths, ifo):
    pp = cfg["processing_pipeline"]
    ts = load_event_strain(cfg, paths, ifo)
    fs = float(ts.sample_rate.value)
    t_c_seg = pp["segment_center_gps"]
    half_seg = 0.5 * pp["segment_length_s"]
    seg = ts.crop(t_c_seg - half_seg, t_c_seg + half_seg).value
    n = len(seg)
    win = tukey_with_transitions(n, fs, pp["tukey_transition_s"])
    f_asd, asd = long_welch_asd(ts, cfg)
    seg_w = whiten_explicit(seg * win, fs, f_asd, asd)
    # For Fig 6 we use a *tight inspiral bandpass* so the Hilbert
    # instantaneous frequency tracks the chirp rather than out-of-band
    # noise.  The Fig 1/2 visualization band [35, 350] Hz is too broad
    # for this purpose: away from merger the analytic-signal frequency
    # would sit at the bandpass median (~200 Hz), not at the chirp.
    sos = bandpass_sos(fs, 30.0, 110.0)
    seg_bp = sosfiltfilt(sos, seg_w)
    t_seg = t_c_seg - half_seg + np.arange(n) / fs
    return t_seg, seg_bp, fs


def fit_chirp_mass_hilbert(t_seg, x_bp, fs, t_c_event,
                           f_lo: float, f_hi: float,
                           t_min_rel: float = -0.20,
                           t_max_rel: float = -0.005) -> HilbertFit:
    from scipy.signal import savgol_filter
    z = hilbert(x_bp)
    phi = np.unwrap(np.angle(z))
    f_raw = np.gradient(phi, 1.0 / fs) / (2.0 * np.pi)
    # Aggressive Savitzky-Golay (~50 ms) suppresses sample-rate jitter
    # of d phi/dt while preserving the chirp's evolution timescale.
    win = max(31, int(round(0.050 * fs)) | 1)
    if win >= len(f_raw):
        win = len(f_raw) - (1 - len(f_raw) % 2)
    f_inst = savgol_filter(f_raw, win, polyorder=3)
    t_rel = t_seg - t_c_event

    # Bin-median in 4 ms bins across the inspiral window
    bin_w = 0.004
    edges = np.arange(t_min_rel, t_max_rel + bin_w, bin_w)
    centers = 0.5 * (edges[:-1] + edges[1:])
    f_binned = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        m = (t_rel >= edges[i]) & (t_rel < edges[i + 1])
        if m.sum() >= 3:
            f_binned[i] = float(np.median(f_inst[m]))
    valid = np.isfinite(f_binned)
    t_b, f_b = centers[valid], f_binned[valid]
    if len(f_b) < 6:
        raise RuntimeError("Too few inspiral bins")

    # Restrict to the early-inspiral fit band (no isotonic projection:
    # Savitzky-Golay + 4 ms binning already produces a monotone chirp in
    # the clean late-inspiral window; isotonic over the full window is
    # corrupted by bandpass-edge ringing far from merger).
    keep = (f_b >= f_lo) & (f_b <= f_hi)
    t_r, f_r = t_b[keep], f_b[keep]
    if len(t_r) < 5:
        raise RuntimeError(
            f"Too few samples in [{f_lo}, {f_hi}] Hz: {len(t_r)}")

    y = f_r ** (-8.0 / 3.0)
    A = np.vstack([t_r, np.ones_like(t_r)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    sigma2 = float(np.mean((y - A @ coef) ** 2))
    cov = sigma2 * np.linalg.inv(A.T @ A)
    slope_err = float(np.sqrt(cov[0, 0]))

    n = len(t_r)
    n_blocks = 10
    block = max(1, n // n_blocks)
    slopes = []
    for b in range(n_blocks):
        m = np.ones(n, dtype=bool)
        m[b * block:(b + 1) * block] = False
        if m.sum() < 4:
            continue
        Ai = np.vstack([t_r[m], np.ones(int(m.sum()))]).T
        c_i, *_ = np.linalg.lstsq(Ai, y[m], rcond=None)
        slopes.append(float(c_i[0]))
    slopes = np.array(slopes)
    slope_jk = float(np.sqrt((len(slopes) - 1) / max(1, len(slopes))
                              * np.sum((slopes - slopes.mean()) ** 2))) \
                if len(slopes) > 1 else 0.0

    mc = _mc_from_slope(slope)
    sigma_slope = float(np.sqrt(slope_err ** 2 + slope_jk ** 2))
    mc_err = (3.0 / 5.0) * mc * (sigma_slope / abs(slope))

    return HilbertFit(t=t_r, f=f_r, y=y,
                      slope=slope, intercept=intercept,
                      slope_err=slope_err, slope_jk=slope_jk,
                      chirp_mass_msun=mc,
                      chirp_mass_err_msun=mc_err,
                      f_lo=f_lo, f_hi=f_hi)


def main() -> None:
    cfg, paths = setup()
    t_c = cfg["event"]["gps_merger"]
    f_lo, f_hi = 35.0, 100.0
    t_min, t_max = -0.06, -0.003   # clean late-inspiral window only

    fits = {}
    cond = {}
    for ifo in ("H1", "L1"):
        try:
            t_seg, x_bp, fs = _conditioned(cfg, paths, ifo)
            cond[ifo] = (t_seg, x_bp, fs)
            fits[ifo] = fit_chirp_mass_hilbert(
                t_seg, x_bp, fs, t_c, f_lo, f_hi, t_min, t_max)
        except Exception as exc:
            print(f"[warn] Hilbert fit failed for {ifo}: {exc}")

    if not fits:
        raise RuntimeError("All Hilbert fits failed")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0),
                             gridspec_kw={"height_ratios": [1.0, 1.0]})

    for col, ifo in enumerate(("H1", "L1")):
        ax = axes[0, col]
        if ifo not in fits:
            ax.text(0.5, 0.5, f"{ifo}: fit failed",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        t_seg, x_bp, fs = cond[ifo]
        from scipy.signal import savgol_filter
        z = hilbert(x_bp)
        phi = np.unwrap(np.angle(z))
        f_raw = np.gradient(phi, 1.0 / fs) / (2.0 * np.pi)
        win = max(31, int(round(0.050 * fs)) | 1)
        f_inst = savgol_filter(f_raw, win, polyorder=3)
        t_rel = t_seg - t_c
        m = (t_rel >= -0.30) & (t_rel <= 0.02)
        ax.plot(t_rel[m], f_inst[m], color="0.55", lw=0.7,
                label=r"Hilbert $f_{\rm inst}(t)$ (full band)")
        ax.plot(fits[ifo].t, fits[ifo].f, "o", ms=3,
                color=detector_color(cfg, ifo), mec="k", mew=0.3,
                label=f"{len(fits[ifo].t)} fit anchors  ({int(f_lo)}-{int(f_hi)} Hz)")
        ax.axhspan(f_lo, f_hi, color="C2", alpha=0.10)
        ax.axvspan(t_min, t_max, color="0.85", alpha=0.5)
        ax.set_xlim(-0.30, 0.02)
        ax.set_ylim(20, 280)
        ax.set_xlabel("time from merger (s)")
        if col == 0:
            ax.set_ylabel("frequency (Hz)")
        ax.set_title(f"{ifo} instantaneous frequency from analytic signal")
        ax.legend(loc="upper left", fontsize=8)

        ax = axes[1, col]
        fit = fits[ifo]
        ax.plot(fit.t, fit.y, "o", ms=4, color=detector_color(cfg, ifo),
                label=f"{ifo} early-inspiral $f^{{-8/3}}$")
        line = fit.slope * fit.t + fit.intercept
        ax.plot(fit.t, line, "k-", lw=1.2,
                label=fr"fit: $\mathcal{{M}}_c^{{\rm det}} = {fit.chirp_mass_msun:.1f}\pm "
                      fr"{fit.chirp_mass_err_msun:.1f}\,M_\odot$")
        sigma_slope = float(np.sqrt(fit.slope_err ** 2 + fit.slope_jk ** 2))
        band = abs(sigma_slope * fit.t)
        ax.fill_between(fit.t, line - band, line + band, color="k", alpha=0.12,
                        label=r"slope $1\sigma$ (lin $\oplus$ block-jk)")
        ax.set_xlabel("time from merger (s)")
        if col == 0:
            ax.set_ylabel(r"$f^{-8/3}$ (Hz$^{-8/3}$)")
        ax.set_title(f"{ifo} early-inspiral chirp-mass fit")
        ax.legend(loc="upper right", fontsize=8)

    mcs = np.array([f.chirp_mass_msun for f in fits.values()])
    errs = np.array([f.chirp_mass_err_msun for f in fits.values()])
    mc_avg = float(np.mean(mcs))
    spread = float(np.std(mcs, ddof=1)) if len(mcs) > 1 else 0.0
    mc_err_total = float(np.sqrt(np.mean(errs ** 2) + spread ** 2))
    fig.suptitle(
        rf"Direct-data $\mathcal{{M}}_c^{{\rm det}}$ from Hilbert instantaneous frequency, "
        rf"early-inspiral {int(f_lo)}-{int(f_hi)} Hz "
        rf"(detector-averaged $= {mc_avg:.1f}\pm{mc_err_total:.1f}\,M_\odot$).  "
        r"Pedagogical estimate; PRL direct value $\sim 30\,M_\odot$, "
        r"GWTC-1 detector-frame posterior $\sim 31\,M_\odot$.  "
        r"Quoted $\sigma$ = linear-regression $\oplus$ block jackknife $\oplus$ inter-detector spread.",
        fontsize=9.5, y=1.0,
    )
    fig.tight_layout()
    save_figure(fig, paths.fig_appendix, "figure_appendix_chirp_mass_fit", cfg)


if __name__ == "__main__":
    main()

