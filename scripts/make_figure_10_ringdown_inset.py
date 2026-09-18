"""Figure 10 — Late-time / ringdown inset with damped-sinusoid scale guide.

Pedagogical only.  Per the GW150914 *basic-physics* pedagogical paper
(Nature/PRD 1608.01940) a Kerr remnant of :math:`M_f\!\sim\!65\,
M_\odot` and :math:`a_f\!\sim\!0.7` rings down at
:math:`f_{\rm RD}\!\approx\!250\text{–}260` Hz with damping time
:math:`\tau\!\approx\!4` ms (≈ one oscillation period).  We do *not*
fit; we overlay a damped sinusoid as a *scale marker*, started a
quarter-period after the merger time so its first crest does not
sit on top of the merger spike.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import setup, load_event_strain, long_welch_asd
from src.preprocess import viz_condition, extract_segment
from src.ringdown import damped_sinusoid
from src.style import detector_color, save_figure


def main() -> None:
    cfg, paths = setup()
    t_c = cfg["event"]["gps_merger"]
    ts = load_event_strain(cfg, paths, "L1")
    fs = float(ts.sample_rate.value)
    f_asd, asd = long_welch_asd(ts, cfg)
    seg = extract_segment(ts, t_c, 0.3).value
    t = np.arange(len(seg)) / fs - 0.3
    cond = viz_condition(seg, fs, cfg, f_asd, asd)

    win = cfg["ringdown"]["inset_window_s"]
    f0 = cfg["ringdown"]["expected_freq_hz"]
    tau = cfg["ringdown"]["expected_tau_ms"] * 1e-3

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(t, cond, color=detector_color(cfg, "L1"), lw=1.0,
            label="L1 whitened, [35,350] Hz")

    # amplitude reference: max of the Hilbert envelope just after merger
    near = (t >= 0.0) & (t <= 0.010)
    if np.any(near):
        from scipy.signal import hilbert
        env = np.abs(hilbert(cond[near]))
        amp = float(np.max(env))
    else:
        amp = 1.0

    # Start the guide a quarter period after merger so its waveform
    # does not visually clash with the merger spike — this is just a
    # scale marker, not a phase-aligned overlay.
    t0 = 0.25 / f0
    g = np.where(t >= t0, damped_sinusoid(t, t0=t0, f_hz=f0, tau_s=tau, amp=amp),
                 np.nan)
    ax.plot(t, g, color="C2", lw=0.9, alpha=0.85,
            label=fr"scale guide: $f_{{\rm RD}}={f0:.0f}$ Hz, $\tau={tau*1e3:.1f}$ ms")
    # exponential envelope to make the τ explicit
    env_pos = amp * np.exp(-(t - t0) / tau)
    env_pos = np.where(t >= t0, env_pos, np.nan)
    ax.plot(t,  env_pos, color="C2", lw=0.5, ls="--", alpha=0.4)
    ax.plot(t, -env_pos, color="C2", lw=0.5, ls="--", alpha=0.4,
            label=r"$\pm A\,e^{-(t-t_0)/\tau}$ envelope")

    ax.axvline(0, color="0.4", ls=":", lw=0.7, label="merger")
    ax.set_xlim(*win)
    cmask = (t >= win[0]) & (t <= win[1])
    ymax = max(0.5, 1.4 * float(np.max(np.abs(cond[cmask]))))
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel("time from merger (s)")
    ax.set_ylabel("whitened strain (a.u.)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title("Ringdown-scale consistency (NOT a ringdown parameter fit)")
    fig.tight_layout()
    save_figure(fig, paths.fig_appendix, "figure_appendix_ringdown_inset", cfg)


if __name__ == "__main__":
    main()

