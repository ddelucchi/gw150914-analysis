"""Figure 11 — GWTC-1 posterior corner plot.

Loads the official ``GW150914_GWTC-1.hdf5`` sample release
(LIGO-P1800370) and shows the joint posterior on detector-frame
component masses, chirp mass, luminosity distance, and effective
inspiral spin, using the IMRPhenomPv2 / Overall posterior group.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import setup
from src.posterior_plots import (
    derive_quantities,
    load_gwtc1_samples,
    select_parameters,
)
from src.style import save_figure


def _placeholder(text, paths, cfg):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=11)
    save_figure(fig, paths.fig_main, "figure_11_posterior_corner", cfg)


def main() -> None:
    cfg, paths = setup()
    h5 = paths.posterior / cfg["posterior"]["gwtc1_filename"]
    if not h5.exists():
        _placeholder(
            f"Missing {h5.name}.\nRun: python -m src.download_data",
            paths, cfg)
        return

    samples = load_gwtc1_samples(h5)
    sel = select_parameters(samples, cfg["posterior_plot"]["parameters"])
    if not sel:
        _placeholder(
            f"No requested parameters found in {h5.name}.\n"
            f"Available raw fields: {sorted(samples)[:8]}",
            paths, cfg)
        return

    labels = list(sel.keys())
    data = np.column_stack([sel[k] for k in labels])

    enriched = derive_quantities(samples)
    mc = enriched["chirp_mass_Msun"]
    chi_eff = enriched["chi_eff"]
    q = enriched["mass_ratio"]
    mc_q = np.quantile(mc, [0.05, 0.5, 0.95])
    chi_q = np.quantile(chi_eff, [0.05, 0.5, 0.95])
    q_q = np.quantile(q, [0.05, 0.5, 0.95])

    try:
        import corner
    except Exception:
        corner = None

    if corner is not None:
        n = len(labels)
        # Slightly tighter canvas (per critic priority 2): the empty
        # upper-right triangle should feel deliberate, not leftover.
        fig = corner.corner(
            data, labels=labels,
            color="#1f3a6b", show_titles=True,
            title_fmt=".2f", quantiles=[0.05, 0.5, 0.95],
            hist_kwargs={"lw": 1.2, "color": "#1f3a6b"},
            label_kwargs={"fontsize": 11},
            title_kwargs={"fontsize": 9.0},
            plot_datapoints=False, plot_density=True, fill_contours=True,
            levels=(0.50, 0.90),
            # Stronger smoothing + finer binning to remove the
            # "stepped coastline" look in the q-related panels.
            smooth=1.5, smooth1d=1.2, bins=50,
            contour_kwargs={"linewidths": 1.0, "colors": "#102a52"},
            fig=plt.figure(figsize=(2.45 * n, 2.45 * n)),
        )
        fig.subplots_adjust(left=0.105, right=0.985,
                            bottom=0.105, top=0.965,
                            wspace=0.06, hspace=0.06)

        # ONE coherent left-aligned annotation block, placed in the
        # visual center of the empty upper-right triangle. A single
        # anchoring convention (left-justified, top-anchored) replaces
        # the prior mix of centered title / right-aligned medians /
        # centered legend that the critic flagged as three competing
        # alignment systems.
        def _fmt(med, lo, hi, fmt="%.2f"):
            return (fmt % med) + r"$^{+%s}_{-%s}$" % (
                fmt % (hi - med), fmt % (med - lo))

        info = (
            "GW150914 detector-frame posterior\n"
            "GWTC-1, IMRPhenomPv2 (Overall)\n"
            "\n"
            r"$\mathcal{M}^{\rm det} = $" + _fmt(mc_q[1], mc_q[0], mc_q[2],
                                                 "%.2f")
            + r"$\,M_\odot$" + "\n"
            r"$q = $" + _fmt(q_q[1], q_q[0], q_q[2], "%.2f")
            + r"  $(q \equiv m_2/m_1 \leq 1)$" + "\n"
            r"$\chi_{\rm eff} = $" + _fmt(chi_q[1], chi_q[0], chi_q[2],
                                          "%.2f")
            + "\n\n"
            "contours: 50% and 90% credible regions\n"
            "dashed: 5th, 50th, 95th percentiles"
        )
        # Anchor inside the empty triangle: column index 1 ~= x 0.50,
        # row index 0 ~= y 0.92. Left-align, top-anchored.
        fig.text(
            0.50, 0.965, info,
            ha="left", va="top", fontsize=9.0,
            color="0.10", linespacing=1.40,
        )
    else:
        n = len(labels)
        fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n))
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                if i == j:
                    ax.hist(data[:, i], bins=40, color="#1f77b4", alpha=0.8,
                            density=True)
                elif j < i:
                    ax.hexbin(data[:, j], data[:, i], gridsize=40, cmap="Blues")
                else:
                    ax.set_visible(False)
                if i == n - 1:
                    ax.set_xlabel(labels[j])
                if j == 0:
                    ax.set_ylabel(labels[i])
        fig.suptitle("GW150914 — GWTC-1 posterior", y=1.0)
        fig.tight_layout()
    save_figure(fig, paths.fig_main, "figure_11_posterior_corner", cfg)


if __name__ == "__main__":
    main()

