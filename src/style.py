"""Consistent matplotlib styling for the GW150914 figure set."""
from __future__ import annotations

from typing import Any, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_style(cfg: Dict[str, Any]) -> None:
    s = cfg["style"]
    mpl.rcParams.update({
        "font.family": s["font_family"],
        "mathtext.fontset": "cm",
        "font.size": s["base_fontsize"],
        "axes.titlesize": s["base_fontsize"] + 1,
        "axes.titleweight": "semibold",
        "axes.labelsize": s["base_fontsize"],
        "axes.labelpad": 4,
        "xtick.labelsize": s["base_fontsize"] - 1,
        "ytick.labelsize": s["base_fontsize"] - 1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.fontsize": s["base_fontsize"] - 1,
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "0.85",
        "lines.linewidth": s["line_width"],
        "lines.solid_capstyle": "round",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "savefig.bbox": "tight",
        "savefig.dpi": s["dpi_png"],
        "image.cmap": s["cmap_tf"],
        "figure.figsize": (s["fig_width_in"], s["fig_height_in"]),
        "figure.facecolor": "white",
    })


def detector_color(cfg: Dict[str, Any], ifo: str) -> str:
    return cfg["style"]["detector_colors"].get(ifo, "#333333")


def save_figure(fig: plt.Figure, out_dir, stem: str, cfg: Dict[str, Any]) -> None:
    """Save figure as both PDF (vector) and PNG (raster)."""
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=cfg["style"]["dpi_png"])
    plt.close(fig)
    print(f"[saved] {pdf}")
    print(f"[saved] {png}")

