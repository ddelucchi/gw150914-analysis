"""Configuration loader and path helpers for the GW150914 figure pipeline."""
from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "gw150914.yaml"


@dataclass
class Paths:
    raw: Path
    intermediate: Path
    posterior: Path
    manifest: Path
    fig_main: Path
    fig_appendix: Path

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "Paths":
        p = cfg["paths"]
        out = cls(
            raw=REPO_ROOT / p["raw_dir"],
            intermediate=REPO_ROOT / p["intermediate_dir"],
            posterior=REPO_ROOT / p["posterior_dir"],
            manifest=REPO_ROOT / p["manifest_dir"],
            fig_main=REPO_ROOT / p["fig_main"],
            fig_appendix=REPO_ROOT / p["fig_appendix"],
        )
        for d in vars(out).values():
            d.mkdir(parents=True, exist_ok=True)
        return out


def load_config(path: os.PathLike | str | None = None) -> Dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_software_versions(manifest_dir: Path) -> Path:
    """Record software versions for reproducibility."""
    versions: Dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for mod in ("numpy", "scipy", "matplotlib", "pandas", "h5py",
                "yaml", "gwpy", "pycbc", "pesummary", "lal", "corner"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception as exc:
            versions[mod] = f"NOT INSTALLED: {exc.__class__.__name__}"
    out = manifest_dir / "software_versions.json"
    out.write_text(json.dumps(versions, indent=2))
    return out

