"""Download GW150914 strain data and posterior products from public archives.

Sources of truth:
  - GWOSC via gwpy.timeseries.TimeSeries.fetch_open_data
  - GWTC-1 PE samples on dcc.ligo.org (P1800370)
  - PESummary public bundle (pesummary_samples.zip)

Files are written to data/raw and data/posterior, with SHA-256 hashes
recorded in data/manifests/data_manifest.json for full provenance.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Dict

import requests

from src.config import REPO_ROOT, Paths, load_config, write_software_versions


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _http_download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already exists")
        return
    print(f"[get ] {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for blk in r.iter_content(1 << 16):
                f.write(blk)


def fetch_strain(cfg: Dict[str, Any], paths: Paths) -> Dict[str, Path]:
    """Fetch H1/L1 strain at master sample rate via GWpy and cache as HDF5."""
    from gwpy.timeseries import TimeSeries

    out: Dict[str, Path] = {}
    t0 = cfg["data"]["fetch_start_gps"]
    t1 = cfg["data"]["fetch_end_gps"]
    sr = cfg["data"]["master_sample_rate"]
    for ifo in cfg["event"]["detectors"]:
        cache = paths.raw / f"{ifo}_strain_{t0}_{t1}_{sr}.hdf5"
        out[ifo] = cache
        if cache.exists():
            print(f"[skip] {cache.name} cached")
            continue
        print(f"[fetch] {ifo} {t0}-{t1} @ {sr} Hz from GWOSC")
        try:
            ts = TimeSeries.fetch_open_data(ifo, t0, t1, sample_rate=sr,
                                            cache=True, verbose=False)
        except Exception as exc:
            print(f"[warn] {sr} Hz fetch failed ({exc}); falling back to "
                  f"{cfg['data']['quicklook_sample_rate']} Hz")
            ts = TimeSeries.fetch_open_data(
                ifo, t0, t1,
                sample_rate=cfg["data"]["quicklook_sample_rate"],
                cache=True, verbose=False)
        ts.write(cache, format="hdf5", overwrite=True)
    return out


def fetch_posterior(cfg: Dict[str, Any], paths: Paths) -> Dict[str, Path]:
    """Download GWTC-1 HDF5 + PESummary bundle."""
    out: Dict[str, Path] = {}

    pe_h5 = paths.posterior / cfg["posterior"]["gwtc1_filename"]
    try:
        _http_download(cfg["posterior"]["gwtc1_url"], pe_h5)
        out["gwtc1_hdf5"] = pe_h5
    except Exception as exc:
        print(f"[warn] GWTC-1 download failed: {exc}")

    zip_url = cfg["posterior"].get("pesummary_zip_url")
    if zip_url:
        zip_path = paths.posterior / "pesummary_samples.zip"
        try:
            _http_download(zip_url, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                inner = cfg["posterior"]["pesummary_inner_dat"]
                for name in zf.namelist():
                    if name.endswith(inner):
                        target = paths.posterior / inner
                        if not target.exists():
                            with zf.open(name) as src, open(target, "wb") as dst:
                                dst.write(src.read())
                        out["pesummary_dat"] = target
                        break
        except Exception as exc:
            print(f"[warn] PESummary bundle download skipped: {exc}")
            print("       (figures fall back to GWTC-1 + TaylorF2 synthesis)")

    return out


def write_manifest(cfg_path: Path, files: Dict[str, Path], paths: Paths) -> Path:
    cfg_abs = Path(cfg_path).resolve()
    try:
        cfg_rel = str(cfg_abs.relative_to(REPO_ROOT))
    except ValueError:
        cfg_rel = str(cfg_abs)
    manifest = {
        "config": cfg_rel,
        "files": {},
    }
    for label, p in files.items():
        if p is None or not Path(p).exists():
            continue
        manifest["files"][label] = {
            "path": str(Path(p).relative_to(REPO_ROOT)),
            "bytes": Path(p).stat().st_size,
            "sha256": _sha256(Path(p)),
        }
    out = paths.manifest / "data_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] {out}")
    return out


def main(cfg_path: Path | None = None) -> None:
    cfg = load_config(cfg_path)
    paths = Paths.from_cfg(cfg)
    write_software_versions(paths.manifest)
    strain = fetch_strain(cfg, paths)
    post = fetch_posterior(cfg, paths)
    write_manifest(cfg_path or Path("configs/gw150914.yaml"),
                   {**strain, **post}, paths)


if __name__ == "__main__":
    main()

