"""Download GW150914 strain data and posterior products from public archives.

Sources of truth:
  - GWOSC via gwpy.timeseries.TimeSeries.fetch_open_data
  - GWTC-1 PE samples on dcc.ligo.org (P1800370)
  - PESummary public bundle (pesummary_samples.zip)

Files are written to data/raw and data/posterior, with SHA-256 hashes
recorded in data/manifests/data_manifest.json for provenance.
"""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict

import requests

from src.config import REPO_ROOT, Paths, load_config, write_software_versions


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _http_download(url: str, dest: Path) -> None:
    """Download atomically so an interrupted transfer never becomes a cache hit."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already exists")
        return

    print(f"[get ] {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    part.unlink(missing_ok=True)

    try:
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with open(part, "wb") as handle:
                for block in response.iter_content(1 << 16):
                    if block:
                        handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())

        if part.stat().st_size == 0:
            raise RuntimeError(f"download produced an empty file: {url}")
        os.replace(part, dest)
    except Exception:
        part.unlink(missing_ok=True)
        raise


def fetch_strain(cfg: Dict[str, Any], paths: Paths) -> Dict[str, Path]:
    """Fetch H1/L1 strain and keep the cache filename truthful about sample rate."""
    from gwpy.timeseries import TimeSeries

    out: Dict[str, Path] = {}
    t0 = cfg["data"]["fetch_start_gps"]
    t1 = cfg["data"]["fetch_end_gps"]
    master_sr = int(cfg["data"]["master_sample_rate"])
    quick_sr = int(cfg["data"]["quicklook_sample_rate"])

    for ifo in cfg["event"]["detectors"]:
        master_cache = paths.raw / f"{ifo}_strain_{t0}_{t1}_{master_sr}.hdf5"
        quick_cache = paths.raw / f"{ifo}_strain_{t0}_{t1}_{quick_sr}.hdf5"

        if master_cache.exists() and master_cache.stat().st_size > 0:
            print(f"[skip] {master_cache.name} cached")
            out[ifo] = master_cache
            continue

        print(f"[fetch] {ifo} {t0}-{t1} @ {master_sr} Hz from GWOSC")
        try:
            ts = TimeSeries.fetch_open_data(
                ifo,
                t0,
                t1,
                sample_rate=master_sr,
                cache=True,
                verbose=False,
            )
            cache = master_cache
        except Exception as exc:
            print(
                f"[warn] {master_sr} Hz fetch failed ({exc}); "
                f"falling back to {quick_sr} Hz"
            )
            if quick_cache.exists() and quick_cache.stat().st_size > 0:
                print(f"[skip] {quick_cache.name} cached")
                out[ifo] = quick_cache
                continue

            ts = TimeSeries.fetch_open_data(
                ifo,
                t0,
                t1,
                sample_rate=quick_sr,
                cache=True,
                verbose=False,
            )
            cache = quick_cache

        ts.write(cache, format="hdf5", overwrite=True)
        out[ifo] = cache

    return out


def fetch_posterior(cfg: Dict[str, Any], paths: Paths) -> Dict[str, Path]:
    """Download GWTC-1 HDF5 and, when available, the PESummary bundle."""
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
            with zipfile.ZipFile(zip_path) as archive:
                inner = cfg["posterior"]["pesummary_inner_dat"]
                for name in archive.namelist():
                    if name.endswith(inner):
                        target = paths.posterior / inner
                        if not target.exists():
                            part = target.with_name(target.name + ".part")
                            part.unlink(missing_ok=True)
                            try:
                                with archive.open(name) as src, open(part, "wb") as dst:
                                    while True:
                                        block = src.read(1 << 20)
                                        if not block:
                                            break
                                        dst.write(block)
                                    dst.flush()
                                    os.fsync(dst.fileno())
                                os.replace(part, target)
                            except Exception:
                                part.unlink(missing_ok=True)
                                raise
                        out["pesummary_dat"] = target
                        break
        except Exception as exc:
            print(f"[warn] PESummary bundle download skipped: {exc}")
            print("       (figures fall back to GWTC-1 + lightweight waveform synthesis)")

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
    for label, path in files.items():
        if path is None or not Path(path).exists():
            continue
        resolved = Path(path)
        manifest["files"][label] = {
            "path": str(resolved.relative_to(REPO_ROOT)),
            "bytes": resolved.stat().st_size,
            "sha256": _sha256(resolved),
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
    posterior = fetch_posterior(cfg, paths)
    write_manifest(
        cfg_path or Path("configs/gw150914.yaml"),
        {**strain, **posterior},
        paths,
    )


if __name__ == "__main__":
    main()
