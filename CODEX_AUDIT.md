# Release audit

Audit date: 2026-09-17

## Included

- central GW150914 configuration
- Python environment metadata
- data/provenance and download utilities
- conditioning and spectral tools
- matched-filter and waveform utilities
- ridge, correlation, ringdown, posterior, and visualization modules
- master figure build and individual main/appendix figure scripts
- network-free core numerical tests

## Observed validation before public hardening

- all retained Python source and figure scripts passed Python compilation
- `scripts/make_all_figures.py --help` completed successfully
- source/configuration scan found no credentials or private keys

Full figure generation was not claimed during curation because it depends on downloaded GWOSC/posterior products and heavyweight scientific packages.

## Public-release verification boundary

The release includes a lightweight test environment that does not require network access or the full gravitational-wave software stack. It checks core configuration and numerical utilities.

The full analysis environment remains an integration path involving public external data and optional heavy dependencies. Reproducibility therefore has two levels:

1. **core numerical verification**, suitable for CI and fresh clones;
2. **full data/figure reproduction**, suitable for an analysis workstation with GWpy/PyCBC/PESummary installed.

## Deliberate exclusions

- downloaded strain and posterior data
- generated manifests and intermediate data
- generated PDF/PNG figures
- virtual environments, caches, and bytecode
- obsolete backup scripts

Public data are fetched from their authoritative archives rather than committed as duplicate binaries.

## Scientific limitations

- visualization conditioning is distinct from the LVC statistical pipeline
- the SciPy/TaylorF2 fallback is a simplified diagnostic model
- successful software tests do not constitute independent confirmation of published astrophysical parameter estimates
- remote archive availability and heavyweight dependency resolution remain external integration dependencies
