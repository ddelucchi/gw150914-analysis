from __future__ import annotations

import numpy as np

from src.config import load_config
from src.preprocess import welch_asd, whiten_explicit
from src.synth_waveform import (
    MSUN_S,
    chirp_mass_msun,
    f_isco,
    matched_filter_snr_series,
    newtonian_chirp,
    ringdown_tail,
    taylorf2_strain,
    time_domain_chirp,
)


def test_master_config_has_consistent_event_window() -> None:
    cfg = load_config()
    merger = float(cfg["event"]["gps_merger"])
    start = float(cfg["data"]["fetch_start_gps"])
    end = float(cfg["data"]["fetch_end_gps"])

    assert cfg["event"]["name"] == "GW150914"
    assert start < merger < end
    assert int(cfg["data"]["master_sample_rate"]) >= int(
        cfg["data"]["quicklook_sample_rate"]
    )
    assert float(cfg["analysis"]["highpass_hz"]) < float(
        cfg["analysis"]["template_f_low_hz"]
    )


def test_chirp_mass_is_symmetric_and_physically_bounded() -> None:
    m12 = chirp_mass_msun(36.0, 29.0)
    m21 = chirp_mass_msun(29.0, 36.0)

    assert np.isclose(m12, m21, rtol=0.0, atol=1e-12)
    assert 0.0 < m12 < 36.0 + 29.0
    assert f_isco(36.0 + 29.0) > 0.0


def test_taylorf2_has_expected_frequency_support() -> None:
    freqs = np.linspace(0.0, 512.0, 4097)
    h = taylorf2_strain(freqs, 36.0, 29.0, f_low=20.0)

    assert h.shape == freqs.shape
    assert np.all(np.isfinite(h.real))
    assert np.all(np.isfinite(h.imag))
    assert np.all(h[freqs < 20.0] == 0.0)
    assert np.any(np.abs(h[freqs >= 20.0]) > 0.0)


def test_time_domain_chirp_is_finite_and_nonzero() -> None:
    t, h = time_domain_chirp(
        fs=1024.0,
        duration_s=2.0,
        m1_msun=36.0,
        m2_msun=29.0,
        f_low=20.0,
    )

    assert len(t) == len(h)
    assert len(h) == 2048
    assert np.all(np.isfinite(h))
    assert np.max(np.abs(h)) > 0.0


def test_welch_asd_and_whitening_are_finite() -> None:
    rng = np.random.default_rng(20150914)
    fs = 1024.0
    x = rng.normal(size=8192)

    freqs, asd = welch_asd(
        x,
        fs=fs,
        seglen_s=1.0,
        step_s=0.5,
        average="mean",
    )

    assert len(freqs) == len(asd)
    assert np.all(np.isfinite(asd))
    assert np.all(asd >= 0.0)

    white = whiten_explicit(x, fs, freqs, asd, rescale_unit_var=True)
    assert np.all(np.isfinite(white))
    assert abs(float(np.mean(white))) < 1e-10
    assert np.isclose(float(np.std(white)), 1.0, atol=1e-10)


def test_ringdown_is_causal() -> None:
    t = np.linspace(-0.01, 0.04, 1000)
    h = ringdown_tail(t, tc=0.0, f_rd=250.0, tau_ms=4.0)

    assert np.all(h[t < 0.0] == 0.0)
    assert np.any(np.abs(h[t >= 0.0]) > 0.0)
    assert np.all(np.isfinite(h))


def test_newtonian_chirp_frequency_obeys_leading_order_chirp_mass_scaling() -> None:
    t = np.array([-1.0])
    tc = 0.0
    m1 = 30.0
    m2 = 30.0
    _, f = newtonian_chirp(t, tc, m1, m2, f_low=0.0)

    mc_s = chirp_mass_msun(m1, m2) * MSUN_S
    tau = tc - t[0]
    expected = (
        (5.0 ** (3.0 / 8.0))
        / (8.0 * np.pi)
        * mc_s ** (-5.0 / 8.0)
        * tau ** (-3.0 / 8.0)
    )

    assert f[0] > 0.0
    assert np.isclose(f[0], expected, rtol=1e-12, atol=0.0)


def test_matched_filter_self_template_peak_has_standard_norm() -> None:
    fs = 1024.0
    n = 4096
    time = np.arange(n) / fs
    template = (
        np.sin(2.0 * np.pi * 64.0 * time)
        + 0.35 * np.sin(2.0 * np.pi * 91.0 * time + 0.2)
    )

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = np.ones_like(freqs)
    f_low = 20.0

    snr = matched_filter_snr_series(
        template,
        template,
        fs,
        freqs,
        psd,
        f_low=f_low,
    )

    hf = np.fft.rfft(template)
    mask = freqs >= f_low
    sigma2 = 4.0 / (fs * n) * np.sum(np.abs(hf[mask]) ** 2)
    expected_peak = np.sqrt(sigma2)

    assert np.all(np.isfinite(snr))
    assert np.isclose(np.max(snr), expected_peak, rtol=1e-12, atol=1e-12)
