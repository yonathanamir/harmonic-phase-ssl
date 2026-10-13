"""Free-field simulation of the synthetic SSL datasets.

Follows Nakadai et al. (IEEE/SICE SII 2020), with the SNR sweep of Itoyama et al.
(Interspeech 2021) and a random start phase per sinusoid:

  * 8-microphone array: circular (R = 0.1 m, the published array) or linear (spacing
    2R sin(pi/8) = 0.0765 m, so both arrays share the spatial-aliasing limit c/(2d))
  * point source at distance r ~ U[0.5, 1.5] m and azimuth theta on a 1 degree grid
  * exact spherical-wave propagation: delay r_m / c and gain 1 / r_m per microphone
  * source: 100 sinusoids, A_i ~ U(0, 1], f_i ~ U(0, 2000] Hz, phi_i ~ U[0, 2 pi)
  * white Gaussian noise per channel, SNR set on the array-average signal power
  * features: one 2048-point FFT per channel (fs = 16 kHz, no window), bins 1..256

Each split is one .npz file:
  phase     (N, 8, 256) float32   phase in (-pi, pi]
  mag       (N, 8, 256) float32   magnitude (used by the weighted embeddings)
  label     (N,)        int16     azimuth in degrees (0..359); training maps it to 72 classes
  theta_deg (N,)        float32   azimuth
  dist_m    (N,)        float32   source distance
  snr_db    (N,)        float32   SNR of the sample (inf = clean)
  meta      ()          str       JSON description of the generation

The source phases come from a second random stream derived from the seed, so the main
stream (labels, distances, SNRs, frequencies, amplitudes, noise) is consumed in a fixed
order: a given seed always gives the same data.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

FS = 16_000                  # sampling rate [Hz]
N_FFT = 2_048                # frame length [samples]
BIN_LO, BIN_HI = 1, 257      # kept rFFT bins: 256 bins up to 2 kHz
N_MICS = 8
ARRAY_RADIUS = 0.10          # [m]
SPEED_OF_SOUND = 343.0       # [m/s]
DIST_RANGE = (0.5, 1.5)      # source distance [m]
N_SINUSOIDS = 100
F_MAX = 2000.0               # sinusoid frequencies U(0, F_MAX] [Hz]
SNRS = (np.inf, 40.0, 30.0, 20.0, 10.0, 0.0)
SNR_NAMES = ("inf", "40dB", "30dB", "20dB", "10dB", "0dB")
PHASE_STREAM = 0x5048415345  # spawn key of the source-phase stream
ULA_SPACING = 2 * ARRAY_RADIUS * np.sin(np.pi / N_MICS)


def circular_array(n=N_MICS, radius=ARRAY_RADIUS):
    a = np.arange(n) * 2 * np.pi / n
    return radius * np.stack([np.cos(a), np.sin(a)], axis=1)


def linear_array(n=N_MICS, spacing=ULA_SPACING):
    x = (np.arange(n) - (n - 1) / 2) * spacing
    return np.stack([x, np.zeros(n)], axis=1)


# Internal names are kept from the experiments, so result files stay comparable.
ARRAYS = {"circular": ("uca-r10", circular_array()), "linear": ("ula-8", linear_array())}


def delays_and_gains(theta_deg, dist_m, mic_pos):
    th = np.deg2rad(theta_deg)
    src = dist_m[:, None] * np.stack([np.cos(th), np.sin(th)], axis=1)
    r = np.linalg.norm(src[:, None, :] - mic_pos[None], axis=2)      # (B, M)
    return r / SPEED_OF_SOUND, 1.0 / r


def synthesize(freqs, amps, phases, theta_deg, dist_m, mic_pos):
    """x_m[n] = sum_i (A_i / r_m) sin(2 pi f_i (n / fs - tau_m) + phi_i), shape (B, M, N_FFT)."""
    tau, gain = delays_and_gains(theta_deg, dist_m, mic_pos)
    w = np.exp(2j * np.pi * freqs / FS).astype(np.complex64)          # (B, I)
    e = np.empty((*w.shape, N_FFT), np.complex64)                     # e^{j 2 pi f_i n / fs}
    e[:] = w[:, :, None]
    e[:, :, 0] = 1.0
    e = np.cumprod(e, axis=2)
    c = (amps[:, None, :]
         * np.exp(1j * (phases[:, None, :] - 2 * np.pi * freqs[:, None, :] * tau[:, :, None]))
         * gain[:, :, None])
    return (c.astype(np.complex64) @ e).imag


def add_noise(x, snr_db, rng):
    """sigma^2 = P 10^(-SNR/10), P = mean signal power over the array."""
    out = x.copy()
    finite = np.isfinite(snr_db)
    if not finite.any():
        return out
    power = np.mean(x[finite] ** 2, axis=(1, 2))
    sigma = np.sqrt(power * 10.0 ** (-snr_db[finite] / 10.0))
    noise = rng.standard_normal(x[finite].shape).astype(np.float32)
    out[finite] += sigma[:, None, None].astype(np.float32) * noise
    return out


def features(x):
    spectrum = np.fft.rfft(x, n=N_FFT, axis=2)[:, :, BIN_LO:BIN_HI]
    return np.angle(spectrum).astype(np.float32), np.abs(spectrum).astype(np.float32)


def generate_split(n, snrs, rng, prng, mic_pos, batch=128, tag=""):
    """n samples; the SNR of each sample is drawn uniformly from `snrs`."""
    phase = np.empty((n, len(mic_pos), BIN_HI - BIN_LO), np.float32)
    mag = np.empty_like(phase)
    label = np.empty(n, np.int16)
    theta = np.empty(n, np.float32)
    dist = np.empty(n, np.float32)
    snr = np.empty(n, np.float32)
    t0 = time.time()
    for lo in range(0, n, batch):
        hi = min(lo + batch, n)
        b = hi - lo
        degrees = rng.integers(0, 360, b)
        r = rng.uniform(*DIST_RANGE, b)
        s = np.asarray(snrs, np.float32)[rng.integers(0, len(snrs), b)]
        freqs = F_MAX * (1.0 - rng.random((b, N_SINUSOIDS)))         # U(0, F_MAX]
        amps = 1.0 - rng.random((b, N_SINUSOIDS))                    # U(0, 1]
        phis = 2 * np.pi * prng.random((b, N_SINUSOIDS))
        x = add_noise(synthesize(freqs, amps, phis, degrees * 1.0, r, mic_pos), s, rng)
        phase[lo:hi], mag[lo:hi] = features(x)
        label[lo:hi], theta[lo:hi], dist[lo:hi], snr[lo:hi] = degrees, degrees, r, s
        if lo // batch % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{tag}] {hi:,}/{n:,}  ({elapsed:.0f}s)", flush=True)
    return dict(phase=phase, mag=mag, label=label, theta_deg=theta, dist_m=dist, snr_db=snr)


def save_split(path: Path, arrays: dict, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")      # a complete file only, never a partial one
    with temporary.open("wb") as stream:
        np.savez(stream, meta=json.dumps(meta), **arrays)
    temporary.replace(path)
    print(f"  wrote {path} ({path.stat().st_size / 2**20:,.0f} MB)", flush=True)


def generate(out: Path, array: str, seed: int, n_train: int, n_val: int, n_test: int,
             splits=("train", "val", "test"), train_name="train"):
    """Write the requested splits of one random stream (train, then val, then the tests)."""
    name, mic_pos = ARRAYS[array]
    rng, prng = np.random.default_rng(seed), np.random.default_rng([seed, PHASE_STREAM])
    meta = {"array": name, "seed": seed, "resolution": 1.0, "fs": FS, "n_fft": N_FFT,
            "mic_xy": np.round(mic_pos, 9).tolist(), "c": SPEED_OF_SOUND,
            "dist_range_m": list(DIST_RANGE), "n_sinusoids": N_SINUSOIDS,
            "source_phase": "random", "snrs": list(SNR_NAMES)}
    if "train" in splits:
        save_split(out / f"{train_name}.npz",
                   generate_split(n_train, SNRS, rng, prng, mic_pos, tag=train_name), meta)
    if "val" in splits:
        save_split(out / "val.npz", generate_split(n_val, SNRS, rng, prng, mic_pos, tag="val"), meta)
    if "test" in splits:
        for snr, snr_name in zip(SNRS, SNR_NAMES):
            save_split(out / f"test_snr_{snr_name}.npz",
                       generate_split(n_test, [snr], rng, prng, mic_pos, tag=f"test {snr_name}"),
                       meta)
