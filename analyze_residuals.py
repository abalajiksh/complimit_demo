#!/usr/bin/env python3
"""
analyze_residuals.py -- null tests, dynamics metrics, M/S degradation, plots.

For every (master, codec) pair:
  1. load reference master and decoded round-trip as float
  2. align via cross-correlation (each codec has its own fixed latency;
     AAC's ~2.6k-sample priming delay is the big one)
  3. residual = decoded - reference   (the null test)
  4. metrics: residual RMS rel. to signal (L/R and Mid/Side), crest factor,
     integrated LUFS, per-1/3-octave-band residual-to-signal ratio
Outputs:
  results.csv                      tidy long-format metrics
  plot_waveforms.png               the sausage progression, 3 masters
  plot_residual_spectrum.png       residual/signal vs frequency, per codec
  plot_ms_degradation.png          Mid vs Side residual bars

deps: pip install numpy scipy soundfile pyloudnorm matplotlib
"""

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import correlate, correlation_lags, stft

MASTERS = ["orig", "moderate", "crushed"]
CODECS = ["sbc", "aptx", "aptxhd", "aac"]
EPS = 1e-12


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load(path):
    x, sr = sf.read(path, dtype="float64", always_2d=True)  # (samples, ch)
    return x, sr


def db(x):
    return 20.0 * np.log10(np.maximum(x, EPS))


def rms(x):
    return np.sqrt(np.mean(x ** 2))


def align(ref, dec, sr, window_s=30.0):
    """Trim ref/dec to a common, sample-aligned length via cross-correlation.

    Correlates the first `window_s` seconds of the mono sums; a clean, sharp
    correlation peak doubles as a sanity check on the pipeline.
    """
    n = min(int(window_s * sr), len(ref), len(dec))
    a = ref[:n].mean(axis=1)
    b = dec[:n].mean(axis=1)
    xc = correlate(b, a, mode="full", method="fft")
    lags = correlation_lags(len(b), len(a), mode="full")
    lag = int(lags[np.argmax(xc)])

    if lag >= 0:
        dec = dec[lag:]
    else:
        ref = ref[-lag:]
    m = min(len(ref), len(dec))
    return ref[:m], dec[:m], lag


def mid_side(x):
    m = (x[:, 0] + x[:, 1]) / 2.0
    s = (x[:, 0] - x[:, 1]) / 2.0
    return m, s


def third_octave_bands(sr, nfft):
    """Return (centers, list of bin-index arrays) for 1/3-octave bands."""
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    centers = 1000.0 * (2.0 ** (np.arange(-18, 14) / 3.0))  # ~15.6 Hz .. 20 kHz
    centers = centers[centers < sr / 2 / 2 ** (1 / 6)]
    bands = []
    for fc in centers:
        lo, hi = fc / 2 ** (1 / 6), fc * 2 ** (1 / 6)
        bands.append(np.where((freqs >= lo) & (freqs < hi))[0])
    return centers, bands


def band_residual_ratio(ref, res, sr, nfft=4096):
    """Residual-to-signal power ratio per 1/3-octave band, in dB (mono sums)."""
    _, _, R = stft(ref.mean(axis=1), sr, nperseg=nfft)
    _, _, E = stft(res.mean(axis=1), sr, nperseg=nfft)
    ref_pow = np.mean(np.abs(R) ** 2, axis=1)
    res_pow = np.mean(np.abs(E) ** 2, axis=1)
    centers, bands = third_octave_bands(sr, nfft)
    ratio = np.array([
        10 * np.log10((res_pow[b].sum() + EPS) / (ref_pow[b].sum() + EPS))
        for b in bands
    ])
    return centers, ratio


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    rows = []
    band_curves = {}   # (master, codec) -> (centers, ratio)
    meter = None

    # --- reference-only metrics (the sausage table) -------------------------
    ref_audio = {}
    for m in MASTERS:
        x, sr = load(f"master_{m}.wav")
        ref_audio[m] = (x, sr)
        if meter is None:
            meter = pyln.Meter(sr)
        crest = db(np.max(np.abs(x))) - db(rms(x))
        rows.append(dict(master=m, codec="(none)", lag=0,
                         lufs=round(meter.integrated_loudness(x), 2),
                         crest_db=round(crest, 2),
                         res_L_db="", res_R_db="", res_M_db="", res_S_db=""))

    # --- codec round-trips ---------------------------------------------------
    for m in MASTERS:
        ref_full, sr = ref_audio[m]
        for c in CODECS:
            try:
                dec, sr2 = load(f"decoded_{c}_{m}.wav")
            except FileNotFoundError:
                print(f"[skip] decoded_{c}_{m}.wav not found")
                continue
            assert sr2 == sr

            ref, dec, lag = align(ref_full, dec, sr)
            res = dec - ref

            rM, rS = mid_side(ref)
            eM, eS = mid_side(res)

            crest_dec = db(np.max(np.abs(dec))) - db(rms(dec))
            row = dict(
                master=m, codec=c, lag=lag,
                lufs=round(meter.integrated_loudness(dec), 2),
                crest_db=round(crest_dec, 2),
                res_L_db=round(db(rms(res[:, 0])) - db(rms(ref[:, 0])), 2),
                res_R_db=round(db(rms(res[:, 1])) - db(rms(ref[:, 1])), 2),
                res_M_db=round(db(rms(eM)) - db(rms(rM)), 2),
                res_S_db=round(db(rms(eS)) - db(rms(rS)), 2),
            )
            rows.append(row)
            band_curves[(m, c)] = band_residual_ratio(ref, res, sr)
            print(f"[ok] {m:9s} {c:7s} lag={lag:+6d}  "
                  f"res M {row['res_M_db']:6.1f} dB  S {row['res_S_db']:6.1f} dB")

    # --- CSV -----------------------------------------------------------------
    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("[ok] results.csv")

    # --- Plot 1: waveform sausage progression --------------------------------
    fig, axes = plt.subplots(len(MASTERS), 1, figsize=(10, 7), sharex=True)
    for ax, m in zip(axes, MASTERS):
        x, sr = ref_audio[m]
        t = np.arange(len(x)) / sr
        step = max(1, len(x) // 8000)
        ax.fill_between(t[::step], x[::step, 0], -x[::step, 0],
                        color="steelblue", linewidth=0)
        ax.set_ylim(-1, 1)
        ax.set_ylabel(m)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Same track, three masters")
    fig.tight_layout()
    fig.savefig("plot_waveforms.png", dpi=150)
    print("[ok] plot_waveforms.png")

    # --- Plot 2: per-band residual, one panel per master ---------------------
    fig, axes = plt.subplots(1, len(MASTERS), figsize=(14, 4.5),
                             sharey=True, sharex=True)
    for ax, m in zip(axes, MASTERS):
        for c in CODECS:
            if (m, c) in band_curves:
                fc, ratio = band_curves[(m, c)]
                ax.semilogx(fc, ratio, label=c)
        ax.set_title(f"master: {m}")
        ax.set_xlabel("frequency (Hz)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("residual / signal (dB)")
    axes[0].legend()
    fig.suptitle("Codec damage vs frequency (closer to 0 dB = more destroyed)")
    fig.tight_layout()
    fig.savefig("plot_residual_spectrum.png", dpi=150)
    print("[ok] plot_residual_spectrum.png")

    # --- Plot 3: Mid vs Side degradation bars --------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    labels, mids, sides = [], [], []
    for m in MASTERS:
        for c in CODECS:
            r = next((r for r in rows if r["master"] == m and r["codec"] == c), None)
            if r:
                labels.append(f"{m}\n{c}")
                mids.append(r["res_M_db"])
                sides.append(r["res_S_db"])
    xpos = np.arange(len(labels))
    ax.bar(xpos - 0.2, mids, 0.4, label="Mid")
    ax.bar(xpos + 0.2, sides, 0.4, label="Side")
    ax.set_xticks(xpos, labels, fontsize=8)
    ax.set_ylabel("residual rel. to signal (dB)")
    ax.set_title("Where the codec throws information away: Mid vs Side")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("plot_ms_degradation.png", dpi=150)
    print("[ok] plot_ms_degradation.png")


if __name__ == "__main__":
    sys.exit(main())
