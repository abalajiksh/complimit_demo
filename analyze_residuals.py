#!/usr/bin/env python3
"""
analyze_residuals.py (v2) -- null tests, dynamics, M/S, micro-dynamics, plots.

New in v2:
  * expanded codec matrix (sbc328/sbc229/aptx/aptxhd/aac256/aac160)
  * micro-dynamics: short-window crest factor (400 ms windows, 100 ms hop)
      - per-master time series plot (shows the limiter flattening in time)
      - per-codec 'transient softening' summary: mean |crest_dec - crest_ref|
  * merges visqol_results.csv into results.csv if run_visqol.py has been run

Outputs:
  results.csv
  plot_waveforms.png            sausage progression
  plot_residual_spectrum.png    residual/signal vs frequency
  plot_ms_degradation.png       Mid vs Side residual bars
  plot_microdynamics.png        short-window crest over time, per master

deps: pip install numpy scipy soundfile pyloudnorm matplotlib
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import correlate, correlation_lags, stft

MASTERS = ["orig", "moderate", "crushed"]
CODECS = ["sbc328", "sbc229", "aptx", "aptxhd", "aac256", "aac160"]
EPS = 1e-12

WIN_S = 0.400   # micro-dynamics window
HOP_S = 0.100


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
    """Sample-align via cross-correlation of the mono sums."""
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
    return (x[:, 0] + x[:, 1]) / 2.0, (x[:, 0] - x[:, 1]) / 2.0


def short_window_crest(x, sr):
    """Crest factor (dB) per short window, mono sum. Returns (t, crest)."""
    mono = x.mean(axis=1)
    win = int(WIN_S * sr)
    hop = int(HOP_S * sr)
    starts = np.arange(0, len(mono) - win, hop)
    t = (starts + win / 2) / sr
    crest = np.empty(len(starts))
    for i, s in enumerate(starts):
        seg = mono[s:s + win]
        crest[i] = db(np.max(np.abs(seg))) - db(rms(seg))
    return t, crest


def third_octave_bands(sr, nfft):
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    centers = 1000.0 * (2.0 ** (np.arange(-18, 14) / 3.0))
    centers = centers[centers < sr / 2 / 2 ** (1 / 6)]
    bands = [np.where((freqs >= fc / 2 ** (1 / 6)) &
                      (freqs < fc * 2 ** (1 / 6)))[0] for fc in centers]
    return centers, bands


def band_residual_ratio(ref, res, sr, nfft=4096):
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


def load_visqol_scores():
    """Return {(master, codec): mos} from visqol_results.csv, if present."""
    scores = {}
    if os.path.exists("visqol_results.csv"):
        with open("visqol_results.csv") as f:
            for row in csv.DictReader(f):
                scores[(row["master"], row["codec"])] = row["moslqo"]
        print(f"[info] merged {len(scores)} ViSQOL scores")
    else:
        print("[info] visqol_results.csv not found -- run run_visqol.py to add MOS")
    return scores


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    rows = []
    band_curves = {}
    crest_series = {}
    visqol = load_visqol_scores()
    meter = None

    ref_audio = {}
    for m in MASTERS:
        x, sr = load(f"master_{m}.wav")
        ref_audio[m] = (x, sr)
        if meter is None:
            meter = pyln.Meter(sr)
        t, cw = short_window_crest(x, sr)
        crest_series[m] = (t, cw)
        crest = db(np.max(np.abs(x))) - db(rms(x))
        rows.append(dict(master=m, codec="(none)", lag=0,
                         lufs=round(meter.integrated_loudness(x), 2),
                         crest_db=round(crest, 2),
                         mdyn_median_db=round(float(np.median(cw)), 2),
                         mdyn_soften_db="",
                         res_L_db="", res_R_db="", res_M_db="", res_S_db="",
                         visqol_moslqo=""))

    for m in MASTERS:
        ref_full, sr = ref_audio[m]
        _, cw_ref = short_window_crest(ref_full, sr)
        for c in CODECS:
            fname = f"decoded_{c}_{m}.wav"
            try:
                dec, sr2 = load(fname)
            except (FileNotFoundError, sf.LibsndfileError):
                print(f"[skip] {fname} not found")
                continue
            assert sr2 == sr

            ref, dec, lag = align(ref_full, dec, sr)
            res = dec - ref

            rM, rS = mid_side(ref)
            eM, eS = mid_side(res)

            # micro-dynamics: how much does the codec alter short-window crest?
            _, cw_dec = short_window_crest(dec, sr)
            nw = min(len(cw_ref), len(cw_dec))
            soften = float(np.mean(np.abs(cw_dec[:nw] - cw_ref[:nw])))

            crest_dec = db(np.max(np.abs(dec))) - db(rms(dec))
            row = dict(
                master=m, codec=c, lag=lag,
                lufs=round(meter.integrated_loudness(dec), 2),
                crest_db=round(crest_dec, 2),
                mdyn_median_db=round(float(np.median(cw_dec)), 2),
                mdyn_soften_db=round(soften, 3),
                res_L_db=round(db(rms(res[:, 0])) - db(rms(ref[:, 0])), 2),
                res_R_db=round(db(rms(res[:, 1])) - db(rms(ref[:, 1])), 2),
                res_M_db=round(db(rms(eM)) - db(rms(rM)), 2),
                res_S_db=round(db(rms(eS)) - db(rms(rS)), 2),
                visqol_moslqo=visqol.get((m, c), ""),
            )
            rows.append(row)
            band_curves[(m, c)] = band_residual_ratio(ref, res, sr)
            print(f"[ok] {m:9s} {c:7s} lag={lag:+6d}  "
                  f"res M {row['res_M_db']:6.1f}  S {row['res_S_db']:6.1f} dB  "
                  f"soften {soften:5.3f} dB  MOS {row['visqol_moslqo'] or '-'}")

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

    # --- Plot 2: per-band residual --------------------------------------------
    fig, axes = plt.subplots(1, len(MASTERS), figsize=(15, 4.5),
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
    axes[0].legend(fontsize=8)
    fig.suptitle("Codec damage vs frequency (closer to 0 dB = more destroyed)")
    fig.tight_layout()
    fig.savefig("plot_residual_spectrum.png", dpi=150)
    print("[ok] plot_residual_spectrum.png")

    # --- Plot 3: Mid vs Side bars ---------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 5))
    labels, mids, sides = [], [], []
    for m in MASTERS:
        for c in CODECS:
            r = next((r for r in rows
                      if r["master"] == m and r["codec"] == c), None)
            if r:
                labels.append(f"{m}\n{c}")
                mids.append(r["res_M_db"])
                sides.append(r["res_S_db"])
    xpos = np.arange(len(labels))
    ax.bar(xpos - 0.2, mids, 0.4, label="Mid")
    ax.bar(xpos + 0.2, sides, 0.4, label="Side")
    ax.set_xticks(xpos, labels, fontsize=7)
    ax.set_ylabel("residual rel. to signal (dB)")
    ax.set_title("Where the codec throws information away: Mid vs Side")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("plot_ms_degradation.png", dpi=150)
    print("[ok] plot_ms_degradation.png")

    # --- Plot 4: micro-dynamics over time --------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    for m in MASTERS:
        t, cw = crest_series[m]
        ax.plot(t, cw, label=m, linewidth=1.2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("short-window crest factor (dB)")
    ax.set_title(f"Micro-dynamics: crest per {int(WIN_S*1000)} ms window "
                 "-- what the limiter actually flattens")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("plot_microdynamics.png", dpi=150)
    print("[ok] plot_microdynamics.png")


if __name__ == "__main__":
    sys.exit(main())
