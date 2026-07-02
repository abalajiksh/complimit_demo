#!/usr/bin/env python3
"""
make_masters.py -- generate 'moderate' and 'crushed' masters from the CD rip.

Chain per master: LSP Compressor Stereo -> Gain (loudness push) -> LSP Limiter Stereo,
with the gain stage iterated until integrated LUFS hits the target.
Falls back to pedalboard's built-in Compressor/Limiter if the LSP VST3
bundle is not installed.

Install LSP for macOS (VST3) from https://lsp-plug.in/?page=download
then confirm the bundle lands in /Library/Audio/Plug-Ins/VST3/.

deps: pip install pedalboard pyloudnorm numpy soundfile
"""

import shutil
import sys

import numpy as np
import pyloudnorm as pyln
from pedalboard import Pedalboard, Gain, Compressor, Limiter
from pedalboard.io import AudioFile

SOURCE = "HorizonOverture_DaftPunk.wav"
VST3_BUNDLE = "/Library/Audio/Plug-Ins/VST3/lsp-plugins.vst3"

# ---------------------------------------------------------------------------
# Targets. Loudness targets are the control variable; compressor settings are
# chosen to imitate real mastering decisions, not a caricature:
#   moderate: gentle 2.5:1 bus compression, slow attack, limiter barely working
#   crushed:  faster/deeper 4:1, then shoved hard into the limiter
# ---------------------------------------------------------------------------
TARGETS = {
    "moderate": {
        "lufs": -11.0,
        "comp": dict(threshold_db=-18.0, ratio=2.5, attack_ms=30.0, release_ms=250.0),
    },
    "crushed": {
        "lufs": -7.0,
        "comp": dict(threshold_db=-24.0, ratio=4.0, attack_ms=10.0, release_ms=120.0),
    },
}

LIMITER_CEILING_DB = -1.0  # true-peak safety margin, also what codecs prefer


# ---------------------------------------------------------------------------
# Plugin loading
# ---------------------------------------------------------------------------
def try_load_lsp():
    """Return (compressor_factory, limiter_factory) using LSP VST3, or None."""
    try:
        from pedalboard import VST3Plugin
    except ImportError:
        return None
    import os

    if not os.path.exists(VST3_BUNDLE):
        print(f"[info] LSP VST3 bundle not found at {VST3_BUNDLE}")
        return None

    names = VST3Plugin.get_plugin_names_for_file(VST3_BUNDLE)
    comp_name = next((n for n in names if "Compressor Stereo" in n and "Sidechain" not in n), None)
    lim_name = next((n for n in names if "Limiter Stereo" in n and "Multiband" not in n), None)
    if not (comp_name and lim_name):
        print("[info] Could not find 'Compressor Stereo' / 'Limiter Stereo' in bundle.")
        print("       Available plugins:")
        for n in names:
            print("        ", n)
        return None

    def set_params(plugin, wanted: dict):
        """Set parameters by fuzzy attribute name; print what's available on miss."""
        available = list(plugin.parameters.keys())
        for key, value in wanted.items():
            matches = [a for a in available if key in a]
            if len(matches) == 1:
                setattr(plugin, matches[0], value)
            else:
                print(f"[warn] parameter '{key}' -> {matches or 'no match'}; "
                      f"inspect plugin.parameters and fix the mapping below.")

    def make_comp(cfg):
        p = VST3Plugin(VST3_BUNDLE, plugin_name=comp_name)
        # LSP parameter names (snake_cased by pedalboard); verify once by
        # printing p.parameters.keys() -- they are stable across LSP versions.
        set_params(p, {
            "attack_threshold": cfg["threshold_db"],
            "ratio": cfg["ratio"],
            "attack_time": cfg["attack_ms"],
            "release_time": cfg["release_ms"],
            "makeup_gain": 0.0,
        })
        return p

    def make_lim():
        p = VST3Plugin(VST3_BUNDLE, plugin_name=lim_name)
        set_params(p, {
            "threshold": LIMITER_CEILING_DB,
            "lookahead": 5.0,
        })
        return p

    print(f"[info] Using LSP plugins: '{comp_name}' + '{lim_name}'")
    return make_comp, make_lim


def builtin_factories():
    def make_comp(cfg):
        return Compressor(threshold_db=cfg["threshold_db"], ratio=cfg["ratio"],
                          attack_ms=cfg["attack_ms"], release_ms=cfg["release_ms"])

    def make_lim():
        return Limiter(threshold_db=LIMITER_CEILING_DB, release_ms=100.0)

    print("[info] Using pedalboard built-in Compressor/Limiter (LSP not loaded).")
    return make_comp, make_lim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    factories = try_load_lsp() or builtin_factories()
    make_comp, make_lim = factories

    with AudioFile(SOURCE) as f:
        audio = f.read(f.frames)          # shape: (channels, samples), float32
        sr = int(f.samplerate)

    meter = pyln.Meter(sr)
    src_lufs = meter.integrated_loudness(audio.T)
    print(f"[info] source: {src_lufs:.2f} LUFS integrated")

    # The untouched reference master -- identical provenance to the others.
    shutil.copyfile(SOURCE, "master_orig.wav")
    print("[ok]   master_orig.wav (verbatim copy)")

    for name, cfg in TARGETS.items():
        gain_db = cfg["lufs"] - src_lufs   # decent first guess
        out = None
        for i in range(8):
            board = Pedalboard([make_comp(cfg["comp"]), Gain(gain_db), make_lim()])
            out = board(audio, sr)
            lufs = meter.integrated_loudness(out.T)
            err = cfg["lufs"] - lufs
            print(f"  [{name}] iter {i}: gain {gain_db:+.2f} dB -> {lufs:.2f} LUFS")
            if abs(err) < 0.2:
                break
            gain_db += err

        peak_db = 20 * np.log10(np.max(np.abs(out)) + 1e-12)
        crest = peak_db - (20 * np.log10(np.sqrt(np.mean(out ** 2)) + 1e-12))
        print(f"  [{name}] final: {lufs:.2f} LUFS, peak {peak_db:.2f} dBFS, "
              f"crest {crest:.1f} dB, gain into limiter {gain_db:+.1f} dB")

        out16 = np.clip(out, -1.0, 1.0)
        fname = f"master_{name}.wav"
        with AudioFile(fname, "w", samplerate=sr,
                       num_channels=out16.shape[0], bit_depth=16) as f:
            f.write(out16)
        print(f"[ok]   {fname}")

    print("\nNext: bash transcode_roundtrips.sh")


if __name__ == "__main__":
    sys.exit(main())
