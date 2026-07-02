# complim -- Compressor/Limiter vs. Bluetooth Codecs

Null-test experiment for the blog post on what mastering compression actually
does to music, versus what Bluetooth-class codecs do to it afterwards.

**Source:** Horizon (Ouverture), Daft Punk, *Random Access Memories* 10th
Anniversary Edition, own CD rip (16-bit/44.1 kHz, gvfs-cdda).

**Hypothesis under test:** heavy mastering compression "protects" music through
lossy Bluetooth codecs (SBC/aptX/AAC) because there is less fine detail left
to destroy -- at the cost of soundstage and micro-dynamics.

**Verdict:** falsified in every tested condition. See Findings.

## Pipeline

```
HorizonOverture_DaftPunk.wav        (CD rip, the untouched reference)
  |  make_masters.py                pedalboard: compressor -> gain -> limiter,
  |                                 LUFS-converged; LSP VST3 if installed,
  |                                 built-ins otherwise
  v
master_{orig,moderate,crushed}.wav  (16-bit; -17.1 / -10.8 / -7.2 LUFS)
  |  transcode_roundtrips.sh        ffmpeg encode + decode per codec
  v
decoded_{codec}_{master}.wav        codec in {sbc328, sbc229, aptx, aptxhd,
  |                                           aac256, aac160}
  |  analyze_residuals.py           cross-correlation alignment, null test,
  |                                 M/S split, short-window crest (400 ms)
  v
results.csv + 4 plots
```

Run order:

```bash
python3 -m venv venv && source venv/bin/activate
pip install pedalboard pyloudnorm numpy scipy soundfile matplotlib
python3 make_masters.py
bash transcode_roundtrips.sh
python3 analyze_residuals.py
```

Environment used: macOS (Apple Silicon), ffmpeg with native sbc/aptx and the
AudioToolbox AAC encoder (`aac_at`) -- i.e. Apple's own encoder, the closest
proxy for the AirPods-era delivery chain.

## The three masters

| master   | LUFS-I | crest factor | chain                                          |
|----------|--------|--------------|------------------------------------------------|
| orig     | -17.1  | 18.6 dB      | untouched rip                                   |
| moderate | -10.8  | 13.7 dB      | 2.5:1, slow attack, limiter barely working      |
| crushed  | -7.2   | 10.1 dB      | 4:1, fast, ~10 dB of gain shoved into limiter   |

(Aside: the rip measures LRA 6.7 LU yet crest 18.6 dB -- LRA and DR/crest are
different quantities, macrodynamics vs. transient headroom.)

## Findings

1. **Crushing does not protect music from waveform codecs.** SBC and aptX
   residuals (relative to signal) are flat across all three masters at both
   SBC operating points (~-44 dB at bitpool 53, ~-36.5 dB at bitpool 35).
   Fixed-rate subband/ADPCM quantizers have signal-proportional error:
   constant SNR, indifferent to mastering.

2. **The perceptual codec punishes crushing.** AAC residuals worsen
   monotonically as the master is crushed: -34.8 -> -33.5 -> -31.9 dB at
   256k, -28.7 -> -28.0 -> -26.9 dB at 160k. A dense, always-loud spectrum
   leaves the psychoacoustic model less masking headroom and fewer bits per
   detail. The hypothesis is not just dead -- for AAC it runs backwards.

3. **Quantization theory, confirmed empirically.** aptX -> aptX HD improves
   the residual by 12.2 dB; ~2 extra bits/subband-sample x 6.02 dB/bit
   predicts 12.0 dB. SBC bitpool 53 -> 35 costs ~8 dB.

4. **No joint-stereo starvation observed.** The Mid/Side residual gap is
   ~1 dB everywhere -- including aptX, which codes L/R as independent mono
   and cannot starve a Side channel it never computes. The gap is a property
   of the track's stereo content, not codec behaviour. The "Bluetooth smears
   the soundstage" claim is unsupported at every operating point tested
   (caveat: file round-trips cannot capture real-link packet loss and
   concealment; n = 1 track).

5. **Mastering vs. codec, in two numbers.** The limiter chain moved crest
   factor by 8.45 dB. The worst codec-induced softening of short-window
   crest in the whole matrix is 0.107 dB -- roughly 80x less. And codecs
   soften the crushed master slightly *more* than the original, not less.

6. **Residual RMS is not audibility.** AAC posts the worst RMS residual of
   all codecs while being, at 256k, perceptually the strongest -- its error
   is masking-shaped and much of the RMS lives in a >16 kHz lowpass.
   Quantifying this properly needs a perceptual metric (ViSQOL); parked.

Sanity notes: measured codec latencies SBC +73, aptX/HD +90 samples; AAC
lag = 0 because aac_at writes priming/edit-list metadata that ffmpeg honours
on decode.

**Thesis for the post:** the destruction attributed to Bluetooth codecs is
overwhelmingly done earlier, at mastering. Mastering is the damage; the
codec is the alibi.
