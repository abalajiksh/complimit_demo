#!/usr/bin/env bash
# transcode_roundtrips.sh -- encode each master through each Bluetooth-class
# codec and decode straight back to PCM. macOS build of ffmpeg: SBC and aptX
# are native, AAC uses Apple's own AudioToolbox encoder (aac_at).
#
# Decoded files are written as 32-bit float so no second quantization is
# stacked on top of the codec's own damage (negligible either way, but free).
set -euo pipefail

MASTERS=(orig moderate crushed)

for m in "${MASTERS[@]}"; do
    in="master_${m}.wav"
    [[ -f "$in" ]] || { echo "missing $in -- run make_masters.py first"; exit 1; }

    echo "=== ${m} ==="

    # --- SBC: A2DP high-quality operating point (bitpool 53 joint stereo,
    # ~328 kbps at 44.1 kHz). The .sbc container is self-describing.
    ffmpeg -hide_banner -loglevel warning -y -i "$in" \
        -c:a sbc -b:a 328k "enc_sbc_${m}.sbc"
    ffmpeg -hide_banner -loglevel warning -y -i "enc_sbc_${m}.sbc" \
        -c:a pcm_f32le "decoded_sbc_${m}.wav"
    echo "  sbc      done"

    # --- aptX: fixed 4:1 ADPCM, no bitrate knob (~352 kbps stereo/44.1).
    # Raw .aptx stream is headerless -> the decoder must be told the format.
    ffmpeg -hide_banner -loglevel warning -y -i "$in" \
        -c:a aptx -f aptx "enc_aptx_${m}.aptx"
    ffmpeg -hide_banner -loglevel warning -y -f aptx -ar 44100 -ac 2 \
        -i "enc_aptx_${m}.aptx" -c:a pcm_f32le "decoded_aptx_${m}.wav"
    echo "  aptx     done"

    # --- aptX HD: same scheme at 24-bit/~576 kbps. Bonus column for the
    # "HD" marketing commentary.
    ffmpeg -hide_banner -loglevel warning -y -i "$in" \
        -c:a aptx_hd -f aptx_hd "enc_aptxhd_${m}.aptxhd"
    ffmpeg -hide_banner -loglevel warning -y -f aptx_hd -ar 44100 -ac 2 \
        -i "enc_aptxhd_${m}.aptxhd" -c:a pcm_f32le "decoded_aptxhd_${m}.wav"
    echo "  aptx_hd  done"

    # --- AAC via Apple's AudioToolbox encoder: the closest thing to the
    # actual AirPods-era encode chain. 256k CBR mirrors Apple Music delivery.
    ffmpeg -hide_banner -loglevel warning -y -i "$in" \
        -c:a aac_at -b:a 256k "enc_aac_${m}.m4a"
    ffmpeg -hide_banner -loglevel warning -y -i "enc_aac_${m}.m4a" \
        -c:a pcm_f32le "decoded_aac_${m}.wav"
    echo "  aac_at   done"
done

echo
echo "All round-trips complete. Next: python3 analyze_residuals.py"
