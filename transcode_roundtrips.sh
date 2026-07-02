#!/usr/bin/env bash
# transcode_roundtrips.sh (v2) -- full codec matrix, now including the
# degraded-link conditions where master-dependence might actually appear:
#
#   sbc328  : A2DP high-quality operating point (bitpool 53, joint stereo)
#   sbc229  : common fallback bitpool (~35) -- "phone on a congested train"
#   aptx    : fixed 4:1 ADPCM, no bitrate knob
#   aptxhd  : same scheme, ~2 extra bits/subband-sample (~576 kbps)
#   aac256  : Apple AudioToolbox encoder, Apple Music delivery rate
#   aac160  : typical Android AAC-over-Bluetooth operating point
#
# Decoded output is 32-bit float so no second quantization stacks on top of
# the codec's own damage.
set -euo pipefail

MASTERS=(orig moderate crushed)

encode_decode () {
    local master="$1" tag="$2"
    local in="master_${master}.wav"

    case "$tag" in
        sbc328)
            ffmpeg -hide_banner -loglevel warning -y -i "$in" \
                -c:a sbc -b:a 328k "enc_${tag}_${master}.sbc"
            ffmpeg -hide_banner -loglevel warning -y -i "enc_${tag}_${master}.sbc" \
                -c:a pcm_f32le "decoded_${tag}_${master}.wav" ;;
        sbc229)
            ffmpeg -hide_banner -loglevel warning -y -i "$in" \
                -c:a sbc -b:a 229k "enc_${tag}_${master}.sbc"
            ffmpeg -hide_banner -loglevel warning -y -i "enc_${tag}_${master}.sbc" \
                -c:a pcm_f32le "decoded_${tag}_${master}.wav" ;;
        aptx)
            ffmpeg -hide_banner -loglevel warning -y -i "$in" \
                -c:a aptx -f aptx "enc_${tag}_${master}.aptx"
            ffmpeg -hide_banner -loglevel warning -y -f aptx -ar 44100 -ac 2 \
                -i "enc_${tag}_${master}.aptx" \
                -c:a pcm_f32le "decoded_${tag}_${master}.wav" ;;
        aptxhd)
            ffmpeg -hide_banner -loglevel warning -y -i "$in" \
                -c:a aptx_hd -f aptx_hd "enc_${tag}_${master}.aptxhd"
            ffmpeg -hide_banner -loglevel warning -y -f aptx_hd -ar 44100 -ac 2 \
                -i "enc_${tag}_${master}.aptxhd" \
                -c:a pcm_f32le "decoded_${tag}_${master}.wav" ;;
        aac256)
            ffmpeg -hide_banner -loglevel warning -y -i "$in" \
                -c:a aac_at -b:a 256k "enc_${tag}_${master}.m4a"
            ffmpeg -hide_banner -loglevel warning -y -i "enc_${tag}_${master}.m4a" \
                -c:a pcm_f32le "decoded_${tag}_${master}.wav" ;;
        aac160)
            ffmpeg -hide_banner -loglevel warning -y -i "$in" \
                -c:a aac_at -b:a 160k "enc_${tag}_${master}.m4a"
            ffmpeg -hide_banner -loglevel warning -y -i "enc_${tag}_${master}.m4a" \
                -c:a pcm_f32le "decoded_${tag}_${master}.wav" ;;
    esac
    echo "  ${tag} done"
}

for m in "${MASTERS[@]}"; do
    [[ -f "master_${m}.wav" ]] || { echo "missing master_${m}.wav"; exit 1; }
    echo "=== ${m} ==="
    for tag in sbc328 sbc229 aptx aptxhd aac256 aac160; do
        encode_decode "$m" "$tag"
    done
done

echo
echo "All round-trips complete."
echo "Next: python3 analyze_residuals.py   (then python3 run_visqol.py)"
