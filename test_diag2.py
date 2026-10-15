#!/usr/bin/env python3
"""Diagnostic: trace ACC and burst levels."""
import sys
import numpy as np
sys.path.insert(0, '/home/user/vhs-emulator')

from vhs.encode import VHSEncoder, generate_test_composite
from vhs.decode import VHSDecoder, VideoStandard, _design_butterworth
import scipy.signal as sp_signal

SAMPLE_RATE = 14_318_180

# Generate and encode
composite = generate_test_composite(
    sample_rate=SAMPLE_RATE, duration_s=262.0 * 63.556e-6,
    pattern="colorbars", standard="NTSC"
)
encoder = VHSEncoder(standard="NTSC")
enc_result = encoder.encode_video(composite, SAMPLE_RATE)
y_fm = enc_result.y_fm
c_under = enc_result.c_under

x_old = np.linspace(0, 1, len(c_under))
x_new = np.linspace(0, 1, len(y_fm))
c_under_full = np.interp(x_new, x_old, c_under)
tape_signal = y_fm + 0.375 * c_under_full

# Run preamp+EQ
n = len(tape_signal)
dec = VHSDecoder(standard=VideoStandard.NTSC)
signal = tape_signal.astype(np.float64)
signal = dec._preamplify_fs(signal, SAMPLE_RATE)
signal = dec._playback_eq(signal, SAMPLE_RATE)

# Extract chrominance
chroma_629k = dec._extract_chroma(signal, SAMPLE_RATE)
print(f"After 629 kHz BPF: rms={np.sqrt(np.mean(chroma_629k**2)):.4f}")

# Upconvert - but manually so we can see the ACC internals
f_lo = dec.f_subcarrier - dec.f_chroma_under
n_samples = len(chroma_629k)
t = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE

# No burst_phase correction for now
lo_signal = 2.0 * np.cos(2.0 * np.pi * f_lo * t)
mixed = chroma_629k * lo_signal

# BPF at subcarrier
f_sub = dec.f_subcarrier
bw = 1.0e6
b_bp, a_bp = _design_butterworth(
    order=4,
    cutoff_hz=[f_sub - bw / 2.0, f_sub + bw / 2.0],
    sample_rate=SAMPLE_RATE,
    btype="band",
)
mixed_bpf = sp_signal.lfilter(b_bp, a_bp, mixed)
print(f"After upconvert BPF: rms={np.sqrt(np.mean(mixed_bpf**2)):.4f}, range=[{mixed_bpf.min():.4f}, {mixed_bpf.max():.4f}]")

# Measure burst on several lines
samples_per_line = SAMPLE_RATE / dec.line_rate
burst_start_us = 5.7
burst_end_us = 8.2
burst_start_samp = int(burst_start_us * 1e-6 * SAMPLE_RATE)
burst_end_samp = int(burst_end_us * 1e-6 * SAMPLE_RATE)
burst_len = burst_end_samp - burst_start_samp

print(f"\n=== Burst measurement (after upconvert BPF) ===")
print(f"  samples_per_line = {samples_per_line:.0f}")
print(f"  burst_start_samp = {burst_start_samp}, burst_end_samp = {burst_end_samp}, burst_len = {burst_len}")

burst_rms_vals = []
for li in range(3, 50):
    ls = int(li * samples_per_line)
    bp_s = ls + burst_start_samp
    bp_e = ls + burst_end_samp
    if bp_e < n_samples:
        burst_region = mixed_bpf[bp_s:bp_e]
        rms = np.sqrt(np.mean(burst_region**2))
        peak = np.max(np.abs(burst_region))
        burst_rms_vals.append(rms)
        if li < 8:
            print(f"  Line {li:3d}: burst_rms={rms:.6f}, burst_peak={peak:.6f}")

burst_rms = float(np.median(burst_rms_vals))
print(f"  Median burst RMS across {len(burst_rms_vals)} lines: {burst_rms:.6f}")
print(f"  ACC target = 0.10, ACC gain = {0.10/max(burst_rms, 1e-12):.4f}")
print(f"  After ACC, expected peak ≈ {0.10/max(burst_rms, 1e-12) * np.max(np.abs(mixed_bpf)):.4f}")

# What would peak-based ACC give?
peak = np.max(np.abs(mixed_bpf))
print(f"\n  Peak of signal: {peak:.4f}")
print(f"  Peak-based ACC gain would be: {0.10/max(peak, 1e-12):.6f}")

# Also check: what about the active video region?
print(f"\n=== Active video chrominance (after upconvert BPF) ===")
blanking_us = 11.7
blanking_samples = blanking_us * 1e-6 * SAMPLE_RATE
line_us = 1e6 / dec.line_rate
active_us = line_us - blanking_us
active_samples = int(active_us * 1e-6 * SAMPLE_RATE)
bar_names = ["White", "Yellow", "Cyan", "Green", "Magenta", "Red", "Blue", "Black"]

check_line = 50
line_start = int(check_line * samples_per_line)
active_start = line_start + int(blanking_samples)
bar_width = active_samples // 8

for i, name in enumerate(bar_names):
    bs = active_start + i * bar_width
    be = bs + bar_width
    if be < n_samples:
        bar = mixed_bpf[bs:be]
        print(f"  {name:10s}: rms={np.sqrt(np.mean(bar**2)):.4f}, peak={np.max(np.abs(bar)):.4f}")
