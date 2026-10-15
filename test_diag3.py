#!/usr/bin/env python3
"""Diagnostic: check why Black bar has more chrominance than Blue."""
import sys
import numpy as np
sys.path.insert(0, '/home/user/vhs-emulator')

from vhs.encode import VHSEncoder, generate_test_composite
from vhs.decode import VHSDecoder, VideoStandard, _design_butterworth
import scipy.signal as sp_signal

SAMPLE_RATE = 14_318_180

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

dec = VHSDecoder(standard=VideoStandard.NTSC)
signal = tape_signal.astype(np.float64)
signal = dec._preamplify_fs(signal, SAMPLE_RATE)
signal = dec._playback_eq(signal, SAMPLE_RATE)

# Extract chrominance at 629 kHz
chroma_629k = dec._extract_chroma(signal, SAMPLE_RATE)

samples_per_line = SAMPLE_RATE / dec.line_rate
blanking_us = 11.7
blanking_samples = blanking_us * 1e-6 * SAMPLE_RATE
line_us = 1e6 / dec.line_rate
active_us = line_us - blanking_us
active_samples = int(active_us * 1e-6 * SAMPLE_RATE)

print("=== 629 kHz BPF output — per-bar analysis (line 50) ===")
check_line = 50
line_start = int(check_line * samples_per_line)
active_start = line_start + int(blanking_samples)
bar_width = active_samples // 8
bar_names = ["White", "Yellow", "Cyan", "Green", "Magenta", "Red", "Blue", "Black"]

for i, name in enumerate(bar_names):
    bs = active_start + i * bar_width
    be = bs + bar_width
    if be < len(chroma_629k):
        bar = chroma_629k[bs:be]
        print(f"  {name:10s}: rms={np.sqrt(np.mean(bar**2)):8.3f}, peak={np.max(np.abs(bar)):8.3f}")

# Now check the encoder's c_under for the same bars
print("\n=== Encoder's c_under signal (line 50, after upsampling) ===")
# The encoder produces c_under at 629 kHz. Let's see what it contains.
c_line_start = int(check_line * (len(c_under_full) / (tape_signal.shape[0] / samples_per_line)))
c_active_start = c_line_start + int(blanking_samples)
c_bar_width = active_samples // 8

for i, name in enumerate(bar_names):
    bs = c_active_start + i * c_bar_width
    be = bs + c_bar_width
    if be < len(c_under_full):
        bar = c_under_full[bs:be]
        print(f"  {name:10s}: rms={np.sqrt(np.mean(bar**2)):8.3f}, peak={np.max(np.abs(bar)):8.3f}")

# Check: what's the actual c_under amplitude in the encoder BEFORE downsampling?
print("\n=== Encoder color-under BEFORE downsampling ===")
# Re-run encoder internals to see
chroma_separated = enc_result  # not directly available, but we can check c_under stats
c_rms_by_bar = []
for i, name in enumerate(bar_names):
    if i * (len(c_under_full) // 8) + (len(c_under_full) // 8) <= len(c_under_full):
        s = i * (len(c_under_full) // 8)
        e = s + (len(c_under_full) // 8)
        bar = c_under_full[s:e]
        print(f"  {name:10s}: rms={np.sqrt(np.mean(bar**2)):8.3f}")

# Check if there's significant energy in c_under where there should be none
print(f"\n=== c_under full statistics ===")
print(f"  range: [{c_under_full.min():.4f}, {c_under_full.max():.4f}]")
print(f"  rms: {np.sqrt(np.mean(c_under_full**2)):.4f}")
print(f"  mean abs: {np.mean(np.abs(c_under_full)):.4f}")

# Most importantly: what does the pure c_under look like after the decode chain?
print("\n=== Pure c_under through decode chain (no FM) ===")
c_pure_eq = dec._playback_eq(c_under_full * 100, SAMPLE_RATE)
c_pure_629 = dec._extract_chroma(c_pure_eq, SAMPLE_RATE)

for i, name in enumerate(bar_names):
    c_line_start2 = int(check_line * (len(c_pure_629) / (tape_signal.shape[0] / samples_per_line)))
    c_active_start2 = c_line_start2 + int(blanking_samples)
    c_bar_width2 = active_samples // 8
    bs = c_active_start2 + i * c_bar_width2
    be = bs + c_bar_width2
    if be < len(c_pure_629):
        bar = c_pure_629[bs:be]
        print(f"  {name:10s}: rms={np.sqrt(np.mean(bar**2)):8.3f}")
