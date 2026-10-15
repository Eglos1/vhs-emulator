#!/usr/bin/env python3
"""Diagnostic: trace signal levels through the decode chain."""
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

# Upsample c_under
x_old = np.linspace(0, 1, len(c_under))
x_new = np.linspace(0, 1, len(y_fm))
c_under_full = np.interp(x_new, x_old, c_under)

tape_signal = y_fm + 0.375 * c_under_full

print(f"=== Raw tape signal ===")
print(f"  FM signal range: [{y_fm.min():.3f}, {y_fm.max():.3f}]")
print(f"  Color-under range: [{c_under_full.min():.3f}, {c_under_full.max():.3f}]")
print(f"  Combined range: [{tape_signal.min():.3f}, {tape_signal.max():.3f}]")

# === Check BPF at 629 kHz rejection ===
print(f"\n=== 629 kHz BPF frequency response ===")
b_bp, a_bp = _design_butterworth(order=6, cutoff_hz=[429e3, 829e3],
                                  sample_rate=SAMPLE_RATE, btype="band")
w, h = sp_signal.freqz(b_bp, a_bp, worN=8192, fs=SAMPLE_RATE)
freqs_to_check = [100e3, 200e3, 429e3, 629e3, 829e3, 1e6, 2e6, 3.4e6, 3.9e6, 4.4e6]
for f in freqs_to_check:
    idx = np.argmin(np.abs(w - f))
    gain_db = 20 * np.log10(max(abs(h[idx]), 1e-20))
    print(f"  {f/1e6:6.2f} MHz: {gain_db:+.1f} dB")

# === Signal levels at each stage ===
print(f"\n=== Decode chain signal levels ===")
n = len(tape_signal)
signal = tape_signal.astype(np.float64)

# Preamp
gain_linear = 10.0 ** (40.0 / 20.0)
signal_preamp = signal * gain_linear
print(f"After preamp (×100): range=[{signal_preamp.min():.1f}, {signal_preamp.max():.1f}], rms={np.sqrt(np.mean(signal_preamp**2)):.1f}")

# Check preamp bandpass
nyquist = SAMPLE_RATE / 2.0
print(f"  Nyquist = {nyquist/1e6:.2f} MHz (preamp BP cutoff is 10 MHz)")
if nyquist > 10.0e6:
    print(f"  → Preamp bandpass IS applied")
else:
    print(f"  → Preamp bandpass NOT applied (nyquist < 10 MHz)")

# EQ
from vhs.decode import VHSDecoder
dec = VHSDecoder(standard=VideoStandard.NTSC)
signal_eq = dec._playback_eq(signal_preamp, SAMPLE_RATE)
print(f"After EQ: range=[{signal_eq.min():.1f}, {signal_eq.max():.1f}], rms={np.sqrt(np.mean(signal_eq**2)):.1f}")

# Check spectral content after EQ
spectrum_eq = np.fft.rfft(signal_eq)
freqs_eq = np.fft.rfftfreq(n, d=1.0/SAMPLE_RATE)
# Find energy in different bands
for band_name, f_lo, f_hi in [("629kHz (chroma)", 400e3, 900e3),
                                ("3.4-4.4MHz (FM luma)", 3.4e6, 4.4e6),
                                ("DC-100kHz", 0, 100e3)]:
    mask = (freqs_eq >= f_lo) & (freqs_eq < f_hi)
    energy = np.sqrt(np.mean(np.abs(spectrum_eq[mask])**2))
    print(f"  Energy in {band_name}: {energy:.1f}")

# Extract chrominance path
chroma_629k = dec._extract_chroma(signal_eq, SAMPLE_RATE)
print(f"\nAfter 629 kHz BPF: range=[{chroma_629k.min():.1f}, {chroma_629k.max():.1f}], rms={np.sqrt(np.mean(chroma_629k**2)):.1f}")

# Upconvert
chroma_up = dec._upconvert_chroma(chroma_629k, SAMPLE_RATE)
print(f"After upconvert+ACC: range=[{chroma_up.min():.4f}, {chroma_up.max():.4f}], rms={np.sqrt(np.mean(chroma_up**2)):.4f}")

# Compare: what is the pure color-under signal (without FM)?
c_under_eq = dec._playback_eq(c_under_full * gain_linear, SAMPLE_RATE)
chroma_pure = dec._extract_chroma(c_under_eq, SAMPLE_RATE)
print(f"\n=== Pure color-under (no FM) through same chain ===")
print(f"After 629 kHz BPF: rms={np.sqrt(np.mean(chroma_pure**2)):.1f}")

# And what is the pure FM signal through the chroma chain?
fm_eq = dec._playback_eq(y_fm * gain_linear, SAMPLE_RATE)
chroma_from_fm = dec._extract_chroma(fm_eq, SAMPLE_RATE)
print(f"FM-only crosstalk at 629 kHz: rms={np.sqrt(np.mean(chroma_from_fm**2)):.1f}")
print(f"Crosstalk/Signal ratio: {np.sqrt(np.mean(chroma_from_fm**2)) / max(np.sqrt(np.mean(chroma_pure**2)), 1e-10):.1f}x")
