#!/usr/bin/env python3
"""Diagnostic after direct upconversion (no 629 kHz BPF)."""
import sys
import numpy as np

sys.path.insert(0, '/home/user/vhs-emulator')

from vhs.encode import VHSEncoder, generate_test_composite
from vhs.decode import VHSDecoder, VideoStandard, _design_butterworth
import scipy.signal as sp_signal

SAMPLE_RATE = 14_318_180

composite = generate_test_composite(
    sample_rate=SAMPLE_RATE,
    duration_s=262.0 * 63.556e-6,
    pattern="colorbars",
    standard="NTSC",
)

encoder = VHSEncoder(standard="NTSC")
enc_result = encoder.encode_video(composite, SAMPLE_RATE)
y_fm = enc_result.y_fm
c_under = enc_result.c_under
if len(c_under) < len(y_fm):
    x_old = np.linspace(0, 1, len(c_under))
    x_new = np.linspace(0, 1, len(y_fm))
    c_under_full = np.interp(x_new, x_old, c_under)
else:
    c_under_full = c_under

tape_signal = y_fm + 0.375 * c_under_full

# Preamp + EQ
decoder = VHSDecoder(standard=VideoStandard.NTSC)
signal_eq = tape_signal.astype(np.float64) * 100.0
signal_eq = decoder._playback_eq(signal_eq, SAMPLE_RATE)

# Direct upconversion (LO at fsc - f_under = 2.95 MHz)
n = len(signal_eq)
t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
f_lo = 3579545.0 - 629375.0
lo_signal = 2.0 * np.cos(2.0 * np.pi * f_lo * t)
mixed = signal_eq * lo_signal

# 3.58 MHz BPF
f_sub = 3579545.0
b_bp, a_bp = _design_butterworth(order=4, cutoff_hz=[f_sub - 0.5e6, f_sub + 0.5e6],
                                  sample_rate=SAMPLE_RATE, btype="band")
chroma_358 = sp_signal.lfilter(b_bp, a_bp, mixed)

print(f"Chroma 3.58 MHz: range [{chroma_358.min():.1f}, {chroma_358.max():.1f}]")
print(f"Chroma 3.58 MHz rms: {np.sqrt(np.mean(chroma_358**2)):.1f}")

# Per-bar
samples_per_line = SAMPLE_RATE / 15734.264
blanking_samples = int(11.7e-6 * SAMPLE_RATE)
bar_names = ["White", "Yellow", "Cyan", "Green", "Magenta", "Red", "Blue", "Black"]
bar_w = (int(samples_per_line) - blanking_samples) // 8

for check_line in [50]:
    line_start = int(check_line * samples_per_line)
    active_start = line_start + blanking_samples
    print(f"\n--- Line {check_line} chroma at 3.58 MHz ---")
    for i, name in enumerate(bar_names):
        bs = active_start + i * bar_w
        be = bs + bar_w
        seg = chroma_358[bs:be]
        print(f"  {name:10s}: rms={np.sqrt(np.mean(seg**2)):.2f}, peak={np.max(np.abs(seg)):.2f}")

# Quadrature demod
carrier_cos = 2.0 * np.cos(2.0 * np.pi * f_sub * t)
carrier_neg_sin = -2.0 * np.sin(2.0 * np.pi * f_sub * t)
I_raw = chroma_358 * carrier_cos
Q_raw = chroma_358 * carrier_neg_sin
b_lp, a_lp = _design_butterworth(order=4, cutoff_hz=1.2e6, sample_rate=SAMPLE_RATE, btype="low")
I_base = sp_signal.lfilter(b_lp, a_lp, I_raw)
Q_base = sp_signal.lfilter(b_lp, a_lp, Q_raw)

# Burst detection
burst_start_us = 5.7
burst_end_us = 8.2
burst_start_samp = int(burst_start_us * 1e-6 * SAMPLE_RATE)
burst_end_samp = int(burst_end_us * 1e-6 * SAMPLE_RATE)

samples_per_line_f = SAMPLE_RATE / 15734.264
line50_start = int(50 * samples_per_line_f)
active50_start = line50_start + blanking_samples

# Burst from several lines
burst_I_vals = []
burst_Q_vals = []
for li in range(3, 30):
    ls = int(li * samples_per_line_f)
    bp_s = ls + burst_start_samp
    bp_e = ls + burst_end_samp
    if bp_e < n:
        burst_I_vals.append(np.mean(I_base[bp_s:bp_e]))
        burst_Q_vals.append(np.mean(Q_base[bp_s:bp_e]))

burst_I = np.mean(burst_I_vals)
burst_Q = np.mean(burst_Q_vals)
burst_phase = np.arctan2(burst_Q, burst_I)
burst_env = np.sqrt(burst_I**2 + burst_Q**2)
print(f"\nBurst: I={burst_I:.4f}, Q={burst_Q:.4f}, env={burst_env:.6f}, phase={np.degrees(burst_phase):.1f}°")

# Per-bar I/Q (no rotation, no scaling)
print(f"\nPer-bar I/Q (line {check_line}, raw quad demod):")
for i, name in enumerate(bar_names):
    bs = active50_start + i * bar_w
    be = bs + bar_w
    I_m = np.mean(I_base[bs:be])
    Q_m = np.mean(Q_base[bs:be])
    env = np.sqrt(I_m**2 + Q_m**2)
    phase = np.degrees(np.arctan2(Q_m, I_m))
    print(f"  {name:10s}: I={I_m:+12.4f}, Q={Q_m:+12.4f}, env={env:8.4f}, phase={phase:+.1f}°")

# Expected
print(f"\nExpected:")
expected = {
    "White": (0.000, 0.000, 0.0), "Yellow": (-0.119, -0.016, -172.3),
    "Cyan": (0.044, 0.148, 73.4), "Green": (-0.075, -0.164, -114.6),
    "Magenta": (0.075, 0.164, 65.4), "Red": (0.154, -0.132, -40.6),
    "Blue": (-0.044, -0.148, -106.6), "Black": (0.000, 0.000, 0.0),
}
for name, (Ie, Qe, pe) in expected.items():
    env = np.sqrt(Ie**2 + Qe**2)
    print(f"  {name:10s}: I={Ie:+12.3f}, Q={Qe:+12.3f}, env={env:8.4f}, phase={pe:+.1f}°")

# Try: measure burst amplitude from ACTIVE VIDEO (use Blue as reference)
# since Blue has the strongest known chrominance
blue_I = np.mean(I_base[active50_start + 6*bar_w : active50_start + 7*bar_w])
blue_Q = np.mean(Q_base[active50_start + 6*bar_w : active50_start + 7*bar_w])
blue_env = np.sqrt(blue_I**2 + blue_Q**2)
blue_phase = np.arctan2(blue_Q, blue_I)
blue_expected_env = 0.1544
blue_expected_phase = np.arctan2(-0.148, -0.044)

print(f"\nBlue ref: env={blue_env:.4f}, phase={np.degrees(blue_phase):.1f}°")
print(f"Blue expected: env={blue_expected_env:.4f}, phase={np.degrees(blue_expected_phase):.1f}°")

# Rotation and scale
cos_rot = np.cos(blue_expected_phase - blue_phase)
sin_rot = np.sin(blue_expected_phase - blue_phase)
scale = blue_expected_env / blue_env

print(f"\nScale={scale:.6f}, rotation={np.degrees(blue_expected_phase - blue_phase):.1f}°")
print(f"\nAfter fix (Blue-reference):")
for i, name in enumerate(bar_names):
    bs = active50_start + i * bar_w
    be = bs + bar_w
    I_m = np.mean(I_base[bs:be])
    Q_m = np.mean(Q_base[bs:be])
    I_r = I_m * cos_rot - Q_m * sin_rot
    Q_r = I_m * sin_rot + Q_m * cos_rot
    I_s = I_r * scale
    Q_s = Q_r * scale
    Y_vals = [0.992, 0.832, 0.688, 0.519, 0.458, 0.301, 0.154, 0.077]
    Y = Y_vals[i]
    R = Y + 0.956 * I_s + 0.621 * Q_s
    G = Y - 0.272 * I_s - 0.647 * Q_s
    B = Y - 1.105 * I_s + 1.703 * Q_s
    print(f"  {name:10s}: I={I_s:+.4f}, Q={Q_s:+.4f} → RGB=({np.clip(R,0,1):.3f},{np.clip(G,0,1):.3f},{np.clip(B,0,1):.3f})")

print("\nDone!")
