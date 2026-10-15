#!/usr/bin/env python3
"""Diagnostic: trace chrominance through the full decode chain."""
import sys
import numpy as np

sys.path.insert(0, '/home/user/vhs-emulator')

from vhs.encode import VHSEncoder, generate_test_composite
from vhs.decode import VHSDecoder, VideoStandard, _design_butterworth
import scipy.signal as sp_signal

SAMPLE_RATE = 14_318_180  # 4 × fsc

print("=== Generating test composite ===")
composite = generate_test_composite(
    sample_rate=SAMPLE_RATE,
    duration_s=262.0 * 63.556e-6,
    pattern="colorbars",
    standard="NTSC",
)

print("\n=== Encoding to tape signal ===")
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
print(f"  y_fm: [{y_fm.min():.4f}, {y_fm.max():.4f}], rms={np.sqrt(np.mean(y_fm**2)):.4f}")
print(f"  c_under_full: [{c_under_full.min():.4f}, {c_under_full.max():.4f}], rms={np.sqrt(np.mean(c_under_full**2)):.4f}")
print(f"  tape_signal: [{tape_signal.min():.4f}, {tape_signal.max():.4f}]")

# Check burst in c_under
samples_per_line = SAMPLE_RATE / 15734.264
blanking_samples = int(11.7e-6 * SAMPLE_RATE)
burst_start = int(5.7e-6 * SAMPLE_RATE)
burst_end = int(8.2e-6 * SAMPLE_RATE)
bar_names = ["White", "Yellow", "Cyan", "Green", "Magenta", "Red", "Blue", "Black"]

print("\n--- Encoder c_under: per-line burst region and active bars (line 50) ---")
line50_start = int(50 * samples_per_line)
active_start_50 = line50_start + blanking_samples
bar_w = (int(SAMPLE_RATE / 15734.264) - blanking_samples) // 8
for i, name in enumerate(bar_names):
    bs = active_start_50 + i * bar_w
    be = bs + bar_w
    seg = c_under_full[bs:be]
    print(f"  {name:10s}: rms={np.sqrt(np.mean(seg**2)):.4f}, peak={np.max(np.abs(seg)):.4f}")

# Burst in c_under
bp_start = line50_start + burst_start
bp_end = line50_start + burst_end
burst_seg = c_under_full[bp_start:bp_end]
print(f"  Burst line50: rms={np.sqrt(np.mean(burst_seg**2)):.4f}, peak={np.max(np.abs(burst_seg)):.4f}")

# === Decoder chain step by step ===
print("\n=== Step 1: Preamp (×100 gain) ===")
signal = tape_signal.astype(np.float64) * 100.0
print(f"  range: [{signal.min():.2f}, {signal.max():.2f}]")
print(f"  rms: {np.sqrt(np.mean(signal**2)):.4f}")

print("\n=== Step 2: Playback EQ ===")
decoder = VHSDecoder(standard=VideoStandard.NTSC)
signal_eq = decoder._playback_eq(signal, SAMPLE_RATE)
print(f"  range: [{signal_eq.min():.2f}, {signal_eq.max():.2f}]")
print(f"  rms: {np.sqrt(np.mean(signal_eq**2)):.4f}")

# Check signal at 629 kHz after EQ
print("\n=== Step 3: 629 kHz BPF ===")
f_center = 629375.0
bw = 400e3
f_low = f_center - bw / 2.0
f_high = f_center + bw / 2.0
b_bp, a_bp = _design_butterworth(order=6, cutoff_hz=[f_low, f_high],
                                  sample_rate=SAMPLE_RATE, btype="band")
chroma_629k = sp_signal.lfilter(b_bp, a_bp, signal_eq)
print(f"  range: [{chroma_629k.min():.2f}, {chroma_629k.max():.2f}]")
print(f"  rms: {np.sqrt(np.mean(chroma_629k**2)):.4f}")

# Per-bar at 629 kHz
print("  Per-bar at 629 kHz (line 50):")
for i, name in enumerate(bar_names):
    bs = active_start_50 + i * bar_w
    be = bs + bar_w
    seg = chroma_629k[bs:be]
    print(f"    {name:10s}: rms={np.sqrt(np.mean(seg**2)):.4f}, peak={np.max(np.abs(seg)):.4f}")

# Burst at 629 kHz
bp_s = line50_start + burst_start
bp_e = line50_start + burst_end
burst_629 = chroma_629k[bp_s:bp_e]
print(f"  Burst 629 kHz: rms={np.sqrt(np.mean(burst_629**2)):.4f}, peak={np.max(np.abs(burst_629)):.4f}")

print("\n=== Step 4: Upconvert to 3.58 MHz ===")
n = len(chroma_629k)
t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
f_lo = 3579545.0 - 629375.0  # 2,950,170 Hz
lo_signal = 2.0 * np.cos(2.0 * np.pi * f_lo * t)
mixed = chroma_629k * lo_signal

# 3.58 MHz BPF
f_sub = 3579545.0
b_bpf, a_bpf = _design_butterworth(order=4, cutoff_hz=[f_sub - 0.5e6, f_sub + 0.5e6],
                                    sample_rate=SAMPLE_RATE, btype="band")
mixed_bpf = sp_signal.lfilter(b_bpf, a_bpf, mixed)
print(f"  range: [{mixed_bpf.min():.2f}, {mixed_bpf.max():.2f}]")
print(f"  rms: {np.sqrt(np.mean(mixed_bpf**2)):.4f}")

# Per-bar after upconversion
print("  Per-bar at 3.58 MHz (line 50):")
for i, name in enumerate(bar_names):
    bs = active_start_50 + i * bar_w
    be = bs + bar_w
    seg = mixed_bpf[bs:be]
    print(f"    {name:10s}: rms={np.sqrt(np.mean(seg**2)):.4f}, peak={np.max(np.abs(seg)):.4f}")

# Burst at 3.58 MHz
burst_358 = mixed_bpf[bp_s:bp_e]
print(f"  Burst 3.58 MHz: rms={np.sqrt(np.mean(burst_358**2)):.4f}, peak={np.max(np.abs(burst_358)):.4f}")

print("\n=== Step 5: ACC ===")
# Measure burst RMS
burst_rms_vals = []
n_lines = min(int(n / samples_per_line), 100)
for line_idx in range(2, n_lines):
    ls = int(line_idx * samples_per_line)
    bp_s2 = ls + burst_start
    bp_e2 = ls + burst_end
    if bp_e2 < n:
        br = mixed_bpf[bp_s2:bp_e2]
        rms = np.sqrt(np.mean(br ** 2))
        burst_rms_vals.append(rms)

burst_rms = float(np.median(burst_rms_vals))
print(f"  Burst RMS (median of {len(burst_rms_vals)} lines): {burst_rms:.4f}")
acc_target = 0.10
acc_gain = acc_target / burst_rms
acc_gain_clamped = np.clip(acc_gain, 0.001, 10.0)
print(f"  ACC gain: {acc_gain:.6f} (clamped: {acc_gain_clamped:.6f})")

after_acc = mixed_bpf * acc_gain_clamped
print(f"  After ACC range: [{after_acc.min():.4f}, {after_acc.max():.4f}]")
print(f"  After ACC rms: {np.sqrt(np.mean(after_acc**2)):.4f}")

# Per-bar after ACC
print("  Per-bar after ACC (line 50):")
for i, name in enumerate(bar_names):
    bs = active_start_50 + i * bar_w
    be = bs + bar_w
    seg = after_acc[bs:be]
    print(f"    {name:10s}: rms={np.sqrt(np.mean(seg**2)):.4f}")

print("\n=== Step 6: Quadrature demod in yc_to_rgb_raster ===")
f_sc = 3579545.0
carrier_cos = 2.0 * np.cos(2.0 * np.pi * f_sc * t)
carrier_neg_sin = -2.0 * np.sin(2.0 * np.pi * f_sc * t)
I_raw = after_acc * carrier_cos
Q_raw = after_acc * carrier_neg_sin
b_lp, a_lp = _design_butterworth(order=4, cutoff_hz=1.2e6, sample_rate=SAMPLE_RATE, btype="low")
I_base = sp_signal.lfilter(b_lp, a_lp, I_raw)
Q_base = sp_signal.lfilter(b_lp, a_lp, Q_raw)

# Per-bar I/Q
print("  Per-bar I/Q after LPF (line 50):")
for i, name in enumerate(bar_names):
    bs = active_start_50 + i * bar_w
    be = bs + bar_w
    I_seg = I_base[bs:be]
    Q_seg = Q_base[bs:be]
    I_mean = np.mean(I_seg)
    Q_mean = np.mean(Q_seg)
    env = np.sqrt(I_mean**2 + Q_mean**2)
    print(f"    {name:10s}: I={I_mean:+.6f}, Q={Q_mean:+.6f}, env={env:.6f}")

# Burst I/Q for phase reference
I_burst = I_base[bp_s:bp_e]
Q_burst = Q_base[bp_s:bp_e]
burst_phase = np.arctan2(float(np.mean(Q_burst)), float(np.mean(I_burst)))
burst_env = np.sqrt(np.mean(I_burst)**2 + np.mean(Q_burst)**2)
print(f"\n  Burst I={np.mean(I_burst):.6f}, Q={np.mean(Q_burst):.6f}")
print(f"  Burst phase: {burst_phase:.4f} rad ({np.degrees(burst_phase):.1f}°)")
print(f"  Burst env: {burst_env:.6f}")

print("\n=== Expected I/Q for NTSC 75% bars ===")
# Standard I/Q values (envelope and phase)
expected = {
    "White":   (0.000, 0.000),
    "Yellow":  (-0.119, -0.016),
    "Cyan":    (0.044, 0.148),
    "Green":   (-0.075, -0.164),
    "Magenta": (0.075, 0.164),
    "Red":     (0.154, -0.132),
    "Blue":    (-0.044, -0.148),
    "Black":   (0.000, 0.000),
}
for name, (Ie, Qe) in expected.items():
    print(f"  {name:10s}: I={Ie:+.3f}, Q={Qe:+.3f}, env={np.sqrt(Ie**2 + Qe**2):.3f}")

print("\nDone!")
