#!/usr/bin/env python3
"""Diagnostic: measure FM crosstalk at 629 kHz for each bar independently."""
import sys
import numpy as np

sys.path.insert(0, '/home/user/vhs-emulator')

from vhs.encode import VHSEncoder, generate_test_composite
from vhs.decode import VHSDecoder, VideoStandard, _design_butterworth
import scipy.signal as sp_signal

SAMPLE_RATE = 14_318_180

# Generate test composite
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

# Upsample c_under
if len(c_under) < len(y_fm):
    x_old = np.linspace(0, 1, len(c_under))
    x_new = np.linspace(0, 1, len(y_fm))
    c_under_full = np.interp(x_new, x_old, c_under)
else:
    c_under_full = c_under

tape_signal = y_fm + 0.375 * c_under_full

# Preamp + EQ
signal_eq = tape_signal.astype(np.float64) * 100.0
decoder = VHSDecoder(standard=VideoStandard.NTSC)
signal_eq = decoder._playback_eq(signal_eq, SAMPLE_RATE)

# 629 kHz BPF
f_center = 629375.0
bw = 400e3
b_bp, a_bp = _design_butterworth(order=6, cutoff_hz=[f_center - bw/2, f_center + bw/2],
                                  sample_rate=SAMPLE_RATE, btype="band")
chroma_629k = sp_signal.lfilter(b_bp, a_bp, signal_eq)

# Now also extract FM-only and c_under-only through the same chain
print("=== FM crosstalk (no chrominance) ===")
fm_only_tape = y_fm  # just FM, no c_under
fm_eq = fm_only_tape.astype(np.float64) * 100.0
fm_eq = decoder._playback_eq(fm_eq, SAMPLE_RATE)
fm_629k = sp_signal.lfilter(b_bp, a_bp, fm_eq)

print("=== Chrominance-only (no FM) ===")
c_only_tape = 0.375 * c_under_full
c_eq = c_only_tape.astype(np.float64) * 100.0
c_eq = decoder._playback_eq(c_eq, SAMPLE_RATE)
c_629k = sp_signal.lfilter(b_bp, a_bp, c_eq)

# Per-bar comparison
samples_per_line = SAMPLE_RATE / 15734.264
blanking_samples = int(11.7e-6 * SAMPLE_RATE)
bar_names = ["White", "Yellow", "Cyan", "Green", "Magenta", "Red", "Blue", "Black"]

for check_line in [25, 50]:
    line_start = int(check_line * samples_per_line)
    active_start = line_start + blanking_samples
    bar_w = (int(samples_per_line) - blanking_samples) // 8
    print(f"\n--- Line {check_line} ---")
    print(f"  {'Bar':10s} {'Total':>10s} {'FM only':>10s} {'C_under':>10s} {'FM/c_under':>12s} {'FM%':>8s}")
    for i, name in enumerate(bar_names):
        bs = active_start + i * bar_w
        be = bs + bar_w
        total_rms = np.sqrt(np.mean(chroma_629k[bs:be]**2))
        fm_rms = np.sqrt(np.mean(fm_629k[bs:be]**2))
        c_rms = np.sqrt(np.mean(c_629k[bs:be]**2))
        ratio = fm_rms / max(c_rms, 1e-10)
        pct = fm_rms / max(total_rms, 1e-10) * 100
        print(f"  {name:10s} {total_rms:10.2f} {fm_rms:10.2f} {c_rms:10.2f} {ratio:12.1f} {pct:7.1f}%")

# Now test: upconvert with ACC using different approaches
print("\n=== Upconvert chain comparison ===")
n = len(chroma_629k)
t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
f_lo = 3579545.0 - 629375.0

# Method 1: Direct (no ACC, no extra filtering)
lo_signal = 2.0 * np.cos(2.0 * np.pi * f_lo * t)
mixed_direct = chroma_629k * lo_signal
f_sub = 3579545.0
b_bpf, a_bpf = _design_butterworth(order=4, cutoff_hz=[f_sub - 0.5e6, f_sub + 0.5e6],
                                    sample_rate=SAMPLE_RATE, btype="band")
mixed_direct_bpf = sp_signal.lfilter(b_bpf, a_bpf, mixed_direct)

# Quadrature demod
carrier_cos = 2.0 * np.cos(2.0 * np.pi * f_sub * t)
carrier_neg_sin = -2.0 * np.sin(2.0 * np.pi * f_sub * t)
I_raw = mixed_direct_bpf * carrier_cos
Q_raw = mixed_direct_bpf * carrier_neg_sin
b_lp, a_lp = _design_butterworth(order=4, cutoff_hz=1.2e6, sample_rate=SAMPLE_RATE, btype="low")
I_base = sp_signal.lfilter(b_lp, a_lp, I_raw)
Q_base = sp_signal.lfilter(b_lp, a_lp, Q_raw)

# Per-bar I/Q for line 50 (no ACC, no normalization)
samples_per_line_f = SAMPLE_RATE / 15734.264
line50_start = int(50 * samples_per_line_f)
active50_start = line50_start + blanking_samples
bar_w = (int(samples_per_line_f) - blanking_samples) // 8

print("\n  Per-bar I/Q (NO ACC, NO normalization) line 50:")
for i, name in enumerate(bar_names):
    bs = active50_start + i * bar_w
    be = bs + bar_w
    I_mean = np.mean(I_base[bs:be])
    Q_mean = np.mean(Q_base[bs:be])
    env = np.sqrt(I_mean**2 + Q_mean**2)
    phase = np.degrees(np.arctan2(Q_mean, I_mean))
    print(f"    {name:10s}: I={I_mean:+10.4f}, Q={Q_mean:+10.4f}, env={env:.4f}, phase={phase:+.1f}°")

# Expected I/Q
print("\n  Expected (NTSC 75% bars):")
expected = {
    "White": (0.000, 0.000), "Yellow": (-0.119, -0.016),
    "Cyan": (0.044, 0.148), "Green": (-0.075, -0.164),
    "Magenta": (0.075, 0.164), "Red": (0.154, -0.132),
    "Blue": (-0.044, -0.148), "Black": (0.000, 0.000),
}
for name, (Ie, Qe) in expected.items():
    env = np.sqrt(Ie**2 + Qe**2)
    phase = np.degrees(np.arctan2(Qe, Ie))
    print(f"    {name:10s}: I={Ie:+10.3f}, Q={Qe:+10.3f}, env={env:.4f}, phase={phase:+.1f}°")

# Compute scaling needed
# Use Blue bar as reference (highest chroma)
blue_bs = active50_start + 6 * bar_w
blue_be = blue_bs + bar_w
blue_I = np.mean(I_base[blue_bs:blue_be])
blue_Q = np.mean(Q_base[blue_bs:blue_be])
blue_env = np.sqrt(blue_I**2 + blue_Q**2)
blue_expected_env = np.sqrt((-0.044)**2 + (-0.148)**2)
blue_phase = np.arctan2(blue_Q, blue_I)
blue_expected_phase = np.arctan2(-0.148, -0.044)

print(f"\n  Blue reference: measured env={blue_env:.4f}, expected env={blue_expected_env:.4f}")
print(f"  Required scale: {blue_expected_env/blue_env:.6f}")
print(f"  Blue measured phase: {np.degrees(blue_phase):.1f}°, expected: {np.degrees(blue_expected_phase):.1f}°")
print(f"  Required hue rotation: {np.degrees(blue_expected_phase - blue_phase):.1f}°")

# Check all bars after applying this fixed scale and rotation
print("\n  After fixed scale + rotation:")
cos_rot = np.cos(blue_expected_phase - blue_phase)
sin_rot = np.sin(blue_expected_phase - blue_phase)
scale = blue_expected_env / blue_env

for i, name in enumerate(bar_names):
    bs = active50_start + i * bar_w
    be = bs + bar_w
    I_m = np.mean(I_base[bs:be])
    Q_m = np.mean(Q_base[bs:be])
    # Rotate
    I_r = I_m * cos_rot - Q_m * sin_rot
    Q_r = I_m * sin_rot + Q_m * cos_rot
    # Scale
    I_s = I_r * scale
    Q_s = Q_r * scale
    # YIQ → RGB
    Y = np.mean([0.992, 0.832, 0.688, 0.519, 0.458, 0.301, 0.154, 0.077][i])  # approximate Y
    Y_vals = [0.992, 0.832, 0.688, 0.519, 0.458, 0.301, 0.154, 0.077]
    Y = Y_vals[i]
    R = Y + 0.956 * I_s + 0.621 * Q_s
    G = Y - 0.272 * I_s - 0.647 * Q_s
    B = Y - 1.105 * I_s + 1.703 * Q_s
    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)
    print(f"    {name:10s}: I={I_s:+.4f}, Q={Q_s:+.4f} → RGB=({R:.3f},{G:.3f},{B:.3f})")

print("\nDone!")
