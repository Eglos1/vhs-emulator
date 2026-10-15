#!/usr/bin/env python3
"""Debug yc_to_rgb_raster burst detection."""
import sys
import numpy as np
import scipy.signal as sp_signal

sys.path.insert(0, '/home/user/vhs-emulator')

from vhs.encode import VHSEncoder, generate_test_composite
from vhs.decode import VHSDecoder, VideoStandard, _design_butterworth

SAMPLE_RATE = 14_318_180

composite = generate_test_composite(
    sample_rate=SAMPLE_RATE, duration_s=262.0 * 63.556e-6,
    pattern="colorbars", standard="NTSC",
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

decoder = VHSDecoder(standard=VideoStandard.NTSC)
signal_eq = tape_signal.astype(np.float64) * 100.0
signal_eq = decoder._playback_eq(signal_eq, SAMPLE_RATE)

# Upconvert directly
n = len(signal_eq)
t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
f_lo = 3579545.0 - 629375.0
mixed = signal_eq * 2.0 * np.cos(2.0 * np.pi * f_lo * t)

f_sub = 3579545.0
b_bp, a_bp = _design_butterworth(order=4, cutoff_hz=[f_sub - 0.5e6, f_sub + 0.5e6],
                                  sample_rate=SAMPLE_RATE, btype="band")
chroma_358 = sp_signal.lfilter(b_bp, a_bp, mixed)

# Quad demod
I_raw = chroma_358 * 2.0 * np.cos(2.0 * np.pi * f_sub * t)
Q_raw = chroma_358 * (-2.0 * np.sin(2.0 * np.pi * f_sub * t))
b_lp, a_lp = _design_butterworth(order=4, cutoff_hz=1.2e6, sample_rate=SAMPLE_RATE, btype="low")
I_base = sp_signal.lfilter(b_lp, a_lp, I_raw)
Q_base = sp_signal.lfilter(b_lp, a_lp, Q_raw)

# What the rasterizer sees:
samples_per_line_f = SAMPLE_RATE / 15734.264

# For line 0, find burst region in I_base/Q_base
line0_start = 0
burst_us_values = [5.0, 5.7, 6.5, 7.0, 7.5, 8.0, 8.2, 9.0, 10.0, 11.0, 11.5, 12.0]
print("Burst I/Q magnitude at various timing offsets (line 0):")
for us in burst_us_values:
    samp = int(line0_start + us * 1e-6 * SAMPLE_RATE)
    # Check a 2μs window
    win_end = int(line0_start + (us + 2.0) * 1e-6 * SAMPLE_RATE)
    if win_end < n:
        I_m = np.mean(I_base[samp:win_end])
        Q_m = np.mean(Q_base[samp:win_end])
        env = np.sqrt(I_m**2 + Q_m**2)
        phase = np.degrees(np.arctan2(Q_m, I_m))
        print(f"  {us:5.1f} μs: I={I_m:+8.3f}, Q={Q_m:+8.3f}, env={env:.3f}, phase={phase:+.1f}°")

# Also check line 3-5 (after filters have settled)
print("\nBurst I/Q on lines 3-5 at various offsets:")
for li in range(3, 6):
    line_start = int(li * samples_per_line_f)
    for us in [7.0, 8.0, 9.0, 10.0, 11.0]:
        samp = line_start + int(us * 1e-6 * SAMPLE_RATE)
        win_end = line_start + int((us + 2.0) * 1e-6 * SAMPLE_RATE)
        if win_end < n:
            I_m = np.mean(I_base[samp:win_end])
            Q_m = np.mean(Q_base[samp:win_end])
            env = np.sqrt(I_m**2 + Q_m**2)
            if env > 0.1:
                phase = np.degrees(np.arctan2(Q_m, I_m))
                print(f"  Line {li}, {us:5.1f} μs: I={I_m:+8.3f}, Q={Q_m:+8.3f}, env={env:.3f}, phase={phase:+.1f}°")

# Now test what yc_to_rgb_raster does internally
print("\n=== Testing rasterizer internals ===")
# Rasterizer uses burst_start_us=6.5, burst_end_us=11.5
burst_start_samp = int(6.5e-6 * SAMPLE_RATE)
burst_end_samp = int(11.5e-6 * SAMPLE_RATE)

# Measure burst amplitude from lines 3-29
burst_amps = []
for li in range(3, 30):
    ls = int(li * samples_per_line_f)
    bp_s = ls + burst_start_samp
    bp_e = ls + burst_end_samp
    if bp_e < n:
        I_b = I_base[bp_s:bp_e]
        Q_b = Q_base[bp_s:bp_e]
        amp = np.sqrt(np.mean(I_b) ** 2 + np.mean(Q_b) ** 2)
        if amp > 1e-6:
            burst_amps.append(amp)
            
global_burst_amp = float(np.median(burst_amps))
target_burst_amp = 0.142
sat_scale = target_burst_amp / global_burst_amp
sat_scale_clipped = np.clip(sat_scale, 0.01, 10.0)
print(f"Burst amplitude measurements: {len(burst_amps)} values")
print(f"  First 5: {[f'{a:.3f}' for a in burst_amps[:5]]}")
print(f"  Median: {global_burst_amp:.3f}")
print(f"  sat_scale: {sat_scale:.6f} (clipped: {sat_scale_clipped:.6f})")

# Check burst phase for line 50
line50_start = int(50 * samples_per_line_f)
bp_s = line50_start + burst_start_samp
bp_e = line50_start + burst_end_samp
I_burst = I_base[bp_s:bp_e]
Q_burst = Q_base[bp_s:bp_e]
burst_phase = np.arctan2(float(np.mean(Q_burst)), float(np.mean(I_burst)))
print(f"\nLine 50 burst: I_mean={np.mean(I_burst):.4f}, Q_mean={np.mean(Q_burst):.4f}")
print(f"  burst_phase={np.degrees(burst_phase):.1f}°")

# What if we use the ACTUAL burst timing (shifted by ~2μs)?
burst_start_samp_shifted = int(7.8e-6 * SAMPLE_RATE)
burst_end_samp_shifted = int(10.3e-6 * SAMPLE_RATE)
bp_s2 = line50_start + burst_start_samp_shifted
bp_e2 = line50_start + burst_end_samp_shifted
I_burst2 = I_base[bp_s2:bp_e2]
Q_burst2 = Q_base[bp_s2:bp_e2]
burst_phase2 = np.arctan2(float(np.mean(Q_burst2)), float(np.mean(I_burst2)))
burst_env2 = np.sqrt(np.mean(I_burst2)**2 + np.mean(Q_burst2)**2)
print(f"\nLine 50 SHIFTED burst (7.8-10.3μs): I_mean={np.mean(I_burst2):.4f}, Q_mean={np.mean(Q_burst2):.4f}")
print(f"  burst_phase={np.degrees(burst_phase2):.1f}°, env={burst_env2:.4f}")

# And the original window
burst_env1 = np.sqrt(np.mean(I_burst)**2 + np.mean(Q_burst)**2)
print(f"  Original window (6.5-11.5μs): env={burst_env1:.4f}")

print("\nDone!")
