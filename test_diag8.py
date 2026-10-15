"""Diagnostic: trace yc_to_rgb_raster internals per bar."""
import numpy as np
import sys
sys.path.insert(0, "/home/user/vhs-emulator")
from vhs.encode import VHSEncoder, generate_test_composite
from vhs.decode import VHSDecoder, VideoStandard

enc = VHSEncoder(standard="NTSC")
dec = VHSDecoder(standard=VideoStandard.NTSC)
SAMPLE_RATE = 14_318_180

composite = generate_test_composite(
    sample_rate=SAMPLE_RATE,
    duration_s=262.0 * 63.556e-6,
    pattern="colorbars",
    standard="NTSC",
)

enc_result = enc.encode_video(composite, SAMPLE_RATE)
y_fm = enc_result.y_fm
c_under = enc_result.c_under
if len(c_under) < len(y_fm):
    x_old = np.linspace(0, 1, len(c_under))
    x_new = np.linspace(0, 1, len(y_fm))
    c_under_full = np.interp(x_new, x_old, c_under)
else:
    c_under_full = c_under
tape_signal = y_fm + 0.375 * c_under_full

# Decode through upconvert_chroma
samples_per_field = int(SAMPLE_RATE / 59.94)
field_signal = tape_signal[:samples_per_field]
result = dec.decode_field(field_signal, SAMPLE_RATE, head_index=0, field_number=0)

luma = result["luma"]
chroma = result["chroma"]

print(f"Luma range: [{luma.min():.4f}, {luma.max():.4f}]")
print(f"Chroma range: [{chroma.min():.4f}, {chroma.max():.4f}]")
print(f"Chroma RMS: {np.sqrt(np.mean(chroma**2)):.4f}")

# Now manually trace yc_to_rgb_raster internals
n = len(luma)
sample_rate = SAMPLE_RATE
samples_per_line_f = sample_rate / dec.line_rate
blanking_us = 11.7
blanking_samples_f = blanking_us * 1e-6 * sample_rate

f_sc = dec.f_subcarrier
t_full = np.arange(n, dtype=np.float64) / sample_rate

# Quad demod
carrier_cos = 2.0 * np.cos(2.0 * np.pi * f_sc * t_full)
carrier_neg_sin = -2.0 * np.sin(2.0 * np.pi * f_sc * t_full)
I_raw = chroma * carrier_cos
Q_raw = chroma * carrier_neg_sin

from vhs.decode import _design_butterworth
import scipy.signal as sp_signal
b_lp, a_lp = _design_butterworth(order=4, cutoff_hz=1.2e6, sample_rate=sample_rate, btype="low")
I_base = sp_signal.lfilter(b_lp, a_lp, I_raw)
Q_base = sp_signal.lfilter(b_lp, a_lp, Q_raw)

print(f"\nI_raw range: [{I_raw.min():.2f}, {I_raw.max():.2f}]")
print(f"I_base range: [{I_base.min():.4f}, {I_base.max():.4f}]")
print(f"Q_base range: [{Q_base.min():.4f}, {Q_base.max():.4f}]")

# Burst window
burst_start_us = 6.5
burst_end_us = 11.5
burst_start_samp = int(burst_start_us * 1e-6 * sample_rate)
burst_end_samp = int(burst_end_us * 1e-6 * sample_rate)
burst_len = burst_end_samp - burst_start_samp

# Global burst amp
burst_amps = []
n_lines_total = int(n / samples_per_line_f)
for li in range(3, min(n_lines_total, 30)):
    ls = int(li * samples_per_line_f)
    bp_s = ls + burst_start_samp
    bp_e = ls + burst_end_samp
    if bp_e < n and burst_len > 4:
        I_b = I_base[bp_s:bp_e]
        Q_b = Q_base[bp_s:bp_e]
        amp = np.sqrt(np.mean(I_b) ** 2 + np.mean(Q_b) ** 2)
        if amp > 1e-6:
            burst_amps.append(amp)

global_burst_amp = float(np.median(burst_amps)) if len(burst_amps) > 3 else 1e-6
target_burst_amp = 0.142
sat_scale = target_burst_amp / global_burst_amp if global_burst_amp > 1e-6 else 1.0
sat_scale = np.clip(sat_scale, 0.01, 10.0)

print(f"\nBurst amplitudes (first 5): {burst_amps[:5]}")
print(f"Global burst amp: {global_burst_amp:.4f}")
print(f"Sat scale: {sat_scale:.4f}")

# Line 50 detailed
row = 50
line_start_f = row * samples_per_line_f
line_start = int(line_start_f)
active_start = line_start + int(blanking_samples_f)
active_end = line_start + int(samples_per_line_f)

bp_start = line_start + burst_start_samp
bp_end = line_start + burst_end_samp
I_burst = I_base[bp_start:bp_end]
Q_burst = Q_base[bp_start:bp_end]
burst_phase = np.arctan2(float(np.mean(Q_burst)), float(np.mean(I_burst)))
print(f"\n=== Line {row} burst ===")
print(f"Burst I mean: {float(np.mean(I_burst)):.6f}")
print(f"Burst Q mean: {float(np.mean(Q_burst)):.6f}")
print(f"Burst phase: {burst_phase:.4f} rad = {np.degrees(burst_phase):.1f} deg")

cos_rot = -np.cos(burst_phase)
sin_rot = np.sin(burst_phase)
print(f"cos_rot={cos_rot:.6f}, sin_rot={sin_rot:.6f}")

# Show the burst waveform to check if it's real
print(f"\nBurst I waveform (every 10th sample): {I_burst[::10]}")
print(f"Burst Q waveform (every 10th sample): {Q_burst[::10]}")

# Per-bar analysis
bar_names = ["White", "Yellow", "Cyan", "Green", "Magenta", "Red", "Blue", "Black"]
n_active = active_end - active_start
bar_width = n_active // 8

print(f"\n=== Line {row} per-bar ===")
print(f"active_start={active_start}, active_end={active_end}, n_active={n_active}, bar_width={bar_width}")
for bi, name in enumerate(bar_names):
    bar_start = active_start + bi * bar_width
    bar_end = bar_start + bar_width
    
    I_bar = I_base[bar_start:bar_end]
    Q_bar = Q_base[bar_start:bar_end]
    
    I_mean = float(np.mean(I_bar))
    Q_mean = float(np.mean(Q_bar))
    Y_mean = float(np.mean(luma[bar_start:bar_end]))
    
    I_rot = I_mean * cos_rot - Q_mean * sin_rot
    Q_rot = I_mean * sin_rot + Q_mean * cos_rot
    
    I_scaled = I_rot * sat_scale
    Q_scaled = Q_rot * sat_scale
    
    R = Y_mean + 0.956 * I_scaled + 0.621 * Q_scaled
    G = Y_mean - 0.272 * I_scaled - 0.647 * Q_scaled
    B = Y_mean - 1.105 * I_scaled + 1.703 * Q_scaled
    
    print(f"{name:10s}: I={I_mean:10.6f} Q={Q_mean:10.6f} Y={Y_mean:.4f}")
    print(f"  rot: I'={I_rot:10.6f} Q'={Q_rot:10.6f}")
    print(f"  scd: I''={I_scaled:10.6f} Q''={Q_scaled:10.6f}")
    print(f"  RGB: ({R:.4f}, {G:.4f}, {B:.4f})")
