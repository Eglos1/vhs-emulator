"""Debug: check encoder burst_phase and the actual chrominance phases."""
import numpy as np
import sys
import math
sys.path.insert(0, "/home/user/vhs-emulator")
from vhs.encode import VHSEncoder, generate_test_composite

SAMPLE_RATE = 14_318_180
enc = VHSEncoder(standard="NTSC")
composite = generate_test_composite(
    sample_rate=SAMPLE_RATE, duration_s=262.0 * 63.556e-6,
    pattern="colorbars", standard="NTSC",
)

# Check encoder's internal state
sync_info = enc._sync_separate(composite, SAMPLE_RATE)
print(f"Encoder sync_separate burst_phase: {sync_info['burst_phase']:.4f} rad = {math.degrees(sync_info['burst_phase']):.1f} deg")
print(f"Encoder subcarrier: {enc.subcarrier_freq:.0f} Hz")

# Now check what happens after _yc_separate + _color_under
luma, chroma = enc._yc_separate(composite, SAMPLE_RATE, sync_info)

# Look at burst on line 5 (back porch area)
samples_per_line = SAMPLE_RATE / enc.h_freq
line5_start = int(5 * samples_per_line)

# Check chroma during burst region (~5.7-8.2 μs after line start)
burst_us = 5.7
burst_end_us = 8.2
burst_start_samp = line5_start + int(burst_us * 1e-6 * SAMPLE_RATE)
burst_end_samp = line5_start + int(burst_end_us * 1e-6 * SAMPLE_RATE)
burst_signal = chroma[burst_start_samp:burst_end_samp]
print(f"\n=== Chroma burst (line 5, {burst_us}-{burst_end_us} μs) ===")
print(f"  Samples: {len(burst_signal)}")
print(f"  RMS: {np.sqrt(np.mean(burst_signal**2)):.6f}")
print(f"  Peak: {np.max(np.abs(burst_signal)):.6f}")

# Measure burst phase from the separated chroma
t_burst = np.arange(burst_start_samp, burst_end_samp) / SAMPLE_RATE
ref_cos = np.cos(2 * np.pi * enc.subcarrier_freq * t_burst)
ref_sin = np.sin(2 * np.pi * enc.subcarrier_freq * t_burst)
i_comp = np.mean(burst_signal * ref_cos)
q_comp = np.mean(burst_signal * ref_sin)
burst_phase_chroma = math.atan2(q_comp, i_comp)
print(f"  Burst I from chroma: {i_comp:.6f}")
print(f"  Burst Q from chroma: {q_comp:.6f}")
print(f"  Burst phase from chroma: {burst_phase_chroma:.4f} rad = {math.degrees(burst_phase_chroma):.1f} deg")

# Check what color-under produces
c_under = enc._color_under(chroma, enc.subcarrier_freq, SAMPLE_RATE, sync_info)
print(f"\n=== Color-under signal ===")
print(f"  Length: {len(c_under)} (original: {len(chroma)})")
print(f"  Peak: {np.max(np.abs(c_under)):.6f}")

# Check chrominance phase for each bar on line 50
print(f"\n=== Per-bar chrominance phases (line 50) ===")
bar_names = ["White", "Yellow", "Cyan", "Green", "Magenta", "Red", "Blue", "Black"]
bar_width = int(samples_per_line * (63.556e-6 - 10.9e-6) / (63.556e-6) / 8)
active_start_offset = int(10.9e-6 * SAMPLE_RATE)

for bi, name in enumerate(bar_names):
    line50_start = int(50 * samples_per_line)
    bar_start = line50_start + active_start_offset + bi * bar_width
    bar_end = bar_start + bar_width
    
    bar_chroma = chroma[bar_start:bar_end]
    t_bar = np.arange(bar_start, bar_end) / SAMPLE_RATE
    ref_c = np.cos(2 * np.pi * enc.subcarrier_freq * t_bar)
    ref_s = np.sin(2 * np.pi * enc.subcarrier_freq * t_bar)
    
    I_bar = np.mean(bar_chroma * ref_c)
    Q_bar = np.mean(bar_chroma * ref_s)
    amp = np.sqrt(I_bar**2 + Q_bar**2)
    phase = math.degrees(math.atan2(Q_bar, I_bar))
    
    print(f"  {name:10s}: I={I_bar:8.5f} Q={Q_bar:8.5f}  amp={amp:.5f}  phase={phase:.1f}°")
