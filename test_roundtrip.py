#!/usr/bin/env python3
"""Quick round-trip test for VHS color bars."""
import sys
import numpy as np

sys.path.insert(0, '/home/user/vhs-emulator')

from vhs.encode import VHSEncoder, generate_test_composite
from vhs.decode import VHSDecoder, VideoStandard

SAMPLE_RATE = 14_318_180  # 4 × fsc

print("=== Generating test composite ===")
composite = generate_test_composite(
    sample_rate=SAMPLE_RATE,
    duration_s=262.0 * 63.556e-6,  # ~2 fields
    pattern="colorbars",
    standard="NTSC",
)

print(f"  Composite shape: {composite.shape}")
print(f"  Range: [{composite.min():.4f}, {composite.max():.4f}]")
print(f"  RMS: {np.sqrt(np.mean(composite**2)):.4f}")

print("\n=== Encoding to tape signal ===")
encoder = VHSEncoder(standard="NTSC")
enc_result = encoder.encode_video(composite, SAMPLE_RATE)

y_fm = enc_result.y_fm
c_under = enc_result.c_under
print(f"  y_fm shape: {y_fm.shape}, range: [{y_fm.min():.4f}, {y_fm.max():.4f}]")
print(f"  c_under shape: {c_under.shape}, range: [{c_under.min():.4f}, {c_under.max():.4f}]")

if len(c_under) < len(y_fm):
    x_old = np.linspace(0, 1, len(c_under))
    x_new = np.linspace(0, 1, len(y_fm))
    c_under_full = np.interp(x_new, x_old, c_under)
else:
    c_under_full = c_under

tape_signal = y_fm + 0.375 * c_under_full
print(f"  tape_signal shape: {tape_signal.shape}")
print(f"  tape_signal range: [{tape_signal.min():.4f}, {tape_signal.max():.4f}]")

print("\n=== Decoding ===")
decoder = VHSDecoder(standard=VideoStandard.NTSC)

# One field
samples_per_field = int(SAMPLE_RATE / 59.94)
field_signal = tape_signal[:samples_per_field]
print(f"  Field signal: {field_signal.shape} samples")

result = decoder.decode_field(field_signal, SAMPLE_RATE, head_index=0, field_number=0)

luma = result["luma"]
chroma = result["chroma"]
quality = result.get("quality", {})

print(f"  Luma shape: {luma.shape}, range: [{luma.min():.4f}, {luma.max():.4f}]")
print(f"  Chroma shape: {chroma.shape}, range: [{chroma.min():.4f}, {chroma.max():.4f}]")
print(f"  Quality: {quality}")

# Check luma at various lines
samples_per_line = SAMPLE_RATE / decoder.line_rate
blanking_us = 11.7
blanking_samples = blanking_us * 1e-6 * SAMPLE_RATE
line_us = 1e6 / decoder.line_rate
active_us = line_us - blanking_us
active_samples = int(active_us * 1e-6 * SAMPLE_RATE)

bar_names = ["White", "Yellow", "Cyan", "Green", "Magenta", "Red", "Blue", "Black"]

# Print luma across all bars for several lines
for check_line in [5, 25, 50, 100]:
    line_start = int(check_line * samples_per_line)
    active_start = line_start + int(blanking_samples)
    bar_width = active_samples // 8
    luma_vals = []
    for i in range(8):
        bs = active_start + i * bar_width
        be = bs + bar_width
        if be <= len(luma):
            luma_vals.append(f"{luma[bs:be].mean():.4f}")
    print(f"  Line {check_line:3d} luma per bar: {' '.join(luma_vals)}")

# Chrominance stats for line 50 (active video)
check_line = 50
line_start = int(check_line * samples_per_line)
active_start = line_start + int(blanking_samples)
bar_width = active_samples // 8

print(f"\n=== Per-bar analysis (line {check_line}) ===")
for i, name in enumerate(bar_names):
    bs = active_start + i * bar_width
    be = bs + bar_width
    if be <= len(chroma) and be <= len(luma):
        bar_chroma = chroma[bs:be]
        bar_luma = luma[bs:be]
        print(f"  {name:10s}: luma={bar_luma.mean():.4f}  chroma_rms={np.sqrt(np.mean(bar_chroma**2)):.6f}")

print("\n=== Converting to RGB raster ===")
# Need enough rows to get past VBI (first ~20 lines) and into active video
raster = decoder.yc_to_rgb_raster(luma, chroma, SAMPLE_RATE, output_width=720, output_height=80)

print(f"  Raster shape: {raster.shape}")
print(f"  Range: [{raster.min():.4f}, {raster.max():.4f}]")

# Use row 50 which is well into active video (past VBI at ~line 20)
check_row = 50
bar_width_px = 720 // 8
expected_rgb = {
    "White":   (1.0, 1.0, 1.0),
    "Yellow":  (1.0, 1.0, 0.0),
    "Cyan":    (0.0, 1.0, 1.0),
    "Green":   (0.0, 1.0, 0.0),
    "Magenta": (1.0, 0.0, 1.0),
    "Red":     (1.0, 0.0, 0.0),
    "Blue":    (0.0, 0.0, 1.0),
    "Black":   (0.0, 0.0, 0.0),
}

print(f"\n{'Bar':>10s}  {'Expected':>24s}  {'Decoded':>24s}  {'Error':>8s}")
print("-" * 75)
for i, name in enumerate(bar_names):
    px_start = i * bar_width_px + bar_width_px // 2
    px_end = px_start + 10
    decoded_rgb = raster[check_row, px_start:px_end, :].mean(axis=0)
    exp = expected_rgb[name]
    err = np.sqrt(sum((d - e)**2 for d, e in zip(decoded_rgb, exp)))
    print(f"{name:>10s}  ({exp[0]:.2f},{exp[1]:.2f},{exp[2]:.2f})    "
          f"({decoded_rgb[0]:.3f},{decoded_rgb[1]:.3f},{decoded_rgb[2]:.3f})  "
          f"{err:.3f}")

print("\nDone!")
