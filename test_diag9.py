"""Quick test: check if rasterizer works with more rows."""
import numpy as np
import sys
sys.path.insert(0, '/home/user/vhs-emulator')
from vhs.encode import VHSEncoder, generate_test_composite
from vhs.decode import VHSDecoder, VideoStandard

SAMPLE_RATE = 14_318_180
enc = VHSEncoder(standard="NTSC")
dec = VHSDecoder(standard=VideoStandard.NTSC)

composite = generate_test_composite(
    sample_rate=SAMPLE_RATE, duration_s=262.0 * 63.556e-6,
    pattern="colorbars", standard="NTSC",
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

samples_per_field = int(SAMPLE_RATE / 59.94)
field_signal = tape_signal[:samples_per_field]
result = dec.decode_field(field_signal, SAMPLE_RATE, head_index=0, field_number=0)

luma = result["luma"]
chroma = result["chroma"]

print(f"Luma shape: {luma.shape}")
print(f"Chroma shape: {chroma.shape}")

# Try rasterizer with many rows
raster = dec.yc_to_rgb_raster(luma, chroma, SAMPLE_RATE, output_width=720, output_height=80)

print(f"\nRaster shape: {raster.shape}")
print(f"Range: [{raster.min():.4f}, {raster.max():.4f}]")

# Check per-row average brightness
for row in range(80):
    avg = raster[row].mean()
    if avg > 0.01 or row < 25:
        print(f"  Row {row:2d}: avg_rgb={avg:.4f}  sample=({raster[row,360,0]:.3f},{raster[row,360,1]:.3f},{raster[row,360,2]:.3f})")

# Check bars at row 50
bar_names = ["White", "Yellow", "Cyan", "Green", "Magenta", "Red", "Blue", "Black"]
expected_rgb = {
    "White": (1.0, 1.0, 1.0), "Yellow": (1.0, 1.0, 0.0),
    "Cyan": (0.0, 1.0, 1.0), "Green": (0.0, 1.0, 0.0),
    "Magenta": (1.0, 0.0, 1.0), "Red": (1.0, 0.0, 0.0),
    "Blue": (0.0, 0.0, 1.0), "Black": (0.0, 0.0, 0.0),
}
bar_w = 720 // 8
mid_row = 50
print(f"\nBar analysis at row {mid_row}:")
for i, name in enumerate(bar_names):
    px = i * bar_w + bar_w // 2
    rgb = raster[mid_row, px-5:px+5, :].mean(axis=0)
    exp = expected_rgb[name]
    err = np.sqrt(sum((a-b)**2 for a, b in zip(rgb, exp)))
    print(f"  {name:10s}: ({rgb[0]:.3f},{rgb[1]:.3f},{rgb[2]:.3f})  err={err:.3f}")
