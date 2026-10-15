#!/usr/bin/env python3
"""Generate a VHS-encoded SMPTE color bars video file."""
import sys
import subprocess
import numpy as np

sys.path.insert(0, "/home/user/vhs-emulator")
from vhs.encode import VHSEncoder, generate_test_composite
from vhs.decode import VHSDecoder, VideoStandard

SAMPLE_RATE = 14_318_180  # 4 x fsc
OUTPUT_FILE = "/home/user/vhs-emulator/vhs_colorbars.avi"
NUM_FIELDS = 30  # ~0.5 seconds at 59.94 Hz

print("=== Generating test composite ===")
encoder = VHSEncoder(standard="NTSC")
decoder = VHSDecoder(standard=VideoStandard.NTSC)

# Generate enough for all fields
duration_s = NUM_FIELDS / 59.94
composite = generate_test_composite(
    sample_rate=SAMPLE_RATE,
    duration_s=duration_s,
    pattern="colorbars",
    standard="NTSC",
)
print(f"  Composite: {len(composite)} samples ({duration_s:.3f}s)")

print("\n=== Encoding to tape signal ===")
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
print(f"  Tape signal: {len(tape_signal)} samples")

# Output dimensions
OUTPUT_W = 720
OUTPUT_H = 480  # NTSC field height (240 per field, doubled)

# Start ffmpeg process
print(f"\n=== Encoding video: {OUTPUT_FILE} ===")
ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-s", f"{OUTPUT_W}x{OUTPUT_H}",
    "-pix_fmt", "rgb48le",
    "-r", "59.94",
    "-i", "-",
    "-an",
    "-c:v", "ffv1",
    "-level", "3",
    OUTPUT_FILE,
]

proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

samples_per_field = int(SAMPLE_RATE / 59.94)

for field_idx in range(NUM_FIELDS):
    start = field_idx * samples_per_field
    end = start + samples_per_field
    if end > len(tape_signal):
        break

    field_signal = tape_signal[start:end]

    # Decode this field
    result = decoder.decode_field(
        field_signal, SAMPLE_RATE, head_index=field_idx % 2, field_number=field_idx
    )
    luma = result["luma"]
    chroma = result["chroma"]

    # Convert to RGB raster (one field = 240 lines)
    raster = decoder.yc_to_rgb_raster(
        luma, chroma, SAMPLE_RATE,
        output_width=OUTPUT_W,
        output_height=240,
    )

    # Interlace: duplicate each line to get 480-line frame
    frame_480 = np.repeat(raster, 2, axis=0)  # (480, 720, 3)

    # Convert float [0,1] to uint16 [0,65535] for rgb48le
    frame_16 = (np.clip(frame_480, 0.0, 1.0) * 65535).astype(np.uint16)

    proc.stdin.write(frame_16.tobytes())

    if (field_idx + 1) % 10 == 0:
        print(f"  Field {field_idx + 1}/{NUM_FIELDS}")

proc.stdin.close()
stderr = proc.stderr.read().decode()
proc.wait()

if proc.returncode == 0:
    print(f"\nDone! Output: {OUTPUT_FILE}")
else:
    print(f"\nffmpeg failed (rc={proc.returncode}):")
    print(stderr[-500:])
