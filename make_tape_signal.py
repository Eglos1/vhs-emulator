#!/usr/bin/env python3
"""Generate a raw VHS tape signal file (the magnetic flux data)."""
import sys
import numpy as np

sys.path.insert(0, "/home/user/vhs-emulator")
from vhs.encode import VHSEncoder, generate_test_composite

SAMPLE_RATE = 14_318_180  # 4 x fsc
OUTPUT_FILE = "/home/user/vhs-emulator/vhs_colorbars.raw"

print("=== Generating SMPTE color bars composite ===")
encoder = VHSEncoder(standard="NTSC")

duration_s = 5.0  # 5 seconds of color bars
composite = generate_test_composite(
    sample_rate=SAMPLE_RATE,
    duration_s=duration_s,
    pattern="colorbars",
    standard="NTSC",
)
print(f"  Composite: {len(composite)} samples")

print("\n=== Encoding to VHS tape signal ===")
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

print(f"  Tape signal: {len(tape_signal)} samples ({len(tape_signal)/SAMPLE_RATE*1000:.1f} ms)")
print(f"  Range: [{tape_signal.min():.4f}, {tape_signal.max():.4f}]")
print(f"  RMS: {np.sqrt(np.mean(tape_signal**2)):.4f}")

# Save as float64 raw binary
tape_signal.astype(np.float64).tofile(OUTPUT_FILE)
print(f"\nSaved: {OUTPUT_FILE}")

# Also save as 16-bit PCM WAV for easier playback
import wave
wav_file = "/home/user/vhs-emulator/vhs_colorbars.wav"
# Normalize to int16 range
peak = np.max(np.abs(tape_signal))
if peak > 0:
    normalized = tape_signal / peak * 0.95
else:
    normalized = tape_signal
pcm = (normalized * 32767).astype(np.int16)

with wave.open(wav_file, "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(int(SAMPLE_RATE))
    wf.writeframes(pcm.tobytes())

print(f"Saved: {wav_file}")
print(f"\nTo play: aplay -f S16_LE -r {int(SAMPLE_RATE)} {wav_file}")
