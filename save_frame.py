#!/usr/bin/env python3
"""Save first frame from the .vhs playback as a PNG for inspection."""
import sys
import numpy as np
sys.path.insert(0, "/home/user/vhs-emulator")
from vhs.emulator import VHSTapeEmulator

emu = VHSTapeEmulator()
result = emu.play("/home/user/vhs-emulator/vhs_colorbars.vhs")

frames = result['video_frames']
print(f"Got {len(frames)} frames, shape={frames[0].shape}, range=[{frames[0].min():.3f}, {frames[0].max():.3f}]")

from PIL import Image
# Save several frames
for idx in [0, 10, 50, 100, 150]:
    if idx < len(frames):
        frame = np.clip(frames[idx], 0, 1)
        img = Image.fromarray((frame * 255).astype(np.uint8))
        path = f"/home/user/vhs-emulator/frame_{idx:04d}.png"
        img.save(path)
        print(f"  Saved {path} — avg RGB per row[50]: {frames[idx][50].mean(axis=0)}")
