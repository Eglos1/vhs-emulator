"""Compute correct SMPTE 75% color bar parameters for the encoder."""
import math

# SMPTE 75% color bars: R, G, B at 75% = 0.75
bars = [
    ("White",   0.75, 0.75, 0.75),
    ("Yellow",  0.75, 0.75, 0.00),
    ("Cyan",    0.00, 0.75, 0.75),
    ("Green",   0.00, 0.75, 0.00),
    ("Magenta", 0.75, 0.00, 0.75),
    ("Red",     0.75, 0.00, 0.00),
    ("Blue",    0.00, 0.00, 0.75),
    ("Black",   0.00, 0.00, 0.00),
]

print("Standard NTSC 75% bars: Y, I, Q, chroma_amp, chroma_phase_deg")
print("-" * 80)

for name, R, G, B in bars:
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    I = 0.596 * R - 0.274 * G - 0.322 * B
    Q = 0.211 * R - 0.523 * G + 0.312 * B
    amp = math.sqrt(I**2 + Q**2)
    phase = math.degrees(math.atan2(Q, I))
    
    # Verify: reconstruct RGB from Y, I, Q
    R_v = Y + 0.956 * I + 0.621 * Q
    G_v = Y - 0.272 * I - 0.647 * Q
    B_v = Y - 1.105 * I + 1.703 * Q
    
    print(f"{name:10s}: Y={Y:.4f} I={I:+.4f} Q={Q:+.4f}  amp={amp:.4f} phase={phase:.1f}°  "
          f"  verify RGB=({R_v:.3f},{G_v:.3f},{B_v:.3f})")
