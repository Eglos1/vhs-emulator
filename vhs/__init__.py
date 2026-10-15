"""
Physically Accurate VHS Tape Emulator

Models the complete VHS recording/playback chain including:
- Magnetic recording physics (Karlqvist field, hysteresis, flux transfer)
- Helical scan drum mechanics (2-head drum, tape transport)
- Internal analog circuitry (FM mod/demod, color-under, pre/de-emphasis)
- Tape degradation (dropout, print-through, stretch, head wear)
- .vhs file format (magnetic flux storage, not video encoding)
"""

__version__ = "1.0.0"
