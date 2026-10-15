"""Recording and playback head model for VHS.

This module models the magnetic recording head, playback (read) head, and
the helical-scan drum head assembly used in VHS video recorders.  All
computations are numpy-vectorised where applicable and use float64
precision.

References:
    - Berov & Poulton, "Video Recording Head Design" (1985)
    - SMPTE Journal, "Helical Scan Recording"
    - IEC 60774-1 (VHS tape speed and track geometry)
    - Karlqvist, "Calculation of the Magnetic Field in the Ferromagnetic
      Layer of a Magnetic Drum" (1954)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

MU_0: float = 4.0 * math.pi * 1e-7  # vacuum permeability (H/m)


# ---------------------------------------------------------------------------
# RecordingHead
# ---------------------------------------------------------------------------

class RecordingHead:
    """Magnetic recording head — models the physical write head.

    The head is a ferrite C-core with a small air gap.  When current flows
    through the coil, a magnetic field is established across the gap that
    magnetises the tape as it passes.

    Parameters
    ----------
    gap_length_um : float
        Head gap length in micrometres.  Typical VHS: ~0.5 μm.
    turns : int
        Number of coil turns.
    core_permeability : float
        Relative permeability of the ferrite core (μ_r).

    Attributes
    ----------
    gap_length_m : float
        Gap length converted to metres.
    """

    def __init__(
        self,
        gap_length_um: float = 0.5,
        turns: int = 100,
        core_permeability: float = 2000.0,
    ) -> None:
        self.gap_length_um = gap_length_um
        self.gap_length_m: float = gap_length_um * 1e-6
        self.turns = turns
        self.core_permeability = core_permeability

    def write_field(
        self,
        current_A: float | NDArray[np.float64],
        x_along_gap_um: float | NDArray[np.float64],
        y_depth_um: float | NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute the magnetic field at the head–tape interface.

        Uses the **Karlqvist approximation** for the field of a
        recording head with a straight gap.

        The Karlqvist equations give the horizontal (Hx) and vertical
        (Hy) components of the field produced by a recording head:

            Hx(x, y) = (Hg / π) · [arctan((g/2 + x) / y) − arctan((g/2 − x) / y)]

            Hy(x, y) = (Hg / (2π)) · ln[((g/2 − x)² + y²) / ((g/2 + x)² + y²)]

        where:
            Hg = N · I / g         (gap field, A/m)
            g  = gap length (m)
            x  = position along the gap direction (m), origin at gap centre
            y  = depth into the tape from the head surface (m)

        Parameters
        ----------
        current_A : float or np.ndarray
            Current through the head coil in amperes.
        x_along_gap_um : float or np.ndarray
            Position along the gap direction in μm.  x = 0 is the gap
            centre, positive x is towards the trailing edge of the head.
        y_depth_um : float or np.ndarray
            Depth into the tape from the head surface in μm.  y = 0 is
            the head surface.

        Returns
        -------
        (Hx, Hy) : tuple of np.ndarray
            Horizontal and vertical field components in A/m.  Arrays are
            broadcast to a common shape from the input arrays.

        Notes
        -----
        The Karlqvist approximation is accurate to within ~5 % for
        points within a few gap lengths of the gap centre and breaks
        down at distances much larger than the gap length.
        """
        current_A = np.asarray(current_A, dtype=np.float64)
        x_um = np.asarray(x_along_gap_um, dtype=np.float64)
        y_um = np.asarray(y_depth_um, dtype=np.float64)

        # Convert to SI
        x_m = x_um * 1e-6
        y_m = np.where(y_um == 0.0, 1e-12, y_um * 1e-6)  # avoid div-by-zero
        g = self.gap_length_m

        # Gap field Hg = N * I / g
        Hg = self.turns * current_A / g

        half_g = g / 2.0

        # Hx — Karlqvist horizontal component
        arg1 = (half_g + x_m) / y_m
        arg2 = (half_g - x_m) / y_m
        Hx = (Hg / math.pi) * (np.arctan(arg1) - np.arctan(arg2))

        # Hy — Karlqvist vertical component
        num = (half_g - x_m) ** 2 + y_m ** 2
        den = (half_g + x_m) ** 2 + y_m ** 2
        # Avoid log(0)
        num = np.where(num == 0.0, 1e-30, num)
        den = np.where(den == 0.0, 1e-30, den)
        Hy = (Hg / (2.0 * math.pi)) * np.log(num / den)

        return Hx, Hy

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"RecordingHead(gap={self.gap_length_um}μm, "
            f"turns={self.turns}, μ_r={self.core_permeability})"
        )


# ---------------------------------------------------------------------------
# PlaybackHead
# ---------------------------------------------------------------------------

class PlaybackHead:
    """Magnetic playback (read) head.

    The playback head converts time-varying magnetic flux from the tape
    into an electrical voltage via Faraday's law.  Its sensitivity is
    shaped by the *head bump* response: a broad resonance around 4–5 MHz
    caused by the mechanical resonance of the head–tape contact, with
    roll-offs below ~1 kHz (eddy-current losses in the ferrite core)
    and above ~8 MHz (spacing loss and gap loss).

    Parameters
    ----------
    gap_length_um : float
        Playback gap length in μm.  Typically ~0.3 μm.
    sensitivity : float
        Base sensitivity in μV per (Wb/s).
    """

    def __init__(
        self,
        gap_length_um: float = 0.3,
        sensitivity: float = 1.0,
    ) -> None:
        self.gap_length_um = gap_length_um
        self.gap_length_m: float = gap_length_um * 1e-6
        self.sensitivity = sensitivity

    def sensitivity_curve(
        self,
        freq_hz: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Head bump response curve (dimensionless gain).

        The model captures three regimes:

        1. **Low-frequency roll-off** (< ~1 kHz): eddy current losses
           in the ferrite core cause a 6 dB/octave slope.
        2. **Head bump peak** (~4–5 MHz): mechanical resonance of the
           head–tape interface gives a broad peak of roughly +3 to +6 dB.
        3. **High-frequency roll-off** (> ~8 MHz): gap loss
           (sin(π f g / v) / (π f g / v)) and spacing loss
           (exp(−2π f d / v)) suppress the signal.

        Parameters
        ----------
        freq_hz : float or np.ndarray
            Frequency in Hz.

        Returns
        -------
        np.ndarray
            Dimensionless sensitivity (linear, not dB).
        """
        freq = np.asarray(freq_hz, dtype=np.float64)
        freq = np.where(freq <= 0.0, 1e-12, freq)

        # --- Low-frequency roll-off (eddy current, 1st-order) ---
        f_low = 1000.0  # corner frequency ~1 kHz
        low_pass = 1.0 / np.sqrt(1.0 + (f_low / freq) ** 2)

        # --- Head bump resonance (2nd-order, Q ≈ 1.5) ---
        f_bump = 4.5e6   # resonant frequency ~4.5 MHz
        Q = 1.5           # quality factor
        r = freq / f_bump
        bump = 1.0 / np.sqrt(
            (1.0 - r ** 2) ** 2 + (r / Q) ** 2
        )
        # Normalise bump to peak at 1.0
        bump_max = Q  # peak of 2nd-order LP at resonance
        bump = bump / bump_max

        # --- Gap loss: sin(π f g / v) / (π f g / v) ---
        v_tape = 33.35e-3  # tape speed in m/s (SP mode)
        x_gap = np.pi * freq * self.gap_length_m / v_tape
        # sinc in numpy is sin(πx)/(πx)
        gap_loss = np.sinc(x_gap / np.pi)

        # --- Spacing loss: exp(−2π f d / v) ---
        d = 0.5e-6  # head–tape spacing ~0.5 μm
        spacing_loss = np.exp(-2.0 * np.pi * freq * d / v_tape)

        return low_pass * bump * gap_loss * spacing_loss

    def read_flux(
        self,
        flux_rate_dPhidt: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert rate of flux change to output voltage.

        By Faraday's law the induced voltage is:

            V = −N · dΦ/dt

        where N is the effective number of turns and dΦ/dt is the time
        derivative of the flux linked by the head.  We incorporate the
        base *sensitivity* factor.

        Parameters
        ----------
        flux_rate_dPhidt : float or np.ndarray
            Rate of change of magnetic flux through the head core
            (Wb/s).

        Returns
        -------
        np.ndarray
            Output voltage in volts.
        """
        flux_rate = np.asarray(flux_rate_dPhidt, dtype=np.float64)
        return -self.sensitivity * flux_rate

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"PlaybackHead(gap={self.gap_length_um}μm, "
            f"sensitivity={self.sensitivity} μV/(Wb/s))"
        )


# ---------------------------------------------------------------------------
# DrumHeadAssembly
# ---------------------------------------------------------------------------

class DrumHeadAssembly:
    """Helical scan drum with two heads.

    VHS uses a 2-head drum:

    * Drum diameter: 74.487 mm (NTSC)
    * Drum RPM: 1800 (NTSC) = 30 rev/s
    * Two heads 180° apart
    * Tape wraps ~180° (π radians) around the drum
    * Each head contacts the tape for ~180° of drum rotation (one field)
    * Scan angle: ~5° 57′ (helical track angle)
    * Each revolution = one field (1/60 s NTSC, 1/50 s PAL)

    Parameters
    ----------
    diameter_mm : float
        Outer diameter of the head drum in millimetres.
    rpm : float
        Drum rotational speed in revolutions per minute.
    num_heads : int
        Number of heads mounted on the drum (default 2 for standard VHS).
    """

    def __init__(
        self,
        diameter_mm: float = 74.487,
        rpm: float = 1800.0,
        num_heads: int = 2,
    ) -> None:
        self.diameter_mm = diameter_mm
        self.drum_circumference_mm: float = math.pi * diameter_mm
        self.rpm = rpm
        self.num_heads = num_heads

        # Derived quantities
        self.rps: float = rpm / 60.0            # revolutions per second
        self.period_s: float = 1.0 / self.rps   # seconds per revolution
        self.omega_rad_s: float = 2.0 * math.pi * self.rps  # angular velocity

        # Tape geometry (VHS standard)
        self.wrap_angle_rad: float = math.pi    # 180° wrap
        self.scan_angle_deg: float = 5.0 + 57.0 / 60.0  # ~5°57′
        self.scan_angle_rad: float = math.radians(self.scan_angle_deg)

        # Track pitch and helix angle
        # The helix angle β relates tape speed, drum circumference,
        # and the scan angle:
        #   tan(β) = v_tape / (π * D_drum * RPS)
        # For NTSC VHS:
        v_tape_sp = 33.35e-3  # m/s
        self.helix_angle_rad: float = math.atan2(
            v_tape_sp, math.pi * diameter_mm * 1e-3 * self.rps
        )
        self.helix_angle_deg: float = math.degrees(self.helix_angle_rad)

    def angle_for_time(self, t_seconds: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute drum angle in radians at time *t*.

        Parameters
        ----------
        t_seconds : float or np.ndarray
            Time in seconds from an arbitrary reference.

        Returns
        -------
        np.ndarray
            Drum angle in radians, modulo 2π.
        """
        t = np.asarray(t_seconds, dtype=np.float64)
        return (self.omega_rad_s * t) % (2.0 * math.pi)

    def head_position_on_tape(
        self,
        head_index: int,
        t_seconds: float | NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute where on the tape each head is at time *t*.

        The tape moves linearly while the drum rotates.  The head traces
        a helical path on the tape surface.  We decompose this into:

        * **x_along_tape_mm**: position along the tape length (slow axis)
        * **y_cross_track_um**: position across the tape width (fast axis,
          from the drum rotation)

        Parameters
        ----------
        head_index : int
            Head number (0 or 1 for a 2-head drum).
        t_seconds : float or np.ndarray
            Time in seconds.

        Returns
        -------
        (x_along_tape_mm, y_cross_track_um) : tuple of np.ndarray
            Position along the tape (mm) and across the tape (μm).

        Notes
        -----
        The head index offsets the initial angle by 2π / num_heads.
        """
        t = np.asarray(t_seconds, dtype=np.float64)

        # Angular offset for this head
        angle_offset = (2.0 * math.pi * head_index) / self.num_heads

        # Drum angle
        theta = (self.omega_rad_s * t + angle_offset) % (2.0 * math.pi)

        # Linear tape position: tape moves at constant speed, but we
        # need to account for the component of the helical scan.
        # The tape velocity along its axis is v_tape (constant).
        # The head's contribution to the x-position comes from the
        # tangential velocity projected onto the tape axis.
        v_tape = 33.35e-3 * 1000.0  # mm/s (SP mode)

        # x_along_tape: linear tape motion + head helical component
        # The head traces ~180° of an arc on the tape; the axial
        # component of the head velocity contributes to the cross-track
        # position.  We model x as pure tape transport.
        x_mm = v_tape * np.asarray(t_seconds, dtype=np.float64)

        # y_cross_track: from drum rotation, projected through the
        # helix angle onto the tape width axis
        # At angle θ, the head position on the drum circumference is
        # R * sin(θ) in the cross-track direction, modulated by the
        # wrap geometry.
        R = self.diameter_mm / 2.0  # mm
        # The effective cross-track position is the projection of the
        # drum circumference onto the tape width:
        #   y = R * (1 − cos(θ))  for θ ∈ [0, π] (within the wrap)
        # This gives 0 at the entry edge and up to 2R at the exit edge,
        # but the actual usable range is limited by the tape width (12.7mm).
        # We normalise to the tape width:
        y_mm = R * (1.0 - np.cos(theta))
        # Clip to tape width
        y_mm = np.clip(y_mm, 0.0, self.width_mm if hasattr(self, 'width_mm') else 12.7)
        y_um = y_mm * 1000.0

        return x_mm, y_um

    @property
    def width_mm(self) -> float:
        """VHS tape width in mm."""
        return 12.7

    def head_switching_times(self) -> NDArray[np.float64]:
        """Compute the times when head switching occurs.

        In a 2-head drum, head 0 records/plays for 180° (π rad) then
        head 1 takes over for the next 180°.  Head switching happens at
        the entry and exit points of the tape wrap.

        Returns
        -------
        np.ndarray
            Array of switching times over one full drum revolution,
            starting at t = 0.  There are *num_heads* switching points
            per revolution, equally spaced by ``period / num_heads``.

        Notes
        -----
        In practice, switching occurs ~6 H-lines before the vertical
        sync pulse to hide the switching transient.  This model gives
        the ideal geometric switching times.
        """
        times = np.arange(self.num_heads, dtype=np.float64) * self.period_s / self.num_heads
        return times

    def track_geometry(self) -> dict[str, float]:
        """Return a dict describing the helical track geometry.

        Returns
        -------
        dict
            Keys:

            ``"track_angle_deg"``
                Helical scan angle in degrees (~5° 57′ for NTSC VHS).
            ``"track_angle_rad"``
                Helical scan angle in radians.
            ``"track_length_mm"``
                Length of one recorded track (one field) along the tape
                surface, in millimetres.
            ``"track_length_um"``
                Same as above, in micrometres.
            ``"overlap_mm"``
                Overlap region between adjacent tracks (from the guard
                band being partially overwritten), in millimetres.
            ``"drum_circumference_mm"``
                Drum circumference in millimetres.
            ``"helical_offset_um"``
                Cross-track offset per revolution (track pitch) in μm.
        """
        # Track length per field: the arc length on the drum that the
        # head sweeps, projected onto the tape surface.
        # Arc length = (wrap_angle / 2π) × drum circumference / cos(scan_angle)
        half_wrap_arc_mm = (self.wrap_angle_rad / (2.0 * math.pi)) * self.drum_circumference_mm
        # Projected onto tape (accounts for helix angle)
        track_length_mm = half_wrap_arc_mm / math.cos(self.scan_angle_rad)

        # Track pitch: for SP mode
        track_pitch_um = 58.0  # μm

        # Overlap: the track pitch is slightly less than the head width
        # so tracks partially overlap.  Typical head width ~30 μm,
        # track pitch ~58 μm for SP → no overlap in SP.
        # In EP, pitch = 19.3 μm, head width ~19 μm → ~1 μm overlap.
        # We return a conservative estimate.
        head_width_um = 30.0
        overlap_um = max(0.0, head_width_um - track_pitch_um)

        return {
            "track_angle_deg": self.scan_angle_deg,
            "track_angle_rad": self.scan_angle_rad,
            "track_length_mm": track_length_mm,
            "track_length_um": track_length_mm * 1000.0,
            "overlap_mm": overlap_um / 1000.0,
            "drum_circumference_mm": self.drum_circumference_mm,
            "helical_offset_um": track_pitch_um,
        }

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"DrumHeadAssembly(diameter={self.diameter_mm}mm, "
            f"rpm={self.rpm}, heads={self.num_heads})"
        )
