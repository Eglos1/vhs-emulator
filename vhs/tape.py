"""Physical VHS tape model.

This module represents the actual magnetic tape as a physical object with
measurable geometric, magnetic, and mechanical properties.  All constants
are for standard consumer VHS (NTSC and PAL where specified).

References:
    - IEC 60094-1 (Magnetic tape recording)
    - SMPTE 170M / ITU-R BT.601 (timing)
    - TDK / Maxell / Sony technical datasheets for tape media
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TapeType(enum.Enum):
    """Standard cassette shell lengths."""
    T60 = "T-60"
    T120 = "T-120"
    T160 = "T-160"


class SpeedMode(enum.Enum):
    """Recording speed modes."""
    SP = "SP"
    LP = "LP"
    EP = "EP"   # also called SLP


class CoatingMaterial(enum.Enum):
    """Magnetic coating formulations used across tape generations."""
    FERRIC_OXIDE = "ferric_oxide"       # Type I – early consumer VHS
    CHROME = "chrome"                     # Type II – higher grade
    METAL_PARTICLE = "metal_particle"     # Type IV – highest grade


# ---------------------------------------------------------------------------
# Physical constants (coating properties)
# ---------------------------------------------------------------------------

# saturation magnetisation (Tesla), coercivity (A/m)
COATING_PROPERTIES: dict[CoatingMaterial, dict[str, float]] = {
    CoatingMaterial.FERRIC_OXIDE: {
        "Ms_T": 0.35,
        "Hc_Am": 300.0,
    },
    CoatingMaterial.CHROME: {
        "Ms_T": 0.32,
        "Hc_Am": 480.0,
    },
    CoatingMaterial.METAL_PARTICLE: {
        "Ms_T": 0.55,
        "Hc_Am": 1200.0,
    },
}

# Speed-mode timing parameters (NTSC)
SPEED_PARAMS: dict[SpeedMode, dict[str, float]] = {
    SpeedMode.SP: {
        "tape_speed_mm_s": 33.35,
        "track_pitch_um": 58.0,
    },
    SpeedMode.LP: {
        "tape_speed_mm_s": 16.675,
        "track_pitch_um": 29.0,
    },
    SpeedMode.EP: {
        "tape_speed_mm_s": 11.117,
        "track_pitch_um": 19.3,
    },
}

# Maximum recording durations per cassette type (minutes, in SP mode)
CASSETTE_DURATIONS: dict[TapeType, dict[SpeedMode, float]] = {
    TapeType.T60: {
        SpeedMode.SP: 60.0,
        SpeedMode.LP: 120.0,
        SpeedMode.EP: 180.0,
    },
    TapeType.T120: {
        SpeedMode.SP: 120.0,
        SpeedMode.LP: 240.0,
        SpeedMode.EP: 360.0,
    },
    TapeType.T160: {
        SpeedMode.SP: 160.0,
        SpeedMode.LP: 320.0,
        SpeedMode.EP: 480.0,
    },
}

# Typical tape lengths in metres for each cassette type (approximate)
CASSETTE_TAPE_LENGTHS_M: dict[TapeType, float] = {
    TapeType.T60:  123.0,
    TapeType.T120: 247.0,
    TapeType.T160: 329.0,
}


# ---------------------------------------------------------------------------
# TapePosition
# ---------------------------------------------------------------------------

@dataclass
class TapePosition:
    """Current position on the tape.

    Attributes
    ----------
    position_mm : float
        Linear distance along the tape from the start (leader).
    track_index : int
        Helical track index (0-based).  VHS records two interleaved tracks
        per head revolution, one per field.
    revolution : int
        Head revolution index since recording began.
    side : str
        ``"A"`` or ``"B"`` — the physical side of the tape currently
        facing the head drum.
    """

    position_mm: float = 0.0
    track_index: int = 0
    revolution: int = 0
    side: str = "A"

    # -- helpers -----------------------------------------------------------

    def advance(self, distance_mm: float) -> None:
        """Move the tape forward by *distance_mm*."""
        self.position_mm += distance_mm

    def advance_revolution(self) -> None:
        """Advance to the next head revolution (one field)."""
        self.revolution += 1
        self.track_index += 1

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"TapePosition(pos={self.position_mm:.2f}mm, "
            f"track={self.track_index}, rev={self.revolution}, "
            f"side={self.side})"
        )


# ---------------------------------------------------------------------------
# VHSTape
# ---------------------------------------------------------------------------

class VHSTape:
    """Physical VHS tape with measurable properties.

    Parameters
    ----------
    tape_type : TapeType
        Cassette shell length classification.
    coating : CoatingMaterial
        Magnetic coating formulation.

    Notes
    -----
    All geometric constants are for standard consumer VHS cassettes.
    """

    # -- geometry ----------------------------------------------------------
    width_mm: float = 12.7
    total_thickness_um: float = 19.0
    coating_thickness_um: float = 7.0
    backing_thickness_um: float = 12.0

    # -- constructor -------------------------------------------------------

    def __init__(
        self,
        tape_type: TapeType = TapeType.T120,
        coating: CoatingMaterial = CoatingMaterial.FERRIC_OXIDE,
    ) -> None:
        self.tape_type = tape_type
        self.coating = coating

        props = COATING_PROPERTIES[coating]
        self.Ms_T: float = props["Ms_T"]
        self.Hc_Am: float = props["Hc_Am"]

        # Derived geometry
        self.tape_length_m: float = CASSETTE_TAPE_LENGTHS_M[tape_type]
        self.tape_length_mm: float = self.tape_length_m * 1000.0

        # Recording surface area (both sides combined for hub-out winding)
        self.recording_area_m2: float = (
            self.tape_length_m * self.width_mm / 1000.0
        )

    # -- factory classmethods ----------------------------------------------

    @classmethod
    def T60(
        cls, coating: CoatingMaterial = CoatingMaterial.FERRIC_OXIDE
    ) -> "VHSTape":
        """Create a T-60 cassette (60 min SP / 120 min LP / 180 min EP)."""
        return cls(tape_type=TapeType.T60, coating=coating)

    @classmethod
    def T120(
        cls, coating: CoatingMaterial = CoatingMaterial.FERRIC_OXIDE
    ) -> "VHSTape":
        """Create a T-120 cassette (120 min SP / 240 min LP / 360 min EP)."""
        return cls(tape_type=TapeType.T120, coating=coating)

    @classmethod
    def T160(
        cls, coating: CoatingMaterial = CoatingMaterial.METAL_PARTICLE
    ) -> "VHSTape":
        """Create a T-160 cassette (160 min SP / 320 min LP / 480 min EP)."""
        return cls(tape_type=TapeType.T160, coating=coating)

    # -- computed properties -----------------------------------------------

    @property
    def total_tracks_possible(self) -> int:
        """Maximum number of helical tracks that fit on the full tape.

        One track per head revolution.  Each revolution produces a 180°
        wrap arc so the effective recording length per revolution equals
        half the drum circumference.  We approximate by dividing the
        total tape length by the tape consumed per revolution at the
        slowest speed (EP) which yields the most tracks.
        """
        # At EP speed, 1/50 s per revolution (PAL) → use 1/60 s (NTSC)
        # tape consumed per revolution = tape_speed * (1/field_rate)
        # For a rough upper bound we simply divide total length by the
        # smallest useful track pitch projected onto the tape.
        # A simpler model: tracks ≈ tape_length_mm / track_pitch_at_SP
        # but tracks are recorded sequentially, so the real bound is
        # determined by tape consumed at a given speed.
        # We return the SP-mode track count as the nominal capacity.
        sp = SPEED_PARAMS[SpeedMode.SP]
        # Each revolution spans roughly the drum circumference / tape_speed
        # For simplicity, approximate: tracks ≈ total_length / (tape_speed * field_period)
        field_period_s = 1.0 / 60.0  # NTSC
        tape_per_rev_mm = sp["tape_speed_mm_s"] * field_period_s
        return int(self.tape_length_mm / tape_per_rev_mm)

    @property
    def coating_cross_section_um2(self) -> float:
        """Cross-sectional area of the magnetic coating layer (μm²)."""
        return self.width_mm * 1000.0 * self.coating_thickness_um

    # -- tape usage computation --------------------------------------------

    def tape_used_mm(
        self,
        duration_s: float,
        speed: SpeedMode = SpeedMode.SP,
    ) -> float:
        """Compute length of tape consumed for a given recording duration.

        Parameters
        ----------
        duration_s : float
            Recording duration in seconds.
        speed : SpeedMode
            Recording speed mode.

        Returns
        -------
        float
            Tape consumed in millimetres.
        """
        tape_speed = SPEED_PARAMS[speed]["tape_speed_mm_s"]
        return tape_speed * duration_s

    def max_duration_s(self, speed: SpeedMode = SpeedMode.SP) -> float:
        """Maximum recording duration in seconds for this cassette type."""
        return CASSETTE_DURATIONS[self.tape_type][speed] * 60.0

    # -- blank tape generation ---------------------------------------------

    def blank_segment(
        self, length_mm: float, seed: int | None = None
    ) -> NDArray[np.float64]:
        """Generate a demagnetised (random-domain) tape segment.

        The returned array represents the normalised magnetisation *m*
        (dimensionless, range [−1, +1]) along the tape for every μm of
        the requested length.  A truly demagnetised tape has zero net
        magnetisation but microscopic domains are randomly oriented, so
        the array values are drawn from a standard normal distribution
        scaled to represent the thermal noise floor of the medium.

        Parameters
        ----------
        length_mm : float
            Segment length in millimetres.
        seed : int or None
            Optional RNG seed for reproducibility.

        Returns
        -------
        np.ndarray
            Shape ``(int(length_mm * 1000),)`` — magnetisation per μm.
        """
        rng = np.random.default_rng(seed)
        n_samples = int(length_mm * 1000)
        # Random domains, zero net magnetisation, small amplitude
        return rng.standard_normal(n_samples).astype(np.float64) * 0.01

    # -- print-through computation -----------------------------------------

    def print_through(
        self,
        magnetisation: NDArray[np.float64],
        hub_outer: bool = True,
    ) -> NDArray[np.float64]:
        """Compute print-through signal for an adjacent tape layer.

        Print-through is the unwanted transfer of recorded signal from
        one layer of wound tape to the next via magnetic coupling.  The
        phenomenon is asymmetric: the *leading* edge of a wound layer
        prints ~2 dB more strongly than the trailing edge because of
        the hub-out winding direction (the outer surface of the wound
        pack is wound first, so its leading edge is closer to the
        adjacent inner layer).

        Parameters
        ----------
        magnetisation : np.ndarray
            1-D normalised magnetisation profile along the tape (per μm).
        hub_outer : bool
            If ``True`` (default), model the outer surface of the wound
            pack; the leading edge is enhanced by +2 dB.  If ``False``,
            model the inner surface (trailing edge baseline).

        Returns
        -------
        np.ndarray
            Print-through signal (same shape as *magnetisation*).
            Typical amplitude is −50 to −60 dB below the original
            signal.

        Notes
        -----
        A simple model couples each sample to its neighbour N layers
        away through a spacing loss function.  The print-through
        coupling coefficient *k* is approximately 0.003 (−50 dB) for
        adjacent layers and falls off rapidly with distance.

        The +2 dB leading-edge asymmetry is applied as a multiplicative
        factor on the positive-slope portions of the signal.
        """
        n = len(magnetisation)
        if n == 0:
            return np.array([], dtype=np.float64)

        # Base coupling coefficient (adjacent layer, ~−50 dB)
        k_base = 0.003

        # Spacing loss model: exponential decay with layer index
        # We model the coupling to the immediately adjacent layer only,
        # since further layers are negligible.
        # Convolve with a simple asymmetric kernel that represents
        # the geometric spacing of the wound pack.
        kernel = np.array([k_base * 0.5, k_base, k_base * 0.5], dtype=np.float64)

        printed = np.convolve(magnetisation, kernel, mode="same")

        # Asymmetric leading-edge enhancement (+2 dB ≈ ×1.259)
        # Detect leading edges: positive slope where magnetisation
        # crosses from negative to positive (or increases sharply).
        slope = np.diff(magnetisation, prepend=magnetisation[0])
        leading_edge_mask = slope > 0.0

        enhancement_db = 2.0
        enhancement_linear = 10.0 ** (enhancement_db / 20.0)

        if hub_outer:
            printed[leading_edge_mask] *= enhancement_linear

        return printed

    # -- helpers -----------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"VHSTape({self.tape_type.value}, "
            f"coating={self.coating.value}, "
            f"length={self.tape_length_m:.0f}m)"
        )
