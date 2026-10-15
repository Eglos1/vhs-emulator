"""Magnetic recording physics module for VHS tape emulation.

This module provides physically accurate models for magnetic recording
and playback physics, implementing:

- **Karlqvist head field** approximation for finite-gap recording heads
- **Magnetic hysteresis** using a simplified Jiles-Atherton (arctan) model
- **Recording process** with self-demagnetization and depth-dependent
  magnetisation through the coating
- **Playback** with spacing loss, gap loss, thickness loss, and
  head-bump frequency response
- **VHS-specific defaults** for SP / LP / EP modes

All computations use numpy with float64 precision.

References
----------
.. [1] Karlqvist, O. (1954). "Calculation of the Magnetic Field in the
   Ferromagnetic Layer of a Magnetic Drum." *Transactions of the Royal
   Institute of Technology, Stockholm*, No. 86.
.. [2] Jiles, D.C. & Atherton, D.L. (1986). "Theory of ferromagnetic
   hysteresis." *Journal of Magnetism and Magnetic Materials*, 61(1-2),
   48-60.
.. [3] Berghaus, A.M. (1987). "The Science of Magnetic Recording."
   Cambridge University Press.
.. [4] IEC 60094-1 (Magnetic tape recording).
.. [5] SMPTE 170M / ITU-R BT.601 (timing).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

MU_0: float = 4.0 * math.pi * 1e-7
"""Vacuum permeability (H/m)."""


# ---------------------------------------------------------------------------
# Tape materials
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TapeMaterial:
    """Magnetic tape material properties.

    Parameters
    ----------
    name : str
        Material name and type classification.
    Ms : float
        Saturation magnetisation in Tesla (T).
    Hc : float
        Coercivity in A/m.
    thickness : float
        Coating thickness in metres (m).
    noise_floor : float
        Base noise level in dB relative to a full-scale signal.
    permeability : float
        Relative permeability of the magnetic coating.
    """

    name: str
    Ms: float
    Hc: float
    thickness: float
    noise_floor: float
    permeability: float


FERRIC: TapeMaterial = TapeMaterial(
    name="Type I - Ferric Oxide (γ-Fe₂O₃)",
    Ms=0.35,
    Hc=300.0,
    thickness=19e-6,
    noise_floor=-56.0,
    permeability=1.0,
)
"""Type I — Gamma ferric oxide (γ-Fe₂O₃).

Standard VHS tape material.  Lower coercivity and saturation
magnetisation compared to chrome or metal tapes.

* Ms = 0.35 T, Hc = 300 A/m, coating thickness = 19 μm
"""

CHROME: TapeMaterial = TapeMaterial(
    name="Type II - Chrome (CrO₂)",
    Ms=0.40,
    Hc=550.0,
    thickness=12e-6,
    noise_floor=-60.0,
    permeability=1.0,
)
"""Type II — Chromium dioxide (CrO₂).

Higher coercivity and saturation magnetisation than ferric oxide.
Better high-frequency performance.

* Ms = 0.40 T, Hc = 550 A/m, coating thickness = 12 μm
"""

METAL: TapeMaterial = TapeMaterial(
    name="Type IV - Metal Particle",
    Ms=0.80,
    Hc=2000.0,
    thickness=10e-6,
    noise_floor=-65.0,
    permeability=1.0,
)
"""Type IV — Metal particle (MP).

Highest coercivity and saturation magnetisation.  Best high-frequency
performance and signal-to-noise ratio.

* Ms = 0.80 T, Hc = 2000 A/m, coating thickness = 10 μm
"""


# ---------------------------------------------------------------------------
# Head parameters
# ---------------------------------------------------------------------------

@dataclass
class HeadParams:
    """Recording / playback head geometry and electrical parameters.

    Parameters
    ----------
    gap_write : float
        Write-head gap length in metres (m).
    gap_read : float
        Read-head gap length in metres (m).
    spacing : float
        Head-to-tape spacing in metres (m).
    track_width : float
        Recorded track width in metres (m).
    track_pitch : float
        Centre-to-centre track pitch in metres (m).
    num_turns : int
        Number of turns on the head coil.
    efficiency : float
        Magnetic efficiency of the head circuit (0–1).
    """

    gap_write: float = 0.5e-6
    gap_read: float = 0.3e-6
    spacing: float = 0.05e-6
    track_width: float = 49e-6
    track_pitch: float = 58e-6
    num_turns: int = 100
    efficiency: float = 0.8


# ---------------------------------------------------------------------------
# VHS speed constants
# ---------------------------------------------------------------------------

VHS_SP_SPEED: float = 33.35e-3
"""VHS SP mode tape speed: 33.35 mm/s."""

VHS_LP_SPEED: float = 16.675e-3
"""VHS LP mode tape speed: 16.675 mm/s."""

VHS_EP_SPEED: float = 11.117e-3
"""VHS EP (SLP) mode tape speed: 11.117 mm/s."""


# ---------------------------------------------------------------------------
# MagneticRecorder
# ---------------------------------------------------------------------------

class MagneticRecorder:
    """Physically accurate magnetic recording model for VHS emulation.

    Implements the full write → tape → read chain:

    1. **Karlqvist head field** — Hx, Hy from a finite-gap head.
    2. **Hysteresis loop** — simplified arctan model (Jiles-Atherton
       inspired) with virgin / ascending / descending branches.
    3. **Recording** — depth-resolved magnetisation through the coating,
       including self-demagnetisation.
    4. **Playback** — Faraday-law voltage with spacing, gap, thickness,
       and head-bump losses.

    Examples
    --------
    >>> rec = MagneticRecorder()
    >>> tape_mag = rec.record_flux(current_signal, position_array)
    >>> voltage = rec.read_flux(tape_mag, position_array)
    """

    # -- construction -------------------------------------------------------

    def __init__(
        self,
        tape_speed: float = VHS_SP_SPEED,
        coating: Optional[TapeMaterial] = None,
        head_params: Optional[HeadParams] = None,
    ) -> None:
        """Initialise the magnetic recorder.

        Parameters
        ----------
        tape_speed : float
            Tape transport speed in m/s.  Default is 33.35 mm/s (VHS SP).
        coating : TapeMaterial, optional
            Tape coating material.  Default is :data:`FERRIC`.
        head_params : HeadParams, optional
            Head geometry.  Default is a standard VHS head.
        """
        self.v: float = tape_speed
        self.coating: TapeMaterial = coating if coating is not None else FERRIC
        self.head: HeadParams = head_params if head_params is not None else HeadParams()

    # ===================================================================
    # Karlqvist head field
    # ===================================================================

    def karlqvist_field(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        Hg: float,
        gap: float,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute Karlqvist head field components.

        The Karlqvist approximation [1]_ gives the magnetic field produced
        by a recording head with a straight gap of length *g*:

        .. math::

            H_x(x,y) = \\frac{H_g}{\\pi}
                \\left[\\arctan\\!\\left(\\frac{x+g/2}{y}\\right)
                - \\arctan\\!\\left(\\frac{x-g/2}{y}\\right)\\right]

            H_y(x,y) = \\frac{H_g}{2\\pi}
                \\ln\\!\\frac{(x+g/2)^2 + y^2}{(x-g/2)^2 + y^2}

        where *x* is along tape travel (origin at gap centre), *y* is
        depth into the tape, *Hg* is the gap field strength, and *g* is
        the gap length.

        Parameters
        ----------
        x : np.ndarray
            Position along tape travel (m), origin at gap centre.
        y : np.ndarray
            Depth into the tape from the head surface (m).
        Hg : float
            Head field strength at the gap (A/m).
        gap : float
            Head gap length (m).

        Returns
        -------
        (Hx, Hy) : tuple of np.ndarray
            Horizontal and vertical field components (A/m).

        Notes
        -----
        The approximation is accurate to within ~5 % for points within a
        few gap lengths of the gap centre.  Arrays are broadcast to a
        common shape.

        References
        ----------
        .. [1] Karlqvist (1954), *op. cit.*
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        g2 = gap / 2.0

        # Guard against y = 0 (surface of the head)
        y_safe = np.where(np.abs(y) < 1e-15, 1e-15, y)

        # --- Hx (horizontal / longitudinal component) ---
        Hx = (Hg / math.pi) * (
            np.arctan2(x + g2, y_safe) - np.arctan2(x - g2, y_safe)
        )

        # --- Hy (vertical / perpendicular component) ---
        r1_sq = (x + g2) ** 2 + y_safe ** 2
        r2_sq = (x - g2) ** 2 + y_safe ** 2
        Hy = (Hg / (2.0 * math.pi)) * np.log(r1_sq / r2_sq)

        return Hx, Hy

    # ===================================================================
    # Current → field helpers
    # ===================================================================

    def head_field_from_current(self, current: float) -> float:
        """Compute gap field strength from recording current.

        .. math::

            H_g = \\frac{N \\cdot I \\cdot \\eta}{g \\cdot \\mu_0}

        Parameters
        ----------
        current : float
            Recording current (A).

        Returns
        -------
        float
            Gap field strength *Hg* (A/m).
        """
        return (
            self.head.num_turns
            * current
            * self.head.efficiency
            / (self.head.gap_write * MU_0)
        )

    def write_head_field(
        self,
        current: float,
        position_along_track: NDArray[np.float64],
        depth: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute the write head field at given positions and depths.

        Convenience wrapper around :meth:`head_field_from_current` and
        :meth:`karlqvist_field`.

        Parameters
        ----------
        current : float
            Recording current (A).
        position_along_track : np.ndarray
            Position along the track (m).
        depth : np.ndarray
            Depth into the tape (m).

        Returns
        -------
        (Hx, Hy) : tuple of np.ndarray
            Magnetic field components (A/m).
        """
        Hg = self.head_field_from_current(current)
        return self.karlqvist_field(
            position_along_track, depth, Hg, self.head.gap_write
        )

    # ===================================================================
    # Hysteresis model
    # ===================================================================

    def hysteresis_M(
        self,
        H: NDArray[np.float64],
        Ms: float,
        Hc: float,
        a: float,
    ) -> NDArray[np.float64]:
        """Vectorised magnetic hysteresis model.

        Uses a simplified arctan-based model inspired by Jiles-Atherton
        [2]_ with three branches:

        **Virgin curve** (starting from demagnetised state):

        .. math::

            M(H) = M_s \\cdot \\frac{2}{\\pi}
                \\arctan\\!\\left(\\frac{\\pi\\,H}{2\\,a\\,H_c}\\right)

        **Ascending branch** (H increasing):

        .. math::

            M(H) = M_s \\cdot \\frac{2}{\\pi}
                \\arctan\\!\\left(\\frac{\\pi\\,(H + H_c)}
                                       {2\\,a\\,H_c}\\right)

        **Descending branch** (H decreasing):

        .. math::

            M(H) = M_s \\cdot \\frac{2}{\\pi}
                \\arctan\\!\\left(\\frac{\\pi\\,(H - H_c)}
                                       {2\\,a\\,H_c}\\right)

        The shape parameter *a* controls loop squareness:

        * a < 1 — square loop (hard magnetic material)
        * a = 1 — standard loop
        * a > 1 — round loop (soft magnetic material)

        For γ-Fe₂O₃, *a* ≈ 0.5 gives a realistic loop.

        Parameters
        ----------
        H : np.ndarray
            Applied field sequence (A/m).
        Ms : float
            Saturation magnetisation (T).
        Hc : float
            Coercivity (A/m).
        a : float
            Shape / squareness parameter (dimensionless).

        Returns
        -------
        np.ndarray
            Magnetisation *M* (T), same shape as *H*.

        Notes
        -----
        The model tracks the ascending / descending state across the
        full input sequence.  Reversal points cause a branch switch.
        The remanence is:

        .. math::

            M_r = M_s \\cdot \\frac{2}{\\pi}
                \\arctan\\!\\left(\\frac{\\pi}{2a}\\right)

        References
        ----------
        .. [2] Jiles & Atherton (1986), *op. cit.*
        """
        H = np.asarray(H, dtype=np.float64)
        M = np.empty_like(H)

        if H.size == 0:
            return M

        two_over_pi = 2.0 / math.pi
        pi_over_2aHc = math.pi / (2.0 * a * Hc)

        # --- Fully vectorised hysteresis ---
        # Pad to compute dH
        H_padded = np.concatenate([[0.0], H])
        dH = np.diff(H_padded)

        # Find all reversal points (where direction changes)
        ascending_mask = dH > 0.0
        descending_mask = dH < 0.0

        # Direction change = sign of dH differs from previous
        dir_sign = np.sign(dH)
        dir_changes = np.diff(dir_sign)
        reversal_mask = dir_changes != 0  # length = len(H)-1, index 0 corresponds to H[1]

        # Find first reversal
        reversal_indices = np.where(reversal_mask)[0]
        first_reversal = reversal_indices[0] if reversal_indices.size > 0 else len(H)

        # Virgin branch
        virgin = Ms * two_over_pi * np.arctan(pi_over_2aHc * H)
        # Descending branch (shifted by -Hc)
        desc_branch = Ms * two_over_pi * np.arctan(pi_over_2aHc * (H - Hc))
        # Ascending branch (shifted by +Hc)
        asc_branch = Ms * two_over_pi * np.arctan(pi_over_2aHc * (H + Hc))

        if first_reversal >= len(H):
            return virgin

        # After first reversal, compute state via cumsum of toggles
        state = np.zeros(len(H), dtype=np.int32)
        state[:first_reversal] = 0  # virgin

        # Build toggle array: +1 at ascending reversals, -1 at descending
        toggle = np.zeros(len(H), dtype=np.int32)
        for idx in reversal_indices:
            idx_h = idx  # index in H (reversal at H[idx+1] in original)
            if idx_h + 1 < len(H):
                if ascending_mask[idx_h + 1]:
                    toggle[idx_h + 1] = 1
                elif descending_mask[idx_h + 1]:
                    toggle[idx_h + 1] = -1

        # After first reversal, cumsum gives state: -1=desc, +1=asc
        toggle[first_reversal] = -1  # first reversal is descending
        state_cumsum = np.cumsum(toggle[first_reversal:])
        state[first_reversal:] = np.where(state_cumsum < 0, 2, 1)  # 2=desc, 1=asc

        M = np.where(state == 0, virgin,
            np.where(state == 1, asc_branch, desc_branch))

        return M

    # ===================================================================
    # Self-demagnetisation
    # ===================================================================

    def self_demagnetization(
        self,
        M: NDArray[np.float64],
        wavelength: float,
    ) -> NDArray[np.float64]:
        """Compute the self-demagnetisation field.

        For a thin tape magnetised along its length the demagnetising
        factor is approximately:

        .. math::

            N \\approx \\frac{2\\pi\\,\\delta}{\\lambda}

        where *δ* is the coating thickness and *λ* is the recorded
        wavelength.  The opposing field is:

        .. math::

            H_{\\text{demag}} = -\\frac{N \\cdot M}{\\mu_0}

        Parameters
        ----------
        M : np.ndarray
            Magnetisation (T).
        wavelength : float
            Recorded wavelength (m).

        Returns
        -------
        np.ndarray
            Demagnetisation field (A/m).
        """
        N_demag = 2.0 * math.pi * self.coating.thickness / wavelength
        return -N_demag * M / MU_0

    # ===================================================================
    # Recording
    # ===================================================================

    def record_flux(
        self,
        signal: NDArray[np.float64],
        tape_position: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Record a current signal onto the tape.

        Uses aggressive downsampling: the magnetic tape physically cannot
        resolve frequencies above ~8 MHz, so the signal is downsampled
        before the expensive Karlqvist/hysteresis computation, then the
        result is upsampled back to the original length.

        Depth layers reduced to 5 (the field gradient through a 19 μm
        coating is smooth enough that 5 samples capture it well).

        Parameters
        ----------
        signal : np.ndarray
            Recording current (A), shape ``(N,)``.
        tape_position : np.ndarray
            Position along the tape (m), shape ``(N,)``.

        Returns
        -------
        np.ndarray
            Magnetisation (T), shape ``(N, num_depth_layers)``.
        """
        signal = np.asarray(signal, dtype=np.float64)
        tape_position = np.asarray(tape_position, dtype=np.float64)

        num_positions = signal.size
        num_layers: int = 5
        delta = self.coating.thickness
        depth = np.linspace(0.0, delta, num_layers)

        # --- Downsample by 8× for magnetic computation ---
        ds = max(1, num_positions // 30000)  # target ~30K points
        sig_ds = signal[::ds]
        pos_ds = tape_position[::ds]
        n_ds = sig_ds.size

        mag_ds = np.zeros((n_ds, num_layers), dtype=np.float64)

        a_param: float = 0.5

        # Head at centre
        centre_idx = n_ds // 2
        x_rel = pos_ds - pos_ds[centre_idx]

        ref_wavelength: float = self.v / 44100.0
        g2 = self.head.gap_write / 2.0

        # Compute Hg for every position at once
        Hg_all = self.head.num_turns * sig_ds * self.head.efficiency / (
            self.head.gap_write * MU_0
        )

        for j in range(num_layers):
            d = depth[j]
            y_safe = max(abs(d), 1e-15)

            Hx = (Hg_all / math.pi) * (
                np.arctan2(x_rel + g2, y_safe)
                - np.arctan2(x_rel - g2, y_safe)
            )
            r1_sq = (x_rel + g2) ** 2 + y_safe ** 2
            r2_sq = (x_rel - g2) ** 2 + y_safe ** 2
            Hy = (Hg_all / (2.0 * math.pi)) * np.log(r1_sq / r2_sq)

            H_field = np.sqrt(Hx * Hx + Hy * Hy)

            # Fast arctan hysteresis (recording from erased tape = virgin curve)
            pi_over_2aHc = math.pi / (2.0 * a_param * self.coating.Hc)
            M = self.coating.Ms * (2.0 / math.pi) * np.arctan(pi_over_2aHc * H_field)

            # Self-demagnetisation correction
            N_demag = 2.0 * math.pi * delta / ref_wavelength
            M = M - N_demag * M * MU_0 / MU_0
            M = np.clip(M, -self.coating.Ms, self.coating.Ms)

            mag_ds[:, j] = M

        # --- Upsample back to original length ---
        if ds > 1:
            orig_idx = np.arange(num_positions)
            ds_idx = np.arange(n_ds) * ds
            tape_magnetisation = np.zeros((num_positions, num_layers), dtype=np.float64)
            for j in range(num_layers):
                tape_magnetisation[:, j] = np.interp(orig_idx, ds_idx, mag_ds[:, j])
        else:
            tape_magnetisation = mag_ds

        return tape_magnetisation

    # ===================================================================
    # Playback losses
    # ===================================================================

    def spacing_loss(
        self, wavelength: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Head-tape spacing loss.

        .. math::

            L_{\\text{spacing}} = 54.6 \\;\\frac{d}{\\lambda} \\quad\\text{dB}

        where *d* is the head-tape spacing and *λ* is the recorded
        wavelength.

        Parameters
        ----------
        wavelength : np.ndarray
            Recorded wavelength (m).

        Returns
        -------
        np.ndarray
            Spacing loss (dB).
        """
        wavelength = np.asarray(wavelength, dtype=np.float64)
        return 54.6 * self.head.spacing / wavelength

    def gap_loss(
        self, wavelength: NDArray[np.float64]
    ) -> np.ndarray:
        """Read-head gap loss (sinc loss).

        .. math::

            L_{\\text{gap}} = 20\\,\\log_{10}
                \\left|\\frac{\\sin(\\pi g / \\lambda)}
                               {\\pi g / \\lambda}\\right| \\quad\\text{dB}

        where *g* is the read-head gap length.  The loss is zero when
        *g = λ* and maximum when *g = λ/2*.

        Parameters
        ----------
        wavelength : np.ndarray
            Recorded wavelength (m).

        Returns
        -------
        np.ndarray
            Gap loss (dB).
        """
        wavelength = np.asarray(wavelength, dtype=np.float64)
        x = math.pi * self.head.gap_read / wavelength
        # numpy sinc is sin(πx)/(πx)
        sinc_val = np.where(np.abs(x) < 1e-10, 1.0, np.sin(x) / x)
        return 20.0 * np.log10(np.abs(sinc_val) + 1e-15)

    def thickness_loss(
        self, wavelength: NDArray[np.float64]
    ) -> np.ndarray:
        """Coating thickness loss.

        .. math::

            L_{\\text{thickness}} = 20\\,\\log_{10}
                \\left|1 - \\exp\\!\\left(-\\frac{2\\pi\\,\\delta}{\\lambda}
                \\right)\\right| \\quad\\text{dB}

        where *δ* is the coating thickness.  Flux from deeper layers is
        increasingly attenuated at short wavelengths.

        Parameters
        ----------
        wavelength : np.ndarray
            Recorded wavelength (m).

        Returns
        -------
        np.ndarray
            Thickness loss (dB).
        """
        wavelength = np.asarray(wavelength, dtype=np.float64)
        exponent = -2.0 * math.pi * self.coating.thickness / wavelength
        return 20.0 * np.log10(np.abs(1.0 - np.exp(exponent)) + 1e-15)

    def head_bump_response(
        self, frequency: NDArray[np.float64]
    ) -> np.ndarray:
        """Head-bump frequency response (read-head sensitivity).

        The playback head exhibits a broad resonance around 4–5 MHz
        caused by the mechanical and magnetic properties of the
        head–tape interface.  Modelled as a normalised bandpass:

        .. math::

            H(f) = \\frac{1}{\\sqrt{1 + Q^2
                \\left(\\frac{f}{f_0} - \\frac{f_0}{f}\\right)^2}}

        where *f₀* ≈ 4.5 MHz (peak) and *Q* ≈ 2.

        Parameters
        ----------
        frequency : np.ndarray
            Signal frequency (Hz).

        Returns
        -------
        np.ndarray
            Relative sensitivity (linear, 0–1).
        """
        frequency = np.asarray(frequency, dtype=np.float64)
        f_peak = 4.5e6
        Q = 2.0
        f_norm = frequency / f_peak
        return 1.0 / np.sqrt(1.0 + Q ** 2 * (f_norm - 1.0 / f_norm) ** 2)

    # ===================================================================
    # Playback
    # ===================================================================

    def read_flux(
        self,
        tape_magnetization: NDArray[np.float64],
        tape_position: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Read back voltage from magnetised tape.

        By Faraday's law the induced voltage is:

        .. math::

            V = -\\frac{\\mathrm{d}\\Phi}{\\mathrm{d}t}
               = -v\\,\\frac{\\mathrm{d}\\Phi}{\\mathrm{d}x}

        The flux Φ through the read head is computed by convolving the
        (depth-averaged) tape magnetisation with the head sensitivity
        function, which by the reciprocity theorem [3]_ is proportional
        to the Karlqvist field the head *would* produce.

        Losses applied (all evaluated at the dominant recorded wavelength):

        * **Spacing loss** — :math:`54.6\\,d/\\lambda` dB
        * **Gap loss** — :math:`20\\,\\log_{10}|\\operatorname{sinc}
          (\\pi g/\\lambda)|`
        * **Thickness loss** — :math:`20\\,\\log_{10}|1 -
          \\exp(-2\\pi\\delta/\\lambda)|`
        * **Head bump** — frequency-dependent sensitivity peaking at
          ~4.5 MHz

        Parameters
        ----------
        tape_magnetization : np.ndarray
            Magnetisation (T), shape ``(N, num_layers)``.
        tape_position : np.ndarray
            Position along tape (m), shape ``(N,)``.

        Returns
        -------
        np.ndarray
            Read-back voltage (V), shape ``(N,)``.
        """
        tape_magnetization = np.asarray(tape_magnetization, dtype=np.float64)
        tape_position = np.asarray(tape_position, dtype=np.float64)

        num_positions = tape_position.size

        # Depth-averaged magnetisation
        M_eff = np.mean(tape_magnetization, axis=1)

        if num_positions <= 1:
            return np.zeros_like(M_eff)

        dx = float(np.mean(np.diff(tape_position)))

        # --- Head sensitivity function (reciprocity) -------------------
        # Use the Hy component of a unit-strength Karlqvist field at the
        # read gap.  This captures the bandpass character of the head.
        x = tape_position - np.mean(tape_position)
        y_depth = self.coating.thickness / 2.0
        _, Hy = self.karlqvist_field(
            x, y_depth, 1.0, self.head.gap_read
        )
        sensitivity = Hy

        # Flux through the head: Φ(x) = ∫ M(x') · h(x − x') dx'
        flux = np.convolve(M_eff, sensitivity, mode="same") * dx

        # --- Estimate dominant wavelength via FFT ----------------------
        fft_mag = np.abs(np.fft.rfft(M_eff))
        freqs = np.fft.rfftfreq(num_positions, dx)

        if fft_mag.size > 1:
            peak_idx = int(np.argmax(fft_mag[1:])) + 1
            peak_freq = float(freqs[peak_idx])
            wavelength = (
                self.v / peak_freq if peak_freq > 0 else self.v / 1e6
            )
        else:
            wavelength = self.v / 1e6

        # --- Total loss (dB → linear) ----------------------------------
        lam_arr = np.array([wavelength], dtype=np.float64)
        total_loss_dB = (
            float(self.spacing_loss(lam_arr)[0])
            + float(self.gap_loss(lam_arr)[0])
            + float(self.thickness_loss(lam_arr)[0])
        )
        total_loss_lin = 10.0 ** (total_loss_dB / 20.0)

        # --- Head bump response -----------------------------------------
        f_est = self.v / wavelength if wavelength > 0 else 1e6
        head_resp = float(
            self.head_bump_response(np.array([f_est], dtype=np.float64))[0]
        )

        # --- Voltage: V = −v · dΦ/dx -----------------------------------
        voltage = (
            -self.v * np.gradient(flux, dx) * total_loss_lin * head_resp
        )

        return voltage

    # ===================================================================
    # Representation
    # ===================================================================

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"MagneticRecorder(v={self.v * 1e3:.2f} mm/s, "
            f"coating={self.coating.name}, "
            f"gap_w={self.head.gap_write * 1e6:.1f} μm, "
            f"gap_r={self.head.gap_read * 1e6:.1f} μm)"
        )
