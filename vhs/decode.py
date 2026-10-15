"""VHS playback (decoding) signal chain.

This module takes the raw magnetic flux read from the tape and decodes it
back into composite video + audio.  It models the actual electronic
circuitry inside a VHS player.

Signal path
-----------
Read head → Preamplifier → Playback EQ → Split into Y / C / Audio
  ├─ Y path: Bandpass → Limiter → FM demod → De-emphasis → Luma out
  ├─ C path: BPF 629 kHz → PLL → Upconvert → ACC → Chroma out
  └─ Audio path: Linear demod (or Hi-Fi FM demod) → De-emphasis

Y + C are recombined into composite video.

References
----------
- VHS Format Specification (JVC, 1976 rev.)
- EIAJ CP-610 (VHS recording standard)
- SMPTE 170M / ITU-R BT.601 (timing)
- NAB / IEC 60094-1 (tape recording)
- Matsushita / Sanyo / Samsung VHS service manuals
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sp_signal

from .encode import VHS_Y_FM_SYNC_HZ_SP, VHS_Y_FM_WHITE_HZ_SP


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class VideoStandard(enum.Enum):
    NTSC = "NTSC"
    PAL = "PAL"


class SpeedMode(enum.Enum):
    SP = "SP"
    LP = "LP"
    EP = "EP"


# ---------------------------------------------------------------------------
# Standard-specific constants
# ---------------------------------------------------------------------------

# All frequencies in Hz unless otherwise noted.

STANDARD_PARAMS: dict[VideoStandard, dict[str, float]] = {
    VideoStandard.NTSC: {
        # Luminance FM carrier
        "f_carrier_sync": 3.4e6,       # frequency at sync tip (Hz)
        "f_carrier_white": 4.4e6,       # frequency at peak white (Hz)
        "f_carrier_center": 3.9e6,      # midpoint carrier (Hz)
        "fm_deviation": 1.0e6,          # peak deviation from center (Hz)
        # Chroma color-under
        "f_chroma_under": 629375.0,     # 629.375 kHz = fsc/4 for NTSC
        "f_subcarrier": 3579545.0,      # NTSC color subcarrier (Hz)
        # Timing
        "line_rate": 15734.264,         # horizontal line frequency (Hz)
        "field_rate": 59.94,            # field rate (Hz)
        "lines_per_field": 262.5,       # interlaced
        "total_lines_per_frame": 525,
        "sync_pulse_us": 4.7,           # horizontal sync width
        "blanking_us": 10.9,            # horizontal blanking total
        "vsync_us": 3.0,                # vertical sync pre-equalizer region
        # Audio
        "audio_carrier_linear": None,   # linear audio, no FM carrier
        "audio_carrier_hifi_l": 1.3e6,  # Hi-Fi L channel FM carrier
        "audio_carrier_hifi_r": 1.7e6,  # Hi-Fi R channel FM carrier
        "audio_fm_deviation": 150e3,    # Hi-Fi deviation ±150 kHz
        "audio_deemph_tc_us": 50.0,     # 50 μs de-emphasis time constant
        # Head switching
        "head_switch_offset_lines": 6,  # lines before vsync
    },
    VideoStandard.PAL: {
        "f_carrier_sync": 3.4e6,
        "f_carrier_white": 4.4e6,
        "f_carrier_center": 3.9e6,
        "fm_deviation": 1.0e6,
        "f_chroma_under": 629375.0,     # fsc/4 PAL ~629 kHz
        "f_subcarrier": 4433618.75,     # PAL color subcarrier (Hz)
        "line_rate": 15625.0,
        "field_rate": 50.0,
        "lines_per_field": 312.5,
        "total_lines_per_frame": 625,
        "sync_pulse_us": 4.7,
        "blanking_us": 12.05,
        "vsync_us": 2.5,
        "audio_carrier_linear": None,
        "audio_carrier_hifi_l": 1.3e6,
        "audio_carrier_hifi_r": 1.7e6,
        "audio_fm_deviation": 150e3,
        "audio_deemph_tc_us": 50.0,
        "head_switch_offset_lines": 6,
    },
}

# Speed-mode tape transport parameters
SPEED_MODE_PARAMS: dict[SpeedMode, dict[str, float]] = {
    SpeedMode.SP: {
        "tape_speed_mm_s": 33.35,
        "track_pitch_um": 58.0,
        "linear_track_width_um": 1.0,  # linear audio track width
    },
    SpeedMode.LP: {
        "tape_speed_mm_s": 16.675,
        "track_pitch_um": 29.0,
        "linear_track_width_um": 0.5,
    },
    SpeedMode.EP: {
        "tape_speed_mm_s": 11.117,
        "track_pitch_um": 19.3,
        "linear_track_width_um": 0.35,
    },
}


# ---------------------------------------------------------------------------
# Filter design helpers
# ---------------------------------------------------------------------------

def _design_butterworth(
    order: int,
    cutoff_hz: float | list[float],
    sample_rate: float,
    btype: str = "low",
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Design a Butterworth digital filter via bilinear transform.

    Parameters
    ----------
    order : int
        Filter order.
    cutoff_hz : float or list of two floats
        Cutoff frequency in Hz.  For bandpass/bandstop provide [low, high].
    sample_rate : float
        Sampling rate in Hz.
    btype : str
        ``"low"``, ``"high"``, ``"band"``, or ``"stop"``.

    Returns
    -------
    b, a : ndarray
        Numerator and denominator transfer-function coefficients.
    """
    nyq = sample_rate / 2.0
    if isinstance(cutoff_hz, (list, tuple)):
        normalized = [c / nyq for c in cutoff_hz]
    else:
        normalized = cutoff_hz / nyq
    # Clip to valid range for bilinear transform
    if isinstance(normalized, list):
        normalized = [min(max(c, 1e-5), 1.0 - 1e-5) for c in normalized]
    else:
        normalized = min(max(normalized, 1e-5), 1.0 - 1e-5)
    return sp_signal.butter(order, normalized, btype=btype)


def _biquad_peak(f0: float, gain_db: float, q: float, sample_rate: float):
    """Design a peaking (parametric EQ) biquad filter.

    Transfer function::

        H(z) = (b0 + b1 z^{-1} + b2 z^{-2}) / (1 + a1 z^{-1} + a2 z^{-2})

    Parameters
    ----------
    f0 : float
        Centre frequency (Hz).
    gain_db : float
        Gain at centre frequency (dB).  Positive = boost, negative = cut.
    q : float
        Quality factor (bandwidth = f0 / Q).
    sample_rate : float
        Sampling rate (Hz).

    Returns
    -------
    b, a : ndarray
        Filter coefficients.
    """
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sample_rate
    alpha = np.sin(w0) / (2.0 * Q(q))
    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A
    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return b, a


def Q(q: float) -> float:
    """Return the Q factor value (identity, for readability)."""
    return q


# ---------------------------------------------------------------------------
# VHSDecoder
# ---------------------------------------------------------------------------

class VHSDecoder:
    """VHS playback decoder — models the signal chain inside a VHS player.

    Parameters
    ----------
    standard : VideoStandard
        NTSC or PAL.
    speed_mode : SpeedMode
        SP, LP, or EP.
    hifi_audio : bool
        If ``True``, decode Hi-Fi stereo audio in addition to (or instead
        of) linear mono audio.

    Notes
    -----
    The decoder state is intentionally minimal so that it can be called
    field-by-field.  Any per-field adaptive state (e.g. ACC, PLL phase)
    is reset between calls to keep the interface simple.
    """

    def __init__(
        self,
        standard: VideoStandard = VideoStandard.NTSC,
        speed_mode: SpeedMode = SpeedMode.SP,
        hifi_audio: bool = False,
    ) -> None:
        self.standard = standard
        self.speed_mode = speed_mode
        self.hifi_audio = hifi_audio

        # Pull standard-specific parameters
        params = STANDARD_PARAMS[standard]
        self.f_carrier_sync: float = params["f_carrier_sync"]
        self.f_carrier_white: float = params["f_carrier_white"]
        self.f_carrier_center: float = params["f_carrier_center"]
        self.fm_deviation: float = params["fm_deviation"]
        self.f_chroma_under: float = params["f_chroma_under"]
        self.f_subcarrier: float = params["f_subcarrier"]
        self.line_rate: float = params["line_rate"]
        self.field_rate: float = params["field_rate"]
        self.lines_per_field: float = params["lines_per_field"]
        self.total_lines: int = int(params["total_lines_per_frame"])
        self.head_switch_offset_lines: int = int(
            params["head_switch_offset_lines"]
        )

        speed_params = SPEED_MODE_PARAMS[speed_mode]
        self.tape_speed_mm_s: float = speed_params["tape_speed_mm_s"]
        self.track_pitch_um: float = speed_params["track_pitch_um"]

        # Derived constants
        self.field_period_s: float = 1.0 / self.field_rate
        self.line_period_s: float = 1.0 / self.line_rate

        # Head switching glitch duration in seconds (~2-4 lines of noise)
        # We model it as 3 lines worth of degraded signal
        self.head_switch_duration_s: float = (
            3.0 * self.line_period_s
        )

    # ------------------------------------------------------------------
    # Main decode entry point
    # ------------------------------------------------------------------

    def decode_field(
        self,
        raw_tape_signal: NDArray[np.float64],
        sample_rate: float,
        head_index: int,
        field_number: int,
    ) -> dict:
        """Decode one field from raw magnetic-flux tape signal.

        Parameters
        ----------
        raw_tape_signal : ndarray
            1-D array of voltage samples as read from the magnetic head.
        sample_rate : float
            Sampling rate of *raw_tape_signal* in Hz.
        head_index : int
            Rotating head index (0 or 1).  Affects head switching timing.
        field_number : int
            Sequential field number (used for dropout seeding, etc.).

        Returns
        -------
        dict
            ``'composite'`` : ndarray — decoded composite video
            ``'luma'``      : ndarray — decoded luminance (Y)
            ``'chroma'``     : ndarray — decoded chrominance (C)
            ``'audio'``      : ndarray — decoded audio waveform
            ``'quality'``    : dict — SNR estimates, dropout counts, etc.
        """
        signal = raw_tape_signal.astype(np.float64)
        quality: Dict[str, float] = {}

        # --- Stage 1: Preamplifier ----------------------------------------
        signal = self._preamplify_fs(signal, sample_rate)
        quality["preamp_gain_db"] = 40.0

        # --- Stage 2: Playback equalization -------------------------------
        signal = self._playback_eq(signal, sample_rate)
        quality["eq_applied"] = True

        # --- Head switching artifact (applied to input-domain signal) ------
        switch_point = self._compute_head_switch_point(
            len(signal), sample_rate, head_index
        )
        signal, dropout_mask = self._head_switch_noise(
            signal, switch_point, sample_rate
        )
        quality["head_switch_samples"] = int(
            np.sum(dropout_mask)
        )

        # --- Split into luminance, chrominance, audio ---------------------

        # FM luminance path
        luma_signal = self._fm_demodulate(signal, sample_rate)
        luma_signal = self._de_emphasize(luma_signal, sample_rate)
        luma_signal = self._clamp_white_level(luma_signal)
        quality["luma_peak_amplitude"] = float(np.max(np.abs(luma_signal)))

        # Chrominance path
        # Instead of: extract at 629 kHz BPF (causes severe ringing that
        # smears energy between color bars), we upconvert the raw EQ'd
        # signal directly to 3.58 MHz.  The LO at 2.95 MHz mixes the
        # 629 kHz color-under up to 3.58 MHz, while the 3.58 MHz BPF
        # rejects the FM luminance content (its difference products at
        # 0.45–1.45 MHz are well outside the 3.08–4.08 MHz passband).
        chroma_upconverted = self._upconvert_chroma(
            signal, sample_rate, burst_phase=0.0
        )
        quality["chroma_amplitude"] = float(np.max(np.abs(chroma_upconverted)))

        # Recombine Y + C into composite
        composite = self._recombine_yc(
            luma_signal, chroma_upconverted, sample_rate
        )

        # Audio path
        audio = self._demodulate_audio(signal, sample_rate)
        quality["audio_rms"] = float(np.sqrt(np.mean(audio ** 2)))

        # --- Dropout model ------------------------------------------------
        dropout_count = self._count_dropouts(dropout_mask, sample_rate)
        quality["dropout_count"] = dropout_count
        quality["dropout_total_us"] = float(
            dropout_count * (1.0 / sample_rate) * 1e6
        )

        return {
            "composite": composite,
            "luma": luma_signal,
            "chroma": chroma_upconverted,
            "audio": audio,
            "quality": quality,
        }

    # ------------------------------------------------------------------
    # Stage 6: Composite → Raster (for display)
    # ------------------------------------------------------------------

    def composite_to_raster(
        self,
        composite: NDArray[np.float64],
        sample_rate: float,
        output_width: int = 720,
        output_height: int | None = None,
    ) -> NDArray[np.float64]:
        """Convert a decoded composite video waveform to a 2D raster image.

        Uses known line timing from the TV standard to extract active video
        from each scanline.

        Parameters
        ----------
        composite : ndarray
            Decoded composite video waveform (1D).
        sample_rate : float
            Sampling rate of the composite signal.
        output_width : int
            Number of pixels per line in the output image.
        output_height : int, optional
            Number of lines. If None, uses lines_per_field for the standard.

        Returns
        -------
        ndarray
            2D array of shape (height, width) with pixel values in [0, 1].
        """
        n = len(composite)
        if output_height is None:
            output_height = int(self.lines_per_field)

        # Use exact fractional samples_per_line to avoid cumulative drift
        samples_per_line_f = sample_rate / self.line_rate
        # Active video starts after sync + blanking.
        # NTSC: sync tip (4.7μs) + back porch + color burst = ~11.7μs total blanking.
        # PAL: ~12.0μs total blanking.
        if self.standard == VideoStandard.NTSC:
            blanking_us = 11.7
        else:
            blanking_us = 12.0
        blanking_samples_f = blanking_us * 1e-6 * sample_rate
        # Active video duration = line period - blanking
        line_period_us = 1e6 / self.line_rate
        active_line_us = line_period_us - blanking_us
        active_samples = int(active_line_us * 1e-6 * sample_rate)

        raster = np.zeros((output_height, output_width), dtype=np.float64)

        for row in range(output_height):
            line_start_f = row * samples_per_line_f
            active_start = int(line_start_f + blanking_samples_f)
            active_end = active_start + active_samples

            if active_end > n:
                break

            active_line = composite[active_start:active_end]

            if len(active_line) == 0:
                continue

            # Resample to output_width
            if len(active_line) != output_width:
                x_old = np.linspace(0, 1, len(active_line))
                x_new = np.linspace(0, 1, output_width)
                active_line = np.interp(x_new, x_old, active_line)

            raster[row, :] = np.clip(active_line, 0.0, 1.0)

        return raster

    def yc_to_rgb_raster(
        self,
        luma: NDArray[np.float64],
        chroma: NDArray[np.float64],
        sample_rate: float,
        output_width: int = 720,
        output_height: int | None = None,
    ) -> NDArray[np.float64]:
        """Convert decoded luma + chrominance to an RGB raster image.

        Uses quadrature demodulation to extract I and Q components from
        the subcarrier, then reconstructs RGB via the ITU-R BT.601 matrix.

        The color burst (present on the back porch of every line) is used
        as a per-line phase and amplitude reference for the I/Q demodulator.
        This corrects for any hue rotation introduced by the decode chain
        and normalizes saturation to a consistent level.

        Parameters
        ----------
        luma : ndarray
            Demodulated luminance, [0, 1].
        chroma : ndarray
            Chrominance at subcarrier frequency (3.58 MHz).
        sample_rate : float
            Sampling rate in Hz.
        output_width : int
            Pixels per line in the output.
        output_height : int, optional
            Number of output lines.

        Returns
        -------
        ndarray
            (height, width, 3) RGB array, values in [0, 1].
        """
        n = len(luma)
        if output_height is None:
            output_height = int(self.lines_per_field)

        samples_per_line_f = sample_rate / self.line_rate
        if self.standard == VideoStandard.NTSC:
            blanking_us = 11.7
        else:
            blanking_us = 12.0
        blanking_samples_f = blanking_us * 1e-6 * sample_rate
        line_period_us = 1e6 / self.line_rate
        active_line_us = line_period_us - blanking_us
        active_samples = int(active_line_us * 1e-6 * sample_rate)

        f_sc = self.f_subcarrier
        t_full = np.arange(n, dtype=np.float64) / sample_rate

        # --- Quadrature demodulation ---
        # Mix chrominance with in-phase and quadrature carriers to extract
        # the baseband I and Q components.
        #
        #   chroma(t) = I(t)*cos(ωt) - Q(t)*sin(ωt)
        #   I_base(t) = LPF{ chroma(t) * 2*cos(ωt) }
        #   Q_base(t) = LPF{ chroma(t) * (-2*sin(ωt)) }
        #
        # The factor of 2 compensates for the 1/2 from product-to-sum.

        carrier_cos = 2.0 * np.cos(2.0 * np.pi * f_sc * t_full)
        carrier_neg_sin = -2.0 * np.sin(2.0 * np.pi * f_sc * t_full)

        I_raw = chroma * carrier_cos
        Q_raw = chroma * carrier_neg_sin

        # Lowpass filter to remove 2× carrier products (~7.16 MHz).
        # Use a 4th-order Butterworth at 1.2 MHz (covers chroma bandwidth).
        lp_cutoff = 1.2e6
        if lp_cutoff < sample_rate / 2.0:
            b_lp, a_lp = _design_butterworth(
                order=4,
                cutoff_hz=lp_cutoff,
                sample_rate=sample_rate,
                btype="low",
            )
            I_base = sp_signal.lfilter(b_lp, a_lp, I_raw)
            Q_base = sp_signal.lfilter(b_lp, a_lp, Q_raw)
        else:
            I_base = I_raw
            Q_base = Q_raw

        # --- Per-line burst detection and correction ---
        # The color burst appears on the back porch at ~5.7–8.2 μs after
        # the start of each line.  However, the 3.58 MHz BPF (~1.3 μs
        # group delay) and the 1.2 MHz LPF in the quad demod (~0.8 μs)
        # shift the burst to ~7.8–10.3 μs in I_base/Q_base.  We use a
        # generous window (6.5–11.5 μs) to capture the delayed burst.
        if self.standard == VideoStandard.NTSC:
            burst_start_us = 6.5
            burst_end_us = 11.5
        else:
            burst_start_us = 6.5
            burst_end_us = 11.5
        burst_start_samp = int(burst_start_us * 1e-6 * sample_rate)
        burst_end_samp = int(burst_end_us * 1e-6 * sample_rate)
        burst_len = burst_end_samp - burst_start_samp

        # Also estimate burst amplitude from multiple lines for robustness
        burst_amps = []
        n_lines_total = int(n / samples_per_line_f)
        for li in range(3, min(n_lines_total, 30)):
            ls = int(li * samples_per_line_f)
            bp_s = ls + burst_start_samp
            bp_e = ls + burst_end_samp
            if bp_e < n and burst_len > 4:
                I_b = I_base[bp_s:bp_e]
                Q_b = Q_base[bp_s:bp_e]
                amp = np.sqrt(np.mean(I_b) ** 2 + np.mean(Q_b) ** 2)
                if amp > 1e-6:
                    burst_amps.append(amp)

        # Global burst amplitude (median of several lines for robustness)
        if len(burst_amps) > 3:
            global_burst_amp = float(np.median(burst_amps))
        else:
            global_burst_amp = 1e-6

        # Target burst amplitude: what we want the burst I/Q magnitude to be
        # after normalization.  The encoder records burst at 0.142 amplitude.
        # After decode chain the level varies; we normalize to a standard level.
        target_burst_amp = 0.142

        raster = np.zeros((output_height, output_width, 3), dtype=np.float64)

        for row in range(output_height):
            line_start_f = row * samples_per_line_f
            line_start = int(line_start_f)
            active_start = line_start + int(blanking_samples_f)
            active_end = active_start + active_samples

            if active_end > n:
                break

            # --- Per-line burst phase ---
            bp_start = line_start + burst_start_samp
            bp_end = line_start + burst_end_samp

            if bp_end < n and burst_len > 4:
                I_burst = I_base[bp_start:bp_end]
                Q_burst = Q_base[bp_start:bp_end]
                burst_phase = np.arctan2(
                    float(np.mean(Q_burst)), float(np.mean(I_burst))
                )
            else:
                burst_phase = 0.0

            # --- Rotate I/Q by burst phase to correct hue ---
            # The burst defines the -I axis (180° in NTSC).  We need to
            # rotate by (π - burst_phase) to align the burst to 180°,
            # NOT by -burst_phase which would align it to 0°.
            #   cos_rot = cos(π - burst_phase) = -cos(burst_phase)
            #   sin_rot = sin(π - burst_phase) =  sin(burst_phase)
            cos_rot = -np.cos(burst_phase)
            sin_rot = np.sin(burst_phase)

            I_active = I_base[active_start:active_end]
            Q_active = Q_base[active_start:active_end]

            I_rot = I_active * cos_rot - Q_active * sin_rot
            Q_rot = I_active * sin_rot + Q_active * cos_rot

            # --- Normalize saturation using burst amplitude ---
            if global_burst_amp > 1e-6:
                sat_scale = target_burst_amp / global_burst_amp
                # Clamp to prevent extreme scaling
                sat_scale = np.clip(sat_scale, 0.01, 10.0)
            else:
                sat_scale = 1.0

            I_scaled = I_rot * sat_scale
            Q_scaled = Q_rot * sat_scale

            # --- Resample to output_width ---
            n_active = len(I_scaled)
            if n_active == 0:
                continue

            if n_active != output_width:
                x_old = np.linspace(0, 1, n_active)
                x_new = np.linspace(0, 1, output_width)
                Y_out = np.interp(x_new, x_old, luma[active_start:active_end])
                I_out = np.interp(x_new, x_old, I_scaled)
                Q_out = np.interp(x_new, x_old, Q_scaled)
            else:
                Y_out = luma[active_start:active_end].copy()
                I_out = I_scaled.copy()
                Q_out = Q_scaled.copy()

            # --- YIQ → RGB (ITU-R BT.601) ---------------------------
            #   R = Y + 0.956·I + 0.621·Q
            #   G = Y − 0.272·I − 0.647·Q
            #   B = Y − 1.105·I + 1.703·Q
            R = Y_out + 0.956 * I_out + 0.621 * Q_out
            G = Y_out - 0.272 * I_out - 0.647 * Q_out
            B = Y_out - 1.105 * I_out + 1.703 * Q_out

            raster[row, :, 0] = np.clip(R, 0.0, 1.0)
            raster[row, :, 1] = np.clip(G, 0.0, 1.0)
            raster[row, :, 2] = np.clip(B, 0.0, 1.0)

        return raster

    # ------------------------------------------------------------------
    # Stage 1: Preamplifier
    # ------------------------------------------------------------------

    def _preamplify(self, signal: NDArray[np.float64]) -> NDArray[np.float64]:
        """Head preamplifier: +40 dB gain, bandpass 10 Hz – 10 MHz.

        The preamplifier is the first active stage.  It must have extremely
        low noise figure (~3 dB) because the head output is on the order
        of microvolts.

        Transfer function modelled as a 4th-order Butterworth bandpass:

        .. math::

            G = 10^{40/20} = 100 \\text{ (voltage gain)}

        Bandpass: 10 Hz to 10 MHz (–3 dB points).

        Parameters
        ----------
        signal : ndarray
            Raw voltage from read head.

        Returns
        -------
        ndarray
            Amplified and bandpass-filtered signal.
        """
        gain_db = 40.0
        gain_linear = 10.0 ** (gain_db / 20.0)  # ×100 voltage

        # Bandpass 10 Hz – 10 MHz
        # We assume the input sample rate is high enough to support 10 MHz
        # bandwidth (≥ 20 MS/s).
        try:
            b_bp, a_bp = _design_butterworth(
                order=4,
                cutoff_hz=[10.0, 10.0e6],
                sample_rate=1.0 / (1.0),  # placeholder, use actual fs
                btype="band",
            )
        except Exception:
            # If sample rate too low for the filter, just apply gain
            return signal * gain_linear

        # NOTE: We need the actual sample rate here; this method is called
        # internally where we always pass sample_rate through decode_field.
        # For standalone use we fall through to simple gain.
        return signal * gain_linear

    def _preamplify_fs(
        self, signal: NDArray[np.float64], sample_rate: float
    ) -> NDArray[np.float64]:
        """Preamplifier with explicit sample rate (internal use).

        Parameters
        ----------
        signal : ndarray
            Raw voltage from read head.
        sample_rate : float
            Sampling rate in Hz.

        Returns
        -------
        ndarray
            Amplified and bandpass-filtered signal.
        """
        gain_db = 40.0
        gain_linear = 10.0 ** (gain_db / 20.0)

        # Only apply bandpass if sample rate supports it
        nyquist = sample_rate / 2.0
        if nyquist > 10.0e6:
            b_bp, a_bp = _design_butterworth(
                order=4,
                cutoff_hz=[10.0, 10.0e6],
                sample_rate=sample_rate,
                btype="band",
            )
            signal = sp_signal.lfilter(b_bp, a_bp, signal)
        elif nyquist > 1.0e6:
            # Can't fit full 10 Hz–10 MHz bandpass, but apply highpass at 10 Hz
            # to remove DC offset.  (Previous elif was dead — duplicated
            # nyquist > 10.0e6 which never fires at 14.3 MHz sample rate.)
            b_hp, a_hp = _design_butterworth(
                order=2, cutoff_hz=10.0, sample_rate=sample_rate, btype="high"
            )
            signal = sp_signal.lfilter(b_hp, a_hp, signal)

        return signal * gain_linear

    # ------------------------------------------------------------------
    # Stage 2: Playback Equalization
    # ------------------------------------------------------------------

    def _playback_eq(
        self, signal: NDArray[np.float64], sample_rate: float
    ) -> NDArray[np.float64]:
        """Playback equalization compensating for recording/playback losses.

        The playback EQ roughly inverts the recording chain losses:

        1. **Spacing loss** — Wallace correction factor:

           .. math::

               L_s = -20 \\log_{10}\\!\\left(\\frac{2\\pi f\\, d}{v}\\right) \\text{ dB}

           where *d* is head-tape spacing and *v* is tape velocity.

        2. **Gap loss** — sinc-shaped response of the head gap:

           .. math::

               L_g = \\text{sinc}\\!\\left(\\frac{f \\cdot g}{v}\\right)

           where *g* is the head gap length.

        3. **Head bump** — the head sensitivity peaks around 4–5 MHz due
           to the head structure; this must be equalized down.

        4. **Recording pre-emphasis compensation** — the recording chain
           boosts high frequencies to improve SNR; playback must de-emphasize
           to restore flat response.

        Parameters
        ----------
        signal : ndarray
            Pre-amplified tape signal.
        sample_rate : float
            Sampling rate in Hz.

        Returns
        -------
        ndarray
            Equalized signal with flat frequency response.
        """
        # Head gap length (m) — typical VHS playback head
        gap_length_um = 0.4
        gap_m = gap_length_um * 1e-6

        # Head-tape spacing (m)
        spacing_um = 0.1
        spacing_m = spacing_um * 1e-6

        # Tape velocity at head surface (helical scan)
        # Effective velocity is tape speed / sin(wrap_angle)
        # VHS wrap angle ≈ 180°, so effective velocity ≈ tape speed
        # but the head moves much faster due to drum rotation.
        # Drum diameter ~62 mm, 1800 rpm (NTSC) → head speed ~5.8 m/s
        drum_diameter_mm = 62.0
        drum_rpm = 1800.0  # NTSC
        head_velocity_ms = (
            np.pi * drum_diameter_mm * 1e-3 * drum_rpm / 60.0
        )

        n = len(signal)
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
        freqs[0] = 1.0  # avoid DC division issues

        # Wallace spacing loss: -20 log10(2*pi*f*d/v)
        # This is a loss, so we *boost* by the inverse
        spacing_loss_linear = np.abs(2.0 * np.pi * freqs * spacing_m / head_velocity_ms)
        spacing_loss_linear = np.clip(spacing_loss_linear, 1e-10, None)
        spacing_boost_db = -20.0 * np.log10(spacing_loss_linear)
        spacing_boost_linear = 10.0 ** (spacing_boost_db / 20.0)

        # Gap loss: sinc(f * gap / v)  — boost inverse
        gap_loss_arg = freqs * gap_m / head_velocity_ms
        gap_loss_sinc = np.sinc(gap_loss_arg)
        gap_loss_sinc = np.where(
            np.abs(gap_loss_sinc) < 1e-6, 1e-6, gap_loss_sinc
        )
        gap_boost_linear = 1.0 / np.abs(gap_loss_sinc)

        # Head bump compensation
        # Head sensitivity peaks around 4.5 MHz with ~3 dB boost
        # Modelled as a notch (inverse bell) centered at 4.5 MHz
        f_head_bump = 4.5e6
        head_bump_q = 2.0
        head_bump_notch_db = -3.0  # cut to flatten

        # De-emphasis for recording pre-emphasis compensation
        # Recording used f1=1.3 MHz, f2=4.5 MHz lag-lead
        # Playback uses the inverse: lead-lag
        f1_deemph = 1.3e6
        f2_deemph = 4.5e6
        tau1 = 1.0 / (2.0 * np.pi * f1_deemph)
        tau2 = 1.0 / (2.0 * np.pi * f2_deemph)

        # Combined equalization in frequency domain
        omega = 2.0 * np.pi * freqs
        # Lag-lead de-emphasis: H(s) = (1 + s*tau1) / (1 + s*tau2)
        deemph_real = (1.0 + (omega * tau1) ** 2) * (1.0 + (omega * tau2) ** 2)
        deemph_real = np.sqrt(deemph_real)
        deemph_norm = np.sqrt((1.0 + (0.0 * tau1) ** 2) * (1.0 + (0.0 * tau2) ** 2))
        deemph_norm = max(deemph_norm, 1e-10)
        deemph_response = deemph_real / deemph_norm

        # Head bump: Gaussian notch at f_head_bump
        head_bump_response = 1.0 - 0.5 * np.exp(
            -0.5 * ((freqs - f_head_bump) / (f_head_bump / head_bump_q)) ** 2
        )
        head_bump_response = np.clip(head_bump_response, 0.3, 1.0)

        # Total equalization curve
        total_eq = spacing_boost_linear * gap_boost_linear * deemph_response * head_bump_response

        # Limit boost to prevent excessive noise amplification (max 20 dB)
        total_eq = np.clip(total_eq, 0.1, 10.0)

        # Apply in frequency domain
        spectrum = np.fft.rfft(signal)
        spectrum *= total_eq
        signal_eq = np.fft.irfft(spectrum, n=n)

        return signal_eq

    # ------------------------------------------------------------------
    # Stage 3: Luminance FM Demodulation
    # ------------------------------------------------------------------

    def _fm_demodulate(
        self, signal: NDArray[np.float64], sample_rate: float
    ) -> NDArray[np.float64]:
        """FM demodulate luminance from carrier using Hilbert discriminator.

        The VHS luminance signal is FM-modulated:

        .. math::

            f_{\\text{sync}} = 3.4 \\text{ MHz},\\quad
            f_{\\text{white}} = 4.4 \\text{ MHz},\\quad
            f_{\\text{center}} = 3.9 \\text{ MHz}

        1. **Bandpass filter** (2.0–6.0 MHz) — isolate FM luminance carrier
        2. **Analytic signal** via Hilbert transform
        3. **Instantaneous phase** → differentiate → instantaneous frequency
        4. **Normalize** using known FM spec bounds (3.4–4.4 MHz)

        Parameters
        ----------
        signal : ndarray
            Equalized tape signal.
        sample_rate : float
            Sampling rate in Hz.

        Returns
        -------
        ndarray
            Demodulated luminance (voltage ∝ instantaneous frequency).
        """
        from scipy.signal import hilbert

        n = len(signal)

        # --- Bandpass: isolate FM luminance carrier ---
        nyquist = sample_rate / 2.0
        if nyquist > 6.0e6:
            bp_low = 2.0e6
            bp_high = min(6.0e6, nyquist - 1000.0)
            b_bp, a_bp = _design_butterworth(
                order=2,
                cutoff_hz=[bp_low, bp_high],
                sample_rate=sample_rate,
                btype="band",
            )
            signal = sp_signal.lfilter(b_bp, a_bp, signal)

        # --- Analytic signal via Hilbert transform ---
        analytic = hilbert(signal)

        # --- Instantaneous phase via arctan2 ---
        inst_phase = np.unwrap(np.angle(analytic))

        # --- Instantaneous frequency = d(phase)/dt ---
        inst_freq = np.diff(inst_phase, append=inst_phase[-1]) * sample_rate / (2.0 * np.pi)

        # --- Normalize to [0, 1] using known VHS FM spec bounds ---
        # Sync tip = 3.4 MHz → black (0.0)
        # White peak = 4.4 MHz → white (1.0)
        freq_min = VHS_Y_FM_SYNC_HZ_SP   # 3.4 MHz
        freq_max = VHS_Y_FM_WHITE_HZ_SP  # 4.4 MHz
        freq_range = freq_max - freq_min
        demod_output = (inst_freq - freq_min) / freq_range

        return np.clip(demod_output, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Stage 3b: De-emphasis
    # ------------------------------------------------------------------

    def _de_emphasize(
        self, signal: NDArray[np.float64], sample_rate: float
    ) -> NDArray[np.float64]:
        """Apply playback de-emphasis (inverse of recording pre-emphasis).

        The recording chain applies a lag-lead network:

        .. math::

            H_{\\text{pre}}(s) = \\frac{1 + s \\cdot \\tau_2}{1 + s \\cdot \\tau_1}

        with :math:`f_1 = \\tau_1^{-1} = 1.3\\text{ MHz}` and
        :math:`f_2 = \\tau_2^{-1} = 4.5\\text{ MHz}`.

        Playback applies the inverse:

        .. math::

            H_{\\text{de}}(s) = \\frac{1 + s \\cdot \\tau_1}{1 + s \\cdot \\tau_2}

        This restores flat frequency response while reducing high-frequency
        noise that was attenuated during recording.

        Parameters
        ----------
        signal : ndarray
            Demodulated luminance.
        sample_rate : float
            Sampling rate in Hz.

        Returns
        -------
        ndarray
            De-emphasized luminance.
        """
        f1 = 1.3e6
        f2 = 4.5e6
        tau1 = 1.0 / (2.0 * np.pi * f1)
        tau2 = 1.0 / (2.0 * np.pi * f2)

        # Analog: H(s) = (1 + s*tau1) / (1 + s*tau2)
        # This is the inverse of pre-emphasis: boosts low freq, attenuates high freq
        b_analog = [tau1, 1.0]   # numerator: tau1*s + 1
        a_analog = [tau2, 1.0]   # denominator: tau2*s + 1

        from scipy.signal import bilinear
        b_dig, a_dig = bilinear(b_analog, a_analog, fs=sample_rate)

        return sp_signal.lfilter(b_dig, a_dig, signal)

    # ------------------------------------------------------------------
    # Stage 3c: White Level Clamp
    # ------------------------------------------------------------------

    def _clamp_white_level(
        self, signal: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Establish reference black and white levels.

        Uses the horizontal blanking interval to clamp the black level.
        The sync tip provides the reference for the bottom of the signal.

        Parameters
        ----------
        signal : ndarray
            De-emphasized luminance.

        Returns
        -------
        ndarray
            Clamped luminance with reference levels established.
        """
        # Use the lower 5th percentile as black level
        black_level = np.percentile(signal, 5.0)
        signal = signal - black_level

        # Normalize to [0, 1] range
        white_level = np.percentile(signal, 95.0)
        if white_level > 1e-12:
            signal = signal / white_level

        signal = np.clip(signal, 0.0, 1.0)
        return signal

    # ------------------------------------------------------------------
    # Stage 4: Chrominance Recovery
    # ------------------------------------------------------------------

    def _extract_chroma(
        self, signal: NDArray[np.float64], sample_rate: float
    ) -> NDArray[np.float64]:
        """Bandpass filter to extract 629 kHz color-under signal.

        The color-under signal is recorded at a down-converted frequency
        of approximately 629 kHz (exactly fsc/4 for NTSC):

        .. math::

            f_{\\text{under}} = \\frac{f_{sc}}{4} = \\frac{3{,}579{,}545}{4}
            \\approx 629{,}375 \\text{ Hz}

        Parameters
        ----------
        signal : ndarray
            Equalized tape signal (before FM demod or combined).
        sample_rate : float
            Sampling rate in Hz.

        Returns
        -------
        ndarray
            Bandpass-filtered chrominance at ~629 kHz.
        """
        f_center = self.f_chroma_under  # 629375 Hz
        bw = 400e3  # bandwidth ~400 kHz (covers chroma sidebands)
        f_low = f_center - bw / 2.0
        f_high = f_center + bw / 2.0

        nyquist = sample_rate / 2.0
        if f_high >= nyquist:
            # Can't filter — sample rate too low for color-under
            return np.zeros_like(signal)

        b_bp, a_bp = _design_butterworth(
            order=6,
            cutoff_hz=[f_low, f_high],
            sample_rate=sample_rate,
            btype="band",
        )
        return sp_signal.lfilter(b_bp, a_bp, signal)

    def _upconvert_chroma(
        self,
        chroma_629k: NDArray[np.float64],
        sample_rate: float,
        burst_phase: float = 0.0,
    ) -> NDArray[np.float64]:
        """Upconvert color-under back to 3.58 MHz subcarrier.

        The chrominance is multiplied by a local oscillator at:

        .. math::

            f_{\\text{LO}} = f_{sc} - f_{\\text{under}}
            = 3{,}579{,}545 - 629{,}375 = 2{,}950{,}170 \\text{ Hz}

        This recovers the original color subcarrier:

        .. math::

            f_{\\text{LO}} + f_{\\text{under}} = f_{sc} = 3{,}579{,}545 \\text{ Hz}

        After mixing, a bandpass centered at fsc removes the image product.

        Parameters
        ----------
        chroma_629k : ndarray
            Color-under signal at ~629 kHz.
        sample_rate : float
            Sampling rate in Hz.
        burst_phase : float
            Phase reference from color burst gating (radians).

        Returns
        -------
        ndarray
            Chrominance at the native subcarrier frequency.
        """
        n = len(chroma_629k)
        t = np.arange(n, dtype=np.float64) / sample_rate

        # Local oscillator frequency
        f_lo = self.f_subcarrier - self.f_chroma_under

        # Mix: multiply chroma by LO to upconvert
        lo_signal = 2.0 * np.cos(2.0 * np.pi * f_lo * t + burst_phase)
        mixed = chroma_629k * lo_signal

        # Bandpass filter around subcarrier to remove image
        f_sub = self.f_subcarrier
        bw = 1.0e6  # chroma bandwidth ~1 MHz
        nyquist = sample_rate / 2.0
        if (f_sub + bw / 2.0) < nyquist:
            b_bp, a_bp = _design_butterworth(
                order=4,
                cutoff_hz=[f_sub - bw / 2.0, f_sub + bw / 2.0],
                sample_rate=sample_rate,
                btype="band",
            )
            mixed = sp_signal.lfilter(b_bp, a_bp, mixed)

        # ACC (Automatic Color Control) — DISABLED
        # The burst-based ACC measured burst RMS from the upconverted signal,
        # but the short 2.5 μs burst gets attenuated by filter ringing far
        # more than the continuous active chrominance bars.  This caused the
        # ACC to over-amplify chrominance.  Normalization is now handled
        # entirely by yc_to_rgb_raster which measures burst I/Q amplitude
        # after quadrature demodulation (much more robust).

        return mixed

    # ------------------------------------------------------------------
    # Stage 5: Y/C Recombination
    # ------------------------------------------------------------------

    def _recombine_yc(
        self,
        luma: NDArray[np.float64],
        chroma: NDArray[np.float64],
        sample_rate: float,
    ) -> NDArray[np.float64]:
        """Add decoded luminance and chrominance back into composite video.

        The luminance signal occupies DC–4.2 MHz and the chrominance is
        centered at the subcarrier.  Recombination is a simple addition:

        .. math::

            Y_{\\text{composite}}(t) = Y(t) + C(t)

        Blanking intervals and sync pulses are inserted to form a valid
        composite video waveform.

        Parameters
        ----------
        luma : ndarray
            Decoded luminance [0, 1].
        chroma : ndarray
            Decoded chrominance at subcarrier frequency.
        sample_rate : float
            Sampling rate in Hz.

        Returns
        -------
        ndarray
            Composite video signal.
        """
        n = len(luma)

        # Ensure same length
        if len(chroma) != n:
            chroma = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(chroma)),
                chroma,
            )

        # Scale chroma to reasonable amplitude relative to luma
        # ACC already normalizes the chrominance peak, so use unity gain.
        chroma_scale = 1.0
        composite = luma + chroma * chroma_scale

        # Insert blanking/sync
        composite = self._insert_sync(composite, sample_rate)

        return composite

    def _insert_sync(
        self, signal: NDArray[np.float64], sample_rate: float
    ) -> NDArray[np.float64]:
        """Insert horizontal and vertical sync into composite video.

        Parameters
        ----------
        signal : ndarray
            Y+C composite without sync.
        sample_rate : float
            Sampling rate in Hz.

        Returns
        -------
        ndarray
            Composite video with sync and blanking intervals.
        """
        n = len(signal)
        samples_per_line = int(sample_rate / self.line_rate)
        total_lines = int(self.lines_per_field)

        # Sync tip level (below black)
        sync_tip = -0.3
        blanking_level = 0.0

        output = signal.copy()

        for line_idx in range(total_lines):
            line_start = line_idx * samples_per_line
            line_end = min(line_start + samples_per_line, n)
            if line_start >= n:
                break

            # Horizontal sync pulse (first ~4.7 μs of each line)
            sync_samples = int(4.7e-6 * sample_rate)
            sync_end = min(line_start + sync_samples, line_end)
            output[line_start:sync_end] = sync_tip

            # Back porch (blanking)
            porch_start = sync_end
            porch_end = min(
                line_start + int(10.9e-6 * sample_rate), line_end
            )
            output[porch_start:porch_end] = blanking_level

        return output

    # ------------------------------------------------------------------
    # Stage 6: Audio Playback
    # ------------------------------------------------------------------

    def _demodulate_audio(
        self, signal: NDArray[np.float64], sample_rate: float
    ) -> NDArray[np.float64]:
        """Demodulate linear or Hi-Fi audio from tape signal.

        **Linear audio** is recorded on a stationary track along the tape
        edge.  Playback is straightforward amplitude demodulation with
        de-emphasis:

        .. math::

            \\tau = 50\\,\\mu\\text{s} \\quad\\Rightarrow\\quad
            f_{-3\\text{dB}} = \\frac{1}{2\\pi\\tau} \\approx 3.18 \\text{ kHz}

        **Hi-Fi audio** uses separate FM carriers:

        .. math::

            f_L = 1.3 \\text{ MHz},\\quad f_R = 1.7 \\text{ MHz}
        .. math::

            \\Delta f = \\pm 150 \\text{ kHz}

        Hi-Fi achieves ~80 dB SNR vs ~43 dB for linear audio.

        Parameters
        ----------
        signal : ndarray
            Equalized tape signal.
        sample_rate : float
            Sampling rate in Hz.

        Returns
        -------
        ndarray
            Demodulated audio waveform.
        """
        nyquist = sample_rate / 2.0

        if self.hifi_audio and nyquist > 2.0e6:
            return self._demodulate_hifi_audio(signal, sample_rate)
        else:
            return self._demodulate_linear_audio(signal, sample_rate)

    def _demodulate_linear_audio(
        self, signal: NDArray[np.float64], sample_rate: float
    ) -> NDArray[np.float64]:
        """Demodulate linear (stationary-head) audio track.

        Linear audio is a simple amplitude-modulated signal on a separate
        track.  We extract it with a lowpass filter and apply de-emphasis.

        Parameters
        ----------
        signal : ndarray
            Equalized tape signal.
        sample_rate : float
            Sampling rate in Hz.

        Returns
        -------
        ndarray
            De-emphasized linear audio.
        """
        nyquist = sample_rate / 2.0
        audio_bw = 15e3  # linear audio bandwidth ~15 kHz

        if audio_bw < nyquist:
            b_lp, a_lp = _design_butterworth(
                order=4,
                cutoff_hz=audio_bw,
                sample_rate=sample_rate,
                btype="low",
            )
            audio = sp_signal.lfilter(b_lp, a_lp, signal)
        else:
            audio = signal.copy()

        # De-emphasis: 50 μs time constant
        # Analog: H(s) = 1 / (1 + s*tau)
        tau = 50.0e-6  # 50 μs
        f_c = 1.0 / (2.0 * np.pi * tau)

        if f_c < nyquist:
            b_de, a_de = _design_butterworth(
                order=1,
                cutoff_hz=f_c,
                sample_rate=sample_rate,
                btype="low",
            )
            audio = sp_signal.lfilter(b_de, a_de, audio)

        return audio

    def _demodulate_hifi_audio(
        self, signal: NDArray[np.float64], sample_rate: float
    ) -> NDArray[np.float64]:
        """Demodulate Hi-Fi stereo FM audio.

        Hi-Fi audio is FM-modulated on two carriers:

        - **L channel**: carrier at 1.3 MHz, deviation ±150 kHz
        - **R channel**: carrier at 1.7 MHz, deviation ±150 kHz

        Each is demodulated independently, then de-emphasized with a
        50 μs time constant.

        A known limitation: **crosstalk from adjacent tracks** degrades
        SNR by ~3 dB when playing back in EP mode (narrower tracks).

        Parameters
        ----------
        signal : ndarray
            Equalized tape signal.
        sample_rate : float
            Sampling rate in Hz.

        Returns
        -------
        ndarray
            Stereo audio (interleaved L, R).
        """
        nyquist = sample_rate / 2.0
        f_l = 1.3e6
        f_r = 1.7e6
        fm_dev = 150e3

        def _fm_demod_carrier(
            sig: NDArray[np.float64],
            f_carrier: float,
        ) -> NDArray[np.float64]:
            """FM demodulate a specific carrier using PLL."""
            n = len(sig)
            loop_bw = 50e3
            dt = 1.0 / sample_rate
            alpha = 2.0 * loop_bw * dt
            vco_phase = 0.0
            lp_int = 0.0
            out = np.zeros(n, dtype=np.float64)

            for i in range(n):
                vco = np.cos(vco_phase)
                err = sig[i] * vco
                lp_int += alpha * err
                out[i] = lp_int
                vco_phase += 2.0 * np.pi * f_carrier * dt + lp_int * dt
                vco_phase = vco_phase % (2.0 * np.pi)

            # Normalize deviation
            out -= np.mean(out)
            peak = np.max(np.abs(out))
            if peak > 1e-12:
                out = out / peak * (fm_dev / sample_rate)

            return out

        # Extract L and R with bandpass filters
        bw = 400e3

        audio_l = np.zeros_like(signal)
        audio_r = np.zeros_like(signal)

        if (f_l + bw / 2.0) < nyquist:
            b_bp, a_bp = _design_butterworth(
                order=6,
                cutoff_hz=[f_l - bw / 2.0, f_l + bw / 2.0],
                sample_rate=sample_rate,
                btype="band",
            )
            l_band = sp_signal.lfilter(b_bp, a_bp, signal)
            audio_l = _fm_demod_carrier(l_band, f_l)

        if (f_r + bw / 2.0) < nyquist:
            b_bp, a_bp = _design_butterworth(
                order=6,
                cutoff_hz=[f_r - bw / 2.0, f_r + bw / 2.0],
                sample_rate=sample_rate,
                btype="band",
            )
            r_band = sp_signal.lfilter(b_bp, a_bp, signal)
            audio_r = _fm_demod_carrier(r_band, f_r)

        # De-emphasis (50 μs)
        tau = 50.0e-6
        f_c = 1.0 / (2.0 * np.pi * tau)
        if f_c < nyquist:
            b_de, a_de = _design_butterworth(
                order=1,
                cutoff_hz=f_c,
                sample_rate=sample_rate,
                btype="low",
            )
            audio_l = sp_signal.lfilter(b_de, a_de, audio_l)
            audio_r = sp_signal.lfilter(b_de, a_de, audio_r)

        # Add crosstalk from adjacent track (~3 dB in EP mode)
        if self.speed_mode == SpeedMode.EP:
            crosstalk_linear = 10.0 ** (-3.0 / 20.0)
            audio_l = audio_l + crosstalk_linear * audio_r
            audio_r = audio_r + crosstalk_linear * audio_l

        # Interleave L, R
        n = len(audio_l)
        stereo = np.zeros(2 * n, dtype=np.float64)
        stereo[0::2] = audio_l
        stereo[1::2] = audio_r

        return stereo

    # ------------------------------------------------------------------
    # Stage 7: Head Switching
    # ------------------------------------------------------------------

    def _compute_head_switch_point(
        self, n_samples: int, sample_rate: float, head_index: int
    ) -> int:
        """Compute sample index where head switching occurs.

        Head switching occurs ~6 lines before vertical sync (by design,
        hidden in the blanking interval).  For a standard VHS head drum
        with 2 heads at 180° separation, each head records one field.

        Parameters
        ----------
        n_samples : int
            Total samples in the field.
        sample_rate : float
            Sampling rate in Hz.
        head_index : int
            Current head (0 or 1).

        Returns
        -------
        int
            Sample index of the head switching point.
        """
        lines_before_vsync = self.head_switch_offset_lines
        samples_per_line = int(sample_rate / self.line_rate)
        # Switch happens this many samples from the start of the field
        switch_point = lines_before_vsync * samples_per_line
        return min(switch_point, n_samples - 1)

    def _head_switch_noise(
        self,
        signal: NDArray[np.float64],
        switch_point: int,
        sample_rate: float,
    ) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Model head switching glitch at field boundary.

        At the boundary where one head leaves the tape and the other
        enters, there is a brief burst of noise.  The glitch occurs
        ~6 lines before vertical sync and lasts ~2–4 lines (~0.1 ms).

        The glitch is modelled as:

        1. **Signal dropout** — reduced amplitude (30–50% of normal)
        2. **Noise burst** — additive random noise (20–40 dB above normal)
        3. **Duration** — approximately 3 lines worth of samples

        Parameters
        ----------
        signal : ndarray
            Equalized tape signal.
        switch_point : int
            Sample index of the switch event.
        sample_rate : float
            Sampling rate in Hz.

        Returns
        -------
        signal_out : ndarray
            Signal with head switching artifact applied.
        dropout_mask : ndarray of bool
            ``True`` where dropout/switching artifact is active.
        """
        n = len(signal)
        signal_out = signal.copy()

        samples_per_line = int(sample_rate / self.line_rate)
        glitch_lines = 3  # ~3 lines of degraded signal
        glitch_samples = glitch_lines * samples_per_line

        # Glitch window: centered on switch_point
        glitch_start = max(0, switch_point - glitch_samples // 2)
        glitch_end = min(n, switch_point + glitch_samples // 2)

        dropout_mask = np.zeros(n, dtype=np.bool_)

        if glitch_end > glitch_start:
            dropout_mask[glitch_start:glitch_end] = True

            # 1. Signal attenuation (dropout-like)
            attenuation = np.random.uniform(0.3, 0.5)
            signal_out[glitch_start:glitch_end] *= attenuation

            # 2. Additive noise burst
            noise_level = 0.3  # ~–10 dB relative to signal
            noise = np.random.randn(glitch_end - glitch_start) * noise_level
            signal_out[glitch_start:glitch_end] += noise

        return signal_out, dropout_mask

    # ------------------------------------------------------------------
    # Stage 8: Dropout Model
    # ------------------------------------------------------------------

    def _apply_dropout(
        self,
        signal: NDArray[np.float64],
        dropout_map: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Apply dropout effects (random signal loss patches).

        Tape dropouts occur when oxide particles detach from the base
        film, causing momentary loss of the magnetic signal.

        Characteristics:

        - **Duration**: 5–50 μs typically (up to 100 μs for severe dropouts)
        - **Appearance**: Horizontal white or black streaks in the image
        - **Frequency**: depends on tape age and quality; new tape may have
          <1 dropout/minute, old tape may have >100/minute

        The dropout is modelled as:

        .. math::

            x_{\\text{out}}(t) = \\begin{cases}
                \\alpha \\cdot x(t) + \\beta \\cdot n(t), & \\text{if dropout} \\\\
                x(t), & \\text{otherwise}
            \\end{cases}

        where :math:`\\alpha \\approx 0.1`–0.3 (signal attenuation) and
        :math:`\\beta` provides noise fill-in.

        Parameters
        ----------
        signal : ndarray
            Video signal.
        dropout_map : ndarray of bool
            ``True`` where dropout is active.

        Returns
        -------
        ndarray
            Signal with dropouts applied.
        """
        output = signal.copy()
        n_dropouts = np.sum(dropout_map)
        if n_dropouts == 0:
            return output

        # Attenuate signal at dropout locations
        attenuation = np.random.uniform(0.05, 0.3)
        noise_fill = np.random.randn(int(n_dropouts)) * 0.2

        output[dropout_map] = (
            output[dropout_map] * attenuation + noise_fill
        )

        return output

    def _count_dropouts(
        self,
        dropout_mask: NDArray[np.bool_],
        sample_rate: float,
    ) -> int:
        """Count distinct dropout events from a boolean mask.

        Parameters
        ----------
        dropout_mask : ndarray of bool
            ``True`` at sample positions where dropout is active.
        sample_rate : float
            Sampling rate in Hz.

        Returns
        -------
        int
            Number of distinct dropout events (contiguous True regions).
        """
        if not np.any(dropout_mask):
            return 0
        # Detect transitions: diff of bool gives +1 at start, -1 at end
        diff = np.diff(dropout_mask.astype(np.int8))
        starts = np.sum(diff == 1)
        return int(starts)

    # ------------------------------------------------------------------
    # Playback Artifact Generators
    # ------------------------------------------------------------------

    def generate_dropout_mask(
        self,
        n_samples: int,
        sample_rate: float,
        density_per_minute: float = 10.0,
        mean_duration_us: float = 15.0,
        seed: int | None = None,
    ) -> NDArray[np.bool_]:
        """Generate a random dropout map for a tape segment.

        Parameters
        ----------
        n_samples : int
            Number of samples in the segment.
        sample_rate : float
            Sampling rate in Hz.
        density_per_minute : float
            Average number of dropout events per minute of tape.
        mean_duration_us : float
            Mean dropout duration in microseconds.
        seed : int or None
            Optional RNG seed for reproducibility.

        Returns
        -------
        ndarray of bool
            ``True`` at positions where dropout is active.
        """
        rng = np.random.default_rng(seed)
        duration_s = n_samples / sample_rate
        duration_min = duration_s / 60.0

        # Expected number of dropouts
        expected_count = density_per_minute * duration_min
        n_dropouts = rng.poisson(expected_count)

        mask = np.zeros(n_samples, dtype=np.bool_)

        for _ in range(n_dropouts):
            # Random position
            pos = rng.integers(0, n_samples)
            # Duration (exponentially distributed)
            dur_samples = int(
                rng.exponential(mean_duration_us * 1e-6 * sample_rate)
            )
            dur_samples = max(dur_samples, 1)
            dur_samples = min(dur_samples, n_samples - pos)
            mask[pos:pos + dur_samples] = True

        return mask

    def generate_tape_stretch_error(
        self,
        n_lines: int,
        max_error_ns: float = 50.0,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        """Generate tape stretch timing errors between tracks.

        Tape stretch causes slight timing misalignment between the
        recording and playback paths.  The error is typically
        quasi-periodic with the cassette hub rotation (~2–5 Hz).

        Parameters
        ----------
        n_lines : int
            Number of horizontal lines.
        max_error_ns : float
            Maximum timing error in nanoseconds.
        seed : int or None
            Optional RNG seed.

        Returns
        -------
        ndarray
            Per-line timing error in nanoseconds.
        """
        rng = np.random.default_rng(seed)

        # Low-frequency wander (tape transport irregularity)
        # Frequency: ~3 Hz (hub rotation)
        lines_per_field = int(self.lines_per_field)
        total_fields = max(1, n_lines // lines_per_field)

        # Create a slowly varying error signal
        t = np.arange(n_lines, dtype=np.float64) / self.line_rate

        # Base wander: sinusoidal at hub rotation frequency
        hub_freq = 3.0  # Hz
        wander = max_error_ns * 0.5 * np.sin(
            2.0 * np.pi * hub_freq * t + rng.uniform(0, 2 * np.pi)
        )

        # Random walk component (short-term jitter)
        jitter = rng.standard_normal(n_lines) * max_error_ns * 0.1
        jitter = np.cumsum(jitter) * 0.01  # slow accumulation
        jitter = np.clip(jitter, -max_error_ns * 0.3, max_error_ns * 0.3)

        return wander + jitter

    def generate_color_phase_error(
        self,
        n_lines: int,
        max_error_deg: float = 5.0,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        """Generate per-line color phase tracking error.

        Adjacent lines may have slightly different color phase due to
        imperfect chroma tracking between record and playback heads.

        Parameters
        ----------
        n_lines : int
            Number of horizontal lines.
        max_error_deg : float
            Maximum phase error in degrees.
        seed : int or None
            Optional RNG seed.

        Returns
        -------
        ndarray
            Per-line phase error in radians.
        """
        rng = np.random.default_rng(seed)

        # Phase error is correlated between adjacent lines
        raw = rng.standard_normal(n_lines) * max_error_deg
        # Low-pass filter for correlation (3-line kernel)
        kernel = np.array([0.25, 0.5, 0.25])
        filtered = np.convolve(raw, kernel, mode="same")

        return np.deg2rad(filtered)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"VHSDecoder(standard={self.standard.value}, "
            f"speed={self.speed_mode.value}, "
            f"hifi={self.hifi_audio})"
        )


# ---------------------------------------------------------------------------
# PlaybackSimulator
# ---------------------------------------------------------------------------

class PlaybackSimulator:
    """Full VHS playback simulation — chains the complete decode pipeline.

    Takes a continuous stream of tape flux data (as would be read from
    a .vhs file) and produces continuous video/audio output, including
    realistic artifacts.

    Parameters
    ----------
    decoder : VHSDecoder
        Configured decoder instance.

    Examples
    --------
    >>> decoder = VHSDecoder(standard=VideoStandard.NTSC, speed_mode=SpeedMode.SP)
    >>> sim = PlaybackSimulator(decoder)
    >>> result = sim.play(tape_flux, sample_rate=48_000_000)
    >>> result["video_frames"][0].shape  # one field of video
    """

    def __init__(self, decoder: VHSDecoder) -> None:
        self.decoder = decoder

    def play(
        self,
        tape_flux_stream: NDArray[np.float64],
        sample_rate: float,
    ) -> dict:
        """Full playback simulation.

        Segments the raw tape flux into per-field chunks and decodes each
        through the full signal chain.

        Parameters
        ----------
        tape_flux_stream : ndarray
            1-D array of voltage samples as read from the tape.
        sample_rate : float
            Sampling rate in Hz.

        Returns
        -------
        dict
            ``'video_frames'`` : list of ndarray — composite video per field
            ``'audio_samples'`` : ndarray — continuous audio (concatenated)
            ``'diagnostics'``   : list of dict — per-field quality metrics
        """
        n_total = len(tape_flux_stream)
        samples_per_field = int(
            sample_rate / self.decoder.field_rate
        )

        video_frames: List[NDArray[np.float64]] = []
        audio_chunks: List[NDArray[np.float64]] = []
        diagnostics: List[Dict[str, float]] = []

        field_idx = 0
        offset = 0

        while offset + samples_per_field <= n_total:
            field_data = tape_flux_stream[offset:offset + samples_per_field]
            head_index = field_idx % 2

            result = self.decoder.decode_field(
                raw_tape_signal=field_data,
                sample_rate=sample_rate,
                head_index=head_index,
                field_number=field_idx,
            )

            video_frames.append(result["composite"])
            audio_chunks.append(result["audio"])
            diagnostics.append(result["quality"])

            offset += samples_per_field
            field_idx += 1

        # Concatenate audio
        if audio_chunks:
            audio_samples = np.concatenate(audio_chunks)
        else:
            audio_samples = np.array([], dtype=np.float64)

        return {
            "video_frames": video_frames,
            "audio_samples": audio_samples,
            "diagnostics": diagnostics,
        }

    def play_with_artifacts(
        self,
        tape_flux_stream: NDArray[np.float64],
        sample_rate: float,
        dropout_density: float = 10.0,
        stretch_error_ns: float = 50.0,
        color_phase_error_deg: float = 5.0,
        seed: int | None = None,
    ) -> dict:
        """Playback with realistic tape artifacts applied.

        Extends ``play()`` by generating and applying:

        - **Dropouts**: random signal loss patches
        - **Tape stretch**: inter-track timing errors
        - **Color phase error**: per-line hue drift

        Parameters
        ----------
        tape_flux_stream : ndarray
            1-D array of voltage samples as read from the tape.
        sample_rate : float
            Sampling rate in Hz.
        dropout_density : float
            Average dropout events per minute.
        stretch_error_ns : float
            Max tape stretch timing error (nanoseconds).
        color_phase_error_deg : float
            Max per-line color phase error (degrees).
        seed : int or None
            RNG seed for reproducibility.

        Returns
        -------
        dict
            Same as ``play()`` plus ``'artifacts'`` key containing:

            ``'dropout_masks'`` : list of ndarray — per-field dropout maps
            ``'stretch_errors'`` : ndarray — per-line timing errors
            ``'phase_errors'``   : ndarray — per-line chroma phase errors
        """
        rng = np.random.default_rng(seed)

        n_total = len(tape_flux_stream)
        samples_per_field = int(
            sample_rate / self.decoder.field_rate
        )

        video_frames: List[NDArray[np.float64]] = []
        audio_chunks: List[NDArray[np.float64]] = []
        diagnostics: List[Dict[str, float]] = []
        dropout_masks: List[NDArray[np.bool_]] = []

        # Pre-generate global artifact maps
        total_lines = int(
            (n_total / sample_rate) * self.decoder.line_rate
        )
        stretch_errors = self.decoder.generate_tape_stretch_error(
            total_lines, max_error_ns=stretch_error_ns, seed=seed
        )
        phase_errors = self.decoder.generate_color_phase_error(
            total_lines, max_error_deg=color_phase_error_deg, seed=seed
        )

        field_idx = 0
        offset = 0

        while offset + samples_per_field <= n_total:
            field_data = tape_flux_stream[offset:offset + samples_per_field]
            head_index = field_idx % 2

            # Apply field-specific dropout mask
            dropout_mask = self.decoder.generate_dropout_mask(
                samples_per_field,
                sample_rate,
                density_per_minute=dropout_density,
                seed=int(rng.integers(0, 2**31)),
            )
            field_data = self.decoder._apply_dropout(field_data, dropout_mask)

            result = self.decoder.decode_field(
                raw_tape_signal=field_data,
                sample_rate=sample_rate,
                head_index=head_index,
                field_number=field_idx,
            )

            video_frames.append(result["composite"])
            audio_chunks.append(result["audio"])
            diagnostics.append(result["quality"])
            dropout_masks.append(dropout_mask)

            offset += samples_per_field
            field_idx += 1

        if audio_chunks:
            audio_samples = np.concatenate(audio_chunks)
        else:
            audio_samples = np.array([], dtype=np.float64)

        return {
            "video_frames": video_frames,
            "audio_samples": audio_samples,
            "diagnostics": diagnostics,
            "artifacts": {
                "dropout_masks": dropout_masks,
                "stretch_errors": stretch_errors,
                "phase_errors": phase_errors,
            },
        }

    def __repr__(self) -> str:
        return f"PlaybackSimulator(decoder={self.decoder!r})"
