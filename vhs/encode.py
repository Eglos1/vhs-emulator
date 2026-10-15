"""
VHS Recording Signal Encoding Chain
====================================

Models the analog signal processing inside a VHS recorder, from composite
video input to the signals actually written onto magnetic tape.

Signal chain:
    Composite In ─► Sync Separator ─► Y/C Separation ─┬─► Y FM Chain ─► Write Head
                                                       ├─► Chroma Under ─► Write Head
                                                       └─► Audio ─► Write Head
                                                         Control Track ─► Stationary Head

References:
    - VHS Technical Manual (Matsushita, 1984)
    - IEC 60774-1: VHS-type magnetic tape cassette system
    - SMPTE 170M: Composite analog video signal (NTSC)
    - ITU-R BT.470-6: Conventional television systems (PAL)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import scipy.fft


# ---------------------------------------------------------------------------
# Power-of-2 padded FFT helpers
# ---------------------------------------------------------------------------
# 238636 (NTSC field at 4×subcarrier) has no small prime factors, making
# scipy.fft use a slow general algorithm (~40ms). Padding to next power of 2
# (262144 = 2^18) gives 6-7x speedup via radix-2 FFT.

def _next_pow2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    return 1 << (n - 1).bit_length()


def rfft_padded(signal: np.ndarray, workers: int = -1):
    """
    Compute rfft with zero-padding to next power of 2.

    Returns (spectrum, n_padded) — pass n_padded to freqs, and original_length to irfft_padded.
    """
    n = len(signal)
    n_padded = _next_pow2(n)
    if n_padded == n:
        return scipy.fft.rfft(signal, workers=workers), n, n
    padded = np.zeros(n_padded, dtype=signal.dtype)
    padded[:n] = signal
    return scipy.fft.rfft(padded, workers=workers), n_padded, n


def irfft_padded(spectrum: np.ndarray, original_length: int, workers: int = -1) -> np.ndarray:
    """
    Compute irfft with power-of-2 size, then truncate to original_length.
    """
    n_padded = _next_pow2(original_length)
    result = scipy.fft.irfft(spectrum, n=n_padded, workers=workers)
    return result[:original_length]


# ---------------------------------------------------------------------------
# Physical constants and VHS specification parameters
# ---------------------------------------------------------------------------

# NTSC subcarrier: 3579545 Hz exactly (f_sc = f_h × 455 / 2, f_h ≈ 15734.264 Hz)
# PAL subcarrier: 4433618.75 Hz (f_h × 284 + 1/4, f_h ≈ 15625 Hz)
NTSC_SUBCARRIER_HZ: float = 3_579_545.0
PAL_SUBCARRIER_HZ: float = 4_433_618.75

# Horizontal line frequencies
NTSC_H_FREQ: float = 15_734.264  # Hz (approx)
PAL_H_FREQ: float = 15_625.0

# Field rates
NTSC_FIELD_RATE: float = 59.94   # Hz (2 × 29.97 fps)
PAL_FIELD_RATE: float = 50.0     # Hz (2 × 25 fps)

# Lines per field (total lines / 2)
NTSC_LINES_PER_FIELD: float = 262.5  # 525/2
PAL_LINES_PER_FIELD: float = 312.5   # 625/2

# Active video time per line (microseconds)
NTSC_ACTIVE_LINE_US: float = 52.6    # ≈ 52.6μs (63.5μs total - 10.9μs blanking)
PAL_ACTIVE_LINE_US: float = 52.0     # ≈ 52.0μs (64.0μs total - 12.0μs blanking)

# Sync widths (microseconds)
NTSC_HSYNC_WIDTH_US: float = 4.7
NTSC_VSYNC_TOTAL_US: float = 570.0   # broad pulses during vertical interval
PAL_HSYNC_WIDTH_US: float = 4.7
PAL_VSYNC_TOTAL_US: float = 640.0    # PAL has wider vertical interval

# Blanking intervals (microseconds per line)
NTSC_HBLANK_US: float = 11.7
PAL_HBLANK_US: float = 12.0

# Vertical blanking (lines)
NTSC_VBLANK_LINES: float = 20.0     # lines of vertical blanking
PAL_VBLANK_LINES: float = 25.0

# ---- VHS Recording Parameters (SP mode) ----

# Luminance FM parameters (IEC 60774)
# SP: sync tip = 3.4 MHz, white peak = 4.4 MHz, deviation = 1.0 MHz
VHS_Y_FM_SYNC_HZ_SP: float = 3.4e6
VHS_Y_FM_WHITE_HZ_SP: float = 4.4e6
VHS_Y_FM_DEVIATION_SP: float = 1.0e6

# LP mode: same carriers but narrower tracks on tape
VHS_Y_FM_SYNC_HZ_LP: float = 3.4e6
VHS_Y_FM_WHITE_HZ_LP: float = 4.4e6

# EP/SLP mode: same carriers
VHS_Y_FM_SYNC_HZ_EP: float = 3.4e6
VHS_Y_FM_WHITE_HZ_EP: float = 4.4e6

# Color-under frequency (NTSC)
# Down-converted chroma carrier: f_sc - f_under = 2.95 MHz → ~629 kHz on tape
# NTSC: 3.579545 MHz → 629.0 kHz difference
VHS_CHROMA_UNDER_NTSC_HZ: float = 629_000.0
VHS_CHROMA_OSC_NTSC_HZ: float = NTSC_SUBCARRIER_HZ - VHS_CHROMA_UNDER_NTSC_HZ  # ~2.95 MHz

# PAL: 4.433619 MHz → 627.0 kHz difference (slightly different for PAL VHS)
VHS_CHROMA_UNDER_PAL_HZ: float = 627_000.0
VHS_CHROMA_OSC_PAL_HZ: float = PAL_SUBCARRIER_HZ - VHS_CHROMA_UNDER_PAL_HZ  # ~3.807 MHz

# Chroma bandwidth before downconversion
VHS_CHROMA_BW_HZ: float = 500_000.0  # 0.5 MHz low-pass

# Chroma-to-luminance current ratio: chroma recorded at ~35-40% of Y current
VHS_CHROMA_CURRENT_RATIO: float = 0.375  # midpoint of 35-40%

# Pre-emphasis curve: lead-lag network
# f₁ = 1.3 MHz (pole — high-frequency boost onset)
# f₂ = 4.5 MHz (zero — rolloff flattens)
VHS_PREEMPH_F1_HZ: float = 1.3e6
VHS_PREEMPH_F2_HZ: float = 4.5e6

# Audio parameters
VHS_AUDIO_TRACK_WIDTH_MM: float = 0.7  # SP mode
VHS_AUDIO_PREEMPH_TC_US: float = 50.0  # 50μs time constant → +10 dB at 10 kHz
VHS_AUDIO_SNR_DB: float = 43.0         # signal-to-noise ratio (SP)
VHS_AUDIO_LOW_HZ: float = 100.0
VHS_AUDIO_HIGH_HZ: float = 10_000.0

# Hi-Fi FM audio carriers (recorded by drum heads, embedded in video signal)
VHS_HIFI_LEFT_CARRIER_HZ: float = 1.3e6
VHS_HIFI_RIGHT_CARRIER_HZ: float = 1.7e6
VHS_HIFI_DEVIATION_HZ: float = 150_000.0  # ±150 kHz deviation

# Common sample rates used internally
SAMPLE_RATE_NTSC: float = 14_318_180.0  # 4× NTSC subcarrier ≈ 14.318 MHz
SAMPLE_RATE_PAL: float = 17_734_475.0   # 4× PAL subcarrier ≈ 17.734 MHz


class SpeedMode(Enum):
    SP = "SP"
    LP = "LP"
    EP = "EP"


class Standard(Enum):
    NTSC = "NTSC"
    PAL = "PAL"


@dataclass
class EncodingResult:
    """Output container for one field of encoded VHS signal."""
    y_fm: np.ndarray          # FM-modulated luminance (instantaneous frequency samples)
    c_under: np.ndarray       # Downconverted chrominance at ~629 kHz
    audio: np.ndarray         # Linear audio track signal
    control: np.ndarray       # Control track pulse train (one pulse per field)
    hifi_left: Optional[np.ndarray] = None   # Hi-Fi FM left channel (if enabled)
    hifi_right: Optional[np.ndarray] = None  # Hi-Fi FM right channel (if enabled)
    metadata: Optional[dict] = None


class VHSEncoder:
    """
    VHS Recording Signal Encoder

    Models the complete signal chain inside a VHS recorder, from composite
    video + audio input to the magnetic write signals.

    Physical model:
        The VHS format records on 1/2-inch (12.65 mm) ferric oxide tape.
        In SP mode, the tape moves at 23.39 mm/s. The recording head drum
        rotates at 1800 rpm (NTSC) or 1500 rpm (PAL), with two heads
        mounted 180° apart. Each head records one field (one half-rotation)
        on a diagonal track across the tape. Track pitch is 49 μm (SP),
        29 μm (LP), or 19.4 μm (EP).

        The luminance signal is frequency-modulated onto a carrier between
        3.4 MHz (sync tip) and 4.4 MHz (white peak). The chrominance is
        downconverted to ~629 kHz ("color-under") and recorded at a lower
        current so it penetrates deeper into the tape coating (longer
        wavelength → deeper penetration, reducing croswipe between adjacent
        tracks). Audio is recorded linearly by a stationary head along the
        bottom tape edge, or optionally as FM by the drum heads (Hi-Fi mode).

    Parameters
    ----------
    standard : str
        'NTSC' or 'PAL'. Determines subcarrier frequency, line count, and
        field rate.
    speed_mode : str
        'SP', 'LP', or 'EP' (also called SLP). Affects tape speed and track
        width, but the FM deviation and color-under frequencies remain the same.
    hifi_audio : bool
        If True, enables Hi-Fi FM audio recording via the drum heads.
    """

    def __init__(
        self,
        standard: str = "NTSC",
        speed_mode: str = "SP",
        hifi_audio: bool = False,
    ):
        self.standard = Standard(standard.upper())
        self.speed_mode = SpeedMode(speed_mode.upper())
        self.hifi_audio = hifi_audio

        # Select parameters based on TV standard
        if self.standard == Standard.NTSC:
            self.subcarrier_freq = NTSC_SUBCARRIER_HZ
            self.h_freq = NTSC_H_FREQ
            self.field_rate = NTSC_FIELD_RATE
            self.lines_per_field = NTSC_LINES_PER_FIELD
            self.hsync_width_us = NTSC_HSYNC_WIDTH_US
            self.hblank_us = NTSC_HBLANK_US
            self.vblank_lines = NTSC_VBLANK_LINES
            self.active_line_us = NTSC_ACTIVE_LINE_US
            self.chroma_under_freq = VHS_CHROMA_UNDER_NTSC_HZ
            self.chroma_osc_freq = VHS_CHROMA_OSC_NTSC_HZ
            self.default_sample_rate = SAMPLE_RATE_NTSC
        else:  # PAL
            self.subcarrier_freq = PAL_SUBCARRIER_HZ
            self.h_freq = NTSC_H_FREQ  # PAL: 15625 Hz, but we keep the attribute name
            self.h_freq = PAL_H_FREQ
            self.field_rate = PAL_FIELD_RATE
            self.lines_per_field = PAL_LINES_PER_FIELD
            self.hsync_width_us = PAL_HSYNC_WIDTH_US
            self.hblank_us = PAL_HBLANK_US
            self.vblank_lines = PAL_VBLANK_LINES
            self.active_line_us = PAL_ACTIVE_LINE_US
            self.chroma_under_freq = VHS_CHROMA_UNDER_PAL_HZ
            self.chroma_osc_freq = VHS_CHROMA_OSC_PAL_HZ
            self.default_sample_rate = SAMPLE_RATE_PAL

        # FM parameters based on speed mode
        # Note: FM deviation is the same for all speed modes in VHS spec.
        # Track width and tape speed change, but carrier frequencies do not.
        self.fm_sync = VHS_Y_FM_SYNC_HZ_SP
        self.fm_white = VHS_Y_FM_WHITE_HZ_SP
        self.fm_deviation = VHS_Y_FM_DEVIATION_SP

        # Derived
        self.line_period_s = 1.0 / self.h_freq
        self.line_period_us = self.line_period_s * 1e6

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_video(
        self,
        composite_signal: np.ndarray,
        sample_rate: float,
        audio_signal: Optional[np.ndarray] = None,
    ) -> EncodingResult:
        """
        Full encoding chain for one field of video.

        Processes a composite video signal through the complete VHS recording
        chain and returns the signals ready for magnetic tape write heads.

        Parameters
        ----------
        composite_signal : np.ndarray
            Composite video signal samples (1D, float64). Normalized so that
            sync tip = 0.0, blanking/black = ~0.077 (7.5 IRE setup),
            white peak = 1.0 for NTSC. For PAL, black = 0.0 (no setup).
        sample_rate : float
            Sampling rate in Hz. Should be at least 2× the luminance bandwidth
            (≥ 6 MHz for NTSC). Ideally 4× subcarrier: 14.318 MHz (NTSC)
            or 17.734 MHz (PAL) for artifact-free processing.
        audio_signal : np.ndarray, optional
            Audio signal samples at sample_rate. If None, silent audio is generated.

        Returns
        -------
        EncodingResult
            Contains:
            - y_fm: FM luminance signal (instantaneous frequency samples)
            - c_under: Downconverted chrominance at chroma_under_freq (~629 kHz)
            - audio: Pre-emphasized linear audio
            - control: Control track pulse train (one pulse per field)
            - hifi_left, hifi_right: Hi-Fi FM audio if enabled
            - metadata: Timing and signal parameters

        Signal flow:
            1. Sync separation → timing reference
            2. Y/C separation via 1H-delay comb filter
            3. Y path: pre-emphasis → white clip → FM modulation
            4. C path: bandpass → downconvert to color-under
            5. Audio: pre-emphasis → (optional) Hi-Fi FM
            6. Control track: vertical sync pulse train
        """
        # Step 1: Sync separation — extract timing
        sync_info = self._sync_separate(composite_signal, sample_rate)

        # Step 2: Y/C separation
        luma, chroma = self._yc_separate(composite_signal, sample_rate, sync_info)

        # Step 3: Luminance recording chain
        luma_emphasized = self._pre_emphasis(luma, sample_rate)
        luma_clipped = self._white_clip(luma_emphasized)
        y_fm = self._fm_modulate(luma_clipped, self.fm_sync, self.fm_white, sample_rate)

        # Step 4: Chrominance recording chain (color-under)
        c_under = self._color_under(
            chroma, self.subcarrier_freq, sample_rate, sync_info
        )
        # Upsample c_under back to y_fm length (it was downsampled 4x internally)
        if len(c_under) < len(y_fm):
            ds = len(y_fm) // len(c_under)
            c_under = np.repeat(c_under, ds)[:len(y_fm)]

        # Step 5: Audio
        if audio_signal is None:
            audio_signal = np.zeros(len(composite_signal), dtype=np.float64)
        audio_out = self._audio_pre_emphasis(audio_signal, sample_rate)

        # Step 5b: Hi-Fi FM audio (optional)
        hifi_left = None
        hifi_right = None
        if self.hifi_audio:
            # Split audio into L/R (here we treat mono input as L=R)
            hifi_left = self._fm_modulate(
                audio_out, 
                VHS_HIFI_LEFT_CARRIER_HZ - VHS_HIFI_DEVIATION_HZ,
                VHS_HIFI_LEFT_CARRIER_HZ + VHS_HIFI_DEVIATION_HZ,
                sample_rate,
            )
            hifi_right = self._fm_modulate(
                audio_out,
                VHS_HIFI_RIGHT_CARRIER_HZ - VHS_HIFI_DEVIATION_HZ,
                VHS_HIFI_RIGHT_CARRIER_HZ + VHS_HIFI_DEVIATION_HZ,
                sample_rate,
            )

        # Step 6: Control track
        num_fields = int(self.field_rate * len(composite_signal) / sample_rate)
        control = self._generate_control_track(num_fields, self.field_rate, sample_rate)

        # Metadata
        metadata = {
            "standard": self.standard.value,
            "speed_mode": self.speed_mode.value,
            "sample_rate": sample_rate,
            "subcarrier_freq": self.subcarrier_freq,
            "h_freq": self.h_freq,
            "field_rate": self.field_rate,
            "fm_sync_hz": self.fm_sync,
            "fm_white_hz": self.fm_white,
            "fm_deviation_hz": self.fm_deviation,
            "chroma_under_hz": self.chroma_under_freq,
            "chroma_osc_hz": self.chroma_osc_freq,
            "num_fields": num_fields,
            "hifi_audio": self.hifi_audio,
            "burst_phase": sync_info.get("burst_phase", 0.0),
        }

        return EncodingResult(
            y_fm=y_fm,
            c_under=c_under,
            audio=audio_out,
            control=control,
            hifi_left=hifi_left,
            hifi_right=hifi_right,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Stage 1: Sync Separator
    # ------------------------------------------------------------------

    def _sync_separate(self, signal: np.ndarray, sample_rate: float) -> dict:
        """
        Extract horizontal sync, vertical sync, and color burst from a
        composite video signal.

        This models the sync separator circuitry in a VHS recorder. In real
        hardware, this is a slicer/clipper circuit that separates sync tips
        from the video signal based on voltage threshold, then a phase-locked
        loop (PLL) recovers clean timing.

        NTSC sync levels:
            - Sync tip: 0.0 (0 IRE, or -40 IRE relative to blanking)
            - Blanking (black): 0.077 (7.5 IRE setup)
            - White peak: 1.0 (100 IRE)

        PAL sync levels (no setup):
            - Sync tip: 0.0
            - Blanking (black): 0.0
            - White peak: 1.0

        The color burst is 8-10 cycles of the subcarrier frequency appearing
        on the back porch of the horizontal sync pulse. Its phase is the
        reference for the chrominance demodulation/conversion.

        Parameters
        ----------
        signal : np.ndarray
            Composite video signal (float64, normalized).
        sample_rate : float
            Sampling rate in Hz.

        Returns
        -------
        dict
            'hsync_samples': Sample indices where horizontal sync tips begin.
            'vsync_start': Sample index where vertical sync begins.
            'vsync_end': Sample index where vertical sync ends.
            'burst_phase': Estimated phase of the color burst (radians).
            'blanking_mask': Boolean array, True during blanking intervals.
            'line_period_samples': Number of samples per horizontal line.
            'num_lines': Total horizontal lines in this signal segment.
        """
        n_samples = len(signal)
        samples_per_line = sample_rate / self.h_freq
        total_lines = int(n_samples / samples_per_line)

        # --- Horizontal sync detection ---
        # Sync tips are the lowest voltage points in the signal.
        # Threshold: halfway between sync tip and blanking level
        if self.standard == Standard.NTSC:
            sync_threshold = 0.038  # between 0.0 (sync tip) and 0.077 (blanking)
        else:
            sync_threshold = -0.05  # PAL: below black level (0.0), allow for noise

        # Find candidate sync tip positions by looking for local minima
        # that fall below threshold
        hsync_samples = []
        i = int(samples_per_line * 0.5)  # start mid-first-line to avoid edge effects
        while i < n_samples - int(samples_per_line):
            # Check if this sample is near a sync tip (minimum in a ~4μs window)
            win_start = max(0, i)
            win_end = min(n_samples, i + int(4.7e-6 * sample_rate))
            window = signal[win_start:win_end]
            if len(window) > 0 and np.min(window) < sync_threshold:
                min_idx = win_start + int(np.argmin(window))
                # Ensure minimum distance between hsyncs (~70% of line period)
                if not hsync_samples or (min_idx - hsync_samples[-1]) > samples_per_line * 0.7:
                    hsync_samples.append(min_idx)
            i += int(samples_per_line * 0.5)  # search every half-line

        hsync_samples = np.array(hsync_samples, dtype=np.int64)

        # --- Vertical sync detection ---
        # Vertical sync consists of broad pulses (pre-equalizer, sync, post-equalizer)
        # In the composite signal, this appears as several consecutive lines where
        # the signal stays low (sync level) for most of each line.
        # Detection: find a run of consecutive lines with high duty-cycle at sync level.
        vsync_start = -1
        vsync_end = -1

        if len(hsync_samples) > 1:
            # Compute the average signal level during the first half of each line
            # (where the sync pulse and active video would be)
            line_avg_levels = []
            for idx in range(len(hsync_samples) - 1):
                s = hsync_samples[idx]
                e = hsync_samples[idx + 1]
                mid = s + int((e - s) * 0.3)
                end = s + int((e - s) * 0.7)
                if end > mid:
                    line_avg_levels.append(np.mean(signal[mid:end]))
                else:
                    line_avg_levels.append(0.5)

            line_avg_levels = np.array(line_avg_levels)
            # Vertical sync: several consecutive lines with very low average
            vsync_threshold = 0.15 if self.standard == Standard.NTSC else 0.1
            low_run = 0
            vsync_start_line = -1
            for idx, level in enumerate(line_avg_levels):
                if level < vsync_threshold:
                    if low_run == 0:
                        vsync_start_line = idx
                    low_run += 1
                else:
                    if low_run >= 3:  # at least 3 lines of low signal = vsync
                        vsync_start = int(hsync_samples[vsync_start_line])
                        vsync_end = int(hsync_samples[min(idx, len(hsync_samples) - 1)])
                        break
                    low_run = 0

        # --- Color burst phase estimation ---
        # The burst sits on the back porch, ~5-9 cycles of subcarrier after hsync.
        # Estimate phase by sampling the signal at the subcarrier frequency
        # near the back porch.
        burst_phase = 0.0
        if len(hsync_samples) > 0:
            # Use the first detected hsync to locate the back porch
            first_hsync = int(hsync_samples[0])
            # Back porch starts ~6μs after sync tip
            porch_start = first_hsync + int(6.0e-6 * sample_rate)
            # Burst is about 8 cycles of subcarrier
            burst_duration_s = 8.0 / self.subcarrier_freq
            porch_end = porch_start + int(burst_duration_s * sample_rate)
            porch_end = min(porch_end, n_samples)

            if porch_end > porch_start:
                burst_segment = signal[porch_start:porch_end]
                # Use absolute time (from porch_start) so the reference
                # carriers are phase-coherent with the subcarrier in the
                # composite signal.  Using t=0 gave a burst phase that
                # depended on where hsync was detected, not on the true
                # subcarrier phase.
                t_burst = (porch_start + np.arange(len(burst_segment))) / sample_rate
                # Correlate with subcarrier to extract phase
                ref_cos = np.cos(2.0 * np.pi * self.subcarrier_freq * t_burst)
                ref_sin = np.sin(2.0 * np.pi * self.subcarrier_freq * t_burst)
                i_component = np.sum(burst_segment * ref_cos)
                q_component = np.sum(burst_segment * ref_sin)
                burst_phase = math.atan2(q_component, i_component)

        # --- Blanking mask ---
        blanking_mask = np.ones(n_samples, dtype=bool)
        for hs in hsync_samples:
            hs = int(hs)
            blank_start = max(0, hs - int(1.5e-6 * sample_rate))
            blank_end = min(n_samples, hs + int(self.hblank_us * 1e-6 * sample_rate))
            blanking_mask[blank_start:blank_end] = False

        # Also blank vertical interval
        if vsync_start >= 0 and vsync_end >= 0:
            vblank_start = max(0, vsync_start - int(0.5e-6 * sample_rate))
            vblank_end = min(n_samples, vsync_end + int(0.5e-6 * sample_rate))
            blanking_mask[vblank_start:vblank_end] = False

        return {
            "hsync_samples": hsync_samples,
            "vsync_start": vsync_start,
            "vsync_end": vsync_end,
            "burst_phase": burst_phase,
            "blanking_mask": blanking_mask,
            "line_period_samples": samples_per_line,
            "num_lines": total_lines,
        }

    # ------------------------------------------------------------------
    # Stage 2: Y/C Separation (Comb Filter)
    # ------------------------------------------------------------------

    def _yc_separate(
        self,
        signal: np.ndarray,
        sample_rate: float,
        sync_info: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Separate luminance (Y) and chrominance (C) using a 1H-delay comb filter.

        In a real VHS recorder, the Y/C separator uses a 1-horizontal-line
        delay and subtracts to extract chrominance. This is the classic
        "comb filter" approach:

            Y_out[n] = (signal[n] + signal[n - 1H]) / 2
            C_out[n] = (signal[n] - signal[n - 1H]) / 2

        The comb filter works because chrominance (on the subcarrier) inverts
        phase from one line to the next (due to the odd-half-line frequency
        offset of the subcarrier). Subtracting adjacent lines cancels the
        luminance (which is correlated line-to-line) and reinforces the
        chrominance, while adding them cancels chrominance and reinforces luma.

        Limitations of the simple 1H comb:
        - Vertical detail in the chroma gets smeared
        - Not as clean as a 3-line adaptive comb filter used in higher-end
          equipment, but matches the VHS recorder's actual circuitry

        Parameters
        ----------
        signal : np.ndarray
            Composite video signal.
        sample_rate : float
            Sampling rate in Hz.
        sync_info : dict
            Output from _sync_separate.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (luma, chrominance) arrays of the same length as input.
            Luma is band-limited to ~3 MHz. Chroma is centered on the
            subcarrier frequency (to be downconverted later).
        """
        n_samples = len(signal)
        # Use EXACT float value for D in the comb filter phase rotation.
        # int() truncation (e.g. 909.88 → 909) causes D mod 4 ≠ 2,
        # making the comb pass only ~70% of chrominance and leaking ~30%
        # luminance into chroma at the subcarrier frequency.
        D_exact = sync_info["line_period_samples"]  # float, e.g. 909.88
        samples_per_line = int(sync_info["line_period_samples"])

        # ── Freq-domain comb filter: compute delay via phase shift ─────
        # Instead of: FFT(signal), FFT(delayed), then add/subtract in time domain,
        # we compute ONE FFT and apply the delay as a phase rotation:
        #   delayed[k] = signal[k] * exp(-j * 2π * k * D / N)
        # where D = samples_per_line.
        # Then:
        #   luma_comb[k]   = (1 + D_k) / 2 * S[k]
        #   chroma_comb[k]  = (1 - D_k) / 2 * S[k]
        # Combined with the filter masks, this gives the final outputs directly
        # from a single forward FFT + two inverse FFTs.
        # Uses power-of-2 padding for 6-7x faster FFTs.

        n = n_samples
        S, n_fft, n_orig = rfft_padded(signal, workers=-1)
        freqs = scipy.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

        # Phase shift for 1H delay: exp(-j * 2π * k * D / N_fft)
        # D_exact preserves sub-sample precision so the comb notch lands
        # exactly on the subcarrier (fsc = sample_rate / 4 → D mod 4 = 2).
        k = np.arange(len(S))
        D = D_exact  # float — keeps comb phase accurate at subcarrier
        delay_phase = np.exp(-1j * 2.0 * math.pi * k * D / n_fft)

        # Combined comb + filter masks:
        #   Luma output = S * (1 + D_phase) / 2 * luma_mask
        #   Chroma output = S * (1 - D_phase) / 2 * chroma_mask
        comb_sum = (1.0 + delay_phase) / 2.0   # luma comb
        comb_diff = (1.0 - delay_phase) / 2.0  # chroma comb

        # Luma: lowpass at 3 MHz with smooth rolloff
        luma_mask = np.ones(len(freqs))
        lp_low, lp_high = 2.7e6, 3.3e6
        lp_roll = (freqs > lp_low) & (freqs < lp_high)
        luma_mask[lp_roll] = 0.5 * (1.0 + np.cos(np.pi * (freqs[lp_roll] - lp_low) / (lp_high - lp_low)))
        luma_mask[freqs >= lp_high] = 0.0

        # Chroma: bandpass ±0.5 MHz around subcarrier
        chroma_mask = np.zeros(len(freqs))
        cbw = VHS_CHROMA_BW_HZ
        bp_low, bp_high = self.subcarrier_freq - cbw, self.subcarrier_freq + cbw
        bp_roll_lo = (freqs >= bp_low - cbw * 0.2) & (freqs < bp_low)
        bp_pass = (freqs >= bp_low) & (freqs <= bp_high)
        bp_roll_hi = (freqs > bp_high) & (freqs <= bp_high + cbw * 0.2)
        chroma_mask[bp_pass] = 1.0
        if np.any(bp_roll_lo):
            chroma_mask[bp_roll_lo] = 0.5 * (1.0 + np.cos(
                np.pi * (bp_low - freqs[bp_roll_lo]) / (cbw * 0.2)))
        if np.any(bp_roll_hi):
            chroma_mask[bp_roll_hi] = 0.5 * (1.0 + np.cos(
                np.pi * (freqs[bp_roll_hi] - bp_high) / (cbw * 0.2)))

        # Apply combined masks and inverse FFT (padded to power of 2)
        luma = irfft_padded(S * comb_sum * luma_mask, n_orig, workers=-1)
        chroma = irfft_padded(S * comb_diff * chroma_mask, n_orig, workers=-1)

        return luma, chroma

    # ------------------------------------------------------------------
    # Stage 3: Luminance Pre-Emphasis
    # ------------------------------------------------------------------

    def _pre_emphasis(self, luma: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        Apply VHS luminance pre-emphasis before FM modulation.

        VHS recording applies a specific pre-emphasis curve to boost high-
        frequency luminance detail. This compensates for the FM modulation
        process, which has a triangular noise spectrum (noise increases with
        frequency). By boosting HF before recording and attenuating it on
        playback, the high-frequency noise is reduced.

        The VHS pre-emphasis is a lead-lag network with:
            f₁ = 1.3 MHz  (pole — begins boosting above this frequency)
            f₂ = 4.5 MHz  (zero — boost flattens out)

        Transfer function (analog prototype):
            H(s) = (1 + s/ω₂) / (1 + s/ω₁)
            where ω₁ = 2π·f₁, ω₂ = 2π·f₂

        In discrete time, implemented via bilinear transform.

        The emphasis is relatively mild (~6 dB maximum boost around 2-3 MHz)
        compared to consumer audio emphasis, because the FM deviation already
        provides good SNR for the luminance band.

        Parameters
        ----------
        luma : np.ndarray
            Separated luminance signal.
        sample_rate : float
            Sampling rate in Hz.

        Returns
        -------
        np.ndarray
            Pre-emphasized luminance signal.
        """
        # Bilinear transform of H(s) = (1 + s/ω₂) / (1 + s/ω₁)
        # Use scipy.signal.bilinear for correct coefficient computation.
        # This guarantees DC gain = 1.0 (no amplification of DC/luma level).
        f1 = VHS_PREEMPH_F1_HZ
        f2 = VHS_PREEMPH_F2_HZ
        omega1 = 2.0 * math.pi * f1
        omega2 = 2.0 * math.pi * f2

        # Analog prototype: H(s) = (1 + s/ω₂) / (1 + s/ω₁)
        b_analog = [1.0 / omega2, 1.0]  # numerator: s/ω₂ + 1
        a_analog = [1.0 / omega1, 1.0]  # denominator: s/ω₁ + 1

        from scipy.signal import bilinear
        b_dig, a_dig = bilinear(b_analog, a_analog, fs=sample_rate)

        # Apply first-order IIR filter (C-implemented, fast for 1st-order)
        from scipy.signal import lfilter
        output = lfilter(b_dig, a_dig, luma)

        return output

    # ------------------------------------------------------------------
    # Stage 4: White Clip
    # ------------------------------------------------------------------

    def _white_clip(self, signal: np.ndarray, clip_level: float = 1.1) -> np.ndarray:
        """
        Clip signal peaks to prevent excessive FM deviation.

        In a real VHS recorder, the white clip circuit prevents the luminance
        signal from exceeding a level that would cause the FM carrier to
        deviate beyond the specification. If the signal exceeds white level
        too much, the FM frequency would go above 4.4 MHz, potentially
        causing interference with the chroma-under signal (~629 kHz) during
        playback demodulation.

        The clip level is set slightly above nominal white (1.0) to allow
        for the pre-emphasis overshoot, typically at 110% of white level.

        Parameters
        ----------
        signal : np.ndarray
            Pre-emphasized luminance signal.
        clip_level : float
            Maximum allowed signal level (default 1.1, i.e., 110% white).

        Returns
        -------
        np.ndarray
            Clipped luminance signal.
        """
        return np.clip(signal, -0.1, clip_level)

    # ------------------------------------------------------------------
    # Stage 5: FM Modulation
    # ------------------------------------------------------------------

    def _fm_modulate(
        self,
        signal: np.ndarray,
        f_sync: float,
        f_white: float,
        sample_rate: float,
    ) -> np.ndarray:
        """
        Frequency-modulate a signal onto a carrier.

        For the luminance channel, the FM modulation maps:
            sync tip (signal ≈ 0.0) → f_sync (3.4 MHz)
            white peak (signal ≈ 1.0) → f_white (4.4 MHz)

        The instantaneous frequency is a linear function of the input:
            f(t) = f_sync + (f_white - f_sync) * signal(t)

        This is then integrated to produce the instantaneous phase:
            φ(t) = 2π ∫₀ᵗ f(τ) dτ

        And the FM output is:
            out(t) = cos(φ(t))

        For luminance, the output is the instantaneous frequency samples
        (for further processing by the write electronics). For Hi-Fi audio,
        the actual cosine wave is produced.

        VHS FM specification (SP mode):
            - Sync tip: 3.4 MHz
            - Blanking (black): ~3.7 MHz
            - White peak: 4.4 MHz
            - Total deviation: 1.0 MHz
            - Pre-emphasis allows the signal to briefly exceed 4.4 MHz
              (white clip catches the worst cases)

        Parameters
        ----------
        signal : np.ndarray
            Baseband signal to modulate (normalized 0-1 for video).
        f_sync : float
            Frequency corresponding to minimum signal level.
        f_white : float
            Frequency corresponding to maximum signal level.
        sample_rate : float
            Output sample rate in Hz.

        Returns
        -------
        np.ndarray
            FM-modulated signal. For video, this is the instantaneous frequency
            waveform. For audio, this is the actual cosine carrier.
        """
        n_samples = len(signal)
        deviation = f_white - f_sync

        # Instantaneous frequency: linear map from signal to frequency
        inst_freq = f_sync + deviation * np.clip(signal, 0.0, 1.0)

        # Integrate to get phase
        # φ[n] = 2π * Σ(inst_freq[k] / sample_rate) for k=0..n
        dt = 1.0 / sample_rate
        phase = 2.0 * np.pi * np.cumsum(inst_freq) * dt

        # Output: cosine of instantaneous phase
        output = np.cos(phase)

        return output

    # ------------------------------------------------------------------
    # Stage 6: Color-Under Downconversion
    # ------------------------------------------------------------------

    def _color_under(
        self,
        chroma: np.ndarray,
        subcarrier_freq: float,
        sample_rate: float,
        sync_info: dict,
    ) -> np.ndarray:
        """
        Downconvert chrominance to the VHS "color-under" frequency.

        VHS records chrominance at a much lower frequency than the original
        subcarrier. This is the key innovation of the VHS format — it allows
        chrominance to be recorded at a longer wavelength (lower frequency),
        which has better signal-to-noise on the ferric oxide tape.

        NTSC color-under:
            Original subcarrier: 3.579545 MHz
            Downconverted: 629.0 kHz
            Local oscillator: 3.579545 - 0.629 = 2.950545 MHz
            Process: mix chroma with LO → bandpass filter at 629 kHz

        PAL color-under:
            Original subcarrier: 4.433619 MHz
            Downconverted: 627.0 kHz
            Local oscillator: 4.433619 - 0.627 = 3.806619 MHz

        The local oscillator is phase-locked to the color burst to ensure
        accurate color reproduction on playback. Any phase error in the LO
        translates directly to a hue error in the decoded picture.

        The downconverted chroma is recorded at a lower tape current than the
        FM luminance — typically 35-40% of the luminance recording current.
        This is because the longer-wavelength chroma signal penetrates deeper
        into the magnetic coating. If recorded at the same current as the
        high-frequency luminance, the chroma would saturate the surface layer
        and produce excessive crosstalk between adjacent tracks.

        The recording current ratio affects the depth of magnetization:
            - Luminance (3.4-4.4 MHz): short wavelength (~6-7 μm), recorded
              in the surface layer of the ~12 μm thick tape coating
            - Chroma (629 kHz): long wavelength (~37 μm at SP speed),
              penetrates through the full coating thickness

        Parameters
        ----------
        chroma : np.ndarray
            Separated chrominance signal (centered on subcarrier_freq).
        subcarrier_freq : float
            Original subcarrier frequency (3.579545 MHz NTSC, 4.433619 MHz PAL).
        sample_rate : float
            Sampling rate in Hz.
        sync_info : dict
            Sync information containing burst_phase for phase-accurate LO.

        Returns
        -------
        np.ndarray
            Downconverted chrominance signal at chroma_under_freq (~629 kHz).
            This is the signal that gets written to tape alongside the FM luma.
        """
        n_samples = len(chroma)
        t = np.arange(n_samples, dtype=np.float64) / sample_rate

        # Local oscillator frequency: difference between original subcarrier
        # and the desired color-under frequency
        lo_freq = subcarrier_freq - self.chroma_under_freq

        # Phase-locked to burst for accurate color reproduction
        # The burst_phase from sync_separate gives us the initial phase offset
        burst_phase = sync_info.get("burst_phase", 0.0)

        # Mix: multiply chroma by LO to produce sum and difference frequencies
        lo_signal = np.cos(2.0 * math.pi * lo_freq * t + burst_phase)
        mixed = chroma * lo_signal

        # The mixed signal contains:
        #   - Difference: subcarrier_freq - lo_freq = chroma_under_freq (~629 kHz)  ← KEEP
        #   - Sum: subcarrier_freq + lo_freq ≈ 7.16 MHz  ← REMOVE

        # ── Lowpass BEFORE downsampling — prevent aliasing of sum product ──
        # The sum product at ~7.16 MHz would alias to ~40 kHz if we downsample
        # first (7.16 mod 7.2 = 0.04).  Lowpass at the original rate first.
        mixed = self._lowpass_filter(mixed, sample_rate, cutoff_hz=800_000, order=4)

        # ── Downsample 4x — the chroma_under is only ~629 kHz ──
        # At full rate (14.3 MHz) we have 238K samples; at 4x downsample (3.6 MHz)
        # we only need 60K samples. The Nyquist at 3.6 MHz is 1.8 MHz, well above
        # the 629 kHz target + 171 kHz BW margin.
        ds = 4
        c_under_ds = mixed[::ds]

        # Scale to appropriate level for tape recording
        # The chroma is recorded at 35-40% of luminance current
        # Normalize peak to this ratio
        peak = np.max(np.abs(c_under_ds))
        if peak > 0:
            c_under_ds = c_under_ds / peak * VHS_CHROMA_CURRENT_RATIO

        return c_under_ds

    # ------------------------------------------------------------------
    # Stage 7: Audio Pre-Emphasis
    # ------------------------------------------------------------------

    def _audio_pre_emphasis(self, audio: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        Apply audio pre-emphasis for the linear audio track.

        VHS linear audio uses a standard pre-emphasis with a 50μs time
        constant, which provides approximately +10 dB of boost at 10 kHz
        relative to 1 kHz.

        Transfer function:
            H(f) = 1 / sqrt(1 + (f · τ)²)   [for playback de-emphasis]
            H_pre(f) = sqrt(1 + (f · τ)²)    [for recording pre-emphasis]

        where τ = 50 μs = 50 × 10⁻⁶ s

        This is the same emphasis curve used in broadcast FM radio (75μs in
        the US, 50μs in Europe/Japan — VHS uses 50μs universally).

        The linear audio track is recorded by a stationary head along the
        bottom edge of the tape:
            - Track width: 0.7 mm (SP mode)
            - Frequency response: 100 Hz – 10 kHz
            - Signal-to-noise ratio: ~43 dB (SP)
            - Wow and flutter: < 0.3% (SP)

        Parameters
        ----------
        audio : np.ndarray
            Audio signal samples.
        sample_rate : float
            Sampling rate in Hz.

        Returns
        -------
        np.ndarray
            Pre-emphasized audio signal.
        """
        tau = VHS_AUDIO_PREEMPH_TC_US * 1e-6  # 50 microseconds
        omega_c = 1.0 / tau  # corner frequency: ~20 kHz

        # Pre-emphasis: H(s) = (1 + s/ω_c) — boost high frequencies
        # Bilinear transform to digital
        T = 1.0 / sample_rate
        omega_c_warped = (2.0 / T) * math.tan(omega_c * T / 2.0)

        # H(z) = (b0 + b1*z⁻¹) / (1 + a1*z⁻¹)
        b0 = (2.0 / T + omega_c_warped) / (2.0 / T + omega_c_warped)  # = 1 at DC... let me redo
        # Proper bilinear for H(s) = (s + ω_c) / ω_c  [unity gain at DC, +6dB/oct above ω_c]
        # At s=jω: |H(jω)| = sqrt(1 + (ω/ω_c)²)
        # Bilinear: s = (2/T)(1-z⁻¹)/(1+z⁻¹)
        # H(z) = [(2/T)(1-z⁻¹)/(1+z⁻¹) + ω_c] / ω_c
        #       = [(2/T + ω_c) - (2/T - ω_c)z⁻¹] / [(2/T + ω_c) + (2/T - ω_c)z⁻¹]
        sum_ct = 2.0 / T + omega_c_warped
        diff_ct = 2.0 / T - omega_c_warped

        b0 = sum_ct / sum_ct  # normalize
        b1 = -diff_ct / sum_ct
        a1 = diff_ct / sum_ct

        # Apply IIR filter (C-implemented, fast for 1st-order)
        from scipy.signal import lfilter
        output = lfilter([b0, b1], [1.0, a1], audio)

        return output

    # ------------------------------------------------------------------
    # Stage 8: Control Track
    # ------------------------------------------------------------------

    def _generate_control_track(
        self,
        num_fields: int,
        field_rate: float,
        sample_rate: float,
    ) -> np.ndarray:
        """
        Generate the VHS control track signal.

        The control track is recorded by a stationary head along the bottom
        edge of the tape. It consists of one pulse per field:
            - NTSC: 60 pulses per second (one per 16.667 ms field)
            - PAL: 50 pulses per second (one per 20 ms field)

        Each pulse is derived from the vertical sync timing. On playback,
        the control track pulses are used by the servo system to:
            1. Lock the drum rotation speed to match recording speed
            2. Align the playback heads with the recorded tracks
            3. Maintain proper tracking (track-to-track alignment)

        The control track pulse is a simple square pulse:
            - Width: ~1-2 ms (approximately 2-4% of field period)
            - Amplitude: determined by recording current (not critical)

        This is the simplest signal in the VHS system — just timing information.

        Parameters
        ----------
        num_fields : int
            Number of fields to generate pulses for.
        field_rate : float
            Field rate in Hz (59.94 for NTSC, 50 for PAL).
        sample_rate : float
            Output sample rate in Hz.

        Returns
        -------
        np.ndarray
            Control track pulse train. Values are 0.0 or 1.0.
        """
        total_samples = int(num_fields * sample_rate / field_rate)
        control = np.zeros(total_samples, dtype=np.float64)

        field_period_samples = sample_rate / field_rate
        pulse_width_samples = int(0.0015 * sample_rate)  # 1.5 ms pulse width

        for i in range(num_fields):
            start = int(i * field_period_samples)
            end = min(start + pulse_width_samples, total_samples)
            control[start:end] = 1.0

        return control

    # ------------------------------------------------------------------
    # Internal: Filter Helpers
    # ------------------------------------------------------------------

    def _lowpass_filter(
        self,
        signal: np.ndarray,
        sample_rate: float,
        cutoff_hz: float,
        order: int = 3,
    ) -> np.ndarray:
        """
        FFT-based low-pass filter. Orders of magnitude faster than
        the naive IIR approach for long signals.

        Applies a smooth Brickwall lowpass (raised-cosine rolloff)
        in the frequency domain.

        Parameters
        ----------
        signal : np.ndarray
            Input signal.
        sample_rate : float
            Sampling rate in Hz.
        cutoff_hz : float
            -3 dB cutoff frequency in Hz.
        order : int
            Controls rolloff steepness (transition width in decades).

        Returns
        -------
        np.ndarray
            Filtered signal.
        """
        n = len(signal)
        spectrum, n_fft, _ = rfft_padded(signal, workers=-1)
        freqs = scipy.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

        # Raised-cosine rolloff for smooth transition
        rolloff_width = cutoff_hz * 0.3  # 30% transition band
        low = cutoff_hz - rolloff_width
        high = cutoff_hz + rolloff_width

        gain = np.ones_like(freqs)
        mask = (freqs > low) & (freqs < high)
        gain[mask] = 0.5 * (1.0 + np.cos(np.pi * (freqs[mask] - low) / (2.0 * rolloff_width)))
        gain[freqs >= high] = 0.0

        spectrum *= gain
        return irfft_padded(spectrum, n, workers=-1)

    def _bandpass_filter(
        self,
        signal: np.ndarray,
        sample_rate: float,
        low_hz: float,
        high_hz: float,
        order: int = 3,
    ) -> np.ndarray:
        """
        FFT-based bandpass filter.

        Parameters
        ----------
        signal : np.ndarray
            Input signal.
        sample_rate : float
            Sampling rate in Hz.
        low_hz : float
            Lower cutoff frequency (-3 dB point).
        high_hz : float
            Upper cutoff frequency (-3 dB point).
        order : int
            Controls rolloff steepness.

        Returns
        -------
        np.ndarray
            Bandpass-filtered signal.
        """
        n = len(signal)
        spectrum, n_fft, _ = rfft_padded(signal, workers=-1)
        freqs = scipy.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

        rolloff = (high_hz - low_hz) * 0.1  # 10% of band as transition
        gain = np.zeros_like(freqs)

        # Flat passband
        gain[(freqs >= low_hz) & (freqs <= high_hz)] = 1.0
        # Low-side rolloff
        low_mask = (freqs >= low_hz - rolloff) & (freqs < low_hz)
        gain[low_mask] = 0.5 * (1.0 + np.cos(np.pi * (low_hz - freqs[low_mask]) / rolloff))
        # High-side rolloff
        high_mask = (freqs > high_hz) & (freqs <= high_hz + rolloff)
        gain[high_mask] = 0.5 * (1.0 + np.cos(np.pi * (freqs[high_mask] - high_hz) / rolloff))

        spectrum *= gain
        return irfft_padded(spectrum, n, workers=-1)


# ===========================================================================
# Test Signal Generator
# ===========================================================================

def generate_test_composite(
    sample_rate: float,
    duration_s: float,
    pattern: str = "colorbars",
    standard: str = "NTSC",
) -> np.ndarray:
    """
    Generate standard test signals for VHS encoder testing.

    Produces a composite video signal (normalized 0.0–1.0) at the specified
    sample rate and duration. The signals conform to SMPTE/EBU standards.

    Parameters
    ----------
    sample_rate : float
        Output sample rate in Hz. Recommended: 4× subcarrier
        (14.318 MHz for NTSC, 17.734 MHz for PAL).
    duration_s : float
        Duration in seconds.
    pattern : str
        Test pattern type:
        - 'colorbars': SMPTE color bars (NTSC) or EBU color bars (PAL)
        - 'gradient': Luminance ramp from 0% to 100%
        - 'pluge': Picture Line-Up Generator Equipment pattern
        - 'multiburst': Multi-frequency burst pattern for frequency response
    standard : str
        'NTSC' or 'PAL'.

    Returns
    -------
    np.ndarray
        Composite video signal samples (float64, 0.0–1.0 normalized).

    Test Pattern Details
    --------------------
    SMPTE Color Bars (75%):
        Standard reference signal for video system alignment.
        Bars from left to right: white, yellow, cyan, green, magenta, red, blue, black
        Each bar occupies 1/8 of the active line time.

        Signal levels (IRE / normalized):
            White (75%):   0.75  (actually 0.75 in normalized)
            Yellow:        0.636
            Cyan:          0.525
            Green:         0.411
            Magenta:       0.333
            Red:           0.222
            Blue:          0.108
            Black:         0.0

        In NTSC, with setup: black = 0.077 (7.5 IRE), white = 1.0

    PLUGE:
        Used for monitor/video level calibration.
        Contains three narrow bars: above-black, black, and below-black.
        The "below-black" bar should be invisible on a properly adjusted display.

    Multi-burst:
        Contains bursts of sine waves at specific frequencies:
        0.5, 1.0, 2.0, 3.0, 3.58 MHz (NTSC) / 4.43 MHz (PAL)
        Each burst occupies 1/8 of the active line.
        Used to measure the frequency response of the recording/playback chain.
    """
    std = standard.upper()
    if std == "NTSC":
        subcarrier = NTSC_SUBCARRIER_HZ
        h_freq = NTSC_H_FREQ
        setup = 0.077  # 7.5 IRE setup (black level)
    else:
        subcarrier = PAL_SUBCARRIER_HZ
        h_freq = PAL_H_FREQ
        setup = 0.0  # PAL has no setup (black = blanking)

    total_samples = int(sample_rate * duration_s)
    signal = np.full(total_samples, setup, dtype=np.float64)

    samples_per_line = sample_rate / h_freq
    active_start_frac = NTSC_HBLANK_US / (1e6 / h_freq)  # fraction of line that is blanking
    # For NTSC: active video starts after blanking (~10.9μs) and occupies ~52.6μs
    active_start_sample = int(active_start_frac * samples_per_line)
    active_duration_samples = int(52.6e-6 * sample_rate) if std == "NTSC" else int(52.0e-6 * sample_rate)

    if pattern == "colorbars":
        signal = _generate_colorbars(
            signal, total_samples, samples_per_line,
            active_start_sample, active_duration_samples,
            subcarrier, h_freq, sample_rate, std, setup,
        )
    elif pattern == "gradient":
        signal = _generate_gradient(
            signal, total_samples, samples_per_line,
            active_start_sample, active_duration_samples, setup,
        )
    elif pattern == "pluge":
        signal = _generate_pluge(
            signal, total_samples, samples_per_line,
            active_start_sample, active_duration_samples, setup,
        )
    elif pattern == "multiburst":
        signal = _generate_multiburst(
            signal, total_samples, samples_per_line,
            active_start_sample, active_duration_samples,
            subcarrier, sample_rate, std,
        )
    else:
        raise ValueError(
            f"Unknown pattern '{pattern}'. "
            f"Valid options: 'colorbars', 'gradient', 'pluge', 'multiburst'"
        )

    # Add horizontal sync pulses
    _add_sync_pulses(signal, total_samples, samples_per_line, sample_rate, std)

    # --- Add color burst on back porch (after sync so it's not wiped) ---
    # The color burst is ~9 cycles of subcarrier placed on the back porch
    # between the sync tip and active video.  It serves as the phase
    # reference for chrominance decoding in the VHS decoder chain.
    if std == "NTSC":
        burst_start_us = 5.7    # μs after line start (after 4.7μs sync)
        burst_duration_us = 2.5  # ~9 cycles of 3.58 MHz
        burst_amplitude = 0.142  # ~40 IRE p-p (±20 IRE above blanking)
        burst_phase_deg = 180.0  # NTSC burst is at -I axis

        for line_start_f in np.arange(0, total_samples, samples_per_line):
            line_start = int(line_start_f)
            burst_start_samp = line_start + int(burst_start_us * 1e-6 * sample_rate)
            burst_end_samp = line_start + int((burst_start_us + burst_duration_us) * 1e-6 * sample_rate)
            burst_end_samp = min(burst_end_samp, total_samples)
            if burst_start_samp < total_samples:
                t_burst = np.arange(burst_start_samp, burst_end_samp) / sample_rate
                burst = burst_amplitude * np.cos(
                    2.0 * math.pi * subcarrier * t_burst + math.radians(burst_phase_deg)
                )
                signal[burst_start_samp:burst_end_samp] += burst

    return signal


def _generate_colorbars(
    signal: np.ndarray,
    total_samples: int,
    samples_per_line: float,
    active_start: int,
    active_duration: int,
    subcarrier: float,
    h_freq: float,
    sample_rate: float,
    standard: str,
    setup: float,
) -> np.ndarray:
    """Generate SMPTE color bars with chrominance on each bar."""
    # SMPTE 75% color bars — standard NTSC Y, I, Q values.
    # Format: (Y_norm, chroma_amplitude, chroma_phase_degrees)
    #
    # Each bar is encoded as:  y_level + amp * cos(2π·fsc·t + phase)
    # which gives I = amp·cos(phase), Q = amp·sin(phase).
    #
    # Standard NTSC 75% bars (R=G=B=0.75):
    #   Y = 0.299·R + 0.587·G + 0.114·B
    #   I = 0.596·R − 0.274·G − 0.322·B
    #   Q = 0.211·R − 0.523·G + 0.312·B
    #   amp = √(I² + Q²),  phase = atan2(Q, I)
    bars_ntsc = [
        (0.7500, 0.0000,    0.0),   # White   (Y=0.75,  I= 0.000, Q= 0.000)
        (0.6645, 0.3363, -44.1),    # Yellow  (Y=0.664, I=+0.242, Q=−0.234)
        (0.5257, 0.4742, -160.5),   # Cyan    (Y=0.526, I=−0.447, Q=−0.158)
        (0.4402, 0.4428, -117.7),   # Green   (Y=0.440, I=−0.206, Q=−0.392)
        (0.3098, 0.4428,  62.3),    # Magenta (Y=0.310, I=+0.206, Q=+0.392)
        (0.2243, 0.4742,  19.5),    # Red     (Y=0.224, I=+0.447, Q=+0.158)
        (0.0855, 0.3363, 135.9),    # Blue    (Y=0.086, I=−0.242, Q=+0.234)
        (setup,  0.0000,    0.0),    # Black   (at setup level, no chroma)
    ]

    bars_pal = [
        (0.562, 0.000, 0.0),    # White
        (0.466, 0.117, 240.0),   # Yellow (PAL UV phases differ)
        (0.382, 0.161, 120.0),   # Cyan
        (0.286, 0.065, 300.0),   # Green
        (0.255, 0.133, 60.0),    # Magenta
        (0.159, 0.080, 0.0),     # Red
        (0.073, 0.208, 180.0),   # Blue
        (setup, 0.000, 0.0),     # Black
    ]

    bars = bars_ntsc if standard == "NTSC" else bars_pal
    bar_width = active_duration // 8

    for line_start_f in np.arange(0, total_samples, samples_per_line):
        line_start = int(line_start_f)
        active_line_start = line_start + active_start

        for bar_idx, (y_level, chroma_amp, chroma_phase) in enumerate(bars):
            b_start = active_line_start + bar_idx * bar_width
            b_end = min(b_start + bar_width, len(signal))
            if b_start >= len(signal):
                break

            t = np.arange(b_start, b_end) / sample_rate
            # Add chrominance (subcarrier modulated by chroma amplitude)
            chroma = chroma_amp * np.cos(
                2.0 * math.pi * subcarrier * t + math.radians(chroma_phase)
            )
            signal[b_start:b_end] = y_level + chroma

    return signal


def _generate_gradient(
    signal: np.ndarray,
    total_samples: int,
    samples_per_line: float,
    active_start: int,
    active_duration: int,
    setup: float,
) -> np.ndarray:
    """Generate a luminance ramp from black (setup) to white (1.0)."""
    for line_start_f in np.arange(0, total_samples, samples_per_line):
        line_start = int(line_start_f)
        active_line_start = line_start + active_start
        active_end = min(active_line_start + active_duration, len(signal))
        if active_line_start >= len(signal):
            break
        n_active = active_end - active_line_start
        gradient = np.linspace(setup, 1.0, n_active)
        signal[active_line_start:active_end] = gradient

    return signal


def _generate_pluge(
    signal: np.ndarray,
    total_samples: int,
    samples_per_line: float,
    active_start: int,
    active_duration: int,
    setup: float,
) -> np.ndarray:
    """
    Generate PLUGE (Picture Line-Up Generator Equipment) pattern.

    PLUGE contains three narrow bars in the center of the screen:
        - Above black: slightly above setup level (should be visible)
        - Black: at setup level (should match the surrounding black)
        - Below black: slightly below setup level (should be invisible)

    The pattern also includes a gray reference bar at 50% (75 IRE for NTSC).
    """
    # Gray bar at 50%
    gray_level = 0.5 + setup / 2.0
    # PLUGE bars: narrow vertical bars centered in the line
    above_black = setup + 0.02   # ~2 IRE above setup
    at_black = setup
    below_black = setup - 0.02  # ~2 IRE below setup

    for line_start_f in np.arange(0, total_samples, samples_per_line):
        line_start = int(line_start_f)
        active_line_start = line_start + active_start
        active_end = min(active_line_start + active_duration, len(signal))
        if active_line_start >= len(signal):
            break

        # Fill with black first
        signal[active_line_start:active_end] = setup

        # Gray bar at 50% across the top 1/3
        n_active = active_end - active_line_start
        third = n_active // 3
        signal[active_line_start:active_line_start + third] = gray_level

        # PLUGE bars in the center portion
        pluge_start = active_line_start + third
        bar_width = n_active // 12  # narrow bars
        center = pluge_start + (n_active - third) // 2

        # Three bars: above-black, black, below-black
        bar_starts = [
            center - bar_width,
            center,
            center + bar_width,
        ]
        bar_levels = [above_black, at_black, below_black]
        for bs, bl in zip(bar_starts, bar_starts):
            be = bs + bar_width
            if active_line_start <= bs < active_end:
                signal[bs:min(be, active_end)] = bl

        # Fix: correct the iteration bug above
        for bs, bl in zip(bar_starts, bar_levels):
            be = bs + bar_width
            if active_line_start <= bs < active_end:
                signal[bs:min(be, active_end)] = bl

    return signal


def _generate_multiburst(
    signal: np.ndarray,
    total_samples: int,
    samples_per_line: float,
    active_start: int,
    active_duration: int,
    subcarrier: float,
    sample_rate: float,
    standard: str,
) -> np.ndarray:
    """
    Generate multi-burst frequency test pattern.

    Contains bursts of sine waves at standard test frequencies.
    Each burst occupies 1/8 of the active line time.

    NTSC frequencies: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.58 MHz
    PAL frequencies: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.43 MHz
    """
    if standard == "NTSC":
        freqs = [0.5e6, 1.0e6, 1.5e6, 2.0e6, 2.5e6, 3.0e6, 3.58e6]
    else:
        freqs = [0.5e6, 1.0e6, 1.5e6, 2.0e6, 2.5e6, 3.0e6, 4.0e6, 4.43e6]

    n_bursts = len(freqs)
    burst_width = active_duration // n_bursts

    for line_start_f in np.arange(0, total_samples, samples_per_line):
        line_start = int(line_start_f)
        active_line_start = line_start + active_start

        for burst_idx, freq in enumerate(freqs):
            b_start = active_line_start + burst_idx * burst_width
            b_end = min(b_start + burst_width, len(signal))
            if b_start >= len(signal):
                break

            t = np.arange(b_start, b_end) / sample_rate
            # 50% amplitude sine burst with cosine envelope (raised cosine taper)
            n_samples_burst = b_end - b_start
            envelope = 0.5 * (1.0 - np.cos(2.0 * math.pi * np.arange(n_samples_burst) / n_samples_burst))
            signal[b_start:b_end] = 0.25 + 0.25 * envelope * np.cos(2.0 * math.pi * freq * t)

    return signal


def _add_sync_pulses(
    signal: np.ndarray,
    total_samples: int,
    samples_per_line: float,
    sample_rate: float,
    standard: str,
) -> None:
    """
    Add horizontal and vertical sync pulses to a composite signal.

    Modifies signal in-place. Sync tips are set to 0.0 (below blanking).
    """
    hsync_width_samples = int(NTSC_HSYNC_WIDTH_US * 1e-6 * sample_rate)

    for line_start_f in np.arange(0, total_samples, samples_per_line):
        line_start = int(line_start_f)
        # Horizontal sync: 0.0 (sync tip) for hsync_width_samples at start of line
        hs_end = min(line_start + hsync_width_samples, len(signal))
        signal[line_start:hs_end] = 0.0

    # Vertical sync: multiple broad pulses
    # In NTSC, vsync is 3 pre-equalizer + 6 broad + 3 post-equalizer lines
    # For simplicity, set the first 20 lines of each field to sync level
    # (this is approximate — real vsync has a specific waveform)
    # We'll add it for the first field only (the signal likely contains one field)
    vsync_lines = 12  # approximate number of vsync lines
    vsync_samples = int(vsync_lines * samples_per_line)
    signal[:vsync_samples] = 0.0  # all sync level during vertical interval
