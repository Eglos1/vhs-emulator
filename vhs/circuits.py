"""Models of internal analog circuitry inside a VHS recorder/player.

These are the actual electronic components on the PCBs, modeled with
real-world parasitics and non-idealities observed in consumer VHS hardware.
"""

import numpy as np
from scipy import signal as scipy_signal


class CircuitComponent:
    """Base class for analog circuit components."""
    pass


class Capacitor(CircuitComponent):
    """Models a real capacitor with ESR and leakage."""
    
    def __init__(self, capacitance_F, esr_ohm=0.01, leakage_A=1e-9):
        self.C = capacitance_F
        self.esr = esr_ohm
        self.leakage = leakage_A
    
    def impedance(self, freq_hz):
        """Complex impedance at given frequency.
        
        Z = ESR + 1/(jωC)
        """
        omega = 2 * np.pi * freq_hz
        return self.esr + 1.0 / (1j * omega * self.C)


class Inductor(CircuitComponent):
    """Models a real inductor with DCR and core losses."""
    
    def __init__(self, inductance_H, dcr_ohm=1.0, core_loss_factor=0.01):
        self.L = inductance_H
        self.dcr = dcr_ohm
        self.core_loss = core_loss_factor
    
    def impedance(self, freq_hz):
        """Complex impedance at given frequency.
        
        Z = DCR + jωL + core_loss(ω)
        """
        omega = 2 * np.pi * freq_hz
        # Core loss increases with frequency
        core_impedance = self.core_loss * omega
        return self.dcr + 1j * omega * self.L + core_impedance


class ActiveFilter:
    """Op-amp based active filter with real component tolerances.
    
    Implements second-order sections (biquad) for numerical stability.
    Uses Sallen-Key topology model for lowpass/highpass, MFB for bandpass.
    """
    
    def __init__(self, filter_type, cutoff_hz, order=2, gain_db=0):
        """
        filter_type: 'lowpass', 'highpass', 'bandpass', 'notch'
        cutoff_hz: -3dB frequency
        order: filter order (must be even for SOS implementation)
        gain_db: passband gain in dB
        """
        self.filter_type = filter_type
        self.cutoff_hz = cutoff_hz
        self.order = order
        self.gain = 10 ** (gain_db / 20.0)
        
        # Design filter coefficients using scipy
        nyq = 28e6 / 2  # Assume 28MHz sample rate for VHS
        self.sos = self._design_filter(nyq)
    
    def _design_filter(self, nyq):
        """Design second-order sections for the filter."""
        wn = self.cutoff_hz / nyq
        if wn >= 1.0:
            wn = 0.9999
        
        if self.filter_type == 'lowpass':
            b, a = scipy_signal.butter(self.order, wn, btype='low')
        elif self.filter_type == 'highpass':
            b, a = scipy_signal.butter(self.order, wn, btype='high')
        elif self.filter_type == 'bandpass':
            # For bandpass, assume bandwidth is 20% of center
            bw = self.cutoff_hz * 0.2
            low = max((self.cutoff_hz - bw / 2) / nyq, 0.001)
            high = min((self.cutoff_hz + bw / 2) / nyq, 0.9999)
            b, a = scipy_signal.butter(self.order, [low, high], btype='band')
        elif self.filter_type == 'notch':
            b, a = scipy_signal.iirnotch(self.cutoff_hz / nyq, 30, fs=nyq * 2)
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
        
        return scipy_signal.sosfiltfilt
    
    def transfer_function(self, freq_hz):
        """H(jω) of the filter.
        
        Evaluates the frequency response at the given frequency.
        """
        w, h = scipy_signal.sosfreqz(self.sos, worN=[freq_hz], fs=28e6)
        return h[0] * self.gain
    
    def apply(self, signal, sample_rate):
        """Apply filter to signal using second-order sections (biquad).
        
        Uses scipy.signal.sosfilt for numerically stable cascaded biquad
        implementation. This avoids the numerical instability of direct
        form I/II for high-order filters.
        """
        # Recompute SOS for actual sample rate if different
        wn = self.cutoff_hz / (sample_rate / 2)
        if wn >= 1.0:
            wn = 0.9999
        
        if self.filter_type == 'lowpass':
            b, a = scipy_signal.butter(self.order, wn, btype='low')
        elif self.filter_type == 'highpass':
            b, a = scipy_signal.butter(self.order, wn, btype='high')
        elif self.filter_type == 'bandpass':
            bw = self.cutoff_hz * 0.2
            low = max((self.cutoff_hz - bw / 2) / (sample_rate / 2), 0.001)
            high = min((self.cutoff_hz + bw / 2) / (sample_rate / 2), 0.9999)
            b, a = scipy_signal.butter(self.order, [low, high], btype='band')
        elif self.filter_type == 'notch':
            b, a = scipy_signal.iirnotch(self.cutoff_hz / (sample_rate / 2), 30, fs=sample_rate)
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
        
        sos = scipy_signal.tf2sos(b, a)
        filtered = scipy_signal.sosfilt(sos, signal)
        return filtered * self.gain


class VHSRecordAmplifier:
    """The recording amplifier that drives current through the write head.
    
    Contains:
    - Pre-emphasis network (lead-lag):
      H(s) = (1 + s/ω₂) / (1 + s/ω₁)
      ω₁ = 2π × 1.3MHz, ω₂ = 2π × 4.5MHz
    - White clip limiter
    - Recording current adjustment
    - Frequency-dependent recording compensation
    """
    
    def __init__(self, f1_hz=1.3e6, f2_hz=4.5e6):
        self.f1 = f1_hz
        self.f2 = f2_hz
        self.w1 = 2 * np.pi * f1_hz
        self.w2 = 2 * np.pi * f2_hz
    
    def pre_emphasize(self, signal, sample_rate):
        """Apply pre-emphasis lead-lag network.
        
        H(s) = (1 + s/ω₂) / (1 + s/ω₁)
        
        This boosts high frequencies before recording to improve SNR.
        The de-emphasis during playback restores flat response.
        """
        # Convert to digital using bilinear transform
        nyq = sample_rate / 2
        
        # Pre-warp frequencies for bilinear transform
        w1_d = np.tan(self.w1 / (2 * sample_rate))
        w2_d = np.tan(self.w2 / (2 * sample_rate))
        
        # Lead-lag transfer function coefficients
        # H(z) = (1 + z^-1) / (1 + a1*z^-1) approximated as biquad
        b0 = (1 + w2_d / w1_d) / 2
        b1 = (1 - w2_d / w1_d) / 2
        a1 = (w1_d - 1) / (w1_d + 1)
        
        # Apply IIR filter
        output = np.zeros_like(signal, dtype=np.float64)
        output[0] = b0 * signal[0]
        for i in range(1, len(signal)):
            output[i] = b0 * signal[i] + b1 * signal[i-1] + a1 * output[i-1]
        
        return output
    
    def white_clip(self, signal, clip_level=1.0):
        """White clip limiter to prevent overmodulation.
        
        Hard clips peaks above clip_level to protect against
        excessive recording current that would cause dropout.
        """
        return np.clip(signal, -clip_level, clip_level)
    
    def recording_current(self, emphasized_signal, max_current_mA=20):
        """Convert voltage signal to recording current.
        
        The record head is an inductive load, so current = integral of voltage.
        In practice, the amplifier has a current feedback loop.
        """
        # Integrate voltage to get current (inductive head model)
        dt = 1.0 / (28e6)  # Assume 28MHz sample rate
        current = np.cumsum(emphasized_signal) * dt
        
        # Scale to mA and clip to max current
        current_mA = current / np.max(np.abs(current) + 1e-10) * max_current_mA
        return np.clip(current_mA, -max_current_mA, max_current_mA)


class VHSPlaybackAmplifier:
    """The playback preamplifier and equalization chain.
    
    Contains:
    - Low-noise preamp (NF ~3dB, Gain ~40dB)
    - Head bump compensation filter
    - Playback de-emphasis (inverse of record pre-emphasis)
    - AGC (Automatic Gain Control)
    """
    
    def __init__(self, gain_db=40, nf_db=3):
        self.gain = 10 ** (gain_db / 20.0)
        self.nf = 10 ** (nf_db / 20.0)
    
    def preamp(self, head_voltage):
        """Low-noise preamplifier stage.
        
        Adds thermal noise and applies gain.
        Head output is typically 0.1-1mV for VHS.
        """
        noise = np.random.normal(0, self.nf * 0.001, head_voltage.shape)
        return (head_voltage + noise) * self.gain
    
    def head_bump_compensation(self, signal, sample_rate):
        """Compensate for head bump frequency response.
        
        The playback head has a natural peak at ~5-8 MHz due to
        its gap geometry and velocity. This filter flattens the response.
        """
        # Peaking filter at 6.5 MHz to compensate head bump
        f_bump = 6.5e6
        q = 2.0
        gain_db = 6.0
        
        nyq = sample_rate / 2
        w0 = f_bump / nyq
        
        # Peaking EQ coefficients (Audio EQ cookbook)
        A = 10 ** (gain_db / 40)
        alpha = np.sin(w0 * np.pi) / (2 * q)
        
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0 * np.pi)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0 * np.pi)
        a2 = 1 - alpha / A
        
        sos = scipy_signal.tf2sos([b0/a0, b1/a0, b2/a0], [1, a1/a0, a2/a0])
        return scipy_signal.sosfilt(sos, signal)
    
    def de_emphasize(self, signal, sample_rate):
        """Apply playback de-emphasis.
        
        Inverse of recording pre-emphasis:
        H(s) = (1 + s/ω₁) / (1 + s/ω₂)
        
        This restores flat frequency response and reduces high-freq noise.
        """
        w1 = 2 * np.pi * 1.3e6
        w2 = 2 * np.pi * 4.5e6
        
        w1_d = np.tan(w1 / (2 * sample_rate))
        w2_d = np.tan(w2 / (2 * sample_rate))
        
        # Inverse lead-lag (de-emphasis)
        b0 = (1 + w1_d / w2_d) / 2
        b1 = (1 - w1_d / w2_d) / 2
        a1 = (w2_d - 1) / (w2_d + 1)
        
        output = np.zeros_like(signal, dtype=np.float64)
        output[0] = b0 * signal[0]
        for i in range(1, len(signal)):
            output[i] = b0 * signal[i] + b1 * signal[i-1] + a1 * output[i-1]
        
        return output
    
    def agc(self, signal, target_level=1.0, attack_ms=10, release_ms=100):
        """Automatic Gain Control.
        
        Adjusts gain to maintain constant output level.
        Attack: how fast gain decreases on loud signals
        Release: how fast gain recovers on quiet signals
        """
        sample_rate = 28e6
        attack_samples = int(attack_ms * sample_rate / 1000)
        release_samples = int(release_ms * sample_rate / 1000)
        
        envelope = np.zeros_like(signal, dtype=np.float64)
        gain = np.ones_like(signal, dtype=np.float64)
        
        # Compute envelope with attack/release
        for i in range(len(signal)):
            level = abs(signal[i])
            if level > envelope[i-1] if i > 0 else 0:
                # Attack: fast decay
                coeff = 1.0 - np.exp(-1.0 / attack_samples)
            else:
                # Release: slow decay
                coeff = 1.0 - np.exp(-1.0 / release_samples)
            
            if i == 0:
                envelope[i] = level
            else:
                envelope[i] = envelope[i-1] + coeff * (level - envelope[i-1])
            
            # Compute gain needed to reach target level
            if envelope[i] > 1e-6:
                gain[i] = target_level / envelope[i]
            else:
                gain[i] = 1.0
        
        # Smooth gain to avoid clicks
        sos = scipy_signal.butter(2, 100 / (sample_rate / 2), btype='low', output='sos')
        gain_smooth = scipy_signal.sosfilt(sos, gain)
        
        return signal * gain_smooth


class FMModulator:
    """Phase-accumulator FM modulator.
    
    Records luminance as FM:
    - Sync tip: 3.4 MHz
    - White peak: 4.4 MHz (SP)
    - Phase accumulator generates instantaneous frequency
    """
    
    def __init__(self, f_sync=3.4e6, f_white=4.4e6, sample_rate=28e6):
        self.f_sync = f_sync
        self.f_white = f_white
        self.sample_rate = sample_rate
        self.f_carrier = (f_sync + f_white) / 2.0
        self.deviation = (f_white - f_sync) / 2.0
    
    def modulate(self, signal):
        """FM modulate the input signal.
        
        Uses phase accumulator method:
        φ[n] = φ[n-1] + 2π * (f_c + k*signal[n]) / f_s
        
        The output is cos(φ[n]).
        """
        n_samples = len(signal)
        phase = np.zeros(n_samples, dtype=np.float64)
        
        # Normalize signal to [-1, 1]
        sig_norm = signal / (np.max(np.abs(signal)) + 1e-10)
        
        # Phase accumulator
        for i in range(1, n_samples):
            inst_freq = self.f_carrier + self.deviation * sig_norm[i]
            phase[i] = phase[i-1] + 2 * np.pi * inst_freq / self.sample_rate
        
        return np.cos(phase)


class FMDemodulator:
    """PLL-based FM demodulator for playback.
    
    Models a real PLL with:
    - VCO center frequency: 3.9 MHz
    - Loop bandwidth: ~100 kHz
    - Capture range: ±500 kHz
    - Phase detector gain
    - Loop filter (2nd order)
    """
    
    def __init__(self, center_freq=3.9e6, sample_rate=28e6, loop_bw=100e3):
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.loop_bw = loop_bw
        
        # PLL parameters
        self.pd_gain = 1.0  # Phase detector gain (V/rad)
        self.vco_gain = 2 * np.pi * center_freq / sample_rate  # VCO gain (rad/sample)
        
        # Loop filter coefficients (2nd order)
        zeta = 0.707  # Damping factor (Butterworth response)
        wn = 2 * np.pi * loop_bw / sample_rate
        
        # proportional + integral controller
        self.kp = 2 * zeta * wn
        self.ki = wn ** 2
    
    def demodulate(self, fm_signal):
        """Demodulate FM signal using PLL.
        
        Returns instantaneous frequency deviation.
        """
        n_samples = len(fm_signal)
        vco_phase = np.zeros(n_samples, dtype=np.float64)
        freq_out = np.zeros(n_samples, dtype=np.float64)
        
        phase_acc = 0.0
        loop_integral = 0.0
        
        for i in range(n_samples):
            # Phase detector (mixer)
            vco_signal = np.cos(vco_phase[i])
            # Multiply input and VCO to get phase error
            phase_error = fm_signal[i] * vco_signal
            
            # Loop filter (PI controller)
            loop_integral += self.ki * phase_error
            vco_control = self.kp * phase_error + loop_integral
            
            # VCO update
            if i < n_samples - 1:
                inst_freq = self.center_freq + vco_control * self.center_freq / (2 * np.pi)
                freq_out[i] = inst_freq - self.center_freq
                vco_phase[i+1] = vco_phase[i] + 2 * np.pi * inst_freq / self.sample_rate
        
        return freq_out


class ColorUnderConverter:
    """Downconverts chroma subcarrier to ~629kHz for recording.
    Upconverts during playback.
    
    Uses a PLL locked to the color burst to generate the mixing LO.
    """
    
    def __init__(self, subcarrier_freq=3.579545e6, color_under_freq=629e3):
        self.subcarrier_freq = subcarrier_freq
        self.color_under_freq = color_under_freq
        self.pll = PLL(center_freq=subcarrier_freq, loop_bw=50e3, sample_rate=28e6)
    
    def downconvert(self, chroma_signal, burst_phase, sample_rate):
        """Downconvert chroma subcarrier to color-under frequency.
        
        Mixing: f_out = f_subcarrier - f_color_under = ~2.95 MHz
        Then lowpass to get baseband color-under signal.
        """
        t = np.arange(len(chroma_signal)) / sample_rate
        lo = np.cos(2 * np.pi * self.subcarrier_freq * t + burst_phase)
        
        mixed = chroma_signal * lo
        
        # Lowpass filter to extract color-under signal
        nyq = sample_rate / 2
        cutoff = self.color_under_freq * 1.5 / nyq
        b, a = scipy_signal.butter(4, cutoff, btype='low')
        return scipy_signal.filtfilt(b, a, mixed)
    
    def upconvert(self, color_under_signal, burst_phase, sample_rate):
        """Upconvert color-under signal back to subcarrier frequency.
        
        Used during playback to restore original chroma.
        """
        t = np.arange(len(color_under_signal)) / sample_rate
        lo = np.cos(2 * np.pi * self.subcarrier_freq * t + burst_phase)
        
        # Mix up
        mixed = color_under_signal * lo
        
        # Bandpass around subcarrier
        nyq = sample_rate / 2
        low = (self.subcarrier_freq - 500e3) / nyq
        high = (self.subcarrier_freq + 500e3) / nyq
        b, a = scipy_signal.butter(4, [low, high], btype='band')
        return scipy_signal.filtfilt(b, a, mixed)


class PLL:
    """Generic Phase-Locked Loop. Used in FM demod and color recovery."""
    
    def __init__(self, center_freq, loop_bw, sample_rate, phase_detector_gain=1.0, vco_gain=1.0):
        self.center_freq = center_freq
        self.loop_bw = loop_bw
        self.sample_rate = sample_rate
        self.pd_gain = phase_detector_gain
        self.vco_gain = vco_gain
        
        # 2nd order loop filter
        zeta = 0.707
        wn = 2 * np.pi * loop_bw / sample_rate
        self.kp = 2 * zeta * wn
        self.ki = wn ** 2
    
    def lock(self, input_signal):
        """Track input frequency. Returns VCO output.
        
        Simulates PLL locking to the input signal's frequency and phase.
        """
        n_samples = len(input_signal)
        vco_phase = np.zeros(n_samples, dtype=np.float64)
        vco_out = np.zeros(n_samples, dtype=np.float64)
        
        loop_integral = 0.0
        
        for i in range(n_samples):
            # VCO output
            vco_out[i] = np.cos(vco_phase[i])
            
            # Phase detector (XOR or multiplier)
            phase_error = input_signal[i] * vco_out[i]
            
            # Loop filter
            loop_integral += self.ki * phase_error
            vco_control = self.kp * phase_error + loop_integral
            
            # VCO update
            if i < n_samples - 1:
                freq = self.center_freq + vco_control * self.center_freq / (2 * np.pi)
                vco_phase[i+1] = vco_phase[i] + 2 * np.pi * freq / self.sample_rate
        
        return vco_out


class NoiseGenerator:
    """Models thermal and tape noise sources.
    
    Tape noise: primarily particle noise from the oxide coating
    - Gaussian, bandwidth-limited
    - PSD: roughly -40 dB at 1 MHz relative to signal (SP mode)
    - SNR: ~43 dB (linear audio), ~40 dB (video luminance SP)
    """
    
    def __init__(self, snr_db=40, bandwidth_hz=6e6, sample_rate=28e6):
        self.snr_db = snr_db
        self.bandwidth_hz = bandwidth_hz
        self.sample_rate = sample_rate
    
    def generate(self, num_samples):
        """Generate bandlimited Gaussian noise with specified SNR.
        
        Creates white noise, filters to bandwidth, then scales
        to achieve desired SNR relative to unit-level signal.
        """
        # Generate white noise
        noise = np.random.normal(0, 1, num_samples).astype(np.float64)
        
        # Bandlimit to specified bandwidth
        nyq = self.sample_rate / 2
        cutoff = min(self.bandwidth_hz / nyq, 0.999)
        b, a = scipy_signal.butter(4, cutoff, btype='low')
        noise_filtered = scipy_signal.filtfilt(b, a, noise)
        
        # Scale to achieve desired SNR
        # SNR = 10*log10(Psignal/Pnoise)
        # For unit-level signal: Pnoise = 10^(-SNR/10)
        noise_rms = np.sqrt(10 ** (-self.snr_db / 10.0))
        current_rms = np.std(noise_filtered)
        
        if current_rms > 0:
            noise_scaled = noise_filtered * (noise_rms / current_rms)
        else:
            noise_scaled = noise_filtered
        
        return noise_scaled
