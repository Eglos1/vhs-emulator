"""Models physical degradation of VHS tapes over time and use.

Degradation mechanisms:
1. Oxide shedding (dropout): magnetic particles detach from binder
2. Print-through: magnetic field from one layer prints onto adjacent layer
3. Tape stretch: mechanical elongation from transport stress
4. Binder hydrolysis: moisture absorption causes oxide to swell and shed
5. Head wear: heads gradually wear, reducing output level

Degradation rates based on tape archivist literature:
- Tape stretch: ~0.01-0.1% per 1000 plays (SMPTE Journal)
- Head wear: ~0.01μm per 1000 hours (Ferrite heads, IEEE Magnetics)
- Binder hydrolysis: Arrhenius model, doubles per 10°C increase
- Dropout density: increases exponentially with age and humidity
- Print-through: ~-50 to -60dB relative to original signal
"""

import numpy as np
from typing import List, Tuple


class TapeDegradation:
    """Models physical degradation of VHS tapes.
    
    Degradation mechanisms:
    1. Oxide shedding (dropout): magnetic particles detach from binder
    2. Print-through: magnetic field from one layer prints onto adjacent layer
    3. Tape stretch: mechanical elongation from transport stress
    4. Binder hydrolysis: moisture absorption causes oxide to swell and shed
    5. Head wear: heads gradually wear, reducing output level
    """
    
    def __init__(self, tape_age_years=0, play_count=0, storage_temp_c=22, storage_humidity_pct=40):
        """
        tape_age_years: years since manufacture
        play_count: number of full passes
        storage_temp_c: storage temperature (affects chemical degradation)
        storage_humidity_pct: affects binder hydrolysis rate
        """
        self.tape_age_years = tape_age_years
        self.play_count = play_count
        self.storage_temp_c = storage_temp_c
        self.storage_humidity_pct = storage_humidity_pct
    
    def generate_dropout_map(self, tape_length_mm, mean_dropout_length_um=50, dropout_density_per_mm=0.001):
        """Generate a map of dropout locations along the tape.
        
        Dropout characteristics:
        - Length: typically 10-500 μm (some up to 2mm for bad tapes)
        - Appears as white/black horizontal streaks in video
        - Density increases with age and play count
        - More common near splices and tape edges
        
        Returns list of (position_mm, length_um) tuples.
        """
        # Adjust density based on age and play count
        # Reference: dropout density increases ~10x per 10 years of age
        age_factor = 10 ** (self.tape_age_years / 10.0)
        # Each play causes ~0.001% additional dropouts
        play_factor = 1.0 + self.play_count * 1e-5
        # High humidity accelerates dropout formation
        humidity_factor = 1.0 + max(0, self.storage_humidity_pct - 50) * 0.02
        
        adjusted_density = dropout_density_per_mm * age_factor * play_factor * humidity_factor
        
        # Generate Poisson-distributed dropouts
        n_dropouts = np.random.poisson(adjusted_density * tape_length_mm)
        
        dropouts = []
        for _ in range(n_dropouts):
            position = np.random.uniform(0, tape_length_mm)
            # Log-normal distribution for dropout length
            length = np.random.lognormal(np.log(mean_dropout_length_um), 0.5)
            length = np.clip(length, 5, 2000)  # 5μm to 2mm
            dropouts.append((position, length))
        
        return sorted(dropouts, key=lambda x: x[0])
    
    def compute_print_through(self, magnetization_profile, tape_thickness_um=19):
        """Compute print-through signal.
        
        Physical model: each magnetized point creates a dipole field.
        The field at an adjacent tape layer (separated by tape thickness)
        partially magnetizes that layer.
        
        Print-through is ~50-60 dB below original signal.
        Hub-out winding: leading print is +2dB stronger than trailing
        (because the tape is wound with the oxide facing out, so
        the inner surface of each wrap prints onto the outer surface
        of the next inner wrap - the leading edge arrives first)
        
        Returns the print-through signal to add to playback.
        """
        # Print-through coupling factor (depends on tape thickness)
        # Typical: -50 to -60 dB, we use -55 dB
        coupling_db = -55.0
        coupling_linear = 10 ** (coupling_db / 20.0)
        
        # Tape thickness in mm
        thickness_mm = tape_thickness_um / 1000.0
        
        # Spatial frequency response of print-through
        # High spatial frequencies are attenuated more
        n = len(magnetization_profile)
        print_signal = np.zeros(n, dtype=np.float64)
        
        # Convolution kernel for print-through (simplified dipole model)
        # The kernel represents the field falloff with distance
        kernel_size = min(n, int(thickness_mm * 100))  # Reasonable kernel size
        if kernel_size < 3:
            kernel_size = 3
        
        # Create print-through kernel (exponential decay with distance)
        x = np.arange(kernel_size)
        kernel = np.exp(-x / (kernel_size * 0.3))
        kernel = kernel / np.sum(np.abs(kernel)) * coupling_linear
        
        # Leading print (appears before the signal)
        leading = np.convolve(magnetization_profile, kernel, mode='same')
        
        # Trailing print is 2dB weaker (hub-out winding)
        trailing_db = coupling_db + 2.0
        trailing_linear = 10 ** (trailing_db / 20.0)
        trailing_kernel = kernel * (trailing_linear / coupling_linear)
        trailing = np.convolve(magnetization_profile, trailing_kernel, mode='same')
        
        # Leading print is inverted relative to signal (magnetic dipole)
        print_signal = -leading + trailing
        
        return print_signal
    
    def compute_tape_stretch(self, original_length_mm):
        """Compute tape elongation factor.
        
        VHS tapes stretch ~0.01-0.1% per 1000 plays.
        Caused by tension on the tape path, especially at splices.
        Stretch causes timing errors and skew between tracks.
        
        Returns stretch factor (1.0 = no stretch, 1.001 = 0.1% stretch)
        """
        # Reference: 0.05% per 1000 plays for typical VHS tape
        # (based on SMPTE and archival literature)
        stretch_per_1000_plays = 0.0005  # 0.05%
        
        # Stretch increases with play count
        play_stretch = self.play_count * stretch_per_1000_plays / 1000.0
        
        # Temperature affects elasticity (higher temp = more stretch)
        temp_factor = 1.0 + max(0, self.storage_temp_c - 22) * 0.001
        
        # Age causes binder to become brittle, reducing stretch tolerance
        age_factor = max(0.5, 1.0 - self.tape_age_years * 0.005)
        
        total_stretch = play_stretch * temp_factor * age_factor
        
        # Clamp to reasonable range (max 0.5% stretch before tape is destroyed)
        total_stretch = min(total_stretch, 0.005)
        
        return 1.0 + total_stretch
    
    def compute_head_wear(self, play_count, head_hardness_vickers=800):
        """Compute head gap widening from wear.
        
        Ferrite heads wear ~0.01 μm per 1000 hours of use.
        Head gap starts at 0.3μm, degrades to ~0.5μm over lifetime.
        Wider gap = less high-frequency output = softer image.
        
        Returns effective gap length in μm.
        """
        # Initial gap length
        initial_gap_um = 0.3
        
        # Wear rate: ~0.01 μm per 1000 hours
        # Assume ~3 hours per play (2-hour tape + setup)
        hours_per_play = 3.0
        wear_per_1000_hours = 0.01  # μm
        
        # Total wear
        total_hours = play_count * hours_per_play
        wear_um = (total_hours / 1000.0) * wear_per_1000_hours
        
        # Hardness factor (harder heads wear slower)
        # Reference: Vickers hardness 800 for MnZn ferrite
        hardness_factor = 800.0 / head_hardness_vickers
        
        wear_um *= hardness_factor
        
        # Gap increases with wear (non-linear at high wear)
        # Gap widens as material is removed from pole tips
        effective_gap = initial_gap_um + wear_um * 1.5
        
        # Maximum useful gap before head needs replacement
        max_gap_um = 0.8
        effective_gap = min(effective_gap, max_gap_um)
        
        return effective_gap
    
    def compute_binder_condition(self):
        """Compute binder degradation factor.
        
        Binder hydrolysis (sticky shed syndrome):
        - Accelerated by high humidity and temperature
        - Causes oxide particles to detach
        - Eventually tape becomes unplayable
        
        Returns output level multiplier (1.0 = perfect, 0.0 = total loss)
        """
        # Arrhenius model for chemical degradation
        # Reference: Q10 = 2 (doubles per 10°C increase)
        # Activation energy for polyurethane hydrolysis: ~0.8 eV
        
        # Temperature acceleration (relative to 22°C reference)
        temp_ref = 22.0
        q10 = 2.0
        temp_acceleration = q10 ** ((self.storage_temp_c - temp_ref) / 10.0)
        
        # Humidity factor (hydrolysis requires water)
        # Below 30% RH: minimal degradation
        # Above 60% RH: rapid degradation
        if self.storage_humidity_pct < 30:
            humidity_factor = 0.1
        elif self.storage_humidity_pct < 50:
            humidity_factor = 1.0
        elif self.storage_humidity_pct < 70:
            humidity_factor = 3.0
        else:
            humidity_factor = 10.0
        
        # Time factor (years of aging)
        # Reference: tapes stored at 22°C, 40% RH last ~30 years
        # before significant degradation
        time_factor = self.tape_age_years / 30.0
        
        # Calculate degradation
        degradation_rate = temp_acceleration * humidity_factor * time_factor
        
        # Play count accelerates degradation (mechanical stress)
        play_stress = self.play_count * 0.0001  # Each play causes 0.01% degradation
        
        total_degradation = degradation_rate + play_stress
        
        # Convert to output level multiplier
        # Exponential decay model
        output_multiplier = np.exp(-total_degradation * 0.5)
        
        # Clamp to valid range
        output_multiplier = np.clip(output_multiplier, 0.0, 1.0)
        
        return output_multiplier
    
    def apply_all_degradation(self, tape_magnetization, tape_position_mm, playback_gap_um):
        """Apply all degradation effects to a tape segment being played back.
        
        Returns dict with:
            'signal': degraded tape magnetization
            'dropout_positions': where signal is lost
            'print_through_signal': noise to add
            'timing_error': stretch-induced timing offset
            'output_level_db': overall level loss from head wear
        """
        # Apply tape stretch (timing error)
        stretch_factor = self.compute_tape_stretch(len(tape_magnetization) * 0.015)  # Assume 15μm pitch
        
        # Apply head wear (output level loss)
        head_gap = self.compute_head_wear(self.play_count)
        # Gap loss model: output decreases as gap approaches signal wavelength
        # For VHS SP, minimum wavelength ~1.5μm
        min_wavelength_um = 1.5
        gap_loss_db = -20 * np.log10(np.sinc(head_gap / min_wavelength_um))
        output_level_db = gap_loss_db
        
        # Apply binder condition
        binder_factor = self.compute_binder_condition()
        
        # Generate dropouts
        segment_length_mm = len(tape_magnetization) * 0.015 / 1000  # Convert samples to mm
        dropouts = self.generate_dropout_map(segment_length_mm)
        
        # Apply dropouts to signal
        degraded_signal = tape_magnetization.copy()
        for pos, length in dropouts:
            # Convert position to sample index
            sample_pos = int(pos / (0.015 / 1000))
            sample_length = int(length / 0.015)
            
            if sample_pos < len(degraded_signal):
                end_pos = min(sample_pos + sample_length, len(degraded_signal))
                degraded_signal[sample_pos:end_pos] = 0.0
        
        # Apply print-through
        print_through = self.compute_print_through(tape_magnetization)
        
        # Apply binder degradation
        degraded_signal *= binder_factor
        
        # Apply head wear loss
        degraded_signal *= 10 ** (output_level_db / 20.0)
        
        return {
            'signal': degraded_signal,
            'dropout_positions': dropouts,
            'print_through_signal': print_through,
            'timing_error': stretch_factor - 1.0,  # Deviation from 1.0
            'output_level_db': output_level_db
        }


def age_tape(tape, years, temperature_c=22, humidity_pct=40):
    """Simulate aging a tape for N years. Returns degraded tape object.
    
    This is a helper function that creates a degradation model and applies
    it to simulate how a tape would age over the specified time period.
    
    Args:
        tape: dict or object with 'magnetization' array and 'length_mm'
        years: number of years to simulate aging
        temperature_c: storage temperature
        humidity_pct: relative humidity
    
    Returns:
        dict with aged tape properties and degradation metrics
    """
    # Create degradation model
    degradation = TapeDegradation(
        tape_age_years=years,
        play_count=0,  # Just aging, no playback stress
        storage_temp_c=temperature_c,
        storage_humidity_pct=humidity_pct
    )
    
    # Get tape properties
    if hasattr(tape, 'magnetization'):
        magnetization = tape.magnetization
        length_mm = tape.length_mm
    else:
        magnetization = tape.get('magnetization', np.zeros(1000))
        length_mm = tape.get('length_mm', 250000)  # ~250m for T-120
    
    # Apply degradation
    result = degradation.apply_all_degradation(
        magnetization,
        tape_position_mm=0,
        playback_gap_um=0.3
    )
    
    # Create aged tape object
    aged_tape = {
        'magnetization': result['signal'],
        'length_mm': length_mm * degradation.compute_tape_stretch(length_mm),
        'print_through': result['print_through_signal'],
        'dropouts': result['dropout_positions'],
        'head_gap_um': degradation.compute_head_wear(0),
        'binder_condition': degradation.compute_binder_condition(),
        'age_years': years,
        'storage_temp_c': temperature_c,
        'storage_humidity_pct': humidity_pct
    }
    
    return aged_tape
