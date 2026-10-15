"""
VHS Tape Emulator — Main orchestrator (multithreaded).

Uses threading for parallelism — numpy/scipy FFT operations release the GIL,
so threads provide real parallelism for the signal processing pipeline.
Each worker thread gets its own encoder instance (no shared state).
"""

import os
import time
import numpy as np
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from .encode import VHSEncoder, generate_test_composite, SAMPLE_RATE_NTSC, SAMPLE_RATE_PAL
from .decode import VHSDecoder, PlaybackSimulator
from .magnetic import MagneticRecorder, FERRIC, CHROME, METAL, HeadParams
from .head import RecordingHead, PlaybackHead, DrumHeadAssembly
from .tape import VHSTape, CoatingMaterial
from .circuits import NoiseGenerator
from .degradation import TapeDegradation
from .file_format import (
    VHSFileWriter, VHSFileReader, VHSHeader, VHSDegradationState,
    print_info as vhs_print_info,
)

# Optimal thread count: 4 threads gives ~2x speedup for encode.
# More threads hurt due to GIL contention on Python-level code.
DEFAULT_WORKERS = min(4, max(1, os.cpu_count() or 4))


# ══════════════════════════════════════════════════════════════════════════
#  TOP-LEVEL WORKER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def _worker_encode_batch(args):
    """
    Worker: encode a BATCH of fields in one call.
    Each call creates its own VHSEncoder (thread-local, no GIL contention).
    """
    composites, sample_rate, standard, speed, hifi, start_idx = args
    encoder = VHSEncoder(standard=standard, speed_mode=speed, hifi_audio=hifi)
    results = []
    for i, composite in enumerate(composites):
        enc_result = encoder.encode_video(composite, sample_rate)
        results.append((start_idx + i, enc_result.y_fm, enc_result.c_under))
    return results


def _worker_decode_batch(args):
    """
    Worker: decode a BATCH of field chunks in one call.
    """
    chunks, sample_rate, standard, speed, hifi = args
    from .decode import VideoStandard as DecStandard, SpeedMode as DecSpeed
    decoder = VHSDecoder(
        standard=DecStandard(standard),
        speed_mode=DecSpeed(speed),
        hifi_audio=hifi,
    )
    results = []
    for flux, head_idx, field_num in chunks:
        result = decoder.decode_field(flux, sample_rate, head_idx, field_num)
        results.append((
            field_num,
            result.get('luma', np.array([])),
            result.get('chroma', np.array([])),
            result.get('audio', np.array([])),
            result.get('quality', {}),
        ))
    return results


class VHSTapeEmulator:
    """
    Physically accurate VHS tape emulator (multithreaded).

    Uses 4 worker threads (optimal for GIL-limited numpy/scipy workloads).
    Each thread creates its own encoder/decoder — zero shared state.
    """

    SPEED_PARAMS = {
        'SP': {'tape_speed': 33.35, 'track_pitch': 58.0, 'track_width': 49.0},
        'LP': {'tape_speed': 16.675, 'track_pitch': 29.0, 'track_width': 24.5},
        'EP': {'tape_speed': 11.117, 'track_pitch': 19.3, 'track_width': 16.0},
    }

    def __init__(
        self,
        standard: str = 'NTSC',
        speed: str = 'SP',
        hifi_audio: bool = False,
        tape_type: str = 'T120',
        coating: str = 'ferric',
        degradation: Optional[TapeDegradation] = None,
        workers: int = 0,
    ):
        self.standard = standard.upper()
        self.speed = speed.upper()
        self.hifi_audio = hifi_audio
        self.tape_type = tape_type
        self.coating_name = coating
        self.num_workers = workers if workers > 0 else DEFAULT_WORKERS

        self.sample_rate = SAMPLE_RATE_NTSC if self.standard == 'NTSC' else SAMPLE_RATE_PAL
        sp = self.SPEED_PARAMS[self.speed]
        self.tape_speed_mm_s = sp['tape_speed']
        self.tape_speed_ms = self.tape_speed_mm_s / 1000.0
        self.track_pitch_um = sp['track_pitch']
        self.track_width_um = sp['track_width']

        tape_factory = {'T60': VHSTape.T60, 'T120': VHSTape.T120, 'T160': VHSTape.T160}
        coating_map_tape = {
            'ferric': CoatingMaterial.FERRIC_OXIDE,
            'chrome': CoatingMaterial.CHROME,
            'metal': CoatingMaterial.METAL_PARTICLE,
        }
        self.tape = tape_factory[tape_type](coating=coating_map_tape[coating])

        drum_rpm = 1800.0 if self.standard == 'NTSC' else 1500.0
        self.drum_rpm = drum_rpm
        self.drum = DrumHeadAssembly(diameter_mm=74.487, rpm=drum_rpm, num_heads=2)

        self.write_head = RecordingHead(gap_length_um=0.5)
        self.read_head = PlaybackHead(gap_length_um=0.3)

        coating_map = {'ferric': FERRIC, 'chrome': CHROME, 'metal': METAL}
        self.tape_material = coating_map[coating]

        self.encoder = VHSEncoder(standard=standard, speed_mode=self.speed, hifi_audio=hifi_audio)

        from .decode import VideoStandard as DecStandard, SpeedMode as DecSpeed
        self.decoder = VHSDecoder(
            standard=DecStandard(self.standard),
            speed_mode=DecSpeed(self.speed),
            hifi_audio=hifi_audio,
        )
        self.playback_sim = PlaybackSimulator(self.decoder)

        snr_db = {'SP': 43, 'LP': 40, 'EP': 37}[self.speed]
        self.noise_gen = NoiseGenerator(
            snr_db=snr_db, bandwidth_hz=6e6, sample_rate=self.sample_rate
        )
        self.degradation = degradation or TapeDegradation()

        self._recording = False
        self._recorded_fields: List[dict] = []
        self._recorded_audio: List[np.ndarray] = []
        self._record_start_time: Optional[float] = None
        self._field_count = 0

    # ════════════════════════════════════════════════════════════════════
    #  RECORDING (multithreaded)
    # ════════════════════════════════════════════════════════════════════

    def record_start(self, duration_sec: float = 60.0):
        self._recording = True
        self._recorded_fields = []
        self._recorded_audio = []
        self._record_start_time = time.time()
        self._field_count = 0
        self._planned_duration = duration_sec
        self._planned_fields = int(duration_sec * (60.0 if self.standard == 'NTSC' else 50.0))
        print(f"[VHS] Recording started — {self.standard} {self.speed}")
        print(f"      Planned: {duration_sec:.1f}s ({self._planned_fields} fields)")
        print(f"      Tape: {self.tape_type}, {self.coating_name} coating")
        print(f"      Drum: {self.drum.rpm:.0f} RPM")
        print(f"      Workers: {self.num_workers} threads")

    def feed_fields_parallel(self, fields: List[np.ndarray], audio: Optional[List[np.ndarray]] = None):
        """Feed multiple fields in parallel using batched threading."""
        if not self._recording:
            raise RuntimeError("Not recording. Call record_start() first.")

        num_fields = len(fields)
        start_idx = self._field_count

        # Split into n_workers batches
        n_workers = min(self.num_workers, num_fields)
        batch_size = max(1, (num_fields + n_workers - 1) // n_workers)
        batches = []
        for i in range(0, num_fields, batch_size):
            batch_fields = fields[i:i + batch_size]
            batches.append((
                batch_fields,
                self.sample_rate,
                self.standard,
                self.speed,
                self.hifi_audio,
                start_idx + i,
            ))

        results_map = {}

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker_encode_batch, b): i
                       for i, b in enumerate(batches)}

            done = 0
            for future in as_completed(futures):
                batch_results = future.result()
                for field_idx, y_fm, c_under in batch_results:
                    results_map[field_idx] = (y_fm, c_under)
                done += 1
                total_done = sum(len(b[0]) for b in batches[:done])
                elapsed = time.time() - self._record_start_time
                fps = total_done / max(0.001, elapsed)
                print(f"\r      [{total_done}/{num_fields}] "
                      f"{elapsed:.1f}s, {fps:.1f} fields/s",
                      end='', flush=True)

        print()

        # Collect in order
        field_rate = 60.0 if self.standard == 'NTSC' else 50.0
        for i in range(num_fields):
            field_idx = start_idx + i
            y_fm, c_under = results_map[field_idx]
            head_idx = field_idx % 2

            angle = self.drum.angle_for_time(field_idx / field_rate)
            start_angle = float(angle[0]) if hasattr(angle, '__len__') else float(angle)
            if hasattr(start_angle, 'item'):
                start_angle = start_angle.item()

            self._recorded_fields.append({
                'field_number': field_idx, 'head_index': head_idx,
                'luma_flux': y_fm, 'chroma_flux': c_under,
                'flux_sample_rate': int(self.sample_rate),
                'start_angle_rad': start_angle, 'end_angle_rad': 0.0,
                'x_start_mm': 0.0,
                'y_start_um': float(head_idx * self.track_pitch_um),
            })
            self._recorded_audio.append(np.zeros(len(y_fm)))

        self._field_count += num_fields

    def feed_field(self, composite_video: np.ndarray, audio: Optional[np.ndarray] = None):
        """Feed a single field (convenience wrapper)."""
        self.feed_fields_parallel([composite_video])

    def record_stop(self, filepath: str) -> str:
        if not self._recording:
            raise RuntimeError("Not recording.")

        self._recording = False
        elapsed = time.time() - self._record_start_time
        total_fields = self._field_count
        rtf = (total_fields / 60.0) / max(0.001, elapsed)

        print(f"\n[VHS] Recording stopped — {total_fields} fields in {elapsed:.1f}s")
        print(f"      Speed: {rtf:.1f}x realtime ({self.num_workers} threads)")

        field_rate = 60.0 if self.standard == 'NTSC' else 50.0
        duration = total_fields / field_rate

        header = VHSHeader(
            video_standard=self.standard, speed_mode=self.speed,
            hifi_audio=self.hifi_audio, duration_seconds=duration,
            tape_length_mm=self.tape_speed_mm_s * duration,
            tape_speed_mm_s=self.tape_speed_mm_s, drum_diameter_mm=self.drum.diameter_mm,
            drum_rpm=self.drum.rpm, track_pitch_um=self.track_pitch_um,
            track_width_um=self.track_width_um,
            head_gap_write_um=0.5, head_gap_read_um=0.3, head_tape_spacing_um=0.05,
            coating_material=self.coating_name,
            coating_thickness_um=self.tape_material.thickness * 1e6,
            carrier_freq_sync_mhz=3.4, carrier_freq_white_mhz=4.4,
            chroma_freq_khz=629.0, audio_track_width_mm=0.7,
            recording_date=int(time.time()),
            tape_label=f"{self.tape_type} {self.coating_name}",
            creator_device="VHS Emulator v1.0",
        )

        head_wear = self.degradation.compute_head_wear(self.degradation.play_count)
        stretch = self.degradation.compute_tape_stretch(self.tape.tape_length_mm)

        deg_state = VHSDegradationState(
            tape_age_years=self.degradation.tape_age_years,
            play_count=self.degradation.play_count,
            storage_temp_c=self.degradation.storage_temp_c,
            storage_humidity_pct=self.degradation.storage_humidity_pct,
            head_wear_um=head_wear, tape_stretch_factor=stretch,
            dropout_density=max(0.0001, 0.001 * (1.0 + self.degradation.tape_age_years * 0.1)),
            print_through_db=-55.0,
        )

        writer = VHSFileWriter(header, deg_state)
        flux_rate = int(self.sample_rate)
        for fd in self._recorded_fields:
            writer.add_video_track(
                field_number=fd['field_number'], head_index=fd['head_index'],
                luma_flux=fd['luma_flux'].astype(np.float16),
                chroma_flux=fd['chroma_flux'].astype(np.float16),
                flux_sample_rate=flux_rate,
                start_angle_rad=fd['start_angle_rad'],
                end_angle_rad=fd['end_angle_rad'],
                x_start_mm=fd['x_start_mm'], y_start_um=fd['y_start_um'],
            )

        if self._recorded_audio:
            audio_all = np.concatenate(self._recorded_audio)
            writer.add_audio_track(audio_all.astype(np.float16), int(self.sample_rate))

        pulse_positions = (np.arange(total_fields) * int(
            self.sample_rate / field_rate
        )).astype(np.uint32)
        writer.add_control_track(pulse_positions)

        writer.save(filepath)
        print(f"[VHS] Saved: {filepath}")
        print(f"      Duration: {duration:.1f}s, Fields: {total_fields}")
        return filepath

    # ════════════════════════════════════════════════════════════════════
    #  PLAYBACK (multithreaded)
    # ════════════════════════════════════════════════════════════════════

    def play(self, filepath: str) -> dict:
        print(f"[VHS] Loading: {filepath}")
        vhs_print_info(filepath)

        reader = VHSFileReader()
        reader.load(filepath)
        header = reader.header
        degradation = reader.degradation

        tracks = reader.read_all_video_tracks()
        print(f"[VHS] Loaded {len(tracks)} video tracks")

        flux_streams = []
        for track in tracks:
            if 'luma_flux' in track:
                luma = track['luma_flux'].astype(np.float64)
                chroma = track.get('chroma_flux', np.zeros_like(luma))
                flux_streams.append(luma + 0.375 * chroma)
            elif 'flux_data' in track:
                flux_streams.append(track['flux_data'].astype(np.float64))

        if not flux_streams:
            raise ValueError("No flux data found in .vhs file")

        tape_flux = np.concatenate(flux_streams)
        tape_flux = self._apply_tape_degradation(tape_flux, degradation)
        tape_flux = tape_flux + self.noise_gen.generate(len(tape_flux))

        playback_rate = float(tracks[0].get('flux_sample_rate', self.sample_rate))
        field_rate = 60.0 if self.standard == 'NTSC' else 50.0
        samples_per_field = int(playback_rate / field_rate)

        chunks = []
        for i in range(0, len(tape_flux) - samples_per_field + 1, samples_per_field):
            flux = tape_flux[i:i + samples_per_field]
            chunks.append((flux, len(chunks) % 2, len(chunks)))

        num_fields = len(chunks)
        n_workers = min(self.num_workers, num_fields)
        batch_size = max(1, (num_fields + n_workers - 1) // n_workers)

        print(f"[VHS] Decoding {num_fields} fields with {n_workers} threads...")

        t0 = time.time()
        video_frames = [None] * num_fields
        chroma_frames = [None] * num_fields
        diagnostics = [None] * num_fields
        audio_chunks = [None] * num_fields

        decode_batches = []
        for i in range(0, num_fields, batch_size):
            batch = chunks[i:i + batch_size]
            decode_batches.append((batch, playback_rate, self.standard, self.speed, self.hifi_audio))

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker_decode_batch, b): i
                       for i, b in enumerate(decode_batches)}

            done = 0
            for future in as_completed(futures):
                batch_results = future.result()
                for field_num, luma, chroma, audio, quality in batch_results:
                    video_frames[field_num] = luma
                    chroma_frames[field_num] = chroma
                    diagnostics[field_num] = quality
                    audio_chunks[field_num] = audio
                done += 1
                total_done = done * batch_size
                print(f"\r      [{min(total_done, num_fields)}/{num_fields}] decoded",
                      end='', flush=True)

        print()
        elapsed = time.time() - t0
        print(f"[VHS] Decoded {num_fields} fields in {elapsed:.1f}s")

        valid_audio = [a for a in audio_chunks if a is not None and len(a) > 0]
        audio_samples = np.concatenate(valid_audio) if valid_audio else np.array([])

        # ── Convert 1D luma+chroma to 2D RGB raster images ────────────
        print("[VHS] Rasterizing frames (YIQ→RGB)...")
        raster_frames = []
        for i in range(num_fields):
            luma = video_frames[i]
            chroma = chroma_frames[i]
            if luma is None or chroma is None:
                continue
            raster = self.decoder.yc_to_rgb_raster(
                luma, chroma, playback_rate, output_width=720
            )
            raster_frames.append(raster)
        print(f"[VHS] Rasterized {len(raster_frames)} frames to 720x{int(self.decoder.lines_per_field)}x3")

        return {
            'video_frames': raster_frames,
            'raw_luma': video_frames,
            'audio_samples': audio_samples,
            'diagnostics': [d for d in diagnostics if d is not None],
            'header': header,
        }

    def _apply_tape_degradation(self, flux, deg_state):
        flux = flux.copy()
        if abs(deg_state.print_through_db) < 60:
            pt_level = 10.0 ** (deg_state.print_through_db / 20.0)
            shift = int(0.001 * len(flux))
            if 0 < shift < len(flux):
                pt = np.zeros_like(flux)
                pt[shift:] = flux[:-shift] * pt_level
                flux = flux + pt
        stretch = deg_state.tape_stretch_factor
        if abs(stretch - 1.0) > 1e-4:
            flux = np.interp(
                np.linspace(0, len(flux) - 1, int(len(flux) * stretch)),
                np.arange(len(flux)), flux)
        if deg_state.dropout_density > 0:
            for _ in range(int(deg_state.dropout_density * len(flux) / 1000)):
                pos = np.random.randint(0, max(1, len(flux) - 100))
                flux[pos:min(pos + np.random.randint(10, 200), len(flux))] = 0.0
        return flux

    # ════════════════════════════════════════════════════════════════════
    #  TEST PATTERNS
    # ════════════════════════════════════════════════════════════════════

    def generate_test_field(self, pattern='colorbars'):
        field_rate = 60.0 if self.standard == 'NTSC' else 50.0
        if pattern == 'black':
            return np.zeros(int(self.sample_rate / field_rate))
        return generate_test_composite(
            sample_rate=self.sample_rate, duration_s=1.0 / field_rate,
            pattern=pattern, standard=self.standard,
        )

    def record_test_pattern(self, filepath, pattern='colorbars', duration_sec=5.0):
        field_rate = 60.0 if self.standard == 'NTSC' else 50.0
        total_fields = int(duration_sec * field_rate)
        batch_size = max(1, (total_fields + self.num_workers - 1) // self.num_workers)

        self.record_start(duration_sec)

        print(f"      Generating fields...")
        t0 = time.time()
        all_fields = [self.generate_test_field(pattern) for _ in range(total_fields)]
        print(f"      Generated {total_fields} fields in {time.time()-t0:.2f}s")

        print(f"      Encoding + recording ({self.num_workers} threads, batch={batch_size})...")
        t_enc = time.time()
        for i in range(0, total_fields, batch_size):
            batch = all_fields[i:i + batch_size]
            self.feed_fields_parallel(batch)

        enc_elapsed = time.time() - t_enc
        fps = total_fields / max(0.001, enc_elapsed)
        print(f"      Encode throughput: {fps:.1f} fields/s ({fps/60:.1f}x realtime)")

        return self.record_stop(filepath)

    def info(self):
        drum_geom = self.drum.track_geometry()
        return {
            'standard': self.standard, 'speed': self.speed,
            'sample_rate': self.sample_rate, 'tape_type': self.tape_type,
            'coating': self.coating_name, 'tape_speed_mm_s': self.tape_speed_mm_s,
            'tape_length_m': self.tape.tape_length_m,
            'track_pitch_um': self.track_pitch_um, 'track_width_um': self.track_width_um,
            'drum_diameter_mm': self.drum.diameter_mm, 'drum_rpm': self.drum.rpm,
            'track_angle_deg': drum_geom['track_angle_deg'],
            'track_length_mm': drum_geom['track_length_mm'],
            'hifi_audio': self.hifi_audio, 'workers': self.num_workers,
        }

    def __repr__(self):
        return (f"VHSTapeEmulator({self.standard} {self.speed}, "
                f"{self.tape_type} {self.coating_name}, "
                f"{self.drum.rpm:.0f}RPM, {self.num_workers} threads)")
