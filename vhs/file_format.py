"""
.vhs binary file format for storing recorded VHS tape data.

The .vhs file is NOT a video file. It stores the actual magnetic state of the
tape — the flux patterns written by the recording heads. When played back, the
emulator reads this magnetic data and processes it through the full physical
playback chain (head response, preamp, EQ, FM demod, chroma recovery, etc.).

This is analogous to how an .flac file stores audio samples rather than "music"
— a .vhs file stores magnetic flux patterns rather than "video".

File layout:
    [8B magic+version] [256B header] [128B degradation] [track index] [track data...] [CRC32 footer]
"""

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, List, Optional, Tuple

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────

VHS_MAGIC = b'VHS!'
VHS_VERSION: int = 1
HEADER_BLOCK_SIZE: int = 256
DEGRADATION_BLOCK_SIZE: int = 128
TRACK_ENTRY_SIZE: int = 27
VIDEO_TRACK_HEADER_SIZE: int = 32

# Mapping helpers for packing/unpacking enum-like fields
STANDARD_MAP = {'NTSC': 0, 'PAL': 1}
STANDARD_DECODE = {0: 'NTSC', 1: 'PAL'}
SPEED_MAP = {'SP': 0, 'LP': 1, 'EP': 2}
SPEED_DECODE = {0: 'SP', 1: 'LP', 2: 'EP'}
COATING_MAP = {'ferric': 0, 'chrome': 1, 'metal': 2}
COATING_DECODE = {0: 'ferric', 1: 'chrome', 2: 'metal'}
TRACK_TYPE_MAP = {'video': 0, 'audio': 1, 'ctrl': 2}
TRACK_TYPE_DECODE = {0: 'video', 1: 'audio', 2: 'ctrl'}


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class VHSHeader:
    """Physical and metadata properties of the recorded tape.

    All numeric fields use native VHS hardware defaults so a freshly-created
    header represents a standard T-120 SP recording without any user overrides.
    """

    video_standard: str = 'NTSC'
    speed_mode: str = 'SP'
    hifi_audio: bool = False
    duration_seconds: float = 0.0

    # Tape geometry
    tape_length_mm: float = 0.0
    tape_speed_mm_s: float = 33.35  # SP default

    # Drum geometry
    drum_diameter_mm: float = 74.487
    drum_rpm: float = 1800.0  # NTSC

    # Head geometry
    track_pitch_um: float = 58.0
    track_width_um: float = 49.0
    head_gap_write_um: float = 0.5
    head_gap_read_um: float = 0.3
    head_tape_spacing_um: float = 0.05

    # Tape coating
    coating_material: str = 'ferric'
    coating_thickness_um: float = 19.0

    # FM / chroma carrier frequencies
    carrier_freq_sync_mhz: float = 3.4
    carrier_freq_white_mhz: float = 4.4
    chroma_freq_khz: float = 629.0

    # Audio
    audio_track_width_mm: float = 0.7

    # Metadata
    recording_date: int = 0  # Unix timestamp
    tape_label: str = ''
    creator_device: str = 'VHS Emulator v1.0'

    # String field sizes (bytes). 48 each keeps the header at exactly 256.
    _LABEL_SIZE: int = 48
    _DEVICE_SIZE: int = 48

    def pack(self) -> bytes:
        """Pack the header into exactly HEADER_BLOCK_SIZE bytes.

        Layout (little-endian, offsets are after the 14-byte file preamble):
            0   u8    video_standard
            1   u8    speed_mode
            2   u8    hifi_audio
            3   5B    reserved
            8   f64   duration_seconds
            16  f64   tape_length_mm
            24  f64   tape_speed_mm_s
            32  f64   drum_diameter_mm
            40  f64   drum_rpm
            48  f64   track_pitch_um
            56  f64   track_width_um
            64  f64   head_gap_write_um
            72  f64   head_gap_read_um
            80  f64   head_tape_spacing_um
            88  u8    coating_material
            89  7B    reserved
            96  f64   coating_thickness_um
            104 f64   carrier_freq_sync_mhz
            112 f64   carrier_freq_white_mhz
            120 f64   chroma_freq_khz
            128 f64   audio_track_width_mm
            136 u32   recording_date
            140 4B    reserved
            144 48B   tape_label (UTF-8, null-padded)
            192 48B   creator_device (UTF-8, null-padded)
            240 16B   reserved (padding to 256)
        """
        buf = bytearray(HEADER_BLOCK_SIZE)
        off = 0

        def _u8(v: int):
            nonlocal off
            struct.pack_into('<B', buf, off, v & 0xFF)
            off += 1

        def _f64(v: float):
            nonlocal off
            struct.pack_into('<d', buf, off, v)
            off += 8

        def _u32(v: int):
            nonlocal off
            struct.pack_into('<I', buf, off, v & 0xFFFFFFFF)
            off += 4

        def _pad(n: int):
            nonlocal off
            off += n

        def _str_field(s: str, size: int):
            nonlocal off
            raw = s.encode('utf-8')[:size]
            buf[off:off + len(raw)] = raw
            buf[off + len(raw):off + size] = b'\x00' * (size - len(raw))
            off += size

        _u8(STANDARD_MAP.get(self.video_standard, 0))
        _u8(SPEED_MAP.get(self.speed_mode, 0))
        _u8(1 if self.hifi_audio else 0)
        _pad(5)  # reserved

        _f64(self.duration_seconds)
        _f64(self.tape_length_mm)
        _f64(self.tape_speed_mm_s)
        _f64(self.drum_diameter_mm)
        _f64(self.drum_rpm)
        _f64(self.track_pitch_um)
        _f64(self.track_width_um)
        _f64(self.head_gap_write_um)
        _f64(self.head_gap_read_um)
        _f64(self.head_tape_spacing_um)

        _u8(COATING_MAP.get(self.coating_material, 0))
        _pad(7)  # reserved

        _f64(self.coating_thickness_um)
        _f64(self.carrier_freq_sync_mhz)
        _f64(self.carrier_freq_white_mhz)
        _f64(self.chroma_freq_khz)
        _f64(self.audio_track_width_mm)

        _u32(self.recording_date)
        _pad(4)  # reserved

        _str_field(self.tape_label, self._LABEL_SIZE)
        _str_field(self.creator_device, self._DEVICE_SIZE)

        assert off <= HEADER_BLOCK_SIZE, f"Header overflow: wrote {off} bytes"
        return bytes(buf)

    @classmethod
    def unpack(cls, data: bytes) -> 'VHSHeader':
        """Unpack a VHSHeader from HEADER_BLOCK_SIZE bytes."""
        if len(data) < HEADER_BLOCK_SIZE:
            raise ValueError(
                f"Header data too short: {len(data)} < {HEADER_BLOCK_SIZE}"
            )

        off = 0

        def _u8() -> int:
            nonlocal off
            v = data[off]
            off += 1
            return v

        def _f64() -> float:
            nonlocal off
            v = struct.unpack_from('<d', data, off)[0]
            off += 8
            return v

        def _u32() -> int:
            nonlocal off
            v = struct.unpack_from('<I', data, off)[0]
            off += 4
            return v

        def _pad(n: int):
            nonlocal off
            off += n

        def _str_field(size: int) -> str:
            nonlocal off
            raw = data[off:off + size]
            end = raw.find(b'\x00')
            if end == -1:
                end = size
            s = raw[:end].decode('utf-8', errors='replace')
            off += size
            return s

        header = cls()
        header.video_standard = STANDARD_DECODE.get(_u8(), 'NTSC')
        header.speed_mode = SPEED_DECODE.get(_u8(), 'SP')
        header.hifi_audio = bool(_u8())
        _pad(5)

        header.duration_seconds = _f64()
        header.tape_length_mm = _f64()
        header.tape_speed_mm_s = _f64()
        header.drum_diameter_mm = _f64()
        header.drum_rpm = _f64()
        header.track_pitch_um = _f64()
        header.track_width_um = _f64()
        header.head_gap_write_um = _f64()
        header.head_gap_read_um = _f64()
        header.head_tape_spacing_um = _f64()

        header.coating_material = COATING_DECODE.get(_u8(), 'ferric')
        _pad(7)

        header.coating_thickness_um = _f64()
        header.carrier_freq_sync_mhz = _f64()
        header.carrier_freq_white_mhz = _f64()
        header.chroma_freq_khz = _f64()
        header.audio_track_width_mm = _f64()

        header.recording_date = _u32()
        _pad(4)

        header.tape_label = _str_field(cls._LABEL_SIZE)
        header.creator_device = _str_field(cls._DEVICE_SIZE)

        return header


@dataclass
class VHSDegradationState:
    """Stores physical degradation parameters that affect playback quality.

    These values drive the degradation model in the playback chain — tape age
    causes high-frequency loss, stretch causes timing jitter, print-through
    causes ghosting, etc.
    """

    tape_age_years: float = 0.0
    play_count: int = 0
    storage_temp_c: float = 22.0
    storage_humidity_pct: float = 40.0
    head_wear_um: float = 0.0
    tape_stretch_factor: float = 1.0
    dropout_density: float = 0.001
    print_through_db: float = -55.0

    def pack(self) -> bytes:
        """Pack into exactly DEGRADATION_BLOCK_SIZE bytes.

        Layout:
            0   f32   tape_age_years
            4   u32   play_count
            8   f32   storage_temp_c
            12  f32   storage_humidity_pct
            16  f32   head_wear_um
            20  f32   tape_stretch_factor
            24  f32   dropout_density
            28  f32   print_through_db
            32  96B   reserved (padding)
        """
        buf = bytearray(DEGRADATION_BLOCK_SIZE)
        off = 0

        def _f32(v: float):
            nonlocal off
            struct.pack_into('<f', buf, off, v)
            off += 4

        def _u32(v: int):
            nonlocal off
            struct.pack_into('<I', buf, off, v & 0xFFFFFFFF)
            off += 4

        _f32(self.tape_age_years)
        _u32(self.play_count)
        _f32(self.storage_temp_c)
        _f32(self.storage_humidity_pct)
        _f32(self.head_wear_um)
        _f32(self.tape_stretch_factor)
        _f32(self.dropout_density)
        _f32(self.print_through_db)

        return bytes(buf)

    @classmethod
    def unpack(cls, data: bytes) -> 'VHSDegradationState':
        """Unpack from DEGRADATION_BLOCK_SIZE bytes."""
        if len(data) < DEGRADATION_BLOCK_SIZE:
            raise ValueError(
                f"Degradation data too short: {len(data)} < {DEGRADATION_BLOCK_SIZE}"
            )

        off = 0

        def _f32() -> float:
            nonlocal off
            v = struct.unpack_from('<f', data, off)[0]
            off += 4
            return v

        def _u32() -> int:
            nonlocal off
            v = struct.unpack_from('<I', data, off)[0]
            off += 4
            return v

        state = cls()
        state.tape_age_years = _f32()
        state.play_count = _u32()
        state.storage_temp_c = _f32()
        state.storage_humidity_pct = _f32()
        state.head_wear_um = _f32()
        state.tape_stretch_factor = _f32()
        state.dropout_density = _f32()
        state.print_through_db = _f32()

        return state


@dataclass
class TrackEntry:
    """One entry in the track index table."""

    track_offset: int = 0  # absolute byte offset in the file
    track_size: int = 0  # bytes of track data (including track header)
    track_type: str = 'video'  # 'video', 'audio', 'ctrl'
    head_index: int = 0  # 0 or 1
    field_number: int = 0  # field index (for video tracks)

    def pack(self) -> bytes:
        """Pack into TRACK_ENTRY_SIZE bytes.

        Layout:
            0   u64   track_offset
            8   u64   track_size
            16  u8    track_type
            17  u8    head_index
            18  u32   field_number
            22  5B    reserved (total = 27)
        """
        buf = bytearray(TRACK_ENTRY_SIZE)
        struct.pack_into('<Q', buf, 0, self.track_offset)
        struct.pack_into('<Q', buf, 8, self.track_size)
        struct.pack_into('<B', buf, 16, TRACK_TYPE_MAP.get(self.track_type, 0))
        struct.pack_into('<B', buf, 17, self.head_index & 0xFF)
        struct.pack_into('<I', buf, 18, self.field_number & 0xFFFFFFFF)
        # bytes 22-26: reserved (already zero)
        return bytes(buf)

    @classmethod
    def unpack(cls, data: bytes) -> 'TrackEntry':
        """Unpack from TRACK_ENTRY_SIZE bytes."""
        if len(data) < TRACK_ENTRY_SIZE:
            raise ValueError(
                f"Track entry too short: {len(data)} < {TRACK_ENTRY_SIZE}"
            )
        return cls(
            track_offset=struct.unpack_from('<Q', data, 0)[0],
            track_size=struct.unpack_from('<Q', data, 8)[0],
            track_type=TRACK_TYPE_DECODE.get(data[16], 'video'),
            head_index=data[17],
            field_number=struct.unpack_from('<I', data, 18)[0],
        )


# ── Helpers ────────────────────────────────────────────────────────────────

def _crc32(data: bytes) -> int:
    """Compute CRC-32 (IEEE) checksum."""
    return zlib.crc32(data) & 0xFFFFFFFF


def _pad_to(buf: bytearray, alignment: int) -> bytearray:
    """Pad *buf* with zero bytes so len(buf) is a multiple of *alignment*."""
    remainder = len(buf) % alignment
    if remainder != 0:
        buf.extend(b'\x00' * (alignment - remainder))
    return buf


# ── Writer ─────────────────────────────────────────────────────────────────

class VHSFileWriter:
    """Writes .vhs binary files.

    Typical usage::

        header = VHSHeader(video_standard='NTSC', tape_label='My Recording')
        degradation = VHSDegradationState(tape_age_years=15.0)
        writer = VHSFileWriter(header, degradation)

        writer.add_video_track(
            field_number=0, head_index=0,
            luma_flux=luma, chroma_flux=chroma,
            flux_sample_rate=3_000_000,
            start_angle_rad=0.0, end_angle_rad=6.283185,
            x_start_mm=0.0, y_start_um=0.0,
        )
        writer.save('recording.vhs')
    """

    def __init__(
        self,
        header: VHSHeader,
        degradation: VHSDegradationState,
    ) -> None:
        self.header = header
        self.degradation = degradation
        self._tracks: List[dict] = []

    # ── Track builders ─────────────────────────────────────────────────

    def add_video_track(
        self,
        field_number: int,
        head_index: int,
        luma_flux: np.ndarray,
        chroma_flux: np.ndarray,
        flux_sample_rate: int,
        start_angle_rad: float,
        end_angle_rad: float,
        x_start_mm: float,
        y_start_um: float,
    ) -> None:
        """Add one field's worth of magnetic flux data.

        Args:
            field_number: 0-based field index.
            head_index: which rotating head (0 or 1).
            luma_flux: 1-D float16 array of lumen flux samples.
            chroma_flux: 1-D float16 array of chroma flux samples.
            flux_sample_rate: samples per second.
            start_angle_rad: drum angle at start of scan (radians).
            end_angle_rad: drum angle at end of scan (radians).
            x_start_mm: horizontal position on tape at scan start.
            y_start_um: vertical position (track centre) on tape.
        """
        if luma_flux.dtype != np.float16:
            luma_flux = luma_flux.astype(np.float16)
        if chroma_flux.dtype != np.float16:
            chroma_flux = chroma_flux.astype(np.float16)
        if luma_flux.ndim != 1 or chroma_flux.ndim != 1:
            raise ValueError("Flux arrays must be 1-D")
        if len(luma_flux) != len(chroma_flux):
            raise ValueError(
                f"Luma/chroma length mismatch: {len(luma_flux)} != {len(chroma_flux)}"
            )

        self._tracks.append({
            'track_type': 'video',
            'field_number': field_number,
            'head_index': head_index,
            'num_flux_samples': len(luma_flux),
            'flux_sample_rate': flux_sample_rate,
            'start_angle_rad': start_angle_rad,
            'end_angle_rad': end_angle_rad,
            'x_start_mm': x_start_mm,
            'y_start_um': y_start_um,
            'luma_flux': luma_flux,
            'chroma_flux': chroma_flux,
        })

    def add_audio_track(
        self,
        audio_flux: np.ndarray,
        flux_sample_rate: int,
    ) -> None:
        """Add continuous audio track data.

        Args:
            audio_flux: 1-D float16 array of audio flux samples.
            flux_sample_rate: samples per second.
        """
        if audio_flux.dtype != np.float16:
            audio_flux = audio_flux.astype(np.float16)
        if audio_flux.ndim != 1:
            raise ValueError("Audio flux array must be 1-D")

        self._tracks.append({
            'track_type': 'audio',
            'field_number': 0,
            'head_index': 0,
            'num_flux_samples': len(audio_flux),
            'flux_sample_rate': flux_sample_rate,
            'start_angle_rad': 0.0,
            'end_angle_rad': 0.0,
            'x_start_mm': 0.0,
            'y_start_um': 0.0,
            'luma_flux': audio_flux,  # single flux array
            'chroma_flux': None,
        })

    def add_control_track(
        self,
        pulse_positions: np.ndarray,
    ) -> None:
        """Add control track pulses (positions as uint16 sample indices).

        Args:
            pulse_positions: 1-D uint16 array of sample positions.
        """
        if pulse_positions.dtype != np.uint16:
            pulse_positions = pulse_positions.astype(np.uint16)
        if pulse_positions.ndim != 1:
            raise ValueError("Control track array must be 1-D")

        self._tracks.append({
            'track_type': 'ctrl',
            'field_number': 0,
            'head_index': 0,
            'num_flux_samples': len(pulse_positions),
            'flux_sample_rate': 0,
            'start_angle_rad': 0.0,
            'end_angle_rad': 0.0,
            'x_start_mm': 0.0,
            'y_start_um': 0.0,
            'luma_flux': pulse_positions,  # uint16 pulse positions
            'chroma_flux': None,
        })

    # ── Serialisation ──────────────────────────────────────────────────

    def _pack_track_data(self, t: dict) -> bytes:
        """Pack a single track's data block (header + flux arrays)."""
        if t['track_type'] == 'video':
            return self._pack_video_track(t)
        elif t['track_type'] == 'audio':
            return self._pack_audio_track(t)
        elif t['track_type'] == 'ctrl':
            return self._pack_ctrl_track(t)
        else:
            raise ValueError(f"Unknown track type: {t['track_type']}")

    def _pack_video_track(self, t: dict) -> bytes:
        """Pack a video track: 32-byte header + luma + chroma float16 arrays."""
        hdr = bytearray(VIDEO_TRACK_HEADER_SIZE)
        struct.pack_into('<I', hdr, 0, t['field_number'])  # track_id
        struct.pack_into('<I', hdr, 4, t['num_flux_samples'])
        struct.pack_into('<I', hdr, 8, t['flux_sample_rate'])
        struct.pack_into('<f', hdr, 12, t['start_angle_rad'])
        struct.pack_into('<f', hdr, 16, t['end_angle_rad'])
        struct.pack_into('<f', hdr, 20, t['x_start_mm'])
        struct.pack_into('<f', hdr, 24, t['y_start_um'])
        # bytes 28-31: reserved
        luma_raw = t['luma_flux'].tobytes()
        chroma_raw = t['chroma_flux'].tobytes()
        return bytes(hdr) + luma_raw + chroma_raw

    def _pack_audio_track(self, t: dict) -> bytes:
        """Pack an audio track: same header + single flux array."""
        hdr = bytearray(VIDEO_TRACK_HEADER_SIZE)
        struct.pack_into('<I', hdr, 0, 0)  # track_id
        struct.pack_into('<I', hdr, 4, t['num_flux_samples'])
        struct.pack_into('<I', hdr, 8, t['flux_sample_rate'])
        struct.pack_into('<f', hdr, 12, 0.0)
        struct.pack_into('<f', hdr, 16, 0.0)
        struct.pack_into('<f', hdr, 20, 0.0)
        struct.pack_into('<f', hdr, 24, 0.0)
        return bytes(hdr) + t['luma_flux'].tobytes()

    def _pack_ctrl_track(self, t: dict) -> bytes:
        """Pack a control track: 32-byte header + uint16 pulse positions."""
        hdr = bytearray(VIDEO_TRACK_HEADER_SIZE)
        struct.pack_into('<I', hdr, 0, 0)
        struct.pack_into('<I', hdr, 4, t['num_flux_samples'])
        struct.pack_into('<I', hdr, 8, 0)
        struct.pack_into('<f', hdr, 12, 0.0)
        struct.pack_into('<f', hdr, 16, 0.0)
        struct.pack_into('<f', hdr, 20, 0.0)
        struct.pack_into('<f', hdr, 24, 0.0)
        return bytes(hdr) + t['luma_flux'].tobytes()

    def save(self, filepath: str) -> None:
        """Write the complete .vhs file to disk.

        File structure (all little-endian):
            [8B]  magic 'VHS!' + version uint16 + file_size uint64
            [256B] header block
            [128B] degradation state block
            [4B]  num_tracks uint32
            [27B * N] track index entries
            [variable] track data blocks
            [4B]  CRC-32 of all track data
            [4B]  magic 'VHS!' footer
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Pre-allocate the full file in a bytearray for CRC calculation,
        # then write in one shot.
        # Layout offsets (will be filled in):
        #   0..13     preamble (magic + version + size)
        #   8..263    header block
        #   264..391  degradation block
        #   392..395  num_tracks
        #   396..396+27*N-1  track index
        #   ...       track data blocks
        #   ...       CRC32 (4 bytes)
        #   ...       footer magic (4 bytes)

        # Phase 1: serialise track data and build the index
        track_data_list: List[bytes] = []
        for t in self._tracks:
            track_data_list.append(self._pack_track_data(t))

        num_tracks = len(self._tracks)

        # Calculate index start
        preamble_size = 14  # magic(4) + version(2) + file_size(8)
        index_start = preamble_size + HEADER_BLOCK_SIZE + DEGRADATION_BLOCK_SIZE + 4
        index_size = num_tracks * TRACK_ENTRY_SIZE
        data_start = index_start + index_size

        # Compute offsets for each track
        entries: List[TrackEntry] = []
        current_offset = data_start
        for i, raw in enumerate(track_data_list):
            t = self._tracks[i]
            entry = TrackEntry(
                track_offset=current_offset,
                track_size=len(raw),
                track_type=t['track_type'],
                head_index=t['head_index'],
                field_number=t['field_number'],
            )
            entries.append(entry)
            current_offset += len(raw)

        # Footer: CRC32 (4B) + magic (4B) = 8 bytes
        footer_start = current_offset
        file_size = footer_start + 8

        # Phase 2: assemble the byte buffer
        buf = bytearray(file_size)
        off = 0

        # Preamble
        struct.pack_into('<4sHQ', buf, off, VHS_MAGIC, VHS_VERSION, file_size)
        off += preamble_size

        # Header block
        header_bytes = self.header.pack()
        buf[off:off + HEADER_BLOCK_SIZE] = header_bytes
        off += HEADER_BLOCK_SIZE

        # Degradation block
        deg_bytes = self.degradation.pack()
        buf[off:off + DEGRADATION_BLOCK_SIZE] = deg_bytes
        off += DEGRADATION_BLOCK_SIZE

        # Track count
        struct.pack_into('<I', buf, off, num_tracks)
        off += 4

        # Track index
        for entry in entries:
            buf[off:off + TRACK_ENTRY_SIZE] = entry.pack()
            off += TRACK_ENTRY_SIZE

        # Track data
        for raw in track_data_list:
            buf[off:off + len(raw)] = raw
            off += len(raw)

        # CRC32 over all track data (from data_start to footer_start)
        crc = _crc32(bytes(buf[data_start:footer_start]))
        struct.pack_into('<I', buf, off, crc)
        off += 4

        # Footer magic
        struct.pack_into('<4s', buf, off, VHS_MAGIC)
        off += 4

        assert off == file_size, f"Size mismatch: wrote {off}, expected {file_size}"

        with open(path, 'wb') as f:
            f.write(buf)


# ── Reader ─────────────────────────────────────────────────────────────────

class VHSFileReader:
    """Reads .vhs binary files and provides access to track data.

    Typical usage::

        reader = VHSFileReader()
        reader.load('recording.vhs')
        info = reader.get_info()
        video = reader.read_all_video_tracks()
    """

    def __init__(self) -> None:
        self.header: Optional[VHSHeader] = None
        self.degradation: Optional[VHSDegradationState] = None
        self.track_index: List[TrackEntry] = []
        self._data: Optional[memoryview] = None  # mmap-like view
        self._filepath: Optional[Path] = None

    def load(self, filepath: str) -> None:
        """Load a .vhs file, parse the preamble, header, and track index.

        The entire file is read into memory for simplicity. For very large
        tapes this could be changed to mmap.

        Args:
            filepath: path to the .vhs file.

        Raises:
            ValueError: if the magic bytes are wrong or the version is unsupported.
        """
        self._filepath = Path(filepath)
        raw = self._filepath.read_bytes()
        self._data = memoryview(raw)

        off = 0

        # ── Preamble ───────────────────────────────────────────────
        magic = bytes(self._data[off:off + 4])
        if magic != VHS_MAGIC:
            raise ValueError(
                f"Invalid .vhs file: expected magic {VHS_MAGIC!r}, got {magic!r}"
            )
        off += 4

        version = struct.unpack_from('<H', self._data, off)[0]
        if version != VHS_VERSION:
            raise ValueError(
                f"Unsupported .vhs version {version} (expected {VHS_VERSION})"
            )
        off += 2

        file_size = struct.unpack_from('<Q', self._data, off)[0]
        if file_size != len(raw):
            raise ValueError(
                f"File size mismatch: header says {file_size}, "
                f"actual {len(raw)}"
            )
        off += 8

        # ── Header block ──────────────────────────────────────────
        self.header = VHSHeader.unpack(bytes(self._data[off:off + HEADER_BLOCK_SIZE]))
        off += HEADER_BLOCK_SIZE

        # ── Degradation block ─────────────────────────────────────
        self.degradation = VHSDegradationState.unpack(
            bytes(self._data[off:off + DEGRADATION_BLOCK_SIZE])
        )
        off += DEGRADATION_BLOCK_SIZE

        # ── Track count + index ───────────────────────────────────
        num_tracks = struct.unpack_from('<I', self._data, off)[0]
        off += 4

        self.track_index = []
        for _ in range(num_tracks):
            entry = TrackEntry.unpack(bytes(self._data[off:off + TRACK_ENTRY_SIZE]))
            self.track_index.append(entry)
            off += TRACK_ENTRY_SIZE

    def _ensure_loaded(self) -> None:
        if self._data is None:
            raise RuntimeError("No file loaded — call load() first")

    def read_track(self, track_number: int) -> dict:
        """Read a specific track's raw data.

        Args:
            track_number: 0-based index into the track index.

        Returns:
            dict with keys 'entry', 'data' (bytes of the track block).
        """
        self._ensure_loaded()
        if track_number < 0 or track_number >= len(self.track_index):
            raise IndexError(
                f"Track {track_number} out of range "
                f"(0..{len(self.track_index) - 1})"
            )
        entry = self.track_index[track_number]
        start = entry.track_offset
        end = start + entry.track_size
        return {'entry': entry, 'data': bytes(self._data[start:end])}

    def _parse_video_track(self, raw: bytes) -> dict:
        """Parse a video track block into its components."""
        hdr = raw[:VIDEO_TRACK_HEADER_SIZE]
        track_id = struct.unpack_from('<I', hdr, 0)[0]
        num_samples = struct.unpack_from('<I', hdr, 4)[0]
        sample_rate = struct.unpack_from('<I', hdr, 8)[0]
        start_angle = struct.unpack_from('<f', hdr, 12)[0]
        end_angle = struct.unpack_from('<f', hdr, 16)[0]
        x_start = struct.unpack_from('<f', hdr, 20)[0]
        y_start = struct.unpack_from('<f', hdr, 24)[0]

        flux_bytes = num_samples * 2  # float16 = 2 bytes each
        luma = np.frombuffer(
            raw[VIDEO_TRACK_HEADER_SIZE:VIDEO_TRACK_HEADER_SIZE + flux_bytes],
            dtype=np.float16,
        ).copy()
        chroma = np.frombuffer(
            raw[VIDEO_TRACK_HEADER_SIZE + flux_bytes:
                VIDEO_TRACK_HEADER_SIZE + 2 * flux_bytes],
            dtype=np.float16,
        ).copy()

        return {
            'track_id': track_id,
            'num_flux_samples': num_samples,
            'flux_sample_rate': sample_rate,
            'start_angle_rad': start_angle,
            'end_angle_rad': end_angle,
            'x_start_mm': x_start,
            'y_start_um': y_start,
            'luma_flux': luma,
            'chroma_flux': chroma,
        }

    def _parse_audio_track(self, raw: bytes) -> dict:
        """Parse an audio track block."""
        hdr = raw[:VIDEO_TRACK_HEADER_SIZE]
        num_samples = struct.unpack_from('<I', hdr, 4)[0]
        sample_rate = struct.unpack_from('<I', hdr, 8)[0]

        audio = np.frombuffer(
            raw[VIDEO_TRACK_HEADER_SIZE:
                VIDEO_TRACK_HEADER_SIZE + num_samples * 2],
            dtype=np.float16,
        ).copy()

        return {
            'num_flux_samples': num_samples,
            'flux_sample_rate': sample_rate,
            'audio_flux': audio,
        }

    def _parse_ctrl_track(self, raw: bytes) -> dict:
        """Parse a control track block."""
        hdr = raw[:VIDEO_TRACK_HEADER_SIZE]
        num_samples = struct.unpack_from('<I', hdr, 4)[0]

        pulses = np.frombuffer(
            raw[VIDEO_TRACK_HEADER_SIZE:
                VIDEO_TRACK_HEADER_SIZE + num_samples * 2],
            dtype=np.uint16,
        ).copy()

        return {
            'num_pulses': num_samples,
            'pulse_positions': pulses,
        }

    def read_all_video_tracks(self) -> List[dict]:
        """Read all video tracks in field order.

        Returns:
            List of dicts (see _parse_video_track) sorted by field_number.
        """
        self._ensure_loaded()
        result = []
        for i, entry in enumerate(self.track_index):
            if entry.track_type != 'video':
                continue
            raw = self.read_track(i)['data']
            parsed = self._parse_video_track(raw)
            result.append(parsed)
        result.sort(key=lambda d: d.get('track_id', 0))
        return result

    def read_audio_track(self) -> Optional[dict]:
        """Read the continuous audio track, if present.

        Returns:
            dict with 'audio_flux' (np.ndarray) and 'flux_sample_rate', or None.
        """
        self._ensure_loaded()
        for i, entry in enumerate(self.track_index):
            if entry.track_type == 'audio':
                raw = self.read_track(i)['data']
                return self._parse_audio_track(raw)
        return None

    def read_control_track(self) -> Optional[dict]:
        """Read the control track, if present.

        Returns:
            dict with 'pulse_positions' (np.ndarray), or None.
        """
        self._ensure_loaded()
        for i, entry in enumerate(self.track_index):
            if entry.track_type == 'ctrl':
                raw = self.read_track(i)['data']
                return self._parse_ctrl_track(raw)
        return None

    def get_info(self) -> dict:
        """Return human-readable metadata about the loaded file.

        Returns:
            dict with header fields, degradation state, and track summary.
        """
        self._ensure_loaded()
        h = self.header
        d = self.degradation

        video_count = sum(1 for e in self.track_index if e.track_type == 'video')
        audio_count = sum(1 for e in self.track_index if e.track_type == 'audio')
        ctrl_count = sum(1 for e in self.track_index if e.track_type == 'ctrl')

        return {
            'filepath': str(self._filepath),
            'file_size_bytes': len(self._data),
            'version': VHS_VERSION,
            'video_standard': h.video_standard,
            'speed_mode': h.speed_mode,
            'hifi_audio': h.hifi_audio,
            'duration_seconds': h.duration_seconds,
            'tape_length_mm': h.tape_length_mm,
            'tape_speed_mm_s': h.tape_speed_mm_s,
            'drum_diameter_mm': h.drum_diameter_mm,
            'drum_rpm': h.drum_rpm,
            'coating_material': h.coating_material,
            'carrier_freq_sync_mhz': h.carrier_freq_sync_mhz,
            'carrier_freq_white_mhz': h.carrier_freq_white_mhz,
            'chroma_freq_khz': h.chroma_freq_khz,
            'tape_label': h.tape_label,
            'creator_device': h.creator_device,
            'recording_date': h.recording_date,
            'tape_age_years': d.tape_age_years,
            'play_count': d.play_count,
            'storage_temp_c': d.storage_temp_c,
            'head_wear_um': d.head_wear_um,
            'tape_stretch_factor': d.tape_stretch_factor,
            'num_video_tracks': video_count,
            'num_audio_tracks': audio_count,
            'num_ctrl_tracks': ctrl_count,
            'total_tracks': len(self.track_index),
        }

    def verify_integrity(self) -> bool:
        """Verify the CRC-32 checksum of all track data.

        Returns:
            True if the checksum matches, False otherwise.
        """
        self._ensure_loaded()
        # Locate the CRC in the file: it sits right before the footer magic.
        crc_offset = len(self._data) - 8  # CRC(4) + magic(4)
        stored_crc = struct.unpack_from('<I', self._data, crc_offset)[0]

        # Data region is everything after the index table up to the CRC.
        preamble = 14 + HEADER_BLOCK_SIZE + DEGRADATION_BLOCK_SIZE + 4
        index_size = len(self.track_index) * TRACK_ENTRY_SIZE
        data_start = preamble + index_size
        data_end = crc_offset

        computed_crc = _crc32(bytes(self._data[data_start:data_end]))
        return stored_crc == computed_crc


# ── Convenience ────────────────────────────────────────────────────────────

def print_info(filepath: str) -> None:
    """Print human-readable metadata for a .vhs file.

    Args:
        filepath: path to a .vhs file.
    """
    reader = VHSFileReader()
    reader.load(filepath)
    info = reader.get_info()

    print(f"=== VHS File: {info['filepath']} ===")
    print(f"  File size:         {info['file_size_bytes']:,} bytes")
    print(f"  Format version:    {info['version']}")
    print()
    print(f"  Standard:          {info['video_standard']}")
    print(f"  Speed:             {info['speed_mode']}")
    print(f"  HiFi audio:        {'yes' if info['hifi_audio'] else 'no'}")
    print(f"  Duration:          {info['duration_seconds']:.1f}s")
    print()
    print(f"  Tape label:        {info['tape_label'] or '(none)'}")
    print(f"  Creator:           {info['creator_device']}")
    if info['recording_date']:
        import datetime
        dt = datetime.datetime.fromtimestamp(info['recording_date'])
        print(f"  Recorded:          {dt.isoformat()}")
    print()
    print(f"  Tape length:       {info['tape_length_mm']:.1f} mm")
    print(f"  Tape speed:        {info['tape_speed_mm_s']:.2f} mm/s")
    print(f"  Drum diameter:     {info['drum_diameter_mm']:.3f} mm")
    print(f"  Drum RPM:          {info['drum_rpm']:.0f}")
    print(f"  Coating:           {info['coating_material']}")
    print()
    print(f"  FM sync freq:      {info['carrier_freq_sync_mhz']:.2f} MHz")
    print(f"  FM white freq:     {info['carrier_freq_white_mhz']:.2f} MHz")
    print(f"  Chroma freq:       {info['chroma_freq_khz']:.1f} kHz")
    print()
    print(f"  Tape age:          {info['tape_age_years']:.1f} years")
    print(f"  Play count:        {info['play_count']}")
    print(f"  Storage temp:      {info['storage_temp_c']:.1f} C")
    print(f"  Head wear:         {info['head_wear_um']:.3f} um")
    print(f"  Stretch factor:    {info['tape_stretch_factor']:.6f}")
    print()
    print(f"  Tracks:            {info['total_tracks']} total")
    print(f"    Video:           {info['num_video_tracks']}")
    print(f"    Audio:           {info['num_audio_tracks']}")
    print(f"    Control:         {info['num_ctrl_tracks']}")
    print()

    ok = reader.verify_integrity()
    print(f"  CRC32 integrity:   {'PASS' if ok else 'FAIL'}")
    print()
