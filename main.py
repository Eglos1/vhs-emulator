#!/usr/bin/env python3
"""
VHS Tape Emulator — Command-Line Interface

Physically accurate VHS recording and playback with real magnetic physics,
helical scan drum mechanics, internal analog circuitry, and tape degradation.

Usage:
    # Record a test pattern to .vhs file
    python main.py record --output my_tape.vhs --pattern colorbars --duration 10

    # Record screen capture to .vhs file
    python main.py record --output screen.vhs --screen --duration 30

    # Play back a .vhs file
    python main.py play my_tape.vhs

    # Play back to window
    python main.py play my_tape.vhs --window

    # Record and immediately play (round-trip test)
    python main.py demo --pattern colorbars --duration 5

    # Show .vhs file info
    python main.py info my_tape.vhs

    # Apply tape aging effects
    python main.py record --output old_tape.vhs --pattern gradient --duration 10 \\
                          --tape-age 20 --play-count 500 --storage-temp 35
"""

import argparse
import sys
import time
import numpy as np


def cmd_record(args):
    """Record video to a .vhs file."""
    from vhs.emulator import VHSTapeEmulator
    from vhs.degradation import TapeDegradation

    # Build degradation model
    deg = TapeDegradation(
        tape_age_years=args.tape_age,
        play_count=args.play_count,
        storage_temp_c=args.storage_temp,
        storage_humidity_pct=args.storage_humidity,
    )

    emu = VHSTapeEmulator(
        standard=args.standard,
        speed=args.speed,
        hifi_audio=args.hifi,
        tape_type=args.tape_type,
        coating=args.coating,
        degradation=deg,
        workers=args.workers,
    )

    print(f"\n{'='*60}")
    print(f"  VHS Tape Emulator — Record Mode")
    print(f"{'='*60}")
    info = emu.info()
    for k, v in info.items():
        print(f"  {k:25s}: {v}")
    print(f"{'='*60}\n")

    if args.screen:
        _record_screen(emu, args)
    else:
        _record_test_pattern(emu, args)


def _record_test_pattern(emu, args):
    """Record a test pattern using the parallel pipeline."""
    print(f"[Record] Pattern: {args.pattern}")
    print(f"[Record] Duration: {args.duration}s, Standard: {args.standard}, Speed: {args.speed}")
    print(f"[Record] Using {emu.num_workers} parallel threads")

    filepath = emu.record_test_pattern(args.output, args.pattern, args.duration)
    print(f"\n[Record] Done! File: {filepath}")

    # Show file info
    from vhs.file_format import print_info
    print(f"\n{'='*60}")
    print_info(filepath)


def _record_screen(emu, args):
    """Record screen capture to .vhs file."""
    try:
        import mss
    except ImportError:
        print("[Error] Screen capture requires 'mss' package: pip install mss")
        sys.exit(1)

    print(f"[Record] Screen capture mode")
    print(f"[Record] Press Ctrl+C to stop recording")

    field_rate = 60.0 if args.standard == 'NTSC' else 50.0
    field_duration = 1.0 / field_rate
    target_dt = field_duration

    emu.record_start(args.duration)
    sct = mss.mss()
    monitor = sct.monitors[1]  # Primary monitor

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            t0 = time.time()
            elapsed = t0 - start_time

            if elapsed >= args.duration:
                break

            # Capture screen
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)[:, :, :3]  # Drop alpha, RGB
            img = img.astype(np.float64) / 255.0

            # Convert to grayscale composite-like signal
            gray = np.mean(img, axis=2)

            # Resize to one scanline's worth of samples
            n_samples = int(emu.sample_rate * field_duration)
            # Use first row as a simple scanline signal (proper NTSC would
            # modulate all rows, but for demo purposes this works)
            scanline = gray[0, :]  # Take first row
            # Resample to our sample rate
            from PIL import Image
            pil_img = Image.fromarray((gray * 255).astype(np.uint8))
            pil_img = pil_img.resize((n_samples // 10, gray.shape[0] // 10))
            field_data = np.array(pil_img).astype(np.float64) / 255.0
            # Flatten to 1D composite signal (simple luma-only representation)
            composite = field_data.flatten()[:n_samples]
            if len(composite) < n_samples:
                composite = np.pad(composite, (0, n_samples - len(composite)))

            emu.feed_field(composite)
            frame_count += 1

            # Progress
            pct = elapsed / args.duration * 100
            print(f"\r  [{('#' * int(pct/2)).ljust(50)}] "
                  f"{pct:5.1f}% — {elapsed:.0f}/{args.duration:.0f}s "
                  f"({frame_count} fields)", end='', flush=True)

            # Frame timing
            dt = time.time() - t0
            if dt < target_dt:
                time.sleep(target_dt - dt)

    except KeyboardInterrupt:
        print("\n[Record] Interrupted by user")

    print()
    filepath = emu.record_stop(args.output)
    print(f"\n[Record] Done! File: {filepath}")


def cmd_play(args):
    """Play back a .vhs file."""
    from vhs.emulator import VHSTapeEmulator
    from vhs.degradation import TapeDegradation

    emu = VHSTapeEmulator()
    result = emu.play(args.input)

    header = result.get('header')
    video_frames = result.get('video_frames', [])
    audio_samples = result.get('audio_samples', np.array([]))
    diagnostics = result.get('diagnostics', [])

    print(f"\n{'='*60}")
    print(f"  VHS Playback Results")
    print(f"{'='*60}")
    if header:
        print(f"  Standard:  {header.video_standard}")
        print(f"  Speed:     {header.speed_mode}")
        print(f"  Duration:  {header.duration_seconds:.1f}s")
        print(f"  Tape:      {header.tape_label}")
    print(f"  Fields:    {len(video_frames)}")
    print(f"  Audio:     {len(audio_samples)} samples")
    print(f"{'='*60}")

    if diagnostics:
        snrs = [d.get('snr_db', 0) for d in diagnostics if 'snr_db' in d]
        if snrs:
            print(f"  Avg SNR:   {np.mean(snrs):.1f} dB")
        dropouts = [d.get('head_switch_samples', 0) for d in diagnostics]
        print(f"  Head switch glitches: {sum(dropouts)}")

    if args.window:
        _play_to_window(video_frames, audio_samples, header)
    elif args.output_video:
        _save_video(video_frames, args.output_video)
    else:
        print(f"\n  Use --window to display, or --output-video to save as AVI")


def _play_to_window(video_frames, audio_samples, header):
    """Display decoded video in a window using OpenGL."""
    try:
        import moderngl
        import glfw
    except ImportError:
        print("[Error] Window display requires: pip install moderngl glfw")
        sys.exit(1)

    print(f"[Play] Opening display window... (press ESC to close)")

    if not glfw.init():
        print("[Error] Could not initialize GLFW")
        sys.exit(1)

    field_rate = 60.0 if (header and header.video_standard == 'NTSC') else 50.0
    width, height = 720, 480  # Approximate VHS resolution

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(width, height, "VHS Playback", None, None)
    glfw.make_context_current(window)

    ctx = moderngl.create_context()
    prog = ctx.program(
        vertex_shader="""
        #version 330
        in vec2 in_pos;
        in vec2 in_uv;
        out vec2 uv;
        void main() {
            gl_Position = vec4(in_pos, 0.0, 1.0);
            uv = in_uv;
        }
        """,
        fragment_shader="""
        #version 330
        uniform sampler2D tex;
        in vec2 uv;
        out vec4 fragColor;
        void main() {
            fragColor = texture(tex, uv);
        }
        """,
    )

    vbo = ctx.buffer(
        np.array([
            -1, -1, 0, 1,
             1, -1, 1, 1,
            -1,  1, 0, 0,
             1,  1, 1, 0,
        ], dtype='f4')
    )
    vao = ctx.simple_vertex_array(prog, vbo, 'in_pos', 'in_uv')

    frame_idx = 0
    last_time = time.time()

    while not glfw.window_should_close(window):
        glfw.poll_events()

        if glfw.get_key(window, glfw.KEY_ESCAPE):
            break

        # Advance frame
        now = time.time()
        if now - last_time >= 1.0 / field_rate:
            frame_idx = (frame_idx + 1) % max(1, len(video_frames))
            last_time = now

        if video_frames:
            frame = video_frames[frame_idx % len(video_frames)]
            # Normalize to 0-1
            frame = np.clip(frame, 0, 1)
            # Handle 2D/3D raster frames
            if frame.ndim == 3 and frame.shape[2] == 3:
                # RGB frame — scale to window size
                from PIL import Image
                rgb_u8 = (frame * 255).astype(np.uint8)
                pil = Image.fromarray(rgb_u8, mode='RGB')
                pil = pil.resize((width, height), Image.NEAREST)
                tex_data = np.array(pil)
            elif frame.ndim == 2:
                h, w = frame.shape
                # Grayscale — scale and convert to RGB
                from PIL import Image
                gray_u8 = (frame * 255).astype(np.uint8)
                pil = Image.fromarray(gray_u8, mode='L')
                pil = pil.resize((width, height), Image.NEAREST)
                gray_scaled = np.array(pil)
                tex_data = np.stack([gray_scaled, gray_scaled, gray_scaled], axis=-1)
            elif frame.ndim == 1:
                gray = np.tile(frame[:width], (height, 1)).astype(np.uint8)
                tex_data = np.stack([gray, gray, gray], axis=-1)
            else:
                tex_data = np.zeros((height, width, 3), dtype=np.uint8)

            texture = ctx.texture((width, height), 3, tex_data.tobytes())
            texture.use()
            vao.render(moderngl.TRIANGLE_STRIP)
            texture.release()
        else:
            # Black frame
            tex_data = np.zeros((height, width, 3), dtype=np.uint8)
            texture = ctx.texture((width, height), 3, tex_data.tobytes())
            texture.use()
            vao.render(moderngl.TRIANGLE_STRIP)
            texture.release()

        glfw.swap_buffers(window)

    glfw.terminate()
    print("[Play] Window closed.")


def _save_video(video_frames, filepath):
    """Save decoded frames as a series of images or numpy archive."""
    if filepath.endswith('.npz'):
        # Save as numpy archive
        data = {f'field_{i}': f for i, f in enumerate(video_frames)}
        np.savez_compressed(filepath, **data)
        print(f"[Play] Saved {len(video_frames)} fields to {filepath}")
    else:
        # Save as images
        from PIL import Image
        import os
        os.makedirs(filepath, exist_ok=True)
        for i, frame in enumerate(video_frames):
            frame_u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            if frame_u8.ndim == 1:
                frame_u8 = np.tile(frame_u8, (480, 1))[:480, :720]
            img = Image.fromarray(frame_u8, mode='L')
            img.save(os.path.join(filepath, f'frame_{i:06d}.png'))
        print(f"[Play] Saved {len(video_frames)} frames to {filepath}/")


def cmd_info(args):
    """Show info about a .vhs file."""
    from vhs.file_format import print_info
    print_info(args.input)


def cmd_demo(args):
    """Record and immediately play back (round-trip test)."""
    from vhs.emulator import VHSTapeEmulator
    from vhs.degradation import TapeDegradation

    print(f"\n{'='*60}")
    print(f"  VHS Tape Emulator — Demo Mode")
    print(f"  Record → .vhs → Playback")
    print(f"{'='*60}\n")

    # Phase 1: Record
    deg = TapeDegradation(
        tape_age_years=args.tape_age,
        play_count=args.play_count,
        storage_temp_c=args.storage_temp,
    )
    emu = VHSTapeEmulator(
        standard=args.standard,
        speed=args.speed,
        coating=args.coating,
        degradation=deg,
    )

    tmp_file = '/tmp/vhs_demo_tape.vhs'
    emu.record_test_pattern(tmp_file, args.pattern, args.duration)

    # Phase 2: Play
    print(f"\n{'─'*60}")
    print(f"  Now playing back...")
    print(f"{'─'*60}\n")

    emu2 = VHSTapeEmulator()
    result = emu2.play(tmp_file)

    video = result.get('video_frames', [])
    print(f"\n  Decoded {len(video)} fields")
    if video:
        print(f"  Frame shape: {video[0].shape}")
        print(f"  Frame range: [{video[0].min():.3f}, {video[0].max():.3f}]")

    diagnostics = result.get('diagnostics', [])
    if diagnostics:
        snrs = [d.get('snr_db', 0) for d in diagnostics if 'snr_db' in diagnostics]
        if snrs:
            print(f"  Average SNR: {np.mean(snrs):.1f} dB")

    print(f"\n  Demo complete! File: {tmp_file}")


def main():
    parser = argparse.ArgumentParser(
        description='VHS Tape Emulator — Physically accurate magnetic recording',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Record colorbars:     python main.py record -o tape.vhs -p colorbars -d 10
  Record screen:        python main.py record -o screen.vhs --screen -d 30
  Play back:            python main.py play tape.vhs --window
  Show file info:       python main.py info tape.vhs
  Round-trip demo:      python main.py demo -p gradient -d 5
  Aged tape:            python main.py record -o old.vhs -p colorbars --tape-age 20
        """
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # ── record ────────────────────────────────────────────────────────
    rec = subparsers.add_parser('record', help='Record video to .vhs file')
    rec.add_argument('-o', '--output', default='output.vhs', help='Output .vhs file path')
    rec.add_argument('-p', '--pattern', default='colorbars',
                     choices=['colorbars', 'gradient', 'pluge', 'multiburst'],
                     help='Test pattern (default: colorbars)')
    rec.add_argument('-d', '--duration', type=float, default=5.0, help='Duration in seconds')
    rec.add_argument('-s', '--standard', default='NTSC', choices=['NTSC', 'PAL'])
    rec.add_argument('--speed', default='SP', choices=['SP', 'LP', 'EP'])
    rec.add_argument('--tape-type', default='T120', choices=['T60', 'T120', 'T160'])
    rec.add_argument('--coating', default='ferric', choices=['ferric', 'chrome', 'metal'])
    rec.add_argument('--hifi', action='store_true', help='Use Hi-Fi stereo audio')
    rec.add_argument('--screen', action='store_true', help='Record screen capture instead')
    rec.add_argument('--tape-age', type=float, default=0, help='Simulate tape age in years')
    rec.add_argument('--play-count', type=int, default=0, help='Simulate number of plays')
    rec.add_argument('--storage-temp', type=float, default=22.0, help='Storage temperature (°C)')
    rec.add_argument('--storage-humidity', type=float, default=40.0, help='Storage humidity (%)')
    rec.add_argument('--workers', type=int, default=0, help='Parallel threads (0=auto, all cores)')

    # ── play ──────────────────────────────────────────────────────────
    play = subparsers.add_parser('play', help='Play back a .vhs file')
    play.add_argument('input', help='Input .vhs file')
    play.add_argument('--window', action='store_true', help='Display in window')
    play.add_argument('--output-video', '--save', help='Save decoded frames (dir or .npz)')

    # ── info ──────────────────────────────────────────────────────────
    info = subparsers.add_parser('info', help='Show .vhs file info')
    info.add_argument('input', help='Input .vhs file')

    # ── demo ──────────────────────────────────────────────────────────
    demo = subparsers.add_parser('demo', help='Record and play back (round-trip test)')
    demo.add_argument('-p', '--pattern', default='colorbars',
                      choices=['colorbars', 'gradient', 'pluge', 'multiburst'])
    demo.add_argument('-d', '--duration', type=float, default=5.0)
    demo.add_argument('-s', '--standard', default='NTSC', choices=['NTSC', 'PAL'])
    demo.add_argument('--speed', default='SP', choices=['SP', 'LP', 'EP'])
    demo.add_argument('--coating', default='ferric', choices=['ferric', 'chrome', 'metal'])
    demo.add_argument('--tape-age', type=float, default=0)
    demo.add_argument('--play-count', type=int, default=0)
    demo.add_argument('--storage-temp', type=float, default=22.0)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    cmd_map = {
        'record': cmd_record,
        'play': cmd_play,
        'info': cmd_info,
        'demo': cmd_demo,
    }
    cmd_map[args.command](args)


if __name__ == '__main__':
    main()
