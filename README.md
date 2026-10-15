# VHS Tape Emulator

Physically accurate **VHS recording and playback** with real magnetic physics,
helical-scan drum mechanics, internal analog circuitry and time-based tape
degradation. Video is written to a self-describing `.vhs` tape file and read
back through the same signal chain, so round-trips reproduce dropouts,
head-switching noise, chroma loss and generation loss.

## Install & run
```sh
pip install -r requirements.txt          # numpy, scipy, Pillow, mss

# record a test pattern to a tape, then play it back
python main.py record --output my_tape.vhs --pattern colorbars --duration 10
python main.py play  my_tape.vhs --window
python main.py demo  --pattern colorbars --duration 5      # record + play
python main.py info  my_tape.vhs

# aged / worn tape
python main.py record --output old_tape.vhs --pattern gradient --duration 10 \
    --tape-age 20 --play-count 500 --storage-temp 35
```

## Layout
- `main.py` — CLI
- `vhs/` — `emulator.py`, `magnetic.py`, `head.py`, `tape.py`,
  `encode.py`, `decode.py`, `circuits.py`, `degradation.py`, `file_format.py`
- `test_*.py` — round-trip and diagnostic checks

## License
MIT
