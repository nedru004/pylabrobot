# Audio Features

This folder contains audio-related functionality for PyLabRobot.

## LED Music Visualizer

The `led_music.py` module provides functionality to synchronize Hamilton STAR loading indicator LEDs with music. The visualizer analyzes audio in real-time using FFT (Fast Fourier Transform) and maps frequency bands to the 54 LEDs on the Hamilton STAR.

### Features

- **Real-time audio analysis**: Uses FFT to analyze audio frequency spectrum
- **Frequency band mapping**: Maps 54 frequency bands (20 Hz - 20 kHz) to LEDs
- **Blinking patterns**: Optional blinking for higher frequencies
- **Multiple input sources**: Supports microphone input or audio file playback
- **Configurable sensitivity**: Adjustable threshold for LED activation

### Requirements

**For microphone input:**
```bash
pip install numpy pyaudio
```

**For audio file playback:**
```bash
pip install numpy pydub  # or scipy for WAV files only
```

**For playing audio through speakers:**
```bash
pip install pygame
```

**Optional but recommended:**
```bash
pip install scipy  # Better audio processing and resampling
```

- `numpy`: Required for FFT analysis
- `pyaudio`: Required for microphone audio capture
- `pydub`: Recommended for audio file loading (supports mp3, wav, etc.)
- `scipy`: Alternative for WAV files, also provides better resampling
- `pygame`: Optional, for playing audio through speakers while visualizing

### Usage

#### From Python code

```python
import asyncio
from pylabrobot.liquid_handling.backends.hamilton import STAR
from pylabrobot.audio.led_music import visualize_music_from_file

async def main():
    backend = STAR()
    await backend.setup()

    # Visualize music from file
    await visualize_music_from_file(backend, "song.mp3", sensitivity=0.5)

    await backend.stop()

asyncio.run(main())
```

#### From command line

```bash
# From microphone
python -m pylabrobot.audio.example_led_music --microphone

# From audio file
python -m pylabrobot.audio.example_led_music --file song.mp3

# With custom sensitivity
python -m pylabrobot.audio.example_led_music --file song.mp3 --sensitivity 0.7

# Disable blinking
python -m pylabrobot.audio.example_led_music --file song.mp3 --no-blink
```

#### Advanced usage

```python
from pylabrobot.audio.led_music import LEDMusicVisualizer

visualizer = LEDMusicVisualizer(
    backend,
    sample_rate=44100,
    chunk_size=4096,
    update_rate=30.0,  # 30 updates per second
    sensitivity=0.6,   # Higher = more sensitive
    use_blink=True,    # Use blinking for high frequencies
)

# Start from microphone
await visualizer.start_from_microphone()

# Or from file
await visualizer.start_from_file("song.mp3")

# Stop when done
await visualizer.stop()
```

### How it works

1. **Audio capture**: Audio is captured from microphone or played from file
2. **FFT analysis**: Each audio chunk is analyzed using FFT to get frequency spectrum
3. **Frequency band mapping**: The spectrum is divided into 54 bands (logarithmically spaced)
4. **LED activation**: LEDs are activated based on the magnitude in each frequency band
5. **Real-time updates**: LEDs are updated at a configurable rate (default: 30 Hz)

### Parameters

- `sensitivity` (0.0-1.0): Threshold for LED activation. Higher values make LEDs more responsive
- `use_blink` (bool): If True, higher frequency LEDs will blink for dynamic effect
- `update_rate` (float): How many times per second to update LEDs (default: 30.0)
- `sample_rate` (int): Audio sample rate in Hz (default: 44100)
- `chunk_size` (int): Number of audio samples per chunk (default: 4096)

### Notes

- The visualizer uses 54 LEDs corresponding to the Hamilton STAR loading indicators
- Frequency bands are logarithmically spaced from 20 Hz to 20 kHz
- On macOS, capturing system audio (for file playback) may require additional setup
- The visualizer gracefully handles missing optional dependencies
