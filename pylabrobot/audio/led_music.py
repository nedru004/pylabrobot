"""LED music visualizer for Hamilton STAR liquid handlers.

This module synchronizes the Hamilton loading indicator LEDs with music by analyzing
audio in real-time and mapping frequency bands to LED patterns.
"""

import asyncio
import logging
import threading
from typing import List, Optional

try:
  import numpy as np

  NUMPY_AVAILABLE = True
except ImportError:
  NUMPY_AVAILABLE = False
  np = None  # type: ignore

try:
  import pyaudio

  PYAUDIO_AVAILABLE = True
except ImportError:
  PYAUDIO_AVAILABLE = False
  pyaudio = None  # type: ignore

try:
  from scipy import signal

  SCIPY_AVAILABLE = True
except ImportError:
  SCIPY_AVAILABLE = False
  signal = None  # type: ignore

try:
  import pygame

  PYGAME_AVAILABLE = True
except ImportError:
  PYGAME_AVAILABLE = False
  pygame = None  # type: ignore

logger = logging.getLogger(__name__)


class LEDMusicVisualizer:
  """Visualizes music on Hamilton STAR loading indicator LEDs."""

  def __init__(
    self,
    backend,
    sample_rate: int = 44100,
    chunk_size: int = 4096,
    update_rate: float = 30.0,  # Updates per second
    sensitivity: float = 0.5,  # 0.0 to 1.0, higher = more sensitive
    use_blink: bool = True,  # Whether to use blinking pattern
  ):
    """Initialize the LED music visualizer.

    Args:
        backend: Hamilton STAR backend instance with set_loading_indicators method
        sample_rate: Audio sample rate in Hz
        chunk_size: Number of audio samples per chunk
        update_rate: How many times per second to update LEDs
        sensitivity: Sensitivity threshold (0.0-1.0), higher values make LEDs more responsive
        use_blink: If True, use blinking pattern for higher frequencies
    """
    if not NUMPY_AVAILABLE:
      raise RuntimeError("numpy is required for LED music visualization")
    if not PYAUDIO_AVAILABLE:
      raise RuntimeError("pyaudio is required for LED music visualization")

    self.backend = backend
    self.sample_rate = sample_rate
    self.chunk_size = chunk_size
    self.update_rate = update_rate
    self.sensitivity = max(0.0, min(1.0, sensitivity))
    self.use_blink = use_blink

    self._running = False
    self._audio_thread: Optional[threading.Thread] = None
    self._update_task: Optional[asyncio.Task] = None
    self._audio_queue: asyncio.Queue = asyncio.Queue()
    self._num_leds = 54

    # Frequency bands: divide audio spectrum into 54 bands
    # Using logarithmic spacing for better visual representation
    self._freq_bands = self._create_frequency_bands()

  def _create_frequency_bands(self) -> List[tuple]:
    """Create frequency bands for LED mapping.

    Returns:
        List of (low_freq, high_freq) tuples for each LED
    """
    # Human hearing range: ~20 Hz to 20 kHz
    # Use logarithmic spacing for better visual representation
    min_freq = 20.0
    max_freq = 20000.0

    bands = []
    for i in range(self._num_leds):
      # Logarithmic spacing
      low = min_freq * (max_freq / min_freq) ** (i / self._num_leds)
      high = min_freq * (max_freq / min_freq) ** ((i + 1) / self._num_leds)
      bands.append((low, high))

    return bands

  def _analyze_audio_chunk(self, audio_data: np.ndarray) -> tuple[List[bool], List[bool]]:
    """Analyze audio chunk and return LED patterns.

    Args:
        audio_data: Audio samples as numpy array

    Returns:
        Tuple of (bit_pattern, blink_pattern) lists, each of length 54
    """
    # Compute FFT
    fft = np.fft.rfft(audio_data)
    fft_magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(len(audio_data), 1.0 / self.sample_rate)

    # Normalize magnitude
    if np.max(fft_magnitude) > 0:
      fft_magnitude = fft_magnitude / np.max(fft_magnitude)

    # Map frequency bands to LEDs
    bit_pattern = [False] * self._num_leds
    blink_pattern = [False] * self._num_leds

    for i, (low_freq, high_freq) in enumerate(self._freq_bands):
      # Find frequencies in this band
      mask = (freqs >= low_freq) & (freqs < high_freq)
      if np.any(mask):
        # Get average magnitude in this band
        band_magnitude = np.mean(fft_magnitude[mask])

        # Apply sensitivity threshold
        threshold = 1.0 - self.sensitivity
        if band_magnitude > threshold:
          bit_pattern[i] = True

          # Use blinking for higher frequencies (more dynamic)
          if self.use_blink and i > self._num_leds // 2:
            # Blink for upper half of LEDs (higher frequencies)
            blink_pattern[i] = True

    return bit_pattern, blink_pattern

  async def _update_leds_loop(self):
    """Main loop for updating LEDs based on audio analysis."""
    while self._running:
      try:
        # Get audio chunk from queue (with timeout)
        try:
          audio_data = await asyncio.wait_for(
            self._audio_queue.get(), timeout=1.0 / self.update_rate
          )
        except asyncio.TimeoutError:
          # No audio data, turn off all LEDs
          bit_pattern = [False] * self._num_leds
          blink_pattern = [False] * self._num_leds
        else:
          # Analyze audio and get LED patterns
          bit_pattern, blink_pattern = self._analyze_audio_chunk(audio_data)

        # Update LEDs
        await self.backend.set_loading_indicators(bit_pattern, blink_pattern)

        # Sleep to maintain update rate
        await asyncio.sleep(1.0 / self.update_rate)

      except Exception as e:
        logger.error(f"Error in LED update loop: {e}", exc_info=True)
        await asyncio.sleep(0.1)

  def _audio_capture_loop(self, stream):
    """Capture audio from stream and put chunks in queue."""
    try:
      while self._running:
        try:
          # Read audio data
          data = stream.read(self.chunk_size, exception_on_overflow=False)
          audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32)
          audio_array = audio_array / 32768.0  # Normalize to [-1, 1]

          # Put in queue (non-blocking)
          try:
            self._audio_queue.put_nowait(audio_array)
          except asyncio.QueueFull:
            # Drop oldest if queue is full
            try:
              self._audio_queue.get_nowait()
              self._audio_queue.put_nowait(audio_array)
            except asyncio.QueueEmpty:
              pass

        except Exception as e:
          if self._running:
            logger.error(f"Error reading audio stream: {e}", exc_info=True)
          break
    finally:
      stream.stop_stream()
      stream.close()

  async def start_from_microphone(self):
    """Start visualization using microphone input.

    This will capture audio from the default microphone and visualize it.
    """
    if not self._running:
      self._running = True

      # Initialize PyAudio
      p = pyaudio.PyAudio()

      # Open audio stream
      stream = p.open(
        format=pyaudio.paInt16,
        channels=1,  # Mono
        rate=self.sample_rate,
        input=True,
        frames_per_buffer=self.chunk_size,
      )

      # Start audio capture thread
      self._audio_thread = threading.Thread(
        target=self._audio_capture_loop, args=(stream,), daemon=True
      )
      self._audio_thread.start()

      # Start LED update task
      self._update_task = asyncio.create_task(self._update_leds_loop())

      logger.info("LED music visualizer started (microphone input)")

  async def start_from_file(self, audio_file: str, play_audio: bool = True):
    """Start visualization using audio file.

    Args:
        audio_file: Path to audio file (mp3, wav, etc.)
        play_audio: If True, also play the audio through speakers (requires pygame)
    """
    if not self._running:
      self._running = True

      # Play audio if requested
      if play_audio:
        if not PYGAME_AVAILABLE:
          logger.warning("pygame not available, audio will not be played")
          play_audio = False
        else:
          pygame.mixer.init(
            frequency=self.sample_rate, size=-16, channels=1, buffer=self.chunk_size
          )
          pygame.mixer.music.load(audio_file)
          pygame.mixer.music.play()

      # Analyze file directly (more reliable than capturing system audio)
      await self._start_from_file_direct(audio_file, play_audio)

  async def _start_from_file_direct(self, audio_file: str, play_audio: bool = False):
    """Analyze audio file directly and visualize on LEDs.

    Supports various audio formats through pydub or soundfile.
    """
    try:
      # Try to load audio using different libraries
      data = None
      rate = self.sample_rate

      # Method 1: Try pydub (supports many formats including mp3)
      try:
        from pydub import AudioSegment

        audio = AudioSegment.from_file(audio_file)
        # Convert to numpy array
        data = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if audio.channels == 2:
          # Convert stereo to mono
          data = data.reshape(-1, 2).mean(axis=1)
        else:
          data = data.reshape(-1)
        # Normalize to [-1, 1]
        if audio.sample_width == 1:
          data = data / 128.0 - 1.0
        elif audio.sample_width == 2:
          data = data / 32768.0
        elif audio.sample_width == 4:
          data = data / 2147483648.0
        rate = audio.frame_rate
        logger.info(f"Loaded audio file using pydub: {rate} Hz, {len(data)} samples")
      except ImportError:
        logger.debug("pydub not available, trying scipy.io.wavfile")
      except Exception as e:
        logger.debug(f"pydub failed: {e}, trying scipy.io.wavfile")

      # Method 2: Try scipy.io.wavfile (WAV only)
      if data is None:
        try:
          from scipy.io import wavfile

          rate, data = wavfile.read(audio_file)

          # Convert to mono if stereo
          if len(data.shape) > 1:
            data = np.mean(data, axis=1)

          # Convert to float
          if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
          elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
          elif data.dtype == np.uint8:
            data = data.astype(np.float32) / 128.0 - 1.0
          else:
            data = data.astype(np.float32)
            if np.max(np.abs(data)) > 1.0:
              data = data / np.max(np.abs(data))

          logger.info(f"Loaded audio file using scipy: {rate} Hz, {len(data)} samples")
        except ImportError:
          logger.error("Neither pydub nor scipy available. Cannot load audio file.")
          raise RuntimeError(
            "Audio file loading requires either pydub or scipy. "
            "Install with: pip install pydub scipy"
          )
        except Exception as e:
          logger.error(f"Failed to load audio file: {e}")
          raise

      # Resample if needed
      if rate != self.sample_rate:
        if SCIPY_AVAILABLE:
          num_samples = int(len(data) * self.sample_rate / rate)
          data = signal.resample(data, num_samples)
          logger.info(f"Resampled from {rate} Hz to {self.sample_rate} Hz")
        else:
          logger.warning(
            "scipy not available, cannot resample audio. " "Audio may not sync correctly with LEDs."
          )

      # Process in chunks
      chunk_samples = int(self.sample_rate / self.update_rate)
      logger.info(f"Processing audio: {len(data)} samples in chunks of {chunk_samples}")

      for i in range(0, len(data), chunk_samples):
        if not self._running:
          break

        # Check if audio is still playing (if we're playing it)
        if play_audio and PYGAME_AVAILABLE:
          if not pygame.mixer.music.get_busy():
            break

        chunk = data[i : i + chunk_samples]
        if len(chunk) < chunk_samples:
          # Pad last chunk with zeros
          chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        # Analyze and update LEDs
        bit_pattern, blink_pattern = self._analyze_audio_chunk(chunk)
        await self.backend.set_loading_indicators(bit_pattern, blink_pattern)

        await asyncio.sleep(1.0 / self.update_rate)

      logger.info("Finished processing audio file")

    except Exception as e:
      logger.error(f"Error in direct file analysis: {e}", exc_info=True)
    finally:
      await self.stop()

  async def stop(self):
    """Stop the visualization."""
    if self._running:
      self._running = False

      # Stop pygame music if playing
      if PYGAME_AVAILABLE:
        try:
          pygame.mixer.music.stop()
        except Exception:
          pass

      # Wait for threads to finish
      if self._audio_thread is not None:
        self._audio_thread.join(timeout=1.0)

      # Cancel update task
      if self._update_task is not None:
        self._update_task.cancel()
        try:
          await self._update_task
        except asyncio.CancelledError:
          pass

      # Turn off all LEDs
      try:
        bit_pattern = [False] * self._num_leds
        blink_pattern = [False] * self._num_leds
        await self.backend.set_loading_indicators(bit_pattern, blink_pattern)
      except Exception as e:
        logger.warning(f"Error turning off LEDs: {e}")

      logger.info("LED music visualizer stopped")


async def visualize_music_from_file(
  backend,
  audio_file: str,
  sensitivity: float = 0.5,
  use_blink: bool = True,
  play_audio: bool = True,
):
  """Convenience function to visualize music from a file.

  Args:
      backend: Hamilton STAR backend instance
      audio_file: Path to audio file (mp3, wav, etc.)
      sensitivity: Sensitivity threshold (0.0-1.0)
      use_blink: Whether to use blinking pattern
      play_audio: If True, also play audio through speakers (requires pygame)

  Example:
      >>> from pylabrobot.liquid_handling import LiquidHandler
      >>> from pylabrobot.liquid_handling.backends.hamilton import STAR
      >>> from pylabrobot.audio.led_music import visualize_music_from_file
      >>>
      >>> backend = STAR()
      >>> await backend.setup()
      >>> await visualize_music_from_file(backend, "song.mp3")
  """
  visualizer = LEDMusicVisualizer(backend, sensitivity=sensitivity, use_blink=use_blink)
  await visualizer.start_from_file(audio_file, play_audio=play_audio)


async def visualize_music_from_microphone(
  backend, sensitivity: float = 0.5, use_blink: bool = True
):
  """Convenience function to visualize music from microphone.

  Args:
      backend: Hamilton STAR backend instance
      sensitivity: Sensitivity threshold (0.0-1.0)
      use_blink: Whether to use blinking pattern

  Example:
      >>> from pylabrobot.liquid_handling import LiquidHandler
      >>> from pylabrobot.liquid_handling.backends.hamilton import STAR
      >>> from pylabrobot.audio.led_music import visualize_music_from_microphone
      >>>
      >>> backend = STAR()
      >>> await backend.setup()
      >>> await visualize_music_from_microphone(backend)
  """
  visualizer = LEDMusicVisualizer(backend, sensitivity=sensitivity, use_blink=use_blink)
  await visualizer.start_from_microphone()
