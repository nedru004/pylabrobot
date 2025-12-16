"""Control Hamilton STAR loading indicator LEDs based on live microphone audio.

This module provides simple "music visualizer" style light shows for the
Hamilton loading LEDs using the `set_loading_indicators` command on the
`STAR` backend. It only uses microphone audio (no files) to avoid
copyright and format issues.

Typical usage (inside an async context where you already have a connected
`STAR` backend instance)::

  from pylabrobot.liquid_handling.backends.hamilton.STAR_backend import STAR
  from pylabrobot.audio.led_music import run_led_music

  backend: STAR = ...  # your connected backend
  await run_led_music(backend, mode="scroll", duration=30.0)

The helper functions in this module do **not** create or connect a
`STAR` backend for you; they only drive LEDs on an existing, connected
backend.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from dataclasses import dataclass
from typing import Literal

try:  # optional dependency
  import numpy as np
  import sounddevice as sd
except ImportError as e:  # pragma: no cover - import error path
  _IMPORT_ERROR = e
  np = None  # type: ignore[assignment]
  sd = None  # type: ignore[assignment]


LEDMode = Literal["scroll", "random", "vu"]


def _check_audio_deps():
  if np is None or sd is None:
    raise RuntimeError(
      "Microphone LED music requires the 'numpy' and 'sounddevice' packages. "
      "Install them with: pip install numpy sounddevice"
    ) from _IMPORT_ERROR


@dataclass
class LEDMusicConfig:
  """Configuration for LED music visualization."""

  num_leds: int = 54  # Hamilton STAR loading indicators
  samplerate: int = 44100
  blocksize: int = 2048
  update_interval_s: float = 0.05  # how often we update LEDs
  mode: LEDMode = "scroll"
  sensitivity: float = (
    1.0  # global gain on audio level (0.5 = less sensitive, 2.0 = very sensitive)
  )
  # Smoothing for the level value passed from the audio callback to the
  # async loop (0 = no smoothing, 1 = very slow response).
  level_smoothing: float = 0.2


class LEDMusicController:
  """Drive Hamilton loading LEDs from microphone input.

  The controller:
  - opens a microphone input stream (using `sounddevice`)
  - computes a simple loudness estimate for each audio block
  - maps loudness to LED patterns using different visualization modes
  - sends patterns to `backend.set_loading_indicators`
  """

  def __init__(self, backend, *, config: LEDMusicConfig | None = None):
    _check_audio_deps()

    self.backend = backend
    self.config = config or LEDMusicConfig()

    self._level_lock = threading.Lock()
    self._current_level: float = 0.0
    self._running = False
    self._update_task: asyncio.Task | None = None

  # ===== Audio handling =====================================================

  def _audio_callback(self, indata, frames, time_info, status):  # pragma: no cover - real-time path
    """Called from sounddevice thread for each audio block."""
    if status:
      # We intentionally don't raise here; just ignore glitches.
      return

    if indata is None or len(indata) == 0:
      return

    # Collapse to mono and compute RMS level
    data = np.asarray(indata, dtype=np.float32)
    if data.ndim > 1:
      data = data.mean(axis=1)

    rms = float(np.sqrt(np.mean(np.square(data))) + 1e-9)

    # Map RMS into a normalized 0..1 range using a fixed reference level.
    # This avoids the AGC making everything look "medium loud" and gives
    # better separation between soft and loud.
    cfg = self.config
    ref_level = 0.2  # tilt a bit more sensitive toward softer sounds
    norm = min(1.0, rms / ref_level)

    # Apply user-configurable sensitivity as a simple linear gain.
    level = max(0.0, min(1.0, norm * cfg.sensitivity))

    with self._level_lock:
      # Smooth the public level value
      self._current_level = (
        1.0 - cfg.level_smoothing
      ) * self._current_level + cfg.level_smoothing * level

  def _get_level(self) -> float:
    with self._level_lock:
      return float(self._current_level)

  # ===== LED pattern generation ============================================

  def _make_empty_patterns(self) -> tuple[list[bool], list[bool]]:
    n = self.config.num_leds
    return [False] * n, [False] * n

  def _pattern_scroll(self, level: float) -> tuple[list[bool], list[bool]]:
    """Scroll newest audio frame in from the front.

    Interpretation:
    - LED index 0 = "front" / newest audio frame.
    - Higher indices = older frames, i.e. history scrolled across the deck.

    On each update:
    - Shift existing state toward higher indices.
    - Insert a new "front" state at index 0 based on current level.
    """

    n = self.config.num_leds

    # Initialize history on first call.
    if not hasattr(self, "_scroll_history_bits"):
      self._scroll_history_bits = [False] * n  # type: ignore[attr-defined]
      self._scroll_history_blinks = [False] * n  # type: ignore[attr-defined]

    # Shift existing history one step to the right (toward larger indices).
    bits_prev = list(self._scroll_history_bits)  # type: ignore[attr-defined]
    blinks_prev = list(self._scroll_history_blinks)  # type: ignore[attr-defined]

    bits_new = [False] * n
    blinks_new = [False] * n
    for i in range(n - 1, 0, -1):
      bits_new[i] = bits_prev[i - 1]
      blinks_new[i] = blinks_prev[i - 1]

    # Determine new "front" LED state (index 0) from current level.
    # Use probabilistic activation so soft sounds give sparse dots and
    # loud sounds give dense, continuous bands. Boost sensitivity so
    # quieter sounds still show up clearly.
    level_boost = max(0.0, min(1.0, level * 2.5))  # boost sensitivity
    p_on = level_boost  # probability of turning the new LED on
    front_on = bool(np.random.rand() < p_on)
    front_blink = False

    bits_new[0] = front_on
    blinks_new[0] = front_blink

    # Save history for next frame.
    self._scroll_history_bits = bits_new  # type: ignore[attr-defined]
    self._scroll_history_blinks = blinks_new  # type: ignore[attr-defined]

    return bits_new, blinks_new

  def _pattern_random(self, level: float) -> tuple[list[bool], list[bool]]:
    """Random twinkling pattern, density follows volume."""
    bit_pattern, blink_pattern = self._make_empty_patterns()
    n = self.config.num_leds

    # Number of active LEDs ~ volume (quieter → much fewer LEDs)
    # Use a nonlinear mapping so that near-silence is almost dark.
    if level < 0.1:
      num_on = 0
    else:
      num_on = int((level**2) * n)
      num_on = max(1, min(n, num_on))
    indices = np.random.choice(n, size=num_on, replace=False)

    for idx in indices:
      bit_pattern[int(idx)] = True
      # Do not use blink; it updates too slowly to be visually useful.
      blink_pattern[int(idx)] = False

    return bit_pattern, blink_pattern

  def _pattern_vu(self, level: float) -> tuple[list[bool], list[bool]]:
    """VU meter radiating from the center in both directions."""
    bit_pattern, blink_pattern = self._make_empty_patterns()
    n = self.config.num_leds

    # Scale so low volumes are visible but loud sounds can still hit
    # the outer LEDs. Use a slight gain to make full-scale reachable.
    level_clipped = max(0.0, min(1.0, level))
    scaled = min(1.0, level_clipped * 1.8)  # 0..1
    # We fill outward from the center: half the LEDs to each side.
    half = n // 2
    num_on_each_side = int(round(scaled * half))
    num_on_each_side = max(0, min(half, num_on_each_side))

    center_left = half - 1
    center_right = half

    for i in range(num_on_each_side):
      left_idx = center_left - i
      right_idx = center_right + i
      if 0 <= left_idx < n:
        bit_pattern[left_idx] = True
      if 0 <= right_idx < n:
        bit_pattern[right_idx] = True

    # No blink pattern; blink is too slow to see meaningful changes.

    return bit_pattern, blink_pattern

  def _make_patterns(self, level: float) -> tuple[list[bool], list[bool]]:
    mode = self.config.mode
    if mode == "scroll":
      return self._pattern_scroll(level)
    if mode == "random":
      return self._pattern_random(level)
    if mode == "vu":
      return self._pattern_vu(level)
    # Fallback to VU if unknown mode
    return self._pattern_vu(level)

  # ===== Public API ========================================================

  async def run(self, duration: float | None = None):
    """Run the LED music controller.

    Args:
      duration: If provided, run for this many seconds. If ``None``,
        run until cancelled (e.g. with Ctrl+C or by cancelling the task).
    """

    self._running = True
    cfg = self.config

    # Open microphone stream; callback runs on a separate thread.
    stream = sd.InputStream(
      channels=1,
      samplerate=cfg.samplerate,
      blocksize=cfg.blocksize,
      callback=self._audio_callback,
    )

    start = time.monotonic()

    try:
      with stream:
        while self._running:
          if duration is not None and time.monotonic() - start >= duration:
            break

          level = self._get_level()
          bit_pattern, blink_pattern = self._make_patterns(level)

          # Fire-and-forget LED updates. If a previous update is still in flight,
          # skip this one to avoid queueing up slow network calls. This keeps
          # the visualizer responsive even if the Hamilton device is slow.
          if self._update_task is None or self._update_task.done():

            async def _send_update():
              try:
                await asyncio.wait_for(
                  self.backend.set_loading_indicators(bit_pattern, blink_pattern),
                  timeout=0.5,  # timeout after 500ms to prevent hanging
                )
              except (asyncio.TimeoutError, Exception):
                # Silently ignore timeouts/errors; we'll try again next frame
                pass

            self._update_task = asyncio.create_task(_send_update())

          await asyncio.sleep(cfg.update_interval_s)
    finally:
      self._running = False
      # Wait for any in-flight update to complete
      if self._update_task is not None and not self._update_task.done():
        try:
          await asyncio.wait_for(self._update_task, timeout=1.0)
        except (asyncio.TimeoutError, Exception):
          self._update_task.cancel()

  def stop(self):
    """Request the controller to stop at the next update tick."""
    self._running = False


async def run_led_music(
  backend,
  *,
  mode: LEDMode = "scroll",
  duration: float | None = None,
  samplerate: int = 44100,
  blocksize: int = 2048,
  update_interval_s: float = 0.05,
  sensitivity: float = 1.0,
) -> None:
  """Convenience helper to run LED music from a microphone.

  Args:
    backend: A connected Hamilton `STAR` backend instance that provides
      :meth:`set_loading_indicators`.
    mode: Visualization mode. One of:
      - ``"scroll"``: scrolling dot with tail, speed and tail follow volume.
      - ``"random"``: random twinkling, density and blinking follow volume.
      - ``"vu"``: simple left-to-right VU meter.
    duration: Number of seconds to run. If ``None``, run until cancelled.
    samplerate: Microphone sampling rate in Hz.
    blocksize: Number of samples per audio block.
    update_interval_s: How often to update LED patterns. If updates feel slow,
      try reducing this (e.g., 0.02 for 50 Hz). Note: very fast rates may be
      limited by network latency to the Hamilton device.
    sensitivity: Global audio sensitivity (0.5 = less sensitive, 2.0 = very
      sensitive).
  """

  cfg = LEDMusicConfig(
    samplerate=samplerate,
    blocksize=blocksize,
    update_interval_s=update_interval_s,
    mode=mode,
    sensitivity=sensitivity,
  )
  controller = LEDMusicController(backend, config=cfg)
  await controller.run(duration=duration)
