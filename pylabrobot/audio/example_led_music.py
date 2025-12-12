"""Example script for LED music visualization on Hamilton STAR.

This script demonstrates how to use the LED music visualizer to sync
Hamilton loading indicator LEDs with music.

Requirements:
    - numpy
    - pyaudio
    - pygame (for file playback)
    - scipy (optional, for better audio processing)

Install dependencies:
    pip install numpy pyaudio pygame scipy

Usage:
    # From microphone
    python -m pylabrobot.audio.example_led_music --microphone

    # From audio file
    python -m pylabrobot.audio.example_led_music --file song.mp3
"""

import argparse
import asyncio
import logging
import sys

# Setup logging
logging.basicConfig(
  level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
  """Main entry point."""
  parser = argparse.ArgumentParser(description="LED music visualizer for Hamilton STAR")
  parser.add_argument(
    "--file",
    type=str,
    help="Path to audio file (mp3, wav, etc.)",
  )
  parser.add_argument(
    "--microphone",
    action="store_true",
    help="Use microphone input instead of file",
  )
  parser.add_argument(
    "--sensitivity",
    type=float,
    default=0.5,
    help="Sensitivity threshold (0.0-1.0, default: 0.5)",
  )
  parser.add_argument(
    "--no-blink",
    action="store_true",
    help="Disable blinking pattern",
  )
  parser.add_argument(
    "--no-play",
    action="store_true",
    help="Don't play audio through speakers (LEDs only)",
  )
  parser.add_argument(
    "--backend-type",
    type=str,
    default="STAR",
    help="Backend type (default: STAR)",
  )

  args = parser.parse_args()

  if not args.file and not args.microphone:
    parser.error("Either --file or --microphone must be specified")

  # Import backend
  try:
    if args.backend_type == "STAR":
      from pylabrobot.liquid_handling.backends.hamilton import STAR

      backend = STAR()
    else:
      logger.error(f"Unknown backend type: {args.backend_type}")
      sys.exit(1)
  except ImportError as e:
    logger.error(f"Failed to import backend: {e}")
    logger.error("Make sure pylabrobot is properly installed")
    sys.exit(1)

  # Import visualizer
  try:
    from pylabrobot.audio.led_music import (
      visualize_music_from_file,
      visualize_music_from_microphone,
    )
  except ImportError as e:
    logger.error(f"Failed to import LED music visualizer: {e}")
    logger.error("Make sure required dependencies are installed: numpy, pyaudio, pygame")
    sys.exit(1)

  try:
    # Setup backend
    logger.info("Setting up Hamilton backend...")
    await backend.setup()

    # Start visualization
    if args.microphone:
      logger.info("Starting visualization from microphone...")
      logger.info("Press Ctrl+C to stop")
      await visualize_music_from_microphone(
        backend, sensitivity=args.sensitivity, use_blink=not args.no_blink
      )
    else:
      logger.info(f"Starting visualization from file: {args.file}")
      await visualize_music_from_file(
        backend,
        args.file,
        sensitivity=args.sensitivity,
        use_blink=not args.no_blink,
        play_audio=not args.no_play,
      )

  except KeyboardInterrupt:
    logger.info("Interrupted by user")
  except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
    sys.exit(1)
  finally:
    # Cleanup
    try:
      await backend.stop()
    except Exception:
      pass


if __name__ == "__main__":
  asyncio.run(main())
