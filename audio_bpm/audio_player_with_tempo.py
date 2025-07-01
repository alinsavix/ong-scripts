#!/usr/bin/env python3
import argparse
import os
import queue
import sys
import threading
import time
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd


class AudioPlayerWithTempo:
    def __init__(self, audio_file, window_size=8.0, hop_length=512, update_interval=2.0):
        """
        Initialize the audio player with tempo detection.

        Args:
            audio_file (str): Path to the audio/MP4 file
            window_size (float): Size of the analysis window in seconds
            hop_length (int): Number of samples between frames
            update_interval (float): How often to update tempo (seconds)
        """
        self.audio_file = audio_file
        self.window_size = window_size
        self.hop_length = hop_length
        self.update_interval = update_interval

        # Audio playback
        self.audio_data = None
        self.sample_rate = None
        self.is_playing = False
        self.audio_queue = queue.Queue(maxsize=100)  # Limit queue size to prevent memory issues
        self.audio_position = 0
        self.audio_lock = threading.Lock()

        # Tempo tracking
        self.current_tempo = None
        self.tempo_history = []
        self.tempo_lock = threading.Lock()

        # Load audio
        self._load_audio()

    def _load_audio(self):
        """Load and prepare the audio file."""
        print(f"Loading audio file: {self.audio_file}")

        try:
            # Load audio with librosa (this will handle MP4 files)
            self.audio_data, self.sample_rate = librosa.load(
                self.audio_file,
                sr=None,  # Keep original sample rate
                mono=True
            )

            # Ensure audio data is float32 and in the correct range
            self.audio_data = self.audio_data.astype(np.float32)

            # Normalize audio to prevent clipping
            max_val = np.max(np.abs(self.audio_data))
            if max_val > 0:
                self.audio_data = self.audio_data / max_val * 0.9

            print("Audio loaded successfully:")
            print(f"  Duration: {len(self.audio_data) / self.sample_rate:.2f} seconds")
            print(f"  Sample rate: {self.sample_rate} Hz")
            print(f"  Data type: {self.audio_data.dtype}")
            print(f"  Max amplitude: {np.max(np.abs(self.audio_data)):.3f}")

        except Exception as e:
            print(f"Error loading audio file: {e}")
            sys.exit(1)

    def _calculate_tempo_segment(self, start_time, end_time):
        """
        Calculate tempo for a specific time segment.

        Args:
            start_time (float): Start time in seconds
            end_time (float): End time in seconds

        Returns:
            float: Estimated tempo in BPM
        """
        try:
            # Convert time to sample indices
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)

            # Ensure we don't go beyond the audio length
            end_sample = min(end_sample, len(self.audio_data))
            start_sample = min(start_sample, end_sample - 1)

            if end_sample <= start_sample:
                return None

            # Extract audio segment
            segment = self.audio_data[start_sample:end_sample]

            if len(segment) < self.sample_rate * 2:  # Need at least 2 seconds
                return None

            # Calculate onset strength
            onset_env = librosa.onset.onset_strength(
                y=segment,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                aggregate=np.median
            )

            # Get tempo directly from onset envelope
            tempo = librosa.feature.tempo(
                onset_envelope=onset_env,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                aggregate=np.median
            )

            # Extract scalar value from numpy array
            return float(tempo.item())

        except Exception as e:
            print(f"Error calculating tempo: {e}")
            return None

    def _tempo_tracker(self):
        """Background thread for tracking tempo."""
        start_time = time.time()

        while self.is_playing:
            current_time = time.time() - start_time

            # Calculate tempo for the current window
            window_start = max(0, current_time - self.window_size)
            window_end = current_time

            tempo = self._calculate_tempo_segment(window_start, window_end)

            if tempo is not None:
                with self.tempo_lock:
                    self.current_tempo = tempo
                    self.tempo_history.append((current_time, tempo))

                    # Keep only recent history
                    if len(self.tempo_history) > 50:
                        self.tempo_history = self.tempo_history[-50:]

                print(f"[{current_time:6.1f}s] Tempo: {tempo:5.1f} BPM")

            time.sleep(self.update_interval)

    def _audio_callback(self, outdata, frames, time, status):
        """Callback for audio playback."""
        if status:
            print(f"Audio callback status: {status}")

        try:
            with self.audio_lock:
                # Check if audio data is loaded
                if self.audio_data is None:
                    outdata.fill(0)
                    self.is_playing = False
                    return

                # Calculate how much audio we need
                end_position = min(self.audio_position + frames, len(self.audio_data))

                if self.audio_position >= len(self.audio_data):
                    # End of audio
                    outdata.fill(0)
                    self.is_playing = False
                    return

                # Get the audio data for this frame
                audio_chunk = self.audio_data[self.audio_position:end_position]

                # Ensure we have the right number of frames
                if len(audio_chunk) < frames:
                    # Pad with zeros if we don't have enough data
                    padded_chunk = np.zeros(frames, dtype=np.float32)
                    padded_chunk[:len(audio_chunk)] = audio_chunk
                    outdata[:, 0] = padded_chunk
                    self.is_playing = False  # End of audio
                else:
                    outdata[:, 0] = audio_chunk

                # Update position
                self.audio_position += frames

        except Exception as e:
            print(f"Error in audio callback: {e}")
            outdata.fill(0)
            self.is_playing = False

    def play(self):
        """Start playing audio with tempo tracking."""
        print("\nStarting playback with tempo tracking...")
        print(f"Analysis window: {self.window_size}s")
        print(f"Update interval: {self.update_interval}s")
        print("-" * 50)

        self.is_playing = True
        self.audio_position = 0

        # Start tempo tracking thread
        tempo_thread = threading.Thread(target=self._tempo_tracker, daemon=True)
        tempo_thread.start()

        try:
            # Start audio playback
            with sd.OutputStream(
                channels=1,
                samplerate=self.sample_rate,
                callback=self._audio_callback,
                dtype=np.float32,
                blocksize=1024,  # Smaller blocksize for better responsiveness
                latency='low'    # Lower latency
            ):
                print("Audio playback started. Press Ctrl+C to stop.")

                # Wait for playback to complete
                while self.is_playing:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nPlayback stopped by user.")
        except Exception as e:
            print(f"Error during playback: {e}")
        finally:
            self.is_playing = False

            # Print final statistics
            if self.tempo_history:
                tempos = [t[1] for t in self.tempo_history]
                print("\nTempo Statistics:")
                print(f"  Average tempo: {np.mean(tempos):.1f} BPM")
                print(f"  Min tempo: {np.min(tempos):.1f} BPM")
                print(f"  Max tempo: {np.max(tempos):.1f} BPM")
                print(f"  Std deviation: {np.std(tempos):.1f} BPM")


def main():
    parser = argparse.ArgumentParser(
        description="Play audio while tracking tempo in real-time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "audio_file",
        help="Path to the audio/MP4 file to play"
    )

    parser.add_argument(
        "--window-size",
        type=float,
        default=10.0,
        help="Size of the analysis window in seconds (default: 8.0)"
    )

    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="Number of samples between frames (default: 512)"
    )

    parser.add_argument(
        "--update-interval",
        type=float,
        default=5.0,
        help="How often to update tempo in seconds (default: 5.0)"
    )

    args = parser.parse_args()

    # Check if file exists
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: File '{args.audio_file}' not found.")
        sys.exit(1)

    # Create and start the player
    player = AudioPlayerWithTempo(
        audio_file=args.audio_file,
        window_size=args.window_size,
        hop_length=args.hop_length,
        update_interval=args.update_interval
    )

    player.play()


if __name__ == "__main__":
    main()
