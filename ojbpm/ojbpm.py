#!/usr/bin/env python3
import argparse
import enum
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, TypeVar

import mido
from tdvutil import ppretty
from tdvutil.argparse import CheckFile

MIDI_CLOCKS_PER_BEAT = 24

# Make timestamps match up with the onglog
os.environ['TZ'] = 'US/Eastern'
time.tzset()

class PlaybackState(enum.Enum):
    UNKNOWN = enum.auto()
    STOPPED = enum.auto()
    STARTED = enum.auto()

# Eventually we'll want to be able to persist some state (so we can restart
# this code in the middle of a loop or something and have it still know the
# state, more or less), so store that data in a single place. Things that we
# don't want to persist we should leave out from here.
@dataclass
class LoopState:
    playback_state: PlaybackState = PlaybackState.UNKNOWN
    looper_slot: str = "0-0"
    playback_start_time: Optional[float] = None
    playback_stop_time: Optional[float] = None
    current_bpm: float = 0

def log(msg: str) -> None:
    now = datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    print(f"{ts} {msg}")

def h(data: List[bytes]) -> str:
    return "".join('%02x' % i for i in data)


T = TypeVar('T')
def group(iterable: Iterable[T], num) -> Iterable[T]:
    iterators = [iter(iterable)] * num
    return zip(*iterators)

def program_to_filenum(program: int) -> int:
    return program + 1

def program_to_slot(program: int) -> str:
    slot_major = (program // 8) + 1
    slot_minor = (program % 8) + 1
    return f"{slot_major}-{slot_minor}"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="parse RC-202 MIDI data")

    parser.add_argument(
        "--device",
        type=str,
        default="Steinberg UR44:Steinberg UR44 MIDI 1 28:0",
        help="MIDI device to use as live MIDI input"
    )

    parser.add_argument(
        "--raw-file", "--file",
        type=Path,
        default=None,
        action=CheckFile(must_exist=True),
        help="Use specified raw midi capture file instead of live MIDI input",
    )

    parser.add_argument(
        "--sample-beats",
        type=int,
        default=8,
        action='store',
        help="Number of beats to sample for BPM calculation",
    )
    parser.add_argument(
        "--fake-bpm",
        type=float,
        default=120.0,
        action='store',
        help="Generate fake midi clock at this bpm for testing purposes",
    )



    return parser.parse_args()

def main() -> int:
    log("Starting up")
    fake_time = 0.0

    def fake_now() -> float:
        nonlocal fake_time
        return fake_time

    args = parse_args()
    if args.raw_file is not None:
        log(f"Using raw file: {args.raw_file}")
        log(f"Using simulated clock timing at {args.fake_bpm} BPM")

        with args.raw_file.open(mode="rb") as f:
            data = f.read()
        midi = mido.Parser()
        midi.feed(data)

        now = fake_now
    else:
        log(f"Using MIDI device: {args.device}")
        try:
            midi = mido.open_input(args.device)
        except Exception as e:
            log(f"ERROR: {e}")
            return 1

        log(f"MIDI device successfully opened: {args.device}")

        now = time.time

    lstate = LoopState()

    clock_tick_count: int = 0
    start_tick_time: float = 0.0

    # Fake clock timing, for testing of raw midi data captures
    fake_time_per_beat = 60 / args.fake_bpm
    fake_time_per_tick = fake_time_per_beat / MIDI_CLOCKS_PER_BEAT

    for msg in midi:
        if msg.type == "clock":
            fake_time += fake_time_per_tick
            if lstate.playback_state == PlaybackState.STOPPED:
                continue
            if clock_tick_count == 0:
                start_tick_time = now()
            clock_tick_count += 1

            # Not sure why the +1 is needed here, really
            if clock_tick_count == (MIDI_CLOCKS_PER_BEAT * args.sample_beats) + 1:
                clock_tick_count = 0
                elapsed = now() - start_tick_time
                time_per_beat = elapsed / args.sample_beats
                lstate.current_bpm = round(60.0 / time_per_beat, 1)
                log(f"BPM: {lstate.current_bpm:0.1f}")

        elif msg.type == "program_change":
            # Changing to a new slot
            lstate.looper_slot = program_to_slot(msg.program)
            log(f"SLOT CHANGE: {lstate.looper_slot}")

        elif msg.type == "start":
            if lstate.playback_state == PlaybackState.STARTED:
                log("START (ignored)")
            else:
                lstate.playback_state = PlaybackState.STARTED
                clock_tick_count = 0
                lstate.playback_start_time = now()
                log("START")

        elif msg.type == "stop":
            if lstate.playback_state == PlaybackState.STOPPED:
                log("STOP (ignored)")
            else:
                lstate.playback_state = PlaybackState.STOPPED
                lstate.playback_stop_time = now()
                log("STOP")

        elif msg.type == "sysex":
            # Drop the checksum byte
            sysex_str = " ".join([h(x) for x in group(msg.data[:-1], 2)])
            log(f"SYSEX: {sysex_str}")

            if lstate.current_bpm > 0:
                loop_length = ((msg.data[8] & 0x0f) << 4) + (msg.data[9] & 0x0f)
                loop_length_s = (60.0 / lstate.current_bpm) * 4 * loop_length

                if loop_length > 0:
                    log(f"LOOP LENGTH: {loop_length} bars ({loop_length_s:0.3f} seconds)")

        else:
            log(f"UNKNOWN: {mido.format_as_string(msg)}")


if __name__ == "__main__":
    sys.exit(main())
