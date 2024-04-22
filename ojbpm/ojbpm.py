#!/usr/bin/env python3
import argparse
import enum
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Literal, Optional, TypeVar

import mido
from tdvutil import ppretty
from tdvutil.argparse import CheckFile

MIDI_CLOCKS_PER_BEAT = 24

# Make timestamps match up with the onglog (unixes only, for some reason)
if hasattr(time, "tzset"):
    os.environ['TZ'] = 'US/Eastern'
    time.tzset()


# Eventually we'll want to be able to persist some state (so we can restart
# this code in the middle of a loop or something and have it still know the
# state, more or less), so store that data in a single place. Things that we
# don't want to persist we should leave out from here.
@dataclass
class LoopState:
    playback_state: Literal["UNKNOWN", "STOPPED", "STARTED"] = "UNKNOWN"
    looper_slot: str = "0-0"
    playback_start_time: float = 0.0
    playback_stop_time: float = 0.0
    current_bpm: float = 0
    updated_t: float = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data) -> 'LoopState':
        s = json.loads(data)
        return cls(**s)


# FIXME: Should these be part of LoopState?
def save_state(statefile: Optional[Path], ls: LoopState):
    if statefile is None:
        return

    ls.updated_t = time.time()
    state_json = ls.to_json()
    tmpfile = statefile.with_suffix(".tmp")
    tmpfile.write_text(state_json)

    tmpfile.replace(statefile)
    log(f"SAVED CURRENT STATE (to {statefile})")


def init_state(statefile: Optional[Path]) -> LoopState:
    if statefile is None:
        return LoopState()

    try:
        state_json = statefile.read_text()
        state = LoopState.from_json(state_json)

        # make sure the state is kinda recent (last 15 mins) or just ignore it
        if time.time() > (state.updated_t + (15 * 60)):
            log("SAVED STATE IGNORED (too old)")
            return LoopState()

        # else
        log(f"SAVED STATE LOADED (from {statefile})")
        return state
    except Exception as e:
        log(f"SAVED STATE NOT AVAILABLE ({e})")
        return LoopState()

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

    parser.add_argument(
        "--state-file", "--statefile",
        type=Path,
        default=None,
        help="Location to persist state during restarts and similar."
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

    lstate = init_state(args.state_file)

    clock_tick_count: int = 0
    start_tick_time: float = 0.0

    # Fake clock timing, for testing of raw midi data captures
    fake_time_per_beat = 60 / args.fake_bpm
    fake_time_per_tick = fake_time_per_beat / MIDI_CLOCKS_PER_BEAT

    for msg in midi:
        if msg.type == "clock":
            fake_time += fake_time_per_tick
            if lstate.playback_state == "STOPPED":
                continue
            if clock_tick_count == 0:
                start_tick_time = now()
            clock_tick_count += 1

            # Not sure why the +1 is needed here, really
            if clock_tick_count == (MIDI_CLOCKS_PER_BEAT * args.sample_beats) + 1:
                # don't count a few ticks so we can make sure we're caught up
                # from saving or whatever before starting again
                clock_tick_count = -20
                elapsed = now() - start_tick_time
                time_per_beat = elapsed / args.sample_beats
                new_bpm = round(60.0 / time_per_beat, 1)

                if lstate.current_bpm != new_bpm:
                    lstate.current_bpm = new_bpm
                    log(f"NEW BPM: {lstate.current_bpm:0.1f}")
                    save_state(args.state_file, lstate)

        elif msg.type == "program_change":
            # Changing to a new slot
            lstate.looper_slot = program_to_slot(msg.program)
            lstate.current_bpm = 0.0
            clock_tick_count = -20
            log(f"SLOT CHANGE: {lstate.looper_slot}")
            save_state(args.state_file, lstate)

        elif msg.type == "start":
            if lstate.playback_state == "STARTED":
                log("START (ignored)")
            else:
                lstate.playback_state = "STARTED"
                clock_tick_count = -20
                lstate.playback_start_time = now()
                log("START")
                save_state(args.state_file, lstate)

        elif msg.type == "stop":
            if lstate.playback_state == "STOPPED":
                log("STOP (ignored)")
            else:
                lstate.playback_state = "STOPPED"
                lstate.playback_stop_time = now()
                log("STOP")
                save_state(args.state_file, lstate)

        elif msg.type == "sysex":
            # Drop the checksum byte
            sysex_str = " ".join([h(x) for x in group(msg.data[:-1], 2)])
            log(f"SYSEX: {sysex_str}")

            # FIXME: should we save the state?
            if lstate.current_bpm > 0:
                loop_length = ((msg.data[8] & 0x0f) << 4) + (msg.data[9] & 0x0f)
                loop_length_s = (60.0 / lstate.current_bpm) * 4 * loop_length

                if loop_length > 0:
                    log(f"LOOP LENGTH: {loop_length} bars ({loop_length_s:0.3f} seconds)")

        else:
            log(f"UNKNOWN: {mido.format_as_string(msg)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
