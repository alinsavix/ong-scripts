#!/usr/bin/env python3
import enum
import os
import sys
import time
from datetime import datetime  # I hate this naming

import mido
from tdvutil import ppretty

MIDI_CLOCKS_PER_BEAT = 24  # I think this is right
BEAT_MULTIPLIER = 8  # how long we want to measure before calculating bpm
MIDI_DEVICE = "Steinberg UR44:Steinberg UR44 MIDI 1 28:0"

# Make timestamps match up with the onglog
os.environ['TZ'] = 'US/Eastern'
time.tzset()

class State(enum.Enum):
    UNKNOWN = enum.auto()
    STOPPED = enum.auto()
    STARTED = enum.auto()


def log(msg: str) -> None:
    now = datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    print(f"{ts} {msg}")

def program_to_filenum(program: int) -> int:
    return program + 1

def program_to_slot(program: int) -> str:
    slot_major = (program // 8) + 1
    slot_minor = (program % 8) + 1
    return f"{slot_major}-{slot_minor}"

# print(program_to_slot(0))
# print(program_to_slot(7))
# print(program_to_slot(8))
# print(program_to_slot(63))

# parsed_out = open("test_miditest_parsed.txt", "a")
# raw_out = open("test_miditest_raw.bin", "ab")


log("Starting up")

# with open("m:/throne.midi", mode="rb") as f:
#     data = f.read()
# p = mido.Parser()
# p.feed(data)'

with open("miditest_raw.bin", mode="rb") as f:
    data = f.read()
p = mido.Parser()
p.feed(data)

# p = mido.open_input(MIDI_DEVICE)
# log(f"MIDI device successfully opened: {MIDI_DEVICE}")

currentstate = State.UNKNOWN
tick_count = 0
start_tick_time = None
slot = None
stop_time = None
start_time = None
current_bpm = 0
fake_time = 0.0

# for 120bpm:
#   Time per beat = 0.5s
#   Time per cllock tick = 0.5 / MIDI_CLOCKS_PER_BEAT = 0.02083333s


for msg in p:
    if msg.type == "clock":
        fake_time += 0.02083333
        if currentstate != State.STARTED:
            continue
        if tick_count == 0:
            start_tick_time = time.time()
            start_tick_time = fake_time
        tick_count += 1

        # Not sure why the +1 is needed here, really
        if tick_count == (MIDI_CLOCKS_PER_BEAT * BEAT_MULTIPLIER) + 1:
            tick_count = 0
            elapsed = time.time() - start_tick_time
            elapsed = fake_time - start_tick_time
            time_per_beat = elapsed / BEAT_MULTIPLIER
            current_bpm = 60 / time_per_beat
            log(f"BPM: {current_bpm:0.3f}")

    elif msg.type == "program_change":
        # Changing to a new slot
        slot = program_to_slot(msg.program)
        log(f"SLOT CHANGE: {slot}")
    elif msg.type == "start":
        if currentstate == State.STARTED:
            log("START (ignored)")
        else:
            currentstate = State.STARTED
            tick_count = 0
            start_time = time.time()
            log("START")
    elif msg.type == "stop":
        if currentstate == State.STOPPED:
            log("STOP (ignored)")
        else:
            currentstate = State.STOPPED
            stop_time = time.time()
            log("STOP")
    elif msg.type == "sysex":
        # print(mido.format_as_string(msg))
        # print(t, "SYSEX", "".join('%02x' % i for i in msg.data), file=parsed_out)
        sysex_str = "".join('%02x' % i for i in msg.data)
        log(f"SYSEX: {sysex_str}")

        if current_bpm > 0:
            loop_length = ((msg.data[8] & 0x0f) << 4) + (msg.data[9] & 0x0f)
            loop_length_s = (60 / current_bpm) * 4 * loop_length
            # log(f"maj/min: {msg.data[8]}/{msg.data[9]} == {loop_length}")
            if loop_length > 0:
                log(f"MAYBE LOOP LENGTH: {loop_length} bars ({loop_length_s:0.3f} seconds)")
    else:
        log(f"UNKNOWN: {mido.format_as_string(msg)}")

    # raw_out.write(msg.bin())

# print(mido.get_input_names())
# x = mido.get_input_names()[3]
# inport = mido.open_input(x)

# for msg in inport:
#     print(msg)
