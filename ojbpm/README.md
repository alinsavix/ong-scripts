# The Orange Jacket Brigade Loopstation BPM Matcher

## What

This is a (small) set of tools for being able to read the tempo (and a
limited amount of other information) from the MIDI output of an RC-202
Loopstation, in real time. It requires a MIDI connection from the
Loopstation to the host the code is running on; it does *not* require
a MIDI connection from the host back to the Loopstation.

The tools attempt to be as lightweight as possible, with the intent to
be runnable on a raspberry pi.

There are two primary tools:

### ojbpm

This is the script that listens to the MIDI data coming from the
Loopstation and attempts to learn things from it, and exports that
data via a set of text files on disk that can be read and utilized
by other software.

See `ojbpm.py --help` for usage.

The information available:

* **Current BPM**: What it says on the tin, the current BPM as detected
based on the MIDI beat clock. When the tempo changes, this can take a
few seconds to settle down completely. On rare occasion it can also
fluctuate slightly (generally by 0.1bpm) with faster tempos
* **Looper Slot**: What slot the Loopstation is currently set on. Note
that this is only updated when the slot is changed, so when this script
is first started, it won't know the current slot.
* **Loop Length in Bars**: The length of the current loop in bars.
Unfortunately, best we can tell, the Loopstation does not provide any
way for these scripts to determine the time signature, so this value
can only be usefully converted to time for songs in 4/4 time. And
possibly not even then -- we haven't tested this one deeply.
* **Playback Start Time**: The start time (in unix epoch time) of the
start of playback of the current loop. This normally represents the
time the 'start' MIDI message was received, but for new loops it
represents the time that recording of the first pass through the 
loop was completed (and started to be played back).
* **Playback State**: The current playback state of the looper, as best
it can be deterimned. "STARTED", "STOPPED", and "UNKNONWN" are the
valid values, currently.

### bpmset

A simple script that watches the output directory of the `ojbpm` script
for updates and changes the playback speed of a media source in OBS
to match, via obs-websocket.

Currently (this will probably change) the media file is expected to
contain an animation that matches a bpm of 100. This script will then
adjust the playback speed accordingly, to match the current BPM.

See `setbpm.py --help` for full usage information.

## Why

Because Alinsa thought it would be neat, mostly. And because that
headbanging pony added during Jon's ~~stream pony~~ pony stream is
really damned compelling.

## Who

Credits go to:

**RobAncalagon** for doing most of the original research to figure out
what MIDI data was available from the Loopstation and figuring out what
some bits of it meant.

**TDV Alinsa** (alinsa_vix on twitch) for the idea, for bugging Jon
until he let her do this, and for the actual code.

**Jonathan Ong** ([JonathanOng](https://twitch.tv/JonathanOng)) for
being a musical genius and providing the amazing music that inspired
(and ultimately feeds) this project.
