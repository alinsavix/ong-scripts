#!/usr/bin/env python3
import argparse
import pprint
import sys
import time
from enum import IntEnum
from math import log
from pathlib import Path
from typing import List

import obsws_python as obs
from tdvutil import ppretty
from tdvutil.argparse import CheckFile
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

update_sources: bool = False

def sources_update():
    global update_sources
    update_sources = True

class BPMHandler(PatternMatchingEventHandler):
    def __init__(self, args: argparse.Namespace, patterns: List[str]):
        super().__init__(patterns=patterns)
        self.args = args

    def on_modified(self, event):
        # print(f"File modified: {event.src_path}")

        with open(event.src_path, "r") as f:
            x = f.readline()

        if len(x) < 2:
            # print("sleep and retry")
            with open(event.src_path, "r") as f:
                time.sleep(2)
                x = f.readline()

        if len(x) < 2:
            print(f"can't read valid data from '{event.src_path}', skipping update")

        if event.src_path.endswith("current_bpm.txt"):
            bpm = float(x)

            print(f"setting BPM to {bpm}:")
            sys.stdout.flush()

            sources = get_sources(self.args.host, self.args.port, self.args.source_prefix)
            set_bpm(self.args.host, self.args.port, sources, bpm)

        elif event.src_path.endswith("looper_slot.txt"):
            print(f"setting looper slot number to {x}")
            sys.stdout.flush()
            set_looper_slot(self.args.host, self.args.port, "Looper Slot Number", x)


def on_input_created(data):
    print("A new source was created, refreshing source list")
    sys.stdout.flush()
    sources_update()

def on_input_name_changed(data):
    print("A source was renamed, refreshing source list")
    sys.stdout.flush()
    sources_update()

def watch_bpm(args: argparse.Namespace):
    # Set up watches so we can update when needed
    # FIXME: It seems like if this gets disconnected (because, say, OBS
    # exited), we have no way to know and we're dead in the water. Not
    # sure how to recover properly from that
    eventclient = None

    while eventclient is None:
        try:
            eventclient = obs.EventClient(
                host=args.host, port=args.port, timeout=5,
                subs=(obs.Subs.INPUTS)
            )
        except OSError as e:
            print(f"ERROR (will retry): {e}")
            sys.stdout.flush()

        time.sleep(60)

    eventclient.callback.register([on_input_created, on_input_name_changed])

    # And now set up the filesystem watch
    path = args.watch_dir
    event_handler = BPMHandler(args, patterns=["current_bpm.txt", "looper_slot.txt"])

    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()

    try:
        while observer.is_alive():
            observer.join(1)
    finally:
        observer.stop()
        observer.join()


def set_looper_slot(host: str, port: int, source: str, slot: str):
    with obs.ReqClient(host=host, port=port, timeout=5) as client:
        try:
            client.set_input_settings(source, {"text": slot.rstrip()}, True)
            print(f"{source} changed slot number to {slot}")
        except Exception as e:
            print(f"WARNING: Failed to set looper slot number for {source}: {e}")

        sys.stdout.flush()


# We could stay connected and have fancy reconnect logic, but bpm changes
# will be rare enough that we're just going to reconnect every time. We
# should probably do this better, later.
def set_bpm(host: str, port: int, sources: List[str], bpm: float):
    if bpm > 200.0:
        bpm = bpm / 2.0
    elif bpm < 25.0:
        bpm = 100.0

    with obs.ReqClient(host=host, port=port, timeout=5) as client:
        # current = client.get_input_settings(source)
        # print(ppretty(current.input_settings))
        # print(ppretty(r.input_settings))
        # sys.stdout.flush()
        vendor = "AdvancedSceneSwitcher"
        vendor_msg = "AdvancedSceneSwitcherMessage"
        client.call_vendor_request(vendor, vendor_msg, {"message": f"BPM:{int(bpm)}"})

        for source in sources:
            try:
                client.set_input_settings(source, {"speed_percent": int(bpm)}, True)
                print(f"{source} changed bpm to {round(bpm)}")
            except Exception as e:
                print(f"WARNING: Failed to set bpm for {source}: {e}")
                sources_update()

            sys.stdout.flush()


def get_sources(host: str, port: int, source_prefix: str) -> List[str]:
    if not hasattr(get_sources, "source_list"):
        get_sources.source_list = []

    if source_prefix is None:
        return get_sources.source_list

    global update_sources
    if not update_sources:
        return get_sources.source_list

    ret: List[str] = []
    try:
        with obs.ReqClient(host=host, port=port, timeout=5) as client:
            r = client.get_input_list("ffmpeg_source")
            for input in r.inputs:
                if input["inputName"].startswith(source_prefix):
                    ret.append(input["inputName"])
    except Exception as e:  # FIXME: narrow this
        # Stick with whatever we had, and don't change update_sources
        print(f"ERROR: Couldn't update source list: {e}", file=sys.stderr)
        return get_sources.source_list

    print(f"INFO: updated list of BPM sources: {ret}")

    update_sources = False
    get_sources.source_list = ret
    return get_sources.source_list


def play_source(host: str, port: int, scene: str, source: str, len: float) -> None:
    with obs.ReqClient(host=host, port=port, timeout=5) as client:
        # r = client2.get_source_active("BPM Headbang")
        # print(f"active: {r.video_active}")
        r = client.get_scene_item_id(scene, source)
        id = r.scene_item_id

        # r = client.get_scene_item_enabled(scene, id)
        # print(f"active: {r.scene_item_enabled}")

        # FIXME: Make this a obs-websocket batch thing instead of sleeping
        client.set_scene_item_enabled(scene, id, True)
        time.sleep(len)
        client.set_scene_item_enabled(scene, id, False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set speed of OBS animation to match a given BPM")

    parser.add_argument(
        "--host",
        type=str,
        default="localhost",  # 192.168.1.152
        help="address or hostname of host running OBS"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=4455,
        help="port number for OBS websocket"
    )

    parser.add_argument(
        "--bpm",
        type=float,
        default=None,
        help="manually specify the BPM to set"
    )

    parser.add_argument(
        "--watch-dir",
        type=Path,
        default=None,
        action=CheckFile(must_exist=True),
        help="ojbpm export path to watch for bpm changes",
    )

    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="name of the scene to look for the source in when manually triggering"
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--source",
        type=str,
        default=None,
        help="name of the source to set bpm for"
    )

    source.add_argument(
        "--source-prefix",
        type=str,
        default=None,
        help="mass-update playback rate for any source with given prefix"
    )

    parser.add_argument(
        "--play",
        default=False,
        action='store_true',
        help="Ask OBS to play BPM-matched source"
    )

    parsed_args = parser.parse_args()

    return parsed_args


def main():
    args = parse_args()

    # get_sources(args.host, args.port, "")
    # sys.exit(0)

    if args.source and not args.source_prefix:
        get_sources.source_list = [args.source]
    elif args.source_prefix:
        print(f"Using source prefix '{args.source_prefix}'")
        sys.stdout.flush()
        sources_update()

    sources = get_sources(args.host, args.port, args.source_prefix)

    if args.watch_dir:
        watch_bpm(args)
    elif args.play:
        if args.bpm:
            set_bpm(args.host, args.port, sources, args.bpm)
        play_source(args.host, args.port, args.scene, args.source, 7)
    elif args.bpm:
        set_bpm(args.host, args.port, sources, args.bpm)
    else:
        print("ERROR: not sure what you want me to do", file=sys.stderr)


        # print(f"Watching {args.watch_dir} for bpm changes")
        # while True:
        #     with open(args.watch_dir, "r") as f:
        #         bpm = float(f.readline())
        #         set_bpm(args.host, args.port, args.source, bpm)
        #     time.sleep(1)


if __name__ == "__main__":
    main()
