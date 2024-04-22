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


class BPMHandler(PatternMatchingEventHandler):
    def __init__(self, args: argparse.Namespace, patterns: List[str]):
        super().__init__(patterns=patterns)
        self.args = args

    def on_modified(self, event):
        print(f"File modified: {event.src_path}")

        with open(event.src_path, "r") as f:
            x = f.readline()

        if len(x) < 2:
            print("sleep and retry")
            with open(event.src_path, "r") as f:
                time.sleep(1)
                x = f.readline()

        if len(x) < 2:
            print("can't read bpm data, skipping")

        bpm = float(x)

        print(f"setting BPM to {bpm}")
        sys.stdout.flush()

        set_bpm(self.args.host, self.args.port, self.args.source, bpm)


def watch_bpm(args: argparse.Namespace):
    path = args.watch_dir
    event_handler = BPMHandler(args, patterns=["current_bpm.txt"])

    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()

    try:
        while observer.is_alive():
            observer.join(1)
    finally:
        observer.stop()
        observer.join()


# We could stay connected and have fancy reconnect logic, but bpm changes
# will be rare enough that we're just going to reconnect every time. We
# should probably do this better, later.
def set_bpm(host: str, port: int, source: str, bpm: float):
    if bpm > 200.0:
        bpm = bpm / 2.0
    elif bpm < 50.0:
        bpm = 100.0

    with obs.ReqClient(host=host, port=port, timeout=5) as client:
        current = client.get_input_settings(source)
        # print(ppretty(r.input_settings))
        # sys.stdout.flush()

        client.set_input_settings(source, {"speed_percent": int(bpm)}, True)
        print(f"{source} changed bpm: {current.speed_percent} to {bpm}")
        sys.stdout.flush()

def play_bpm(host: str, port: int, scene: str, source: str, len: float) -> None:
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

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        required=True,
        help="name of the source to set bpm for"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.watch_dir:
        watch_bpm(args)

        # print(f"Watching {args.watch_dir} for bpm changes")
        # while True:
        #     with open(args.watch_dir, "r") as f:
        #         bpm = float(f.readline())
        #         set_bpm(args.host, args.port, args.source, bpm)
        #     time.sleep(1)


if __name__ == "__main__":
    main()
