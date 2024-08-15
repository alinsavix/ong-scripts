#!/usr/bin/env python3
# Do some OBS monitoring things running under inputs.execd in telegraf
import argparse
import logging
import os
import re
import sys
import time
from typing import Dict, List, Optional, Union

import obsws_python as obs
from obsws_python.error import (OBSSDKError, OBSSDKRequestError,
                                OBSSDKTimeoutError)
from tdvutil import hms_to_sec, ppretty
from websocket._exceptions import WebSocketTimeoutException


# flush stdin so that we don't have a backlog before we go into our main loop
def flush_input():
    try:
        # For windows
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError:
        import sys
        import termios  # for linux/unix
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)

def now():
    return int(time.time())

def log(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.stderr.flush()

def printmetric(metric_name: str, ts: int, value: int | float, tags: Dict[str, str] | None = None):
    tags_list = ""
    if tags:
        tags_list = " ".join([f"{k}={v}" for k, v in tags.items()])

    # we have to do this the roundabout way because if we use \n normally,
    # it will output newline + carriage return on windows, and the wavefront
    # input plugin on telegraf will explode on the \r character (sigh)
    sys.stdout.buffer.write(
        f"obs.{metric_name} {value} {ts} {tags_list}\n".encode())

def normalize_name(name: str) -> str:
    name = re.sub(r"[^\w-]+", "_", name)
    return name.lower()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gather OBS stats with a telegraf input processor")

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

    parsed_args = parser.parse_args()

    return parsed_args


def main():
    # make obsws-python not really output any logging on its own
    logging.basicConfig(level=logging.FATAL)

    args = parse_args()

    while True:
        try:
            if not run(args):
                log("Unexpected exit from metrics loop, continuing in 60 seconds.")
        except KeyboardInterrupt:
            log("Keyboard interrupt, exiting")
            sys.exit(0)
        except ConnectionError as e:
            log(f"OBS connection error, trying again in 60 seconds: {e}")
        except (Exception) as e:
            log(f"UNKNOWN EXCEPTION: {ppretty(e)}")

        time.sleep(60)


# Returns true if the outer loop shouldn't print an error. Yeah, it's sloppy.
def run(args: argparse.Namespace) -> bool:
    try:
        client = obs.ReqClient(host=args.host, port=args.port, timeout=5)
        eventclient = obs.EventClient(host=args.host, port=args.port,
                                      timeout=5, subs=(obs.Subs.GENERAL))
    except (ConnectionRefusedError, OBSSDKTimeoutError, WebSocketTimeoutException) as e:
        printmetric("active", now(), 0, {})
        sys.stdout.flush()
        return True

    def on_exit_started(data):
        log(f"Got OBS exit signal, disconnecting")

        # the clients have 'disconnect' methods, but they don't actually work
        # when called from an event handler. We really want to disconnect NOW,
        # though, so we'll just null out the client objects and python will
        # do its normal cleanup and shut down the connections. Open to better
        # ideas on how best to handle this.
        # client = None
        # eventclient = None
        # shutdown = True

        # except that didn't work reliably, how about just exit, and let
        # telegraf or whatever restart us? And we can't even just use exit()
        # because of the way threads are implemented in obsws
        os._exit(0)

    eventclient.callback.register(on_exit_started)

    flush_input()

    tags = {}
    r = client.get_version()
    tags["version"] = r.obs_version
    # tags["platform"] = r.platform

    # r = client2.get_scene_collection_list()
    # tags["scene_collection"] = r.current_scene_collection_name

    count = 0
    while True:
        # telegraf will give us a newline whenever it's expecting us to
        # generate some metrics, and will close that fd when it's time
        # for us to exit.
        x = sys.stdin.readline()
        if len(x) == 0 or sys.stdin.closed:
            log("stdin closed, exiting")
            printmetric("active", now(), 0, {})
            sys.exit(0)

        # We only want to occasionally generate metrics for things that aren't
        # active, so keep a counter of how many times we've been asked for
        # metrics that we can reference later.
        count += 1

        r = client.get_stats()
        ts = now()
        printmetric("active", ts, 1, {})

        # Main stats
        printmetric("usage.cpu_pct", ts, r.cpu_usage, tags)
        printmetric("usage.memory_mb", ts, r.memory_usage, tags)
        printmetric("fps", ts, r.active_fps, tags)
        printmetric("frames.render.time_avg_ms", ts, r.average_frame_render_time, tags)
        printmetric("frames.render.skipped", ts, r.render_skipped_frames, tags)
        printmetric("frames.render.total", ts, r.render_total_frames, tags)
        printmetric("frames.output.skipped", ts, r.output_skipped_frames, tags)
        printmetric("frames.output.total", ts, r.output_total_frames, tags)
        printmetric("websocket.messages.incoming", ts,
                    r.web_socket_session_incoming_messages, tags)
        printmetric("websocket.messages.outgoing", ts,
                    r.web_socket_session_outgoing_messages, tags)

        # Stats for each output. The ideal here is that we would query the
        # list of outputs, and then query the status of each, but a recent
        # OBS crash showed an issue when iterating over the list of outputs,
        # so we're going to just specify a few outputs that we care about
        # specifically, and only query those, with the hope we don't trigger
        # the same crash again.
        # r = client.get_output_list()
        # outputs = r.outputs
        outputs = ["simple_stream", "simple_file_output", "adv_stream", "adv_file_output"]

        for output_name in outputs:
            try:
                r = client.get_output_status(output_name)
            except obs.error.OBSSDKRequestError:
                continue

            output_tags = {"output": normalize_name(output_name)}
            printmetric("output.active", ts, 1 if r.output_active else 0, tags | output_tags)
            printmetric("output.reconnecting", ts,
                        1 if r.output_reconnecting else 0, tags | output_tags)

            # no reason to dump stats often for something that's not active or reconnecting
            if not any([r.output_active, r.output_reconnecting]) and count % 10 > 0:
                continue

            # FIXME: need to check the format on this. Is it SMPTE? Just HH:MM:SS.sss?
            # printmetric("output.timecode", ts, hms_to_sec(r.output_timecode), tags | output_tags)
            printmetric("output.duration_s", ts, r.output_duration / 1000, tags | output_tags)
            printmetric("output.congestion", ts, r.output_congestion, tags | output_tags)
            printmetric("output.bytes", ts, r.output_bytes, tags | output_tags)
            printmetric("output.frames.skipped", ts,
                        r.output_skipped_frames, tags | output_tags)
            printmetric("output.frames.total", ts, r.output_total_frames, tags | output_tags)

        sys.stdout.flush()


if __name__ == "__main__":
    main()
