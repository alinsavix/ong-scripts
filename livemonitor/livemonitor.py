#!/usr/bin/env python3
# DISCLAIMER: This is some of the worst code I have ever written.
#
# This is a hopefully simple/basic script that watches an OBS instance and
# waits for it to go live, and starts or stops a systemd unit based on that,
# to trigger other things that need to run based on that.
#
# Current idea/design: two different systemd targets, ong-online.target and
# ong-offline.target, which are started based on stream status. The two will
# conflict with each other, so starting one stops the other, and gives us a
# good way to have certain things that only run when the stream is online,
# and some that only run when it isn't.
#
# FIXME: this is probably a great place for some asyncio action, maybe, but
# Alinsa is not smert enough for asyncio yet.
import argparse
import atexit
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional

import obsws_python as obs
from obsws_python.error import OBSSDKRequestError
from tdvutil import ppretty

state: Literal["WAITING", "STREAMING", "COOLDOWN"] = "WAITING"

def log(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.stderr.flush()


# FIXME: is there some pypi module that could do this instead?
def start_target(target: str) -> bool:
    cmd = ["/usr/bin/systemctl", "start", target]

    try:
        res = subprocess.run(cmd, shell=False, capture_output=True, text=True, timeout=300)
    except FileNotFoundError as e:
        log(f"ERROR: couldn't execute systemctl: {e}")
        return False  # FIXME: might it be better to exit?
    except Exception as e:
        log(f"ERROR: unknown error calling systemctl: {e}")
        return False
    else:
        if res.returncode != 0:
            log(f"ERROR: couldn't start {target}: {res.stderr}")
            return False

        log(f"INFO: started systemd target {target}")
        return True

client = None

def connect_obs(host: str, port: int) -> obs.ReqClient:
    global client

    # the logic here sucks
    try:
        client = obs.ReqClient(host=host, port=port, timeout=5)
    except OBSSDKRequestError as e:
        code = e.code
    except (OSError):
        # log("ERROR: Connection Refused")
        return None
    except Exception as e:
        log(f"ERROR: couldn't connect to OBS: {e}")
        return None
    else:
        # only reached if no exceptions raised (not sure if I like this style)
        return client

    if code != 207:
        log(f"WARNING: Unknown OBS response code {code}")
        return None

    log("WARNING: OBS not ready, retrying in 30 seconds")
    time.sleep(30)
    return connect_obs(host, port)


# returns true if we're streaming
def check_streaming(args: argparse.Namespace) -> bool:
    global client

    if args.force:
        return True

    if client is None:
        client = connect_obs(args.host, args.port)

    if client is None:
        return False

    try:
        r = client.get_stream_status()
    except Exception as e:  # FIXME: make more granular
        connect_obs(args.host, args.port)
        r = client.get_stream_status()
    return r.output_active


def get_obs_uptime(host: Optional[str], port: int) -> Optional[int]:
    global client

    if host is None:
        return None

    if client is None:
        client = connect_obs(host, port)

    if client is None:
        return None

    try:
        r = client.get_stream_status()
        uptime = int(r.output_duration) // 1000

        if uptime == 0:
            return None
        else:
            return uptime
    except Exception:
        return None


# def play_source(args: argparse.Namespace, scene: str, source: str) -> None:
#     global client

#     if client is None:
#         client = connect_obs(args.host, args.port)

#     if client is None:
#         return

#     try:
#         r = client.get_scene_item_id(scene, source)
#         id = r.scene_item_id

#         client.set_scene_item_enabled(scene, id, True)
#         time.sleep(5)
#         client.set_scene_item_enabled(scene, id, False)
#     except Exception as e:
#         log(f"ERROR: couldn't play timecode source: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect if OBS is streaming, and Do Stuff™")

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
        "--online-start-unit",
        type=str,
        default="ong-online.target",
        help="unit/target to start when OBS starts streaming"
    )

    parser.add_argument(
        "--offline-start-unit",
        type=str,
        default="ong-offline.target",
        help="unit/target to start when OBS stops streaming"
    )

    parser.add_argument(
        "--force",
        default=False,
        action="store_true",
        help="pretend OBS is streaming even if it isn't"
    )

    return parser.parse_args()


def main():
    global state

    args = parse_args()

    # This just shuts up the obsws module
    logging.basicConfig(level=logging.FATAL)

    # atexit.register(kill_ffmpeg)

    if args.force:
        log("WARNING: force mode is enabled, actual OBS status will be ignored")
    else:
        log("INFO: in startup, waiting for OBS to start streaming")


    cooldown_start = 0.0

    # FIXME: we should also have an 'unknown' state, probably, and if it
    # doesn't resolve to a proper state within a timeout, assume that we're
    # offline, to avoid accidentally recording things when we don't intend
    # to.
    while True:
        # print(state)

        if state == "WAITING":
            if check_streaming(args):
                log("INFO: OBS is streaming, switching to ong-online.target")

                # Only change state if we successfully started the target
                if start_target(args.online_start_unit):
                    state = "STREAMING"
                else:
                    log("ERROR: couldn't start ong-online.target, will retry")

        elif state == "STREAMING":
            if not check_streaming(args):
                log("INFO: OBS has stopped streaming, entering cooldown")
                cooldown_start = time.time()
                state = "COOLDOWN"
                continue

        elif state == "COOLDOWN":
            # Are we streaming again? If so, just go straight back into
            # streaming
            if check_streaming(args):
                log("INFO: OBS is streaming again, continuing")
                state = "STREAMING"
                continue

            # Otherwise, we can be in cooldown for 5 mins before we kill ffmpeg
            if (time.time() - cooldown_start) > (5 * 60):
                log("INFO: OBS cooldown has expired, switching to ong-offline.target")
                if start_target(args.offline_start_unit):
                    state = "WAITING"
                else:
                    # FIXME: what should we do if the systemctl call fails?
                    pass

        time.sleep(5)


if __name__ == "__main__":
    main()
