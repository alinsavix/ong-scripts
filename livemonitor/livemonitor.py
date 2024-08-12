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

state: Literal["WAITING", "STREAMING", "UNKNOWN"]

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
def check_streaming(args: argparse.Namespace) -> Optional[bool]:
    global client

    if args.force:
        return True

    if client is None:
        client = connect_obs(args.host, args.port)

    if client is None:
        return None

    try:
        r = client.get_stream_status()
    except Exception as e:  # FIXME: make more granular
        r = None

    # if we failed to get the status for whatever reason, reconnect and
    # try again
    if r is None:
        try:
            connect_obs(args.host, args.port)
            r = client.get_stream_status()
        except Exception as e:  # FIXME: make more granular
            # log(f"ERROR: couldn't get OBS stream status: {e}")
            return None

    return r.output_active


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
        log("INFO: in startup, initial state UNKNOWN")


    state = "UNKNOWN"
    cooldown_start = time.time()

    # State flow:
    # If we can get an absolute "yes" or "no" from check_streaming, go directly
    # to STREAMING or WAITING, respectively.
    #
    # If we don't know what's actually happening because we can't talk to OBS,
    # sit in an unknown state for ~5 minutes, then transition to WAITING.
    #
    # FIXME: could probably do a good bit of code deduplication here
    while True:
        # print(state)

        is_streaming = check_streaming(args)

        if state == "WAITING":
            # Only check to see if we're streaming now. We don't care if
            # is_streaming is None, because there's never any reason to
            # transition to UNKNOWN from WAITING.
            if is_streaming is True:
                log("INFO: OBS is streaming, switching to ong-online.target")

                # Only change state if we successfully started the target
                if start_target(args.online_start_unit):
                    state = "STREAMING"
                else:
                    log("ERROR: couldn't start ong-online.target, will retry")

        elif state == "STREAMING":
            if is_streaming is False:
                log("INFO: OBS has stopped streaming, switching to ong-offline.target")
                # Only change state if we successfully started the target
                if start_target(args.offline_start_unit):
                    state = "WAITING"
                else:
                    log("ERROR: couldn't start ong-offline.target, will retry")

            elif is_streaming is None:
                log("WARNING: OBS status unknown, entering cooldown")
                cooldown_start = time.time()
                state = "UNKNOWN"

        elif state == "UNKNOWN":
            # Are we streaming (again)? If so, go straight to STREAMING. Go
            # ahead and start the target, too, because we don't know if it's
            # actually started and it's harmless to start it twice.
            if is_streaming is True:
                log("INFO: UNKNOWN > STREAMING, starting ong-online.target")
                if start_target(args.online_start_unit):
                    state = "STREAMING"
                else:
                    log("ERROR: couldn't start ong-online.target, will retry")

            elif is_streaming is False:
                log("INFO: UNKNOWN > WAITING, starting ong-offline.target")
                if start_target(args.offline_start_unit):
                    state = "WAITING"
                else:
                    log("ERROR: couldn't start ong-offline.target, will retry")

            # if not true or false, it's unknown still
            else:
                # how long have we been unknown?
                if (time.time() - cooldown_start) > (5 * 60):
                    log("INFO: UNKNOWN > WAITING (expired timer), switching to ong-offline.target")

                    if start_target(args.offline_start_unit):
                        state = "WAITING"
                    else:
                        log("ERROR: couldn't start ong-offline.target, will retry")

        time.sleep(5)


if __name__ == "__main__":
    main()
