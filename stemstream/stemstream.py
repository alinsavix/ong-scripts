#!/usr/bin/env python3
# DISCLAIMER: This is some of the worst code I have ever written. I'm sorry.
# I was in a hurry. I'll probably do better at some point. That point is
# not going to be today.
#
# Seriously, don't model your code after *anything* you see here. Just
# look away. Please.
import argparse
import atexit
import logging
import subprocess
import sys
import time
from typing import List, Literal

import obsws_python as obs
from tdvutil import ppretty

state: Literal["WAITING", "STREAMING", "COOLDOWN"] = "WAITING"

subproc = None

def run_ffmpeg(args: argparse.Namespace):
    global subproc

    userpass = f"{args.stream_user}:{args.stream_pass}"
    icecast_url = f"icecast://{userpass}@{args.stream_host}:{args.stream_port}/ong-stems.mp3"
    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner",
        "-f", "alsa", "-channels", "6", "-sample_rate", "48000", "-c:a", "pcm_s24le", "-channel_layout", "6.0",
        "-i", "hw:CARD=UR44,DEV=0", "-af", "pan=2c|c0=c4|c1=c5",
        "-c:a", "libmp3lame", "-q:a", "2",
        "-f", "mp3", icecast_url,
    ]
    # ffmpeg_cmd = [ "sleep", "5"]

    try:
        print("INFO: ffmpeg startup, output logged to /tmp/stemstream.log", file=sys.stderr)
        with open('/tmp/stemstream.log', "a") as logfile:
            subproc = subprocess.Popen(
                ffmpeg_cmd, shell=False,
                stdin=subprocess.DEVNULL, stdout=logfile,
                stderr=subprocess.STDOUT
            )
    except FileNotFoundError:
        print("ERROR: couldn't execute ffmpeg, please make sure it exists in your PATH", file=sys.stderr)
        sys.exit(1)
    # except subprocess.TimeoutExpired:
    #     print(f"ERROR: remux process timed out after {args.timeout} seconds", file=sys.stderr)
    #     tmpfile.unlink(missing_ok=True)
    #     sys.exit(1)
    # except subprocess.CalledProcessError as e:
    #     print(
    #         f"ERROR: remux process failed with ffmpeg exit code {e.returncode}", file=sys.stderr)
    #     tmpfile.unlink(missing_ok=True)
    #     sys.exit(1)
    except Exception as e:
        print(f"ERROR: unknown error during streaming: {e}", file=sys.stderr)
        sys.exit(1)

def kill_ffmpeg():
    global subproc


    if subproc is None:
        print("INFO: asked to kill ffmpeg, but it was already dead", file=sys.stderr)
        return

    if subproc.poll() is None:
        print("INFO: asking ffmpeg to exit", file=sys.stderr)
        subproc.terminate()
        subproc.wait(5)

    if subproc.poll() is None:
        print("INFO: outright demanding ffmpeg to exit", file=sys.stderr)
        subproc.kill()

    print("INFO: I bask in your favor, I have killed the king^W ffmpeg", file=sys.stderr)
    subproc = None


def check_ffmpeg():
    global subproc

    if subproc is None:
        return False

    ret = subproc.poll()
    if ret is not None:  # has exited
        print(f"WARNING: ffmpeg has exited, code {ret}", file=sys.stderr)
        subproc = None
        return False

    # otherwise, it's still running
    return True


client = None

def connect_obs(host: str, port: int) -> obs.ReqClient:
    global clilent
    try:
        client = obs.ReqClient(host=host, port=port, timeout=5)
        return client
    except Exception as e:
        print(f"ERROR: couldn't connect to OBS: {e}")
        return None

# returns true if we're streaming
def get_streaming(args: argparse.Namespace) -> bool:
    global client

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream or record audio, when OBS is streaming")

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
        "--stream-host",
        type=str,
        default="localhost",
        help="host to which to stream audio"
    )

    parser.add_argument(
        "--stream-port",
        type=int,
        default=8000,
        help="host to which to stream audio"
    )

    parser.add_argument(
        "--stream-user",
        type=str,
        default="source",
        help="icecast source name to use when streaming",
    )

    parser.add_argument(
        "--stream-pass",
        type=str,
        default=None,
        help="icecast password to use when streaming"
    )

    return parser.parse_args()


def main():
    global state

    logging.basicConfig(level=logging.FATAL)
    args = parse_args()
    atexit.register(kill_ffmpeg)

    print("INFO: in startup, waiting for OBS to start streaming", file=sys.stderr)
    cooldown_start = 0.0

    while True:
        # print(state)
        time.sleep(5)

        if state == "WAITING":
            if get_streaming(args):
                print("INFO: OBS is streaming, starting ffmpeg", file=sys.stderr)
                run_ffmpeg(args)
                state = "STREAMING"

        elif state == "STREAMING":
            if not check_ffmpeg():
                # ffmpeg has died, sigh
                print("WARNING: ffmpeg died when we didn't expect it, restarting", file=sys.stderr)
                state = "WAITING"  # will restart next check
                continue

            if not get_streaming(args):
                print("INFO: OBS has stopped streaming, entering cooldown", file=sys.stderr)
                cooldown_start = time.time()
                state = "COOLDOWN"
                continue

        elif state == "COOLDOWN":
            # Are we streaming again? If so, just go straight back into
            # streaming
            if get_streaming(args):
                print("INFO: OBS is streaming again, continuing", file=sys.stderr)
                state = "STREAMING"
                continue

            # Otherwise, we can be in cooldown for 5 mins before we kill ffmpeg
            if (time.time() - cooldown_start) > (5 * 60):
                print("INFO: OBS cooldown has expired, killing ffmpeg", file=sys.stderr)
                kill_ffmpeg()
                state = "WAITING"

            continue


if __name__ == "__main__":
    main()
