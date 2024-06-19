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
from pathlib import Path
from typing import List, Literal, Optional

import obsws_python as obs
import toml
from discord_webhook import DiscordWebhook
from obsws_python.error import OBSSDKRequestError
from tdvutil import ppretty
from tdvutil.argparse import CheckFile

state: Literal["WAITING", "STREAMING", "COOLDOWN"] = "WAITING"

def log(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.stderr.flush()


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
    # ffmpeg_cmd = ["py", "./sleep.py"]

    try:
        log("INFO: ffmpeg startup, output logged to /tmp/stemstream.log")
        with open('/tmp/stemstream.log', "a") as logfile:
            # with open('d:/temp/stemstream.log', "a") as logfile:
            subproc = subprocess.Popen(
                ffmpeg_cmd, shell=False,
                stdin=subprocess.DEVNULL, stdout=logfile,
                stderr=subprocess.STDOUT
            )
    except FileNotFoundError as e:
        log(f"ERROR: couldn't execute ffmpeg: {e}")
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
        log(f"ERROR: unknown error during streaming: {e}")
        sys.exit(1)

def kill_ffmpeg():
    global subproc


    if subproc is None:
        log("INFO: asked to kill ffmpeg, but it was already dead")
        return

    if subproc.poll() is None:
        log("INFO: asking ffmpeg to exit")
        subproc.terminate()
        subproc.wait(5)

    if subproc.poll() is None:
        log("INFO: outright demanding ffmpeg to exit")
        subproc.kill()

    log("INFO: I bask in your favor, I have killed the king^W^W ffmpeg")
    subproc = None


def check_ffmpeg():
    global subproc

    if subproc is None:
        return False

    ret = subproc.poll()
    if ret is not None:  # has exited
        log(f"WARNING: ffmpeg has exited, code {ret}")
        subproc = None
        return False

    # otherwise, it's still running
    return True

def get_webhook_url(cfgfile: Path) -> str:
    log(f"INFO: loading config from {cfgfile}")
    config = toml.load(cfgfile)

    try:
        return config["stemstream"]["webhook_url"]
    except KeyError:
        log("ERROR: missing 'webhook_url' in config")
        sys.exit(1)

# send a message to discord, of a given type, but only if the last
# message of that type was different
def send_discord(webhook_url: Optional[str], msg_type: str, msg: str) -> None:
    # a stupid trick for persistent function variables
    if not hasattr(send_discord, "last_sent"):
        send_discord.last_sent = {}  # msg type -> message

    if webhook_url is None:
        return

    checkfile = Path(__file__).parent / "no_discord"
    if checkfile.exists():
        log(f"safe mode, not sending discord updates (to resume: rm {checkfile})")
        return

    # if msg in last_sent and now() - last_sent[msg] < 60:
    if msg == send_discord.last_sent.get(msg_type, ""):  # already sent this one!
        log(f"DUPLICATE: {msg}")
        return

    send_discord.last_sent[msg_type] = msg

    webhook = DiscordWebhook(url=webhook_url, content=msg)
    response = webhook.execute()
    if response.status_code == 200:
        log(f"SENT: {msg}")
    else:
        log(f"FAILED: (status {response.status_code}): {msg}")

    return


client = None

def connect_obs(host: str, port: int) -> obs.ReqClient:
    global client

    # the logic here sucks
    try:
        client = obs.ReqClient(host=host, port=port, timeout=5)
        return client
    except OBSSDKRequestError as e:
        code = e.code
    except (ConnectionRefusedError, OSError):
        # log("ERROR: Connection Refused"
        return None
    except Exception as e:
        log(f"ERROR: couldn't connect to OBS: {e}")
        return None

    if code != 207:
        log(f"WARNING: Unknown OBS response code {code}")
        return None


    log(f"WARNING: OBS not ready, retrying in 30 seconds")
    time.sleep(30)
    return connect_obs(host, port)


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

    # FIXME: move these to credentials file?
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

    parser.add_argument(
        "--credentials-file", "-c",
        type=Path,
        default=None,
        action=CheckFile(must_exist=True),
        help="file with discord credentials"
    )

    return parser.parse_args()


def main():
    global state

    logging.basicConfig(level=logging.FATAL)
    args = parse_args()

    if args.credentials_file is not None:
        webhook_url = get_webhook_url(args.credentials_file)
    else:
        webhook_url = None

    atexit.register(kill_ffmpeg)

    log("INFO: in startup, waiting for OBS to start streaming")
    cooldown_start = 0.0

    while True:
        time.sleep(5)

        if state == "WAITING":
            if get_streaming(args):
                log("INFO: OBS is streaming, starting ffmpeg")
                send_discord(webhook_url, "status", "Stream online, starting stem stream")
                run_ffmpeg(args)
                state = "STREAMING"

        elif state == "STREAMING":
            if not check_ffmpeg():
                # ffmpeg has died, sigh
                log("WARNING: ffmpeg died when we didn't expect it, restarting")
                send_discord(webhook_url, "status",
                             "Unexpected termination of stem stream (will retry)")
                state = "WAITING"  # will restart next check
                continue

            if not get_streaming(args):
                log("INFO: OBS has stopped streaming, entering cooldown")
                send_discord(webhook_url, "status", "Stream offline, starting cooldown")
                cooldown_start = time.time()
                state = "COOLDOWN"
                continue

        elif state == "COOLDOWN":
            # Are we streaming again? If so, just go straight back into
            # streaming
            if get_streaming(args):
                log("INFO: OBS is streaming again, continuing")
                send_discord(webhook_url, "status",
                             "Stream back online, continuing with stem stream")
                state = "STREAMING"
                continue

            # Otherwise, we can be in cooldown for 5 mins before we kill ffmpeg
            if (time.time() - cooldown_start) > (5 * 60):
                log("INFO: OBS cooldown has expired, killing ffmpeg")
                send_discord(webhook_url, "status", "Cooldown ended, terminating stem stream")
                kill_ffmpeg()
                state = "WAITING"

            continue


if __name__ == "__main__":
    main()
