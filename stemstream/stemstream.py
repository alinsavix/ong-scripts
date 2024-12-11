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
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional

import toml
from discord_webhook import DiscordWebhook
from obsws_python.error import OBSSDKRequestError
from tdvutil import ppretty
from tdvutil.argparse import CheckFile


def log(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.stderr.flush()


subproc = None

def run_ffmpeg(args: argparse.Namespace):
    global subproc

    now = int(time.time())
    datestr = time.strftime("%Y-%m-%d %Hh%Mm%Ss", time.localtime(now))

    # ong-ffmpeg -loop 1 -i looper-thumbnail.jpg -re -i /tmp/test.mp3 -vn -c:a libfdk_aac -vbr 5 -cutoff 18000 -f flv rtmp://localhost/audio/stems

    ffmpeg_cmd = [
        "ong-ffmpeg", "-hide_banner", "-stats_period", "60",
        "-f", "alsa", "-channels", "6", "-sample_rate", "48000", "-c:a", "pcm_s24le", "-channel_layout", "6.0",
        "-i", "hw:CARD=UR44,DEV=0", "-af", "pan=2c|c0=c4|c1=c5",
        "-c:a", "libfdk_aac", "-vbr", "5", "-cutoff", "18000",
        "-vn", "-f", "flv", args.stream_url,
    ]
    # ffmpeg_cmd = ["py", "./sleep.py"]

    try:
        logpath = f"/ong/stems/stems {datestr}.log"
        log(f"INFO: ffmpeg startup, output logged to {logpath}")
        with open(logpath, "a") as logfile:
            print(f"COMMAND: {' '.join(ffmpeg_cmd)}", file=logfile)

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

    try:
        subproc.wait(15)
    except subprocess.TimeoutExpired:
        log("INFO: ffmpeg didn't exit on its own, outright killing it")
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


should_terminate = False
def handle_signal(signum, _frame):
    global should_terminate
    should_terminate = True
    log(f"INFO: caught signal {signum}, flagging for shutdown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream or record audio, when OBS is streaming")

    parser.add_argument(
        "--stream-url",
        type=str,
        help="host to which to stream audio"
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

    signal.signal(signal.SIGTERM, handle_signal)
    atexit.register(kill_ffmpeg)

    # Unconditionally start ffmpeg, then handle post-start things.
    # FIXME: handle ffmpeg failing to run
    run_ffmpeg(args)
    time.sleep(15)

    if not check_ffmpeg():
        log("ERROR: ffmpeg failed to start")
        send_discord(webhook_url, "status",
                     "Stem stream failed to start :(")
        sys.exit(1)

    # Looks like it at least started ok
    send_discord(webhook_url, "status",
                 "Stream online, stem stream started")

    # Now just... keep an eye on it
    while True:
        time.sleep(5)

        global should_terminate
        if should_terminate:
            log("INFO: shutting down ffmpeg")
            kill_ffmpeg()
            send_discord(webhook_url, "status",
                         "Stem stream ended normally")
            sys.exit(0)

        if not check_ffmpeg():
            # ffmpeg has died, sigh
            log("WARNING: ffmpeg died when we didn't expect it")
            send_discord(webhook_url, "status",
                         "Stem stream terminated unexpectedly")
            sys.exit(1)


if __name__ == "__main__":
    main()
