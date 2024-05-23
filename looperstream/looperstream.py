#!/usr/bin/env python3
# DISCLAIMER: This is still some of the worst code I have ever written. I'm
# sorry. I was in a hurry. I'll probably do better at some point. Maybe
# combine with the stemstream code.
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

subproc = None

def run_ffmpeg(args: argparse.Namespace):
    global subproc

    drawtext_conf = "font=mono:fontsize=48:y=h-text_h-15:box=1:boxcolor=black:boxborderw=10:fontcolor=white:expansion=normal"

    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner",
        "-f", "v4l2", "-framerate", str(args.stream_fps), "-video_size", args.stream_res,
        "-i", args.camera_device,
        "-vf", f"drawtext=x=15:text='%{{localtime}}':{drawtext_conf},drawtext=x=w-text_w-15:text='%{{n}}':{drawtext_conf}",
        "-c:v", "libx264", "-r", str(args.stream_fps),
        "-b:v", "0", "-maxrate", args.stream_bitrate, "-bufsize", "2000k",
        "-g", "10", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "veryfast",
        "-an", "-f", "flv", args.stream_url,
    ]
    # ffmpeg_cmd = ["py", "./sleep.py"]

    try:
        print("INFO: ffmpeg startup, output logged to /tmp/looperstream.log", file=sys.stderr)
        with open('/tmp/looperstream.log', "a") as logfile:
            # with open('d:/temp/looperstream.log', "a") as logfile:
            subproc = subprocess.Popen(
                ffmpeg_cmd, shell=False,
                stdin=subprocess.DEVNULL, stdout=logfile,
                stderr=subprocess.STDOUT
            )
    except FileNotFoundError as e:
        print(
            f"ERROR: couldn't execute ffmpeg: {e}", file=sys.stderr)
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

def get_webhook_url(cfgfile: Path) -> str:
    print(f"INFO: loading config from {cfgfile}", file=sys.stderr)
    config = toml.load(cfgfile)

    try:
        return config["looperstream"]["webhook_url"]
    except KeyError:
        print("ERROR: missing 'webhook_url' in config", file=sys.stderr)
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
        print(
            f"safe mode, not sending discord updates (to resume: rm {checkfile})", file=sys.stderr)
        return

    # if msg in last_sent and now() - last_sent[msg] < 60:
    if msg == send_discord.last_sent.get(msg_type, ""):  # already sent this one!
        print(f"DUPLICATE: {msg}", file=sys.stderr)
        return

    send_discord.last_sent[msg_type] = msg

    webhook = DiscordWebhook(url=webhook_url, content=msg)
    response = webhook.execute()
    if response.status_code == 200:
        print(f"SENT: {msg}", file=sys.stderr)
    else:
        print(f"FAILED: (status {response.status_code}): {msg}", file=sys.stderr)

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
        # print("ERROR: Connection Refused", file=sys.stderr)
        return None
    except Exception as e:
        print(f"ERROR: couldn't connect to OBS: {e}", file=sys.stderr)
        return None
    if code != 207:
        print(f"WARNING: Unknown OBS response code {code}", file=sys.stderr)
        return None


    print(f"WARNING: OBS not ready, retrying in 30 seconds", file=sys.stderr)
    time.sleep(3)
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


def play_source(args: argparse.Namespace, scene: str, source: str) -> None:
    global client

    if client is None:
        client = connect_obs(args.host, args.port)

    if client is None:
        return

    try:
        r = client.get_scene_item_id(scene, source)
        id = r.scene_item_id

        client.set_scene_item_enabled(scene, id, True)
        time.sleep(5)
        client.set_scene_item_enabled(scene, id, False)
    except Exception as e:
        print(f"ERROR: couldn't play timecode source: {e}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream the loopercam, when OBS is streaming")

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
        "--timecode-scene",
        type=str,
        help="scene of timecode source in OBS, for automatic display"
    )

    parser.add_argument(
        "--timecode-source",
        type=str,
        help="source name of timecode source in OBS, for automatic display"
    )

    parser.add_argument(
        "--camera-device",
        type=str,
        default="/dev/video0",
        help="which camera to stream from"
    )

    parser.add_argument(
        "--stream-url",
        type=str,
        help="host to which to stream audio",
        required=True,
    )

    parser.add_argument(
        "--stream-res",
        type=str,
        default="1280x720",
        help="resolution at which to stream"
    )

    parser.add_argument(
        "--stream-fps",
        type=int,
        default=5,
        help="framerate at which to stream",
    )

    parser.add_argument(
        "--stream-bitrate",
        type=str,
        default="600k",
        help="bitrate at which to stream"
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

    print("INFO: in startup, waiting for OBS to start streaming", file=sys.stderr)
    cooldown_start = 0.0

    while True:
        # print(state)
        time.sleep(5)

        if state == "WAITING":
            if get_streaming(args):
                print("INFO: OBS is streaming, starting ffmpeg", file=sys.stderr)
                send_discord(webhook_url, "status", "Stream online, starting looper stream")
                run_ffmpeg(args)
                state = "STREAMING"
                if args.timecode_scene is not None:
                    time.sleep(30)   # this sucks
                    play_source(args, args.timecode_scene, args.timecode_source)

        elif state == "STREAMING":
            if not check_ffmpeg():
                # ffmpeg has died, sigh
                print("WARNING: ffmpeg died when we didn't expect it, restarting", file=sys.stderr)
                send_discord(webhook_url, "status",
                             "Unexpected termination of looper stream (will retry)")
                state = "WAITING"  # will restart next check
                continue

            if not get_streaming(args):
                print("INFO: OBS has stopped streaming, entering cooldown", file=sys.stderr)
                send_discord(webhook_url, "status", "Stream offline, starting cooldown")
                cooldown_start = time.time()
                state = "COOLDOWN"
                continue

        elif state == "COOLDOWN":
            # Are we streaming again? If so, just go straight back into
            # streaming
            if get_streaming(args):
                print("INFO: OBS is streaming again, continuing", file=sys.stderr)
                send_discord(webhook_url, "status",
                             "Stream back online, continuing with looper stream")
                state = "STREAMING"
                continue

            # Otherwise, we can be in cooldown for 5 mins before we kill ffmpeg
            if (time.time() - cooldown_start) > (5 * 60):
                print("INFO: OBS cooldown has expired, killing ffmpeg", file=sys.stderr)
                send_discord(webhook_url, "status", "Cooldown ended, terminating looper stream")
                kill_ffmpeg()
                state = "WAITING"

            continue


if __name__ == "__main__":
    main()
