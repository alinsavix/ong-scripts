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
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional

import obsws_python as obs
import toml
from discord_webhook import DiscordWebhook
from obsws_python.error import OBSSDKRequestError
from tdvutil import ppretty, sec_to_shortstr
from tdvutil.argparse import CheckFile

state: Literal["WAITING", "STREAMING", "COOLDOWN"] = "WAITING"

def log(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.stderr.flush()


def config_camera(script: Path, device: Path):
    config_cmd = [
        script, device
    ]

    log("INFO: running camera configuration script")

    try:
        subprocess.run(config_cmd, shell=False, check=True, timeout=15)
        log("INFO: Camera configuration script completed ok")
    except subprocess.TimeoutExpired:
        log(f"WARNING: Camera configuration script '{script}' timed out")
    except subprocess.CalledProcessError as e:
        log(f"WARNING: Camera config script exited with an error: {e}")
    except Exception as e:
        log(f"WARNING: Camera config script execution failed with unknown error: {e}")


subproc = None

def run_ffmpeg(args: argparse.Namespace):
    global subproc


    now = int(time.time())
    datestr = time.strftime("%Y-%m-%d %Hh%Mm%Ss", time.localtime(now))

    drawtext_conf = "font=mono:fontsize=48:y=h-text_h-15:box=1:boxcolor=black:boxborderw=10:fontcolor=white:expansion=normal"

    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner",  # "-loglevel", "error",
        "-stats", "-stats_period", "60", "-hwaccel", "auto",

        # Needed to get the actual timestamp of the first frame in the logs.
        # Unfortunately, not in the video, and our drawtext filter won't use
        # this value, but we can replace/overwrite it later.
        "-use_wallclock_as_timestamps", "1",

        # mjpeg is required to get more than like 7.5fps from this (USB2),
        # so we use it. The camera actually embeds a h264 stream in the mjpeg
        # stream, which would be great to use, but ffmpeg doesn't seem to
        # support actually extracting it.
        "-f", "v4l2", "-input_format", "mjpeg",
        "-framerate", str(args.stream_fps), "-video_size", args.stream_res,
        "-i", args.camera_device,
        "-r", str(args.stream_fps), "-fps_mode", "cfr",

        # Burn in an approximate timestamp. Note that this timestamp won't be
        # very precise, because it's the timestamp of the time a frame is
        # processed (which doesn't quite happen in realtime) rather than when
        # the frame is received.
        #
        # Sending the output to [v0out] as a separate stream lets us have a
        # separate stream to record, and still be able to copy the original
        # stream to the loopback video device we want to use.
        "-filter_complex", f"drawtext=x=15:text='RTC %{{localtime\\:%Y-%m-%d %T.%3N}}':{drawtext_conf}[v0out]",

        # FIXME: See if there are better settings for our specific use case,
        # which is an atypical one.
        "-c:v", "h264_nvenc", "-preset", "p5", "-pix_fmt", "yuv420p",
        "-b:v", "0", "-maxrate", args.stream_bitrate, "-bufsize", "2000k",

        # Keep a relatively small GOP to preserve seekability
        "-g", str(args.stream_fps * 2),

        # This is the main stream (the one with burned in timestamp)
        "-map", "[v0out]", f"/ong/looper/looper-{datestr}.flv",
    ]

    if args.mirror_device is not None:
        # Also copy the original stream to a v4l2 loopback device, so that we
        # can get at it with e.g. ustreamer or other things that need access
        # to the same video stream.
        ffmpeg_cmd += ["-c:v", "copy", "-f", "v4l2", "-map", "0:v", "-y", str(args.mirror_device)]

        # f"/ong/looper/looper-{datestr}.mp4|[f=flv:onfail=ignore:fifo_options=attempt_recovery=1\\\\:recover_any_error=1\\\\:drop_pkts_on_overflow=1\\\\:fifo_format=flv]{args.stream_url}"
        # "-f", "flv", args.stream_url,

    # ffmpeg_cmd = ["py", "./sleep.py"]

    try:
        logpath = f"/ong/looper/looper-{datestr}.log"
        log(f"INFO: ffmpeg startup, output logged to {logpath}")
        with open(logpath, "a") as logfile:
            print(f"COMMAND: {' '.join(ffmpeg_cmd)}", file=logfile)

            # with open('d:/temp/looperstream.log', "a") as logfile:
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
        subproc.wait(15)

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
        return config["looperstream"]["webhook_url"]
    except KeyError:
        log("ERROR: missing 'webhook_url' in config")
        sys.exit(1)

# send a message to discord, of a given type, but only if the last
# message of that type was different
def send_discord(args: argparse.Namespace, webhook_url: Optional[str], msg_type: str, msg: str) -> None:
    # a stupid trick for persistent function variables
    if not hasattr(send_discord, "last_sent"):
        send_discord.last_sent = {}  # msg type -> message

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
    except OBSSDKRequestError as e:
        code = e.code
    except (OSError):
        # log("ERROR: Connection Refused")
        return None
    except Exception as e:
        log(f"ERROR: couldn't connect to OBS: {e}")
        return None
    else:
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
    except Exception:
        return None
    else:
        if uptime == 0:
            return None

        return uptime


def play_source(args: argparse.Namespace, scene: str, source: str) -> None:
    global client

    if client is None:
        client = connect_obs(args.host, args.port)

    if client is None:
        return

    try:
        r = client.get_scene_item_id(scene, source)
        sceneid = r.scene_item_id

        client.set_scene_item_enabled(scene, sceneid, True)
        time.sleep(5)
        client.set_scene_item_enabled(scene, sceneid, False)
    except Exception as e:
        log(f"ERROR: couldn't play timecode source: {e}")


should_terminate = False
def handle_signal(signum, _frame):
    global should_terminate
    should_terminate = True
    log(f"INFO: caught signal {signum}, flagging for shutdown")


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

    # FIXME: Make optional
    parser.add_argument(
        "--mirror-device",
        type=Path,
        default=None,
        help="v4l2 loopback device to copy stream to"
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

    parser.add_argument(
        "--camera-config-script",
        type=Path,
        default=None,
        action=CheckFile(must_exist=True),
        help="script to execute to configure camera after it has been opened"
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

    if args.mirror_device is not None:
        if not args.mirror_device.exists() or not args.mirror_device.is_char_device():
            log(f"ERROR: mirror device {args.mirror_device} does not exist or is not a character device")
            sys.exit(1)

    signal.signal(signal.SIGTERM, handle_signal)
    atexit.register(kill_ffmpeg)

    # Unconditionally start ffmpeg, then handle post-start things.
    # FIXME: handle ffmpeg failing to run
    run_ffmpeg(args)
    time.sleep(15)

    if not check_ffmpeg():
        log("ERROR: ffmpeg failed to start")
        send_discord(args, webhook_url, "status",
                     "Looper recording failed to start :(")
        sys.exit(1)

    # Looks like it at least started ok
    send_discord(args, webhook_url, "status",
                 "Stream online, looper recording started")

    if args.camera_config_script:
        time.sleep(10)  # this sucks
        log("INFO: configuring camera")
        config_camera(args.camera_config_script, args.camera_device)

    if args.timecode_scene is not None:
        time.sleep(30)  # this also sucks
        log("INFO: flashing timecode on screen")
        # make sure this never fails
        # FIXME: Do better
        try:
            uptime = get_obs_uptime(args.host, args.port)
            if uptime is not None and uptime > 25 and uptime < 300:
                play_source(args, args.timecode_scene, args.timecode_source)
        except Exception as e:
            pass

    # Now just... keep an eye on it
    while True:
        # print(state)
        time.sleep(5)

        global should_terminate
        if should_terminate:
            log("INFO: shutting down ffmpeg")
            kill_ffmpeg()
            send_discord(args, webhook_url, "status",
                         "Looper recording ended normally")
            sys.exit(0)

        if not check_ffmpeg():
            # ffmpeg has died, sigh
            log("WARNING: ffmpeg died when we didn't expect it")
            send_discord(args, webhook_url, "status",
                         "Looper recording terminated unexpectedly")
            sys.exit(1)


if __name__ == "__main__":
    main()
