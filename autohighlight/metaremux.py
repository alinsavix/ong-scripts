#!/usr/bin/env python3
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import ffmpeg
from PIL import Image
from tdvutil import hms_to_sec, ppretty
from tdvutil.argparse import CheckFile

# ! WARNING: The time handling in this code is *very* fucky-wucky.
# Really, we should do better, because it's *bad*. The problem is that we
# always want our video timestamps and timecode to represent local time
# where the recording was taken, but a lot of the various time conversion
# routines assume UTC. So a lot of the time manipulations, we do manually,
# just so we get the right timestmaps on the other side. We keep the
# time values in string form most of the time to help with this.
#
# Really do need to figure out a better way to deal with it, because did
# I mention that the code here suuuuuucks?


if platform.system() == "Windows":
    os.environ["TESSDATA_PREFIX"] = "C:/Program Files/Tesseract-OCR/tessdata"
import tesserocr

# give ourselves a place to stuff our indexes
script_dir = Path(__file__).parent.resolve()
INDEX_DIR = script_dir / "indexes"
INDEX_DIR.mkdir(exist_ok=True)


@dataclass
class VideoMeta:
    filename: str
    video_type: str
    start_time: Optional[str]
    end_time: Optional[str]
    duration: float
    fps: int  # only support integer fps for now
    size_bytes: int

# FIXME: do we need this?
@dataclass
class FoundTimestamp:
    ocr_ts: str
    frame_ts: float
    fps: float


def add_duration(timestamp: str, duration: float) -> str:
    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
    dt = dt + timedelta(seconds=duration)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")


# Regex for finding our burned-in timecodes
re_timecode = re.compile(r"""
    RTC
    \s
    (?P<full_timestamp>
        (?P<year>\d{4}) - (?P<month>\d{2}) - (?P<day>\d{2})
        \s
        (?P<time_only>
            (?P<hour>\d{2}) : (?P<minute>\d{2}) : (?P<second>\d{2}) \. (?P<millisecond>\d{3})
        )
    )
""", re.VERBOSE)


# modeled after https://medium.com/nanonets/a-comprehensive-guide-to-ocr-with-tesseract-opencv-and-python-fd42f69e8ca8
# FIXME: make this a class so we can have a persistent tesseract object
ocrapi = None
def ocr_frame(frame: cv2.typing.MatLike) -> Optional[str]:
    global ocrapi

    if ocrapi is None:
        ocrapi = tesserocr.PyTessBaseAPI()
        # ocrapi.SetVariable("tessedit_do_invert", "1")
        ocrapi.SetPageSegMode(tesserocr.PSM.SINGLE_BLOCK)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # snip out the place we expect the timecode, based on which feed we think
    # it is, based on the frame size
    #
    # FIXME: This is kinda a stupid way to do this
    if frame.shape[0] > 1000:
        # Main stream feed, probably
        subimage = gray[800:900, 165:]
    else:
        # Probably the looper feed
        subimage = gray[660:710, 5:805]

    # We probably don't actually *need* to do a threshold on the time text,
    # but this will at least clean up any compression artifacting.
    subimage = cv2.threshold(subimage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # cv2.imshow("frame", subimage)
    # cv2.waitKey(1)

    # Raw, slow pytesseract, here as a fallback in case we need it
    # custom_config = r'--oem 3 --psm 6'
    # ocr = pytesseract.image_to_string(subimage, config=custom_config)

    # We should really see if we can find a way to read these frames directly
    # into PIL instead of going through opencv's numpy arrays first
    pil_image = Image.fromarray(subimage)
    ocrapi.SetImage(pil_image)

    ocr = ocrapi.GetUTF8Text()

    m = re_timecode.search(ocr)
    if not m:
        return None

    # cv2.imshow("frame", subimage)
    # cv2.waitKey(10000)

    # print(f"OCR'd timestamp string as: {ocr}")
    try:
        full_timestamp = datetime.strptime(m.group("full_timestamp"), "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return None

    full_timestamp_str = m.group("full_timestamp")
    # return full_timestamp.timestamp()
    return full_timestamp_str


# returns ocr'd timestamp info, or None if not found
def find_timestamp_in_range(args: argparse.Namespace, filename: Path, start_time: float, search_len: float) -> Optional[FoundTimestamp]:
    # frame_offset = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    # cap = VideoGear(source=str(args.filenames[0]), logging=True).start()
    cap = cv2.VideoCapture(str(filename), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        # FIXME: Is this the right exception?
        raise OSError(f"Couldn't open video stream from {filename}")

    cap.set(cv2.CAP_PROP_POS_MSEC, int(start_time * 1000))
    if args.debug:
        print(f"INFO: Looking for timestamp at offset {start_time}s...", end="")
        sys.stdout.flush()

    fps = cap.get(cv2.CAP_PROP_FPS)
    framenum = -1
    while cap.isOpened():
        ret, frame = cap.read()

        # Couldn't read the next frame, bail
        if frame is None:
            return None

        framenum += 1

        frame_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if frame_ts > (start_time + search_len):
            cap.release()
            if args.debug:
                print("")
            return None

        if args.debug:
            print(".", end="")
            sys.stdout.flush()

        ts = ocr_frame(frame)
        if ts is not None:
            cap.release()
            if args.debug:
                print(f"FOUND: {ts}")
            return FoundTimestamp(ts, frame_ts, fps)

    cap.release()

    if args.debug:
        print("")

    return None


def dt_to_secs(dt: str) -> float:
    tc = dt.split(" ")[1]
    # dt.strftime("%H:%M:%S.%f")
    return hms_to_sec(tc)


def dt_to_timecode(dt: str, fps: float) -> str:
    tc = dt.split(" ")[1]
    # dt.strftime("%H:%M:%S")
    frac = float(tc.split(":")[2]) % 1
    frames = int(frac * fps)

    tc = tc.split(".")[0]

    return f"{tc}:{frames:02d}"


# FIXME: Only expected to work with integer framerates.
def remux_with_timecode(args: argparse.Namespace, video_file: Path, remux_file: Path, metadata: VideoMeta) -> bool:
    # Not writing actual timecode right now
    # if metadata.start_time is None:
    #     print(f"WARNING: No start time found for '{video_file}', skipping")
    #     return False

    # timecode = dt_to_timecode(metadata.start_time, metadata.fps)
    # secs = dt_to_secs(metadata.start_time)
    # print(f"INFO: using time {secs} -> {timecode}")

    timeout = 30 * 60

    fh, _tmpfile = tempfile.mkstemp(suffix=".mp4", prefix="remux_", dir=remux_file.parent)
    os.close(fh)
    tmpfile = Path(_tmpfile)

    # log(f"remuxing w/ timecode using tmpfile {tmpfile}")
    print(f"REMUXING '{video_file.name}'")

    # don't do a burned-in timecode right now
    burn_in = False
    if not burn_in:
        cmd = [
            "ffmpeg", "-hide_banner", "-hwaccel", "auto",
            # Something is weird when inserting the timecode, it gets the
            # wrong fps for it (giving 62.5 instead of 60), so skip the timecode
            # for now and just do a straight remux
            # "-i", str(video_file), "-map", "0", "-map", "-0:d", "-c", "copy",
            # "-timecode", timecode, "-video_track_timescale", "30000",
            "-i", str(video_file), "-map", "0", "-c", "copy",
            "-y", str(tmpfile)
        ]
    # else:
    #     drawtext_conf = "font=mono:fontsize=48:y=h-text_h-15:box=1:boxcolor=black:boxborderw=10|300:fontcolor=white:expansion=normal"

    #     if platform.system() == "Windows":
    #         encoder = ["-c:v", "h264_nvenc", "-preset", "p5"]
    #     elif platform.system() == "Darwin":
    #         encoder = ["-c:v", "h264_videotoolbox", "-coder", "cabac"]
    #     else:
    #         encoder = ["-c:v", "libx264", "-preset", "medium"]

    #     cmd = [
    #         "ffmpeg", "-hide_banner", "-stats_period", "10", "-hwaccel", "auto",
    #         "-i", str(video_file), "-map", "0", "-map", "-0:d",
    #         # "-vf", f"drawtext=x=15:text='RTC %{{localtime\\:%Y-%m-%d %T.%3N}}':{drawtext_conf},drawtext=x=w-text_w-15:text='%{{n}}':{drawtext_conf}",
    #         "-vf", f"drawtext=x=15:text='RTC %{{pts\\:hms\\:{secs}}}':{drawtext_conf}",
    #         "-r", str(metadata.fps), "-fps_mode", "cfr",
    #         *encoder, "-b:v", "1000k",
    #         "-timecode", timecode, "-video_track_timescale", "30000",
    #         "-y", str(tmpfile)
    #     ]

    if args.dry_run:
        print("DRY RUN only, remux command would have been: " + " ".join(cmd))
        tmpfile.unlink(missing_ok=True)
        return True

    print(f"Running command: {' '.join(cmd)}")

    try:
        subprocess.run(args=cmd, shell=False, stdin=subprocess.DEVNULL, check=True, timeout=timeout)
    except FileNotFoundError:
        print("ERROR: couldn't execute ffmpeg, please make sure it exists in your PATH")
        tmpfile.unlink(missing_ok=True)
        return False
    except subprocess.TimeoutExpired:
        print(f"ERROR: remux-with-timecode process timed out after {timeout} seconds")
        tmpfile.unlink(missing_ok=True)
        return False
    except subprocess.CalledProcessError as e:
        print(f"ERROR: remux-with-timecode process failed with ffmpeg exit code {e.returncode}")
        tmpfile.unlink(missing_ok=True)
        return False
    except KeyboardInterrupt:
        print("ERROR: remux-with-timecode process interrupted")
        tmpfile.unlink(missing_ok=True)
        return False
    except Exception:
        print("ERROR: unknown error during remux-with-timecode process, aborting")
        tmpfile.unlink(missing_ok=True)
        return False

    # seems like it ran ok, rename the temp file
    # log(f"DEBUG: replacing {vidfile} with {tmpfile}")
    tmpfile.replace(remux_file)

    if not args.keep_original:
        video_file.unlink()

    return True


def gen_metadata(args: argparse.Namespace, video_file: Path) -> Optional[VideoMeta]:
    try:
        probeinfo = ffmpeg.probe(video_file)
    except ffmpeg.Error as e:
        print(f"ffmpeg couldn't open {video_file}, skipping")
        return

    # Find the first video stream
    video_stream = next((stream for stream in probeinfo['streams'] if stream['codec_type'] == 'video'), None)
    if video_stream is None:
        print(f"No video stream found in {video_file}, skipping")
        return None

    framerate_str = video_stream['r_frame_rate']
    framerate_num, framerate_den = map(int, framerate_str.split('/'))
    framerate = round(framerate_num / framerate_den)

    filename = video_file.name
    duration = float(probeinfo['format']['duration'])
    size_bytes = int(probeinfo['format']['size'])

    # Extract video type from filename
    video_type = filename.split(' ')[0]

    timestamp = find_timestamp_in_range(args, video_file, 55, 30)

    if timestamp is None:
        # print("No timestamp found")
        start_time = None
        end_time = None
    else:
        # print(f"Found timestamp: {timestamp}")
        start_time = add_duration(timestamp.ocr_ts, -timestamp.frame_ts)
        end_time = add_duration(start_time, duration)

    metadata = VideoMeta(filename, video_type, start_time, end_time, duration, framerate, size_bytes)
    return metadata


def process_video(args: argparse.Namespace, video_file: Path, remuxed_file: Path, metadata_file: Path):
    # if we already have metadata, and we're not forcing, skip
    if metadata_file.exists() and not args.force:
        print(f"Skipping '{video_file}': metadata already exists")
        return

    print(f"Processing '{video_file}':")

    metadata = gen_metadata(args, video_file)
    if metadata is None:
        print(f"Failed to generate metadata for '{video_file}'")
        print("WARNING: File will not be remuxed")
        return

    if metadata.start_time is None:
        print(f"WARNING: No start time found for '{video_file}'")

    # if metadata.start_time is not None:
    #     print(f"Wrote metadata for '{video_file}' (timestamp found)")
    # else:
    #     print(f"Wrote metadata for '{video_file}' (no timestamp found)")

    ret = remux_with_timecode(args, video_file, remuxed_file, metadata)
    if ret is True:
        print(f"Wrote remuxed video to '{remuxed_file}'")
        metadata.filename = remuxed_file.name
        metadata_json = json.dumps(asdict(metadata), indent=4)

        with metadata_file.open("w") as f:
            f.write(metadata_json)
        print(f"Wrote metadata to '{metadata_file}'")
    else:
        print(f"ERROR: Failed to remux video for '{video_file}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ginerate Ong video metadata",
    )

    parser.add_argument(
        "--remux-dest-dir",
        type=Path,
        default=Path("."),
        action=CheckFile(must_exist=True),
        help="Directory to remux video files into"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing metadata",
    )

    parser.add_argument(
        "--force-remux",
        action="store_true",
        default=False,
        help="Overwrite existing remuxed video",
    )

    parser.add_argument(
        "--keep-original",
        action="store_true",
        default=True,
        help="Keep original video file",
    )

    parser.add_argument(
        "filenames",
        type=Path,
        nargs="+",
        metavar="filename",
        # action=CheckFile(extensions=valid_extensions, must_exist=True),
        help="video file(s) to process",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug output",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Dry run, don't actually remux anything",
    )

    parsed_args = parser.parse_args()

    return parsed_args


def main():
    args = parse_args()

    for src_file in args.filenames:
        remuxed_file = args.remux_dest_dir / (src_file.stem + '.mp4')
        # print(f"Remuxing '{src_file}' to '{remuxed_file}'")
        if remuxed_file.exists():
            # print(f"Skipping '{filename}': '{remuxed_file}' already exists")
            continue

        metadata_file = remuxed_file.with_suffix(remuxed_file.suffix + '.meta')
        if metadata_file.exists():
            # print(f"Skipping '{filename}': '{metadata_file}' already exists")
            continue

        process_video(args, src_file, remuxed_file, metadata_file)

    global ocrapi
    if ocrapi is not None:
        ocrapi.End()



if __name__ == "__main__":
    main()
