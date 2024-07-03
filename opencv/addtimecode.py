#!python3
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

import cv2
import numpy as np
import tesserocr
from PIL import Image
# import skimage
from tdvutil import hms_to_sec, ppretty, sec_to_timecode, timecode_to_sec
from tdvutil.argparse import CheckFile, NegateAction

# from vidgear.gears import VideoGear, WriteGear

# FIXME: should these be int?
Point: TypeAlias = Tuple[int, int]

@dataclass
# UL, UR, BR, BL
class MarkerCorners:
    UL: Point
    UR: Point
    BR: Point
    BL: Point

    def as_array(self):
        return [self.UL, self.UR, self.BR, self.BL]


REFERENCE_IMAGE = "Looper Marker Reference.png"
REFERENCE_MASK1 = "Looper Channel 1 Mask.png"
REFERENCE_MASK2 = "Lopper Channel 2 Mask.png"

mask1_roi = ((660, 485), (755, 582))

reference_markers: Dict[int, MarkerCorners] = {
    1: MarkerCorners(
        BL=(432, 412),
        BR=(468, 384),
        UL=(403, 381),
        UR=(438, 355)
    ),
    2: MarkerCorners(
        BL=(744, 106),
        BR=(775, 129),
        UL=(771, 83),
        UR=(801, 107)
    ),
    9: MarkerCorners(
        BL=(539, 531),
        BR=(561, 553),
        UL=(564, 509),
        UR=(586, 533)
    )
}

reference_dims = (1280, 720)

do_trace = False
def trace(*args: Any):
    if do_trace:
        print("TRACE:", *args)  # should this go to stderr?
        sys.stdout.flush()

def log(*args: Any):
    print(*args)
    sys.stdout.flush()


# FIXME: Find a way to do this without a remux. If there is one. It's vaguely
# possible mp4box can do it (at least if there's an existing timecode chunk)
# but I'm not smart enough to figure out how to do that. So... remux.
#
# FIXME: Only expected to work with integer framerates.
def remux_with_timecode(args: argparse.Namespace, vidfile: Path, ts: float, fps: int):
    ts = ts % (24 * 60 * 60)
    timecode = sec_to_timecode(ts, fps)
    timeout = 30 * 60

    fh, _tmpfile = tempfile.mkstemp(suffix=vidfile.suffix, prefix="remux_", dir=vidfile.parent)
    os.close(fh)

    tmpfile = Path(_tmpfile)
    # log(f"remuxing w/ timecode using tmpfile {tmpfile}")
    log(f"REMUXING '{vidfile.name}' to add timecode metadata of '{timecode}'")

    cmd = [
        "ffmpeg", "-hide_banner",
        "-i", str(vidfile), "-map", "0", "-map", "-0:d", "-c", "copy",
        "-timecode", timecode, "-video_track_timescale", "30000",
        "-y", str(tmpfile)]

    if args.dry_run:
        log("DRY RUN only, remux command would have been: " + " ".join(cmd))
        return

    try:
        subprocess.run(args=cmd, shell=False, stdin=subprocess.DEVNULL, check=True, timeout=timeout)
    except FileNotFoundError:
        log("ERROR: couldn't execute ffmpeg, please make sure it exists in your PATH")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        log(f"ERROR: remux-with-timecode process timed out after {timeout} seconds")
        tmpfile.unlink(missing_ok=True)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        log(f"ERROR: remux-with-timecode process failed with ffmpeg exit code {e.returncode}")
        tmpfile.unlink(missing_ok=True)
        sys.exit(1)
    except Exception:
        log("ERROR: unknown error during remux-with-timecode process, aborting")
        tmpfile.unlink(missing_ok=True)
        sys.exit(1)

    # seems like it ran ok, rename the temp file
    # log(f"DEBUG: replacing {vidfile} with {tmpfile}")
    tmpfile.replace(vidfile)

    log(f"\nDONE: completed remux-with-timecode to {vidfile}")


# Our timecode string will look like: RTC 2024-06-25 12:14:45.904
re_timestamp = re.compile(r"""
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

ocrapi = None
def ocr_frame(frame: cv2.typing.MatLike) -> Optional[float]:
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
        subimage = gray[900:1020, 165:]
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

    m = re_timestamp.search(ocr)
    if not m:
        return None

    # print(f"OCR'd timestamp string as: {ocr}")
    # full_timestamp = datetime.strptime(m.group("full_timestamp"), "%Y-%m-%d %H:%M:%S.%f")
    return hms_to_sec(m.group("time_only"))


@dataclass
class FoundTimestamps:
    ocr_ts: float
    frame_ts: float
    fps: float

# returns (ocr'd timestamp, frame timestamp, fps) tuple, or None if not found
def find_timestamp_in_range(
    args: argparse.Namespace, filename: Path,
    start_time: float = 30.0, search_len: float = 15
) -> Optional[FoundTimestamps]:
    # frame_offset = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    # cap = VideoGear(source=str(args.filenames[0]), logging=True).start()
    cap = cv2.VideoCapture(str(filename), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        # FIXME: Should probably throw an exception or something
        print("Error opening video stream or file")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_POS_MSEC, int(start_time * 1000))
    print(f"Looking for timestamp at offset {start_time}s...", end="")
    sys.stdout.flush()

    fps = cap.get(cv2.CAP_PROP_FPS)

    framenum = -1
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            return None

        framenum += 1

        frame_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if frame_ts > (start_time + search_len):
            cap.release()
            print("")
            return None

        print(".", end="")
        sys.stdout.flush()

        ts = ocr_frame(frame)
        if ts is not None:
            cap.release()
            print(f"FOUND: {ts}")
            return FoundTimestamps(ts, frame_ts, fps)

    cap.release()
    print("")

    return None


valid_extensions = {"mkv", "mp4", "flv"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find a burned-in timecode and set the video's metadata to match",
    )

    parser.add_argument(
        "filenames",
        type=Path,
        # action="append",
        nargs="+",
        metavar="filename",
        # action=CheckFile(extensions=valid_extensions, must_exist=True),
        help="image file(s) to process",
    )

    parser.add_argument(
        "--trace",
        default=False,
        action="store_true",
        help="enable trace logging",
    )

    parser.add_argument(
        "--dry-run", "--dryrun",
        default=False,
        action="store_true",
        help="don't actually write any files",
    )

    parsed_args = parser.parse_args()

    return parsed_args


def main():
    global ocrapi
    args = parse_args()
    if args.trace:
        global do_trace
        do_trace = True

    searches = [
        (5, 1),
        (40, 2),
        (32, 5),
        (27, 5),
        (37, 3),
        (42, 5),
        (20, 7),
        (42, 5),
        (47, 13)
    ]

    timestamps = None

    # FIXME: work with more than a single file
    # (ocr_ts, frame_ts, fps)
    for search_start, search_len in searches:
        timestamps = find_timestamp_in_range(args, args.filenames[0], search_start, search_len)
        if timestamps is not None:
            break

    if timestamps is None:
        print("No timestamp found")
        sys.exit(1)

    # FIXME: Sloppy
    if ocrapi is not None:
        ocrapi.End()

    starting_timestamp = timestamps.ocr_ts - timestamps.frame_ts
    print(f"found {timestamps.ocr_ts} at {
          timestamps.frame_ts} => video starts at {starting_timestamp}")
    print(f"Video proports to be {timestamps.fps} fps (rounding to {round(timestamps.fps)})")

    remux_with_timecode(args, args.filenames[0], starting_timestamp, round(timestamps.fps))

    sys.exit(0)

if __name__ == "__main__":
    main()
