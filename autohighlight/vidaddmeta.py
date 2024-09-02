#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import ffmpeg
import tesserocr
from PIL import Image
from tdvutil import ppretty

video_file = "clean 2024-08-30 12h32m56s.flv"

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


@dataclass
class VideoMeta:
    filename: str
    start_time: Optional[str]
    end_time: Optional[str]
    duration: float
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
    full_timestamp = datetime.strptime(m.group("full_timestamp"), "%Y-%m-%d %H:%M:%S.%f")
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


def process_video(args: argparse.Namespace, video_file: Path):
    try:
        probeinfo = ffmpeg.probe(video_file)
    except ffmpeg.Error as e:
        print(f"ffmpeg couldn't open {video_file}, skipping")
        return

    filename = video_file.name
    duration = float(probeinfo['format']['duration'])
    size_bytes = int(probeinfo['format']['size'])

    timestamp = find_timestamp_in_range(args, video_file, 55, 30)

    if timestamp is None:
        # print("No timestamp found")
        start_time = None
        end_time = None
    else:
        # print(f"Found timestamp: {timestamp}")
        start_time = add_duration(timestamp.ocr_ts, -timestamp.frame_ts)
        end_time = add_duration(start_time, duration)

    metadata = VideoMeta(filename, start_time, end_time, duration, size_bytes)
    metadata_json = json.dumps(asdict(metadata), indent=4)

    metafile = video_file.with_suffix(".meta")
    with metafile.open("w") as f:
        f.write(metadata_json)

    if start_time is not None:
        print(f"Wrote metadata for '{video_file}' (timestamp found)")
    else:
        print(f"Wrote metadata for '{video_file}' (no timestamp found)")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ginerate Ong video metadata",
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

    parsed_args = parser.parse_args()

    return parsed_args


def main():
    args = parse_args()

    for filename in args.filenames:
        # print(f"Processing '{filename}'")
        process_video(args, filename)


if __name__ == "__main__":
    main()
