#!/usr/bin/env python3
import argparse
import io
import re
import subprocess
import sys
from enum import IntEnum
from pathlib import Path
from typing import List

import dateparser
import gspread
from PIL import Image, ImageDraw, ImageFont
from tdvutil import hms_to_sec
from tdvutil.argparse import CheckFile

# NOTE: You will need to set up a file with your google cloud credentials
# as noted in the documentation for the "gspread" module


# Where to find the onglog. Sure, we could take these as config values or
# arguments or something, but... why?
ONG_SPREADSHEET_ID = "14ARzE_zSMNhp0ZQV34ti2741PbA-5wAjsXRAW8EgJ-4"
ONG_SPREADSHEET_URL = f"https://docs.google.com/spreadsheets/d/{ONG_SPREADSHEET_ID}/edit"
ONG_RANGE_NAME = 'Songs!A2:I'

# If modifying these scopes, delete the file token.pickle.
# SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

# column offsets for the various onglog fields
class Col(IntEnum):
    FIRST = 0
    DATE = 0
    UPTIME = 1
    ORDER = 2
    REQUESTER = 3
    TITLE = 4
    GENRE = 5
    TYPE = 6
    LINKS = 7
    LOOPER = 8
    LAST = 8


# Convert seconds to HH:MM:SS.SSS format. Sure, this could use strftime
# or datetime.timedelta, but both of those have their own issues when
# you want a consistent format involving milliseconds.
#
# This is stolen from tdvutils and tweaked to be what we want; we should
# roll some of those changes back into tdvutil
def sec_to_hms(secs: float) -> str:
    hours = int(secs // (60 * 60))
    secs %= (60 * 60)

    minutes = int(secs // 60)
    secs %= 60

    ms = int((secs % 1) * 1000)
    secs = int(secs)

    ret = ""
    if hours > 0:
        ret = f"{hours:02d}:"

    ret += f"{minutes:02d}:{secs:02d}"
    return ret


# stupid simple filename normalization
# FIXME: do better
def normalize_filename(name: str) -> str:
    return re.sub(r"\s*:\s*", " - ", name)


# helper function to validate and parse a provided time string, and
# raise an appropriate argparse exception if we fail.
def offset_str(arg_value: str) -> float:
    offset_re = re.compile(r"^(\d+:)?(\d+:)?(\d+)(\.\d+)?$")

    if not offset_re.match(arg_value):
        raise argparse.ArgumentTypeError

    # else
    return hms_to_sec(arg_value)


def mkthumbnail(date: str, background="thumbnail_template.png", fontfile="mtcorsva_0.ttf", center_x=606, baseline=650, size=220, color="#fe9a0d") -> None:
    img = Image.open(background)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(fontfile, size)

    size_x = draw.textlength(date, font=font)
    centering_x = center_x - (size_x / 2)

    draw.text((centering_x, baseline), date, font=font, fill=color)
    # img.show()
    img.save(f"{date}.jpg", quality=95)


# Argument parsing (I know, shocking)
def parse_arguments(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a video file based on timestamps in the onglog",
        allow_abbrev=True,
    )

    parser.add_argument(
        "--time-offset", "-t",
        default=0.0,
        metavar="timestring",
        type=offset_str,
        help="Time offset of the start of the VoD file compared to onglog",
    )

    parser.add_argument(
        "--credsfile",
        default="credentials.json",
        type=Path,
        action=CheckFile(must_exist=True),
        help="credentials file to use",
    )

    # parser.add_argument(
    #     "--debug",
    #     action='store_true',
    #     default=False,
    #     help="Enable debugging output",
    # )

    parser.add_argument(
        "--concert-grand", "--cg",
        default=False,
        action='store_true',
        help="Treat stream as timer'd concert grand scene",
    )

    # positional arguments
    parser.add_argument(
        "lineno",
        type=int,
        # nargs="1",
        help="line number starting the appropriate line in the onglog"
    )

    parsed_args = parser.parse_args(argv)
    return parsed_args


def main(argv: List[str]) -> int:
    args = parse_arguments(argv[1:])

    # gc = gspread.oauth(scopes=gspread.auth.READONLY_SCOPES)
    gc = gspread.service_account(filename=args.credsfile)

    onglog = gc.open_by_url(ONG_SPREADSHEET_URL)
    ws = onglog.worksheet("Songs")

    # This gets a list of all rows that start a new onglog chunk. This
    # is from some old code, we probably don't need it anymore now that
    # we're having the user explicitly specify a line number.
    # FIXME: figure out if we can do better
    #
    # Annoyingly, the gsheets interface has columns 1-based instead of 0-based,
    # so we'll see all sorts of "Col.SOMETHING + 1" type constructs in this code
    cell_list = ws.findall("0", in_column=Col.ORDER + 1)

    # Start at the end, work backwards until we find our matching stream
    # start, then save the line numbers
    for i in range(len(cell_list) - 1, 0, -1):
        if cell_list[i].row == args.lineno:
            first_row = cell_list[i].row + 1

            # If this is the last chunk (and there's not, effectively, a null
            # chunk after the one we're caring about), pick an unreasonable max
            # size for the chunk and use that. There's probably a better way.
            if i >= len(cell_list) - 1:
                last_row = first_row + 500
            else:
                last_row = cell_list[i + 1].row - 1
            break
    else:
        print(f"ERROR: Failed to find a stream start on line {args.lineno}", file=sys.stderr)
        return 1  # FIXME: use a proper constant here

    # Get ourselves a chunk of rows to work with
    upleft = gspread.utils.rowcol_to_a1(first_row, Col.FIRST + 1)
    botright = gspread.utils.rowcol_to_a1(last_row, Col.LAST + 1)
    range_query = "%s:%s" % (upleft, botright)

    rows = ws.get(range_query, major_dimension="ROWS")

    # Output header
    print("\nAPPROXIMATE start times of each segment:\n")
    print("00:00 Stream Start and Warmup")

    # if we're in a concert grand segment (which means we ignore entries
    # until we hit the end or run into something that isn't piano)
    in_concert_grand = False
    has_concert_grand = False

    # The stream end time is the date we'll use for the date of the stream.
    # It's all kind of twisted because Jon is on Australia time, but the
    # OngLog is kept in US time (EST, I think?). We want to generate the VOD
    # title and thumbnail based on Jon time, though, so we have to get fancy.
    # We could do the date math properly if we really want, but in 99%
    # of cases, just taking the date the stream ended in the US will give
    # the date of the stream in Australia, so we just go with that.
    log_end_time = None

    for i, row in enumerate(rows):
        # Parse the start time for a song in onglog-standard (but not gsheets
        # standard, heh) date format. Skip rows without a parsable timestamp.
        ts = dateparser.parse(row[Col.DATE], settings={"DATE_ORDER": "YMD"})
        if ts is None or len(row) < Col.UPTIME or not row[Col.UPTIME]:
            continue
        log_end_time = ts

        # make sure the requestor name was something other than a single "-" or
        # similar
        if len(row[Col.REQUESTER]) <= 2:
            row[Col.REQUESTER] = "no_user"

        # adjust by our time offset, to allow us to still have proper times
        # if Jon forgot to start the recording on time
        onglog = hms_to_sec(row[Col.UPTIME]) + args.time_offset

        start_time_hms = sec_to_hms(onglog)

        if len(row) <= Col.TITLE or not row[Col.TITLE]:
            continue

        reqby_str = ""
        if row[Col.REQUESTER] != "no_user":
            reqby_str = f" (req'd by {row[Col.REQUESTER]})"

        # Skip tier 3 resub songs when making timestamps. Ideally we'd
        # auto-skip warmup songs and such as well, but we don't have a
        # really good way to identify those.
        if len(row) > Col.LINKS and "tier 3" in row[Col.LINKS].lower():
            continue

        if not args.concert_grand:
            if not in_concert_grand:
                if len(row) > Col.LINKS and "concert grand" in row[Col.LINKS].lower():
                    in_concert_grand = True
                    has_concert_grand = True
                    print(f"{start_time_hms} Concert Grand")
                    continue
            else:
                # if we're in a concert grand segment, see if we should exit
                if len(row) > Col.TYPE and "piano" in row[Col.TYPE].lower():
                    continue
                else:
                    in_concert_grand = False

        # generate the actual output
        print(f"{start_time_hms} {row[Col.TITLE]}{reqby_str}")


    # If we have a date for the stream, generate a title line for it, and
    # generate a thumbnail
    if log_end_time is not None:
        # Sucks we have to go through these gymnastics to get the date formatted
        # the way we want, because somehow there's not a strftime substitution
        # for a day of month without a leading zero, nor is there one that will
        # automatically handle the "1st", "2nd", and "3rd" prefixes. Wonder if
        # there's a module that handles that better
        if 4 <= log_end_time.day <= 20 or 24 <= log_end_time.day <= 30:
            daysuffix = "th"
        else:
            daysuffix = ["st", "nd", "rd"][log_end_time.day % 10 - 1]

        monthname = log_end_time.strftime("%B")

        if args.concert_grand:
            grandstr = " (Timer'd Concert Grand Stream)"
        elif has_concert_grand:
            grandstr = " (incl. Concert Grand)"
        else:
            grandstr = ""

        titlestr = f"VOD for the {log_end_time.day}{daysuffix} of {monthname} {log_end_time.year}{grandstr}"
        print(f"\n\nTITLE: {titlestr}")

        datestr = log_end_time.strftime("%Y-%m-%d")
        print(f"\nGenerating thumbnail for date: {datestr}")

        if args.concert_grand:
            mkthumbnail(background="thumbnail_concertgrand_template.png",
                        date=datestr, center_x=960)
        else:
            mkthumbnail(background="thumbnail_template.png", date=datestr, center_x=606)

    return 0


if __name__ == "__main__":
    # make sure our output streams are properly encoded so that we can
    # not screw up Frédéric Chopin's name and such.
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding="utf-8")

    sys.exit(main(sys.argv))
