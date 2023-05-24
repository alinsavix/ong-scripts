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
from tdvutil import hms_to_sec
from tdvutil.argparse import CheckFile

# If modifying these scopes, delete the file token.pickle.
# SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']



ONG_SPREADSHEET_URL = 'https://docs.google.com/spreadsheets/d/14ARzE_zSMNhp0ZQV34ti2741PbA-5wAjsXRAW8EgJ-4/edit?ts=5a3893f5#gid=0'
ONG_SPREADSHEET_ID = '14ARzE_zSMNhp0ZQV34ti2741PbA-5wAjsXRAW8EgJ-4'
ONG_RANGE_NAME = 'Songs!A2:I'

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

def normalize_filename(name: str) -> str:
    return re.sub(r"\s*:\s*", " - ", name)

# helper function to validate and parse a provided time string
def offset_str(arg_value: str) -> float:
    offset_re = re.compile(r"^(\d+:)?(\d+:)?(\d+)(\.\d+)?$")

    if not offset_re.match(arg_value):
        raise argparse.ArgumentTypeError

    # else
    return hms_to_sec(arg_value)


def parse_arguments(argv: List[str]):
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


    # positional arguments
    parser.add_argument(
        "lineno",
        type=int,
        # nargs="1",
        help="line number starting the appropriate line in the onglog"
    )

    # parser.add_argument(
    #     "file",
    #     type=Path,
    #     action=CheckFile(must_exist=True),
    #     # default=Path("training.log"),
    #     # nargs="1",
    #     help="video file to split",
    # )

    parsed_args = parser.parse_args(argv)
    return parsed_args


def main(argv: List[str]) -> int:
    args = parse_arguments(argv[1:])

    # gc = gspread.oauth(scopes=gspread.auth.READONLY_SCOPES)
    gc = gspread.service_account(filename=args.credsfile)

    start_time = gc.open_by_url(ONG_SPREADSHEET_URL)
    ws = start_time.worksheet("Songs")

    # Annoyingly, the gsheets interface has columns 1-based instead of 0-based
    cell_list = ws.findall("0", in_column=Col.ORDER + 1)
    # print([x.row for x in cell_list])
    # print(f"{len(cell_list)} stream start markers")

    # penultimate_row = cell_list[-2].row
    # last_row = cell_list[-1].row - 1
    # print("Last spreadsheet segment: %d to %d" % (penultimate_row, last_row))

    # Start at the end, work backwards until we find our row
    for i in range(len(cell_list) - 1, 0, -1):
        # print("checking index %d, holds row %d" % (i, cell_list[i].row))
        if cell_list[i].row == args.lineno:
            first_row = cell_list[i].row + 1
            last_row = cell_list[i + 1].row - 1
            break
    else:
        print(f"ERROR: Failed to find a show start on line {args.lineno}")
        sys.exit(1)

    # This should be equivalent, but japparently isn't. Not sure why.
    # for i, cell in enumerate(reversed(cell_list)):
    #     if cell.row == SHEET_ROW:
    #         first_row = cell.row + 1
    #         last_row = cell_list[i + 1].row
    #         break
    # else:
    #     print(f"ERROR: Failed to find a show start on line {SHEET_ROW}")
    #     sys.exit(1)

    # print(f"Using spreadsheet rows {first_row} to {last_row}\n", file=sys.stderr)

    upleft = gspread.utils.rowcol_to_a1(first_row, Col.FIRST + 1)
    botright = gspread.utils.rowcol_to_a1(last_row, Col.LAST + 1)
    range_query = "%s:%s" % (upleft, botright)

    # ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(args.file)]

    # rows = ws.range(cell_list[-2].row, 1, cell_list[-1].row, 9)
    rows = ws.get(range_query, major_dimension="ROWS")

    print("\nAPPROXIMATE start times of each segment:\n")
    print("00:00 Stream Start and Warmup")

    log_end_time = None

    for i, row in enumerate(rows):
        log_end_time = dateparser.parse(row[Col.DATE], settings={"DATE_ORDER": "YMD"})
        # print(f"parsed end date: {log_end_time}")
        # print(row[Col.TITLE])
        # lastindex = len(rows) - 1
        # for i in range(lastindex):
        # row = rows[i]
        # print(row)
        if len(row[Col.REQUESTER]) <= 2:
            row[Col.REQUESTER] = "no_user"

        start_time = hms_to_sec(row[Col.UPTIME]) - args.time_offset
        # if i < (lastindex - 1):
        if i < len(rows) - 1:
            end_time = hms_to_sec(rows[i + 1][Col.UPTIME]) - args.time_offset
        else:
            end_time = -1
        # print("i is %d len is %d" % (i, len(rows)))

        # print("Track %02d: %s to %s - %s" % (i, start_time, end_time, row[Col.TITLE]))
        start_time_hms = sec_to_hms(start_time).replace(".000", "")

        reqby_str = ""
        if row[Col.REQUESTER] != "no_user":
            reqby_str = f" (req'd by {row[Col.REQUESTER]})"

        if len(row) > Col.LINKS and "tier 3" in row[Col.LINKS].lower():
            continue

        print(f"{start_time_hms} {row[Col.TITLE]}{reqby_str}")
        # end_time_hms = sec_to_hms(end_time) if end_time > 0 else "end"
        # print(f"Track {i}: {start_time_hms} to {end_time_hms} - {row[Col.TITLE]}")
        # ffmpeg_cmd.extend(["-ss", start_time_hms])
        # # if end_time_hms != "end":
        # #     ffmpeg_cmd.extend(["-to", rows[i + 1][Col.UPTIME]])
        # if end_time_hms != "end":
        #     ffmpeg_cmd.extend(["-to", end_time_hms])
        # ffmpeg_cmd.extend(["-c", "copy"])
        # ffmpeg_cmd.extend(["-sn", "%s - %s.mp4" %
        #                   (row[Col.ORDER], normalize_filename(row[Col.TITLE]))])

        # print(row[3])
        # print("%s - %s - %s - \"%s\" %s" % (row[2], row[1], row[3], row[4], row[6]))
        # print(row[0])
        # print(":::")
        # print("entry %d - %s" % (i, " ".join(ffmpeg_cmd)))
    # row = rows[8]
    # pprint(row)
    # print("executing: " + " ".join(ffmpeg_cmd))
    # subprocess.run(ffmpeg_cmd)

    if log_end_time is not None:
        # Sucks we have to go through these gymnastics to get the date formatted
        # the way we want, because somehow there's not a strftime substitution
        # for a day of month without a leading zero, nor is there one that will
        # automatically handle the "1st", "2nd", and "3rd" prefixes. Wonder if
        # there's a module that handles that better
        # log_end_time = datetime(2023, 1, 1)
        if 4 <= log_end_time.day <= 20 or 24 <= log_end_time.day <= 30:
            daysuffix = "th"
        else:
            daysuffix = ["st", "nd", "rd"][log_end_time.day % 10 - 1]

        monthname = log_end_time.strftime("%B")

        titlestr = f"VOD for the {log_end_time.day}{daysuffix} of {monthname} {log_end_time.year}"
        print(f"\n\nTITLE: {titlestr}")

        datestr = log_end_time.strftime("%Y-%m-%d")
        print(f"\nGenerating thumbnail for date: {datestr}")
        subprocess.run(["python3", "mkthumbnail.py", datestr])



if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding="utf-8")
    sys.exit(main(sys.argv))
