#!/usr/bin/env python3
import argparse
import io
import re
import sys
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Tuple

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

# Track the row for the start of the stream we last processed.
STATE_FILE = Path("last_processed_row.txt")

# How many actual song entries a chunk of the onglog needs to have before we
# believe it's a real stream. Sometimes a placeholder row (with a 0 in the
# order column) gets added ahead of the stream actually happening, and we
# don't want to generate anything for those.
MIN_VALID_ROWS = 2

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


# The row number (1-based, as the spreadsheet counts them) of the order=0 row
# that started the last stream we generated data for, or None if we've never
# been told.
def read_last_processed_row(statefile: Path) -> Optional[int]:
    if not statefile.exists():
        return None

    contents = statefile.read_text(encoding="utf-8").strip()
    if not contents.isdigit():
        print(f"ERROR: state file {statefile} should hold only an onglog row "
              f"number, but holds: {contents!r}", file=sys.stderr)
        sys.exit(1)  # FIXME: use a proper constant here

    return int(contents)


def write_last_processed_row(statefile: Path, row: int) -> None:
    statefile.write_text(f"{row}\n", encoding="utf-8")


# gspread hands us short rows when the trailing columns are empty, so pad
# everything out to a consistent width and save ourselves a pile of length
# checks further down.
def pad_row(row: List[str]) -> List[str]:
    if len(row) > Col.LAST:
        return row

    return row + [""] * (Col.LAST + 1 - len(row))


# Find every chunk of the onglog that starts a stream (the rows with a 0 in
# the order column), and work out which rows belong to each of them. Returns
# (start_row, first_row, last_row) triples of 1-based onglog row numbers,
# where start_row is the order=0 row itself, and first_row..last_row are the
# rows that follow it, up to the start of the next stream.
def find_stream_chunks(values: List[List[str]]) -> List[Tuple[int, int, int]]:
    start_rows = [i + 1 for i, row in enumerate(values) if row[Col.ORDER].strip() == "0"]

    chunks = []
    for i, start_row in enumerate(start_rows):
        if i + 1 < len(start_rows):
            last_row = start_rows[i + 1] - 1
        else:
            last_row = len(values)

        chunks.append((start_row, start_row + 1, last_row))

    return chunks


# A real stream has actual songs logged in it. A placeholder that someone
# added before the stream happened will have the 0 in the order column, but
# nothing useful after it.
def is_real_stream(rows: List[List[str]]) -> bool:
    valid = [r for r in rows if r[Col.REQUESTER] and r[Col.TITLE] and r[Col.TYPE]]
    return len(valid) >= MIN_VALID_ROWS


# Generate the info file and the thumbnail for a single stream, given the
# onglog rows that make up that stream. Returns the date string used for the
# generated files, or None if we couldn't figure out what day the stream was.
def gen_stream_info(args: argparse.Namespace, rows: List[List[str]]) -> Optional[str]:
    # The lines that will make up our info file. This all used to just go to
    # stdout and get redirected by hand, so the format is what it always was.
    out: List[str] = []

    # Output header
    out.append("")
    out.append("APPROXIMATE start times of each segment:")
    out.append("")
    out.append("00:00 Stream Start and Warmup")

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

    for row in rows:
        # Parse the start time for a song in onglog-standard (but not gsheets
        # standard, heh) date format. Skip rows without a parsable timestamp.
        if not row[Col.DATE]:
            continue

        ts = dateparser.parse(row[Col.DATE], settings={"DATE_ORDER": "YMD"})
        if ts is None or not row[Col.UPTIME]:
            continue
        log_end_time = ts

        # make sure the requestor name was something other than a single "-" or
        # similar
        requester = row[Col.REQUESTER]
        if len(requester) <= 2:
            requester = "no_user"

        # adjust by our time offset, to allow us to still have proper times
        # if Jon forgot to start the recording on time
        onglog = hms_to_sec(row[Col.UPTIME]) - args.time_offset

        start_time_hms = sec_to_hms(onglog)

        if not row[Col.TITLE]:
            continue

        reqby_str = ""
        if requester != "no_user":
            reqby_str = f" (req'd by {requester})"

        # Skip tier 3 resub songs when making timestamps. Ideally we'd
        # auto-skip warmup songs and such as well, but we don't have a
        # really good way to identify those.
        if row[Col.LINKS].strip().lower() in ["tier 3", "tier3", "credits"]:
            continue

        if not args.concert_grand:
            if not in_concert_grand:
                if any([
                    "concert grand" in row[Col.LINKS].lower(),
                    "concert grand" in row[Col.TITLE].lower()
                ]):
                    in_concert_grand = True
                    has_concert_grand = True
                    out.append(f"{start_time_hms} Concert Grand")
                    continue
            else:
                # if we're in a concert grand segment, see if we should exit
                if "piano" in row[Col.TYPE].lower():
                    continue
                else:
                    in_concert_grand = False

        # generate the actual output
        out.append(f"{start_time_hms} {row[Col.TITLE]}{reqby_str}")

    # Without a date for the stream we've got nothing to name our files after,
    # so there's nothing useful we can do here.
    if log_end_time is None:
        return None

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
    out.append("")
    out.append("")
    out.append(f"TITLE: {titlestr}")

    datestr = log_end_time.strftime("%Y-%m-%d")
    out.append("")
    out.append(f"Generating thumbnail for date: {datestr}")

    Path(f"{datestr}.txt").write_text("\n".join(out) + "\n", encoding="utf-8")

    if args.concert_grand:
        mkthumbnail(background="thumbnail_concertgrand_template.png",
                    date=datestr, center_x=960)
    else:
        mkthumbnail(background="thumbnail_template.png", date=datestr, center_x=606)

    return datestr


# Argument parsing (I know, shocking)
def parse_arguments(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate VOD info files and thumbnails from the onglog",
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

    parser.add_argument(
        "--state-file",
        default=STATE_FILE,
        type=Path,
        help="file holding the onglog row of the last stream we processed",
    )

    parser.add_argument(
        "--start-row", "--row", "-r",
        default=None,
        type=int,
        metavar="lineno",
        help="process only the stream starting at this onglog row, and leave"
             " the last-processed row untouched",
    )

    parser.add_argument(
        "--set-last-row",
        default=None,
        type=int,
        metavar="lineno",
        help="record this onglog row as the last stream processed, and exit",
    )

    parsed_args = parser.parse_args(argv)

    if parsed_args.start_row is not None and parsed_args.set_last_row is not None:
        parser.error("--start-row and --set-last-row don't make sense together")

    return parsed_args


def main(argv: List[str]) -> int:
    args = parse_arguments(argv[1:])

    if args.set_last_row is not None:
        write_last_processed_row(args.state_file, args.set_last_row)
        print(f"Recorded onglog row {args.set_last_row} as the last stream processed")
        return 0

    # Work out where we're picking up from before we go to the trouble of
    # fetching the onglog.
    last_row = None
    if args.start_row is None:
        last_row = read_last_processed_row(args.state_file)
        if last_row is None:
            print(f"ERROR: no state file at {args.state_file}, so we don't know where"
                  " to start. Use --set-last-row to tell us, or --start-row to"
                  " process a single stream.", file=sys.stderr)
            return 1  # FIXME: use a proper constant here

    # gc = gspread.oauth(scopes=gspread.auth.READONLY_SCOPES)
    gc = gspread.service_account(filename=args.credsfile)

    onglog = gc.open_by_url(ONG_SPREADSHEET_URL)
    ws = onglog.worksheet("Songs")

    # One fetch of the whole sheet, and then we can work everything else out
    # locally. Rows are 1-based in the spreadsheet, so onglog row N lives at
    # values[N - 1].
    values = [pad_row(row) for row in ws.get_all_values()]
    chunks = find_stream_chunks(values)

    # Are we doing a single stream by hand, or picking up where we left off?
    if args.start_row is not None:
        todo = [c for c in chunks if c[0] == args.start_row]
        if not todo:
            print(f"ERROR: Failed to find a stream start on line {args.start_row}",
                  file=sys.stderr)
            return 1  # FIXME: use a proper constant here

        # The user asked for this one specifically, so don't second-guess
        # whether it looks like a real stream.
        check_valid = False
        track_state = False
    else:
        assert last_row is not None
        todo = [c for c in chunks if c[0] > last_row]
        if not todo:
            print(f"No streams in the onglog after row {last_row}, nothing to do")
            return 0

        check_valid = True
        track_state = True

    # Only worth fetching the backup copy of the onglog once, no matter how
    # many streams we end up generating data for.
    backupdir = Path("backups")
    backup_bytes = None

    generated = 0
    for start_row, first_row, last_row in todo:
        rows = values[first_row - 1:last_row]

        if check_valid and not is_real_stream(rows):
            print(f"Skipping onglog row {start_row}: looks like a placeholder rather"
                  " than a real stream")
            continue

        print(f"Processing stream starting at onglog row {start_row}...")
        datestr = gen_stream_info(args, rows)

        if datestr is None:
            print(f"WARNING: couldn't find a date for the stream at onglog row"
                  f" {start_row}, skipping it", file=sys.stderr)
            continue

        print(f"Wrote {datestr}.txt and {datestr}.jpg")
        generated += 1

        # Back up the log if we have a backups directory
        if backupdir.exists():
            if backup_bytes is None:
                backup_bytes = gc.export(file_id=ONG_SPREADSHEET_ID,
                                         format=gspread.utils.ExportFormat.EXCEL)

            backup_path = backupdir / f"{datestr}.xlsx"
            backup_path.write_bytes(backup_bytes)
            print(f"Saved onglog backup as {backup_path}")

        if track_state:
            write_last_processed_row(args.state_file, start_row)

    if generated == 0:
        print("No new streams to generate data for")

    return 0


if __name__ == "__main__":
    # make sure our output streams are properly encoded so that we can
    # not screw up Frédéric Chopin's name and such. Keep them line buffered
    # while we're at it, so that our progress output actually shows up as it
    # happens, even when we're being redirected to a file or a pipe.
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding="utf-8", line_buffering=True)

    sys.exit(main(sys.argv))
