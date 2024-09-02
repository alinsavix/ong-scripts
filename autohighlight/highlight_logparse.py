#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tdvutil.argparse import CheckFile

logline_re = re.compile(r"""
    ^(?P<logtime>\d{2}:\d{2}:\d{2}\.\d{3})
    \s*
    :
    \s*
    (?P<logline>.*)
""", re.VERBOSE)

def parse_logfile(logfile_path) -> List[datetime]:
    with open(logfile_path, 'r') as file:
        lines = file.readlines()

    current_date = None
    highlight_times = []
    prev_time = None

    for line in lines:
        m = logline_re.match(line)
        if not m:
            continue

        logtime, logline = m.group('logtime'), m.group('logline')

        if "Current Date/Time" in line:
            date_str = logline.split(": ")[1].strip()
            current_date = datetime.strptime(date_str, "%Y-%m-%d, %H:%M:%S").date()
            prev_time = datetime.strptime(date_str, "%Y-%m-%d, %H:%M:%S").time()

            continue

        # Wasn't a date/time line, and we haven't seen one, so just
        # skip until we have one
        if not prev_time:
            continue

        # We could put this rollover checking in only where the
        # highlight times are added, and save a lot of processing,
        # but if we ever have a reeeeeally long stream, or OBS has
        # been up for a really long time, the highlight requests
        # are sparse enough that we might actually *miss* a rollover
        # and have a wrong timestamp, so just... check it on every
        # log entry.
        #
        # Am I overthinking this? Yes. Yes I am.
        time_obj = datetime.strptime(logtime, "%H:%M:%S.%f").time()

        if time_obj < prev_time:
            current_date += timedelta(days=1)

        prev_time = time_obj

        if "HIGHLIGHT SEGMENT REQUEST" in line:
            time_obj = datetime.strptime(logtime, "%H:%M:%S.%f").time()

            highlight_times.append(datetime.combine(current_date, time_obj))

    # for highlight_time in highlight_times:
    #     print(highlight_time.strftime("%Y-%m-%d %H:%M:%S.%f"))
    return highlight_times


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse a log file and find highlight requests"
    )

    parser.add_argument(
        "--index",
        type=Path,
        default="logindex.csv",
        action=CheckFile(extensions={"csv"}, must_exist=True),
        help="The ongoing index to read/update"
    )

    parser.add_argument(
        "logfiles",
        metavar="LOGFILE",
        type=Path,
        nargs='+',
        help="The log file(s) to process"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    highlight_times = []

    for logfile in args.logfiles:
        if not logfile.exists():
            print(f"file '{logfile}' does not exist", file=sys.stderr)
            continue

        highlight_times.append(parse_logfile(logfile))


if __name__ == "__main__":
    main()
# # Example usage
# logfile_path = '2024-08-30 12-22-08.txt'
# parse_logfile(logfile_path)
