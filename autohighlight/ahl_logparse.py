#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tdvutil.argparse import CheckFile

# give ourselves a place to stuff our indexes
script_dir = Path(__file__).parent.resolve()
INDEX_DIR = script_dir / "indexes"
INDEX_DIR.mkdir(exist_ok=True)


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
        if not prev_time or not current_date:
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
        default=INDEX_DIR / "ahlindex.csv",
        action=CheckFile(extensions={"csv"}, must_exist=True),
        help="The ongoing autohighlightindex to read/update"
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

    highlight_times: List[datetime] = []

    for logfile in args.logfiles:
        if not logfile.exists():
            print(f"file '{logfile}' does not exist", file=sys.stderr)
            continue

        log_highlights = parse_logfile(logfile)
        # if log_highlights:
        highlight_times.extend(log_highlights)

    # print(highlight_times)
    # Sort the highlight times
    highlight_times.sort()

    # Prepare the data for CSV
    csv_filename = args.index
    fieldnames = ['timestamp', 'highlight_id', 'completed', 'video_filename']

    existing_highlights = set()
    csv_data = []

    # Read existing CSV file if it exists
    if csv_filename.exists():
        with csv_filename.open('r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                existing_highlights.add(row['timestamp'])
                csv_data.append(row)

    # Generate new entries for highlights that don't exist yet
    next_id = max([int(row['highlight_id']) for row in csv_data], default=0) + 1
    for ht in highlight_times:
        print(ht)
        timestamp = ht.strftime("%Y-%m-%d %H:%M:%S.%f")
        if timestamp not in existing_highlights:
            csv_data.append({
                'timestamp': timestamp,
                'highlight_id': str(next_id),
                'completed': 'N',
                'video_filename': ''
            })
            next_id += 1

    # Sort the combined data by timestamp
    csv_data.sort(key=lambda x: x['timestamp'])

    # Write to CSV file
    with csv_filename.open('w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)

    print(f"Highlight requests updated in {csv_filename}")


if __name__ == "__main__":
    main()
# # Example usage
# logfile_path = '2024-08-30 12-22-08.txt'
# parse_logfile(logfile_path)
