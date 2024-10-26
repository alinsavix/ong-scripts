#!/usr/bin/env python
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from onglog import find_onglog_entry_by_datetime
from tdvutil import ppretty, sec_to_hms, sec_to_shortstr
from tdvutil.argparse import CheckFile

# give ourselves a place to stuff our indexes
script_dir = Path(__file__).parent.resolve()
INDEX_DIR = script_dir / "indexes"
INDEX_DIR.mkdir(exist_ok=True)


def log(msg):
    print(msg)
    sys.stdout.flush()


def load_metadata(remux_dir):
    metadata = {}
    for meta_file in Path(remux_dir).glob('*.meta'):
        log(f"INFO: Loading metadata from {meta_file}")
        try:
            with open(meta_file, 'r') as f:
                data = json.load(f)
            metadata[data['filename']] = data
        except json.JSONDecodeError:
            log(f"WARNING: Couldn't decode metadata from {meta_file}")

    return metadata


def load_highlight_requests(index_file):
    requests = []
    with open(index_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            requests.append(row)
    return requests


def find_highlights(metadata, requests, content_class):
    highlights = []
    for request in requests:
        if request["completed"] == "Y":
            continue

        request_time = datetime.strptime(request['timestamp'], "%Y-%m-%d %H:%M:%S.%f")
        for filename, data in metadata.items():
            if "content_class" not in data or data['content_class'] != content_class:
                continue
            if not data["start_time"]:
                continue
            start_time = datetime.strptime(data['start_time'], "%Y-%m-%d %H:%M:%S.%f")
            end_time = datetime.strptime(data['end_time'], "%Y-%m-%d %H:%M:%S.%f")
            # print(f"Comparing: {start_time} <= {request_time} <= {end_time}")
            if start_time <= request_time <= end_time:
                highlights.append((filename, request['timestamp'], request['highlight_id'], content_class))
    return highlights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Ong video metadata",
    )

    parser.add_argument(
        "--index",
        type=Path,
        default=INDEX_DIR / "ahlindex.csv",
        action=CheckFile(extensions={"csv"}, must_exist=True),
        help="The ongoing autohighlight index to read/update"
    )

    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path("."),
        action=CheckFile(must_exist=True),
        help="Directory to extract video files into"
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("."),
        action=CheckFile(must_exist=True),
        help="Directory to look for source videos to extract"
    )

    parser.add_argument(
        "--content-class",
        type=str,
        default="clean",
        help="Content class to extract"
    )

    parser.add_argument(
        "--extra-head",
        type=int,
        default=3,
        help="Extra seconds to add to the start of the highlight"
    )

    parser.add_argument(
        "--extract-length", "--extract-len",
        type=int,
        default=60,
        help="Number of seconds to extract as highlight"
    )

    # parser.add_argument(
    #     "--trash-dir",
    #     type=Path,
    #     default=None,
    #     action=CheckFile(must_exist=True),
    #     help="Directory to move video files into when done",
    # )


    parser.add_argument(
        "--ffmpeg-bin",
        type=str,
        default="ffmpeg",
        help="ffmpeg binary name or full path to use",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing output files",
    )

    parser.add_argument(
        "--force-metadata",
        action="store_true",
        default=False,
        help="Overwrite existing metadata files",
    )

    # parser.add_argument(
    #     "--force-remux",
    #     action="store_true",
    #     default=False,
    #     help="Overwrite existing remuxed video",
    # )

    # parser.add_argument(
    #     "--keep-original",
    #     action="store_true",
    #     default=True,
    #     help="Keep original video file",
    # )

    # parser.add_argument(
    #     "filenames",
    #     type=Path,
    #     nargs="+",
    #     metavar="filename",
    #     # action=CheckFile(extensions=valid_extensions, must_exist=True),
    #     help="video file(s) to process",
    # )

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

    metadata = load_metadata(args.source_dir)
    requests = load_highlight_requests(args.index)

    if args.debug:
        log(f"DEBUG: metadata: {ppretty(metadata)}\n")
        log(f"DEBUG: requests: {ppretty(requests)}\n")

    highlights = find_highlights(metadata, requests, args.content_class)

    log(f"INFO: Found {len(highlights)} highlight segments out of {len(requests)} requests")

    if args.debug:
        log("DEBUG: Videos containing requested highlights:")
        for filename, timestamp, description, content_class in highlights:
            log(f"  - {filename}: {timestamp} - {description}")

    for filename, timestamp, highlight_id, content_class in highlights:
        output_meta = {}
        video_path = args.source_dir / filename
        request_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
        start_time = datetime.strptime(metadata[filename]['start_time'], "%Y-%m-%d %H:%M:%S.%f")
        start_date = metadata[filename]['start_time'].split(" ")[0]

        time_offset = (request_time - start_time).total_seconds() + 3 - args.extra_head
        # offset_str = sec_to_hms(time_offset).split(".")[0]

        input_ext = Path(filename).suffix

        output_filename = f"highlight_{highlight_id}_{content_class}_{start_date}{input_ext}"
        output_path = args.dest_dir / output_filename

        metadata_filename = f"highlight_{highlight_id}_meta_{start_date}.txt"
        metadata_path = args.dest_dir / metadata_filename

        if output_path.exists() and not args.force:
            log(f"INFO: Highlight {highlight_id} from {filename} already exists. Skipping.")
            continue

        ffmpeg_cmd = [
            args.ffmpeg_bin, "-hide_banner", "-hwaccel", "auto",
            "-ss", f"{time_offset}",
            "-t", str(args.extra_head + args.extract_length),
            "-i", str(video_path),
            "-vsync", "vfr",
            "-c:v", "h264_nvenc",
            "-preset", "p3",
            "-rc", "constqp",
            "-qp", "16",
            "-b:v", "0",
            "-c:a", "libfdk_aac", "-vbr", "5", "-cutoff", "18000",
            "-shortest",
            "-y",
            str(output_path)
        ]

        log(f"INFO: Extracting highlight {highlight_id} from {filename} at offset {time_offset}")
        if args.debug:
            log(f"DEBUG: Command: {' '.join(ffmpeg_cmd)}")

        if args.dry_run:
            log(f"DRY RUN: Would have extracted highlight to {output_path}")
            continue

        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            log(f"INFO: Successfully extracted highlight to {output_filename}")
        except subprocess.CalledProcessError as e:
            log(f"ERROR: Couldn't extract highlight: {e}")
            log(f"ERROR: ffmpeg stderr: {e.stderr}")
            continue

        output_meta["highlight_id"] = highlight_id
        output_meta["request_time"] = str(request_time)
        output_meta["request_uptime"] = sec_to_hms(time_offset)
        output_meta["original_filename"] = filename
        output_meta["extra_head"] = args.extra_head
        output_meta["extract_length"] = args.extract_length
        logentry = find_onglog_entry_by_datetime(request_time)

        if logentry:
            if args.debug:
                log(f"DEBUG: found onglog entry: {ppretty(logentry)}")
            output_meta["title"] = logentry.title
            output_meta["requester"] = logentry.requester
            output_meta["onglog_line"] = logentry.rowid
        else:
            log(f"WARNING: Couldn't find onglog entry for {request_time}")

        if not metadata_path.exists() or args.force_metadata:
            metadata_path.write_text(json.dumps(output_meta, indent=4))

    log("\nINFO: Highlight extraction complete.")
    # output_index_file.write_text("\n".join(meta_lines))

if __name__ == "__main__":
    main()
