#!/usr/bin/env python

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import ffmpeg
from tdvutil import ppretty
from tdvutil.argparse import CheckFile

# give ourselves a place to stuff our indexes
script_dir = Path(__file__).parent.resolve()
INDEX_DIR = script_dir / "indexes"
INDEX_DIR.mkdir(exist_ok=True)


@dataclass
class MediaMeta:
    filename: str
    content_class: str
    media_types: str
    exact_times: bool
    start_time: Optional[str]
    end_time: Optional[str]
    duration: float
    fps: int
    size_bytes: int


def load_metadata(meta_file: Path) -> Optional[MediaMeta]:
    """Load metadata from a .meta file"""
    try:
        with meta_file.open() as f:
            meta_dict = json.load(f)
            return MediaMeta(**meta_dict)
    except Exception as e:
        print(f"Error loading metadata from {meta_file}: {e}")
        return None


def get_time_offset(base_time: str, target_time: str) -> float:
    """Calculate time offset between two timestamp strings"""
    base_dt = datetime.strptime(base_time, "%Y-%m-%d %H:%M:%S.%f")
    target_dt = datetime.strptime(target_time, "%Y-%m-%d %H:%M:%S.%f")
    return (target_dt - base_dt).total_seconds()




# FIXME: It's annoying we pass time around as a string.
def merge_media_files(args: argparse.Namespace, base_time: str, base_file: Path, base_meta: MediaMeta, additional_files: List[Path], output_file: Path, metadata_list: List[MediaMeta]):
    # print(f"Base file: {base_file}")
    # print(f"Additional files: {[file for file in additional_files]}")

    # probeinfo = ffmpeg.probe(base_file)

    if base_meta.start_time is None:
        print(f"Error: Base file '{base_file}' has no start time metadata")
        return False

    # duration = probeinfo['format']['duration']
    duration = base_meta.duration

    inputs = []
    encoders = []

    base_offset = get_time_offset(base_time, base_meta.start_time)
    if base_offset <= 0:
        base_pre_offset = int(abs(base_offset) * 1000)
    else:
        print(f"ERROR: Base file starts after the target time {base_meta.start_time} > {base_time}")
        return False

    print(f"Base video file (offset {base_pre_offset:>05d}ms): {base_pre_offset}")

    inputs.append(["-ss", f"{base_pre_offset}ms", "-i", str(base_file)])
    # encoders.append(["-avoid_negative_ts", "make_zero", "-shortest"])
    encoders.append(["-map", "0:v", "-c:v:0", "copy"])

    # Add additional audio streams with proper delays
    for i, meta in enumerate(metadata_list):
        if not meta.start_time:
            print(f"Warning: Skipping file {additional_files[i]} - no start time metadata")
            continue

        output_num = i  # FIXME: is there a better way?
        input_num = i + 1

        offset = get_time_offset(base_time, meta.start_time)
        if offset <= 0:
            pre_offset = int(abs(offset) * 1000)
        else:
            print(f"ERROR: Additional file {additional_files[i]} starts after the target time")
            return False

        print(f"Audio track (offset {pre_offset:>5d}ms): {additional_files[i]}")
        sys.stdout.flush()

        # We can just leave in the -ss option, because having it set to 0 is
        # harmless
        inputs.append(["-ss", f"{pre_offset}ms", "-i", str(additional_files[i])])

        # But for the filter block, we don't want to leave the filter in place
        # if it's not needed, because it'll use a lot of CPU to do nothing
        # if filter_offset > 0:
        #     filters.append(f"[{input_num}:a]adelay={filter_offset}|{filter_offset}[a{input_num}]")
        #     encoders.append(["-map", f"[a{input_num}]", f"-c:a:{input_num}", "aac", f"-b:a:{input_num}", "320k"])
        # else:
        #     encoders.append(["-map", f"{input_num}:a", f"-c:a:{input_num}", "copy"])
        #     encoders.append(["-map", f"{input_num}:a", f"-c:a:{input_num}", "aac", f"-b:a:{input_num}", "320k"])
        encoders.append(["-map", f"{input_num}:a", f"-c:a:{output_num}", "copy"])

    # Build ffmpeg command
    cmd = ["ffmpeg", "-hide_banner", "-hwaccel", "auto"]

    for input in inputs:
        cmd.extend(input)

    # if len(filters) > 0:
    #     cmd.extend(["-filter_complex", ";".join(filters)])

    for encoder in encoders:
        cmd.extend(encoder)

    # Output file
    # cmd.extend(["-movflags", "+faststart"])
    if args.duration:
        cmd.extend(["-t", str(args.duration)])
    else:
        cmd.extend(["-t", str(duration)])

    cmd.extend(["-y", str(output_file)])

    # print(cmd)
    if args.dry_run:
        print("Dry run, would have executed:")
        print(" ".join(cmd))
        sys.stdout.flush()
        return True

    if args.debug:
        print("Running ffmpeg command:", " ".join(cmd))
        sys.stdout.flush()

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg: {e}")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple media files based on metadata timing",
    )

    parser.add_argument(
        "filenames",
        type=Path,
        nargs="+",
        metavar="filename",
        help="media files to merge (first file provides video)",
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output file path",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration of the output file",
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
        help="Dry run the ffmpeg command",
    )

    return parser.parse_args()


def find_latest_start(metadata_list: List[MediaMeta]) -> Optional[str]:
    latest_time = None

    for meta in metadata_list:
        if meta.start_time is None:
            continue

        # print(f"Meta start time checking: {meta.start_time}")

        if latest_time is None or meta.start_time > latest_time:
            latest_time = meta.start_time

    return latest_time


def main():
    args = parse_args()

    if len(args.filenames) < 2:
        print("Error: At least two input files are required")
        return

    # Load metadata for all files
    metadata_list = []
    for media_file in args.filenames:
        meta_file = media_file.with_suffix(media_file.suffix + '.meta')
        if not meta_file.exists():
            print(f"No metadata file found for {media_file}")
            return

        metadata = load_metadata(meta_file)
        if metadata is None:
            return
        metadata_list.append(metadata)

        if args.debug:
            print(f"Loaded metadata for {media_file}:")
            print(f"  Content class: {metadata.content_class}")
            print(f"  Media types: {metadata.media_types}")
            print(f"  Start time: {metadata.start_time}")
            print(f"  Duration: {metadata.duration}")

    # Merge the files
    base_file = args.filenames[0]
    additional_files = args.filenames[1:]

    latest_start = find_latest_start(metadata_list)
    if latest_start is None:
        print("Error: No valid start times found")
        return

    print(f"Using base time: {latest_start}")

    if merge_media_files(args, latest_start, base_file, metadata_list[0], additional_files, args.output, metadata_list[1:]):
        print(f"Successfully merged files to {args.output}")
    else:
        print("Failed to merge files")


if __name__ == "__main__":
    main()
