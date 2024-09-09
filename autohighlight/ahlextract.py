#!/usr/bin/env python3
import csv
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


def load_metadata(remux_dir):
    metadata = {}
    for meta_file in Path(remux_dir).glob('*.meta'):
        with open(meta_file, 'r') as f:
            data = json.load(f)
            metadata[data['filename']] = data
    return metadata

def load_highlight_requests(index_file):
    requests = []
    with open(index_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            requests.append(row)
    return requests

def find_highlights(metadata, requests):
    highlights = []
    for request in requests:
        request_time = datetime.strptime(request['timestamp'], "%Y-%m-%d %H:%M:%S.%f")
        for filename, data in metadata.items():
            start_time = datetime.strptime(data['start_time'], "%Y-%m-%d %H:%M:%S.%f")
            end_time = datetime.strptime(data['end_time'], "%Y-%m-%d %H:%M:%S.%f")
            if start_time <= request_time <= end_time:
                highlights.append((filename, request['timestamp'], request['highlight_id']))
    return highlights

def main():
    remux_dir = Path("x:/zTEMP/remux_tmp")
    index_file = Path("x:/zTEMP/logs_tmp/ahlindex.csv")
    autohighlight_dir = Path("x:/zTEMP/autohighlight_tmp")

    metadata = load_metadata(remux_dir)
    requests = load_highlight_requests(index_file)

    print("metadata: " + str(metadata))
    print("requests: " + str(requests))
    print()
    highlights = find_highlights(metadata, requests)

    print("Videos containing requested highlights:")
    for filename, timestamp, description in highlights:
        print(f"- {filename}: {timestamp} - {description}")


    for filename, timestamp, highlight_id in highlights:
        video_path = remux_dir / filename
        request_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
        start_time = datetime.strptime(metadata[filename]['start_time'], "%Y-%m-%d %H:%M:%S.%f")

        time_offset = (request_time - start_time).total_seconds() + 5

        output_filename = f"highlight_{highlight_id}_{filename}"
        output_path = autohighlight_dir / output_filename

        ffmpeg_cmd = [
            "ffmpeg",
            "-ss", f"{time_offset}",
            "-t", "60",
            "-i", str(video_path),
            # "-t", "60",
            "-shortest",
            "-c", "copy",
            "-y",
            str(output_path)
        ]

        print(f"Extracting highlight {highlight_id} from {filename} at offset {time_offset}")
        # print(f"Command: {' '.join(ffmpeg_cmd)}")

        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            print(f"Successfully extracted highlight to {output_filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error extracting highlight: {e}")
            print(f"ffmpeg stderr: {e.stderr}")

    print("\nHighlight extraction complete.")

if __name__ == "__main__":
    main()