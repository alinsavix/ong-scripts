#!/usr/bin/env python3
import csv
import json
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
    remux_dir = Path("remux")
    index_file = Path("ahlindex.csv")

    metadata = load_metadata(remux_dir)
    requests = load_highlight_requests(index_file)

    print("metadata: " + str(metadata))
    print("requests: " + str(requests))
    print()
    highlights = find_highlights(metadata, requests)

    print("Videos containing requested highlights:")
    for filename, timestamp, description in highlights:
        print(f"- {filename}: {timestamp} - {description}")

if __name__ == "__main__":
    main()
