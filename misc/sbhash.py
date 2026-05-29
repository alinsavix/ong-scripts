#!/usr/bin/env python3
"""Print a deterministic checksum of a streamer.bot config directory.

Expects a directory containing: actions.json, commands.json, obs.json, settings.json

Ignores:
- Save timestamps ("t" field)
- UI state (collapsedGroups, window positions)
- Timed action counters
- Ordering of commands/actions arrays
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

CONFIG_FILES = ["actions.json", "commands.json", "obs.json", "settings.json"]

# Top-level keys to strip from all files
GLOBAL_STRIP = {"t"}

# Per-file top-level keys to strip
FILE_STRIP = {
    "actions.json": {"collapsedGroups"},
    "commands.json": {"collapsedGroups"},
}

# Keys in each file whose array values should be sorted by "id"
SORT_BY_ID = {
    "actions.json": {"actions"},
    "commands.json": {"commands"},
}


def strip_volatile(data, filename):
    """Strip volatile fields from a parsed config file."""
    # Remove global volatile keys
    data = {k: v for k, v in data.items() if k not in GLOBAL_STRIP}

    # Remove per-file volatile keys
    file_keys = FILE_STRIP.get(filename, set())
    if file_keys:
        data = {k: v for k, v in data.items() if k not in file_keys}

    # Sort arrays by id for stable ordering
    sort_keys = SORT_BY_ID.get(filename, set())
    for key in sort_keys:
        if key in data and isinstance(data[key], list):
            data = dict(data)
            data[key] = sorted(data[key], key=lambda x: x.get("id", ""))

    # Strip timer counters in settings.json
    if filename == "settings.json":
        if "timedActions" in data:
            ta = data["timedActions"]
            if "timers" in ta:
                ta = dict(ta)
                ta["timers"] = [
                    {k: v for k, v in timer.items() if k != "counter"}
                    for timer in ta["timers"]
                ]
                data = dict(data)
                data["timedActions"] = ta

        # Strip window positions (UI layout state)
        if "windowSettings" in data:
            ws = dict(data["windowSettings"])
            ws.pop("positions", None)
            data = dict(data)
            data["windowSettings"] = ws

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Print a deterministic checksum of a streamer.bot config directory."
    )
    parser.add_argument("config_dir", help="Path to directory containing the config files")
    parser.add_argument("-w", "--write", action="store_true",
                        help="Write version string to sbdata_ver.txt in the config directory")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)

    combined = {}
    for filename in CONFIG_FILES:
        filepath = config_dir / filename
        if not filepath.exists():
            print(f"Missing: {filepath}", file=sys.stderr)
            sys.exit(1)
        with open(filepath, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        combined[filename] = strip_volatile(data, filename)

    canonical = json.dumps(combined, sort_keys=True, ensure_ascii=True)
    checksum = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]

    actions_path = config_dir / "actions.json"
    mtime = os.path.getmtime(actions_path)
    ts = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y%m%d.%H%M%S")
    version = f"{ts}-{checksum}"

    if args.write:
        out_path = config_dir / "sbdata_ver.txt"
        out_path.write_text(version + "\n")
    else:
        print(checksum)


if __name__ == "__main__":
    main()
