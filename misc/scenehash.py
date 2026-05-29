#!/usr/bin/env python3
"""Print a deterministic checksum of an OBS scene collection JSON file.

Ignores:
- Source/sceneitem visibility
- Contents of text sources
- Current scene selection
"""

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

TEXT_SOURCE_IDS = {"text_gdiplus", "text_gdiplus_v2", "text_gdiplus_v3",
                   "text_ft2_source", "text_ft2_source_v2"}


# Hotkey keys with numeric item IDs that change on re-add
_HOTKEY_ITEM_RE = re.compile(r"^libobs\.(show|hide)_scene_item\.\d+$")


def strip_volatile(obj):
    """Recursively strip volatile fields from a scene collection object."""
    if isinstance(obj, dict):
        obj = {k: strip_volatile(v) for k, v in obj.items()
               if k not in ("current_scene", "current_program_scene")}

        # Strip text content from text sources
        if obj.get("id") in TEXT_SOURCE_IDS or obj.get("versioned_id") in TEXT_SOURCE_IDS:
            settings = obj.get("settings")
            if isinstance(settings, dict):
                obj["settings"] = {k: v for k, v in settings.items() if k != "text"}

        # Strip id_counter from scene settings
        settings = obj.get("settings")
        if isinstance(settings, dict) and "id_counter" in settings:
            obj["settings"] = {k: v for k, v in settings.items() if k != "id_counter"}

        # Strip hotkey keys with numeric scene item IDs
        hotkeys = obj.get("hotkeys")
        if isinstance(hotkeys, dict):
            obj["hotkeys"] = {k: v for k, v in hotkeys.items()
                              if not _HOTKEY_ITEM_RE.match(k)}

        # Strip visibility and id from scene items
        items = obj.get("items")
        if isinstance(items, list):
            obj["items"] = [strip_visible(item) for item in items]

        return obj
    elif isinstance(obj, list):
        return [strip_volatile(item) for item in obj]
    return obj


def strip_visible(item):
    """Strip the visible and id keys from a scene item dict."""
    if isinstance(item, dict):
        return {k: strip_volatile(v) for k, v in item.items()
                if k not in ("visible", "id")}
    return strip_volatile(item)


def main():
    parser = argparse.ArgumentParser(
        description="Print a deterministic checksum of an OBS scene collection JSON file."
    )
    parser.add_argument("scene_collection", help="Path to the scene collection JSON file")
    parser.add_argument("-w", "--write", action="store_true",
                        help="Write version string to <scenename>_ver.txt")
    args = parser.parse_args()

    filepath = Path(args.scene_collection)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = strip_volatile(data)
    canonical = json.dumps(cleaned, sort_keys=True, ensure_ascii=True)
    checksum = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]

    mtime = os.path.getmtime(filepath)
    ts = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y%m%d.%H%M%S")
    version = f"{ts}-{checksum}"

    if args.write:
        scene_name = filepath.stem
        out_path = filepath.parent / f"{scene_name}_ver.txt"
        out_path.write_text(version + "\n")
        # print(f"{version} -> {out_path}")
    else:
        print(checksum)


if __name__ == "__main__":
    main()
