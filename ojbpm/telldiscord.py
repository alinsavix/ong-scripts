#!/usr/bin/env python3
import argparse
import pprint
import sys
import time
import tomllib
from pathlib import Path
from typing import Dict, List

from discord_webhook import DiscordWebhook
from tdvutil import ppretty
from tdvutil.argparse import CheckFile
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer


#
# utility functions
#
def log(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.stderr.flush()

def now() -> int:
    return int(time.time())

def get_webhook_url(cfgfile: Path) -> str:
    log(f"loading config from {cfgfile}")
    with cfgfile.open("rb") as f:
        config = tomllib.load(f)

    try:
        return config["telldiscord"]["webhook_url"]
    except KeyError:
        log("ERROR: missing 'webhook_url' in config")
        sys.exit(1)

def file_read(fp: Path) -> str | None:
    with fp.open("r") as f:
        x = f.readline()
        if len(x) < 2:
            time.sleep(2)
            x = f.readline()

    if len(x) < 2:
        return None

    return x

def file_age(fp: Path) -> int:
    return now() - int(fp.stat().st_mtime)

#
# the good stuff
#
# send a message to discord, of a given type, but only if the last
# message of that type was different
def send_discord(webhook_url: str, msg_type: str, msg: str) -> None:
    # a stupid trick for persistent function variables
    if not hasattr(send_discord, "last_sent"):
        send_discord.last_sent = {}  # msg type -> message

    # if msg in last_sent and now() - last_sent[msg] < 60:
    if msg == send_discord.last_sent.get(msg_type, ""):  # already sent this one!
        log(f"DUPLICATE: {msg}")
        return

    send_discord.last_sent[msg_type] = msg

    webhook = DiscordWebhook(url=webhook_url, content=msg)
    response = webhook.execute()
    if response.status_code == 200:
        log(f"SENT: {msg}")
    else:
        log(f"FAILED: (status {response.status_code}): {msg}")

    return


# global to keep track of what files have changed. There may be a better
# way to do this, not sure
changed: Dict[str, int] = {}
class OJBPMHandler(PatternMatchingEventHandler):
    def __init__(self, args: argparse.Namespace, patterns: List[str]):
        super().__init__(patterns=patterns)
        self.args = args

    def on_modified(self, event):
        global changed

        # bn = Path(event.src_path).name
        bn = event.src_path
        log(f"INFO: File modified: {bn}")
        changed[bn] = now()

        return


# Require a file to not have changed in this long before we process it, so
# that if the looper is being flipped through slots, or the bpm is still
# settling, we don't spam discord
min_ages = {
    "current_bpm.txt": 20,  # Might take a bit to settle
    "looper_slot.txt": 5,
}

def watch_ojbpm(args: argparse.Namespace, webhook_url: str):
    path = args.watch_dir
    event_handler = OJBPMHandler(args, patterns=["current_bpm.txt", "looper_slot.txt"])

    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()

    # we're gonna get events now, which will set the `changed` var, which we
    # now handle in this loop
    while True:
        for fp in list(changed.keys()):
            age = file_age(Path(fp))
            bn = Path(fp).name

            if age < min_ages[bn]:
                continue

            # The file is old enough, remove it from the list. Yes, there
            # is a minor race condition here.
            changed.pop(fp)

            if bn == "current_bpm.txt":
                bpm = file_read(Path(fp))
                if bpm is None:
                    log(f"WARNING: Can't read bpm data from {fp}, skipping update")
                    continue

                send_discord(webhook_url, "bpm", f"Looper BPM: {bpm}")

            elif bn == "looper_slot.txt":
                slot = file_read(Path(fp))
                if slot is None:
                    log(f"WARNING: Can't read slot data from {fp}, skipping update")
                    continue

                send_discord(webhook_url, "slot", f"Looper SLOT: {slot}")

            else:
                log(f"ERROR: Unknown file changed: {fp}")

        time.sleep(4)

    # try:
    #     while observer.is_alive():
    #         observer.join(1)
    # finally:
    #     observer.stop()
    #     observer.join()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Share to discord some of the changes ojbpm has detected")

    parser.add_argument(
        "--credentials-file", "-c",
        type=Path,
        default=None,
        action=CheckFile(must_exist=True),
        help="file with discord credentials"
    )

    parser.add_argument(
        "--watch-dir",
        type=Path,
        default=None,
        action=CheckFile(must_exist=True),
        required=True,
        help="ojbpm export path to watch for bpm changes",
    )

    parsed_args = parser.parse_args()

    if parsed_args.credentials_file is None:
        parsed_args.credentials_file = Path(__file__).parent / "credentials.toml"

    return parsed_args


def main():
    args = parse_args()

    webhook_url = get_webhook_url(args.credentials_file)
    watch_ojbpm(args, webhook_url)


if __name__ == "__main__":
    main()
