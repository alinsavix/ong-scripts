#!/usr/bin/env python

# from jinokimijin:
# List of available output devices (* = default):
#  * Game (TC-Helicon GoXLR)
#    Speakers (Realtek(R) Audio)
#    Chat (TC-Helicon GoXLR)
#    Music (TC-Helicon GoXLR)
#    Sample (TC-Helicon GoXLR)
#    System (TC-Helicon GoXLR)
#    Speakers (Voice.ai Audio Cable)
#    Realtek Digital Output (Realtek(R) Audio)
#    Speakers (USB DAC - E01)
# Changing default device to Speakers (USB DAC - E01)...
# Updated list of available output devices (* = default):
# * Game (TC-Helicon GoXLR)
#   Speakers (Realtek(R) Audio)
#   Chat (TC-Helicon GoXLR)
#   Music (TC-Helicon GoXLR)
#   Sample (TC-Helicon GoXLR)
#   System (TC-Helicon GoXLR)
#   Speakers (Voice.ai Audio Cable)
#   Realtek Digital Output (Realtek(R) Audio)
#   Speakers (USB DAC - E01)


# So we basically need to map...
#
# audio capture source -> input device
# output capture source -> output device
# monitor -> output device

import argparse
import configparser
import json
import msvcrt
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

from pycaw.constants import DEVICE_STATE, EDataFlow
from pycaw.pycaw import AudioUtilities
from pycaw.utils import AudioDevice
from tdvutil import ppretty

OBS_DIR = Path("C:/obs/OBS 30.1.2-1")
OBS_PROFILE = "MainJon"
# OBS_PROFILE = "Untitled"
OBS_SCENE_COLLECTION = "OngMain2024Rework"

# the maps are "if you see the device on the left, assign it the first OBS
# source on the right that exists".
INPUT_MAP = {
    # Jon's bits
    "Line (Steinberg UR44)": ["UR44 Audio"],
    "VR-4HD(Audio)  (VR-4HD(Audio))": ["VR4HD Audio [maybe]"],   # not used, but set it anyhow

    # Alinsa's bits, for testing
    "Chat Mic (TC-Helicon GoXLR)": ["UR44 Audio"],
}

OUTPUT_CAPTURE_MAP = {
    # Jon's
    "Speakers (USB Audio CODEC )": ["nightbot"],

    # Alinsa's
    "System (TC-Helicon GoXLR)": ["nightbot"],
}

# A little different from the above, since the monitoring device is actually
# stored in the profile, not the scene collection, and there's only one of
# that target device.
MONITOR_DEVICES = [
    "Speakers (VR-4HD(Audio))",
    "Game (TC-Helicon GoXLR)",
]


def get_default_output_device():
    return AudioUtilities.GetSpeakers()


def get_active_output_devices():
    with warnings.catch_warnings():  # suppress COMError warnings
        warnings.simplefilter("ignore", UserWarning)
        return AudioUtilities.GetAllDevices(data_flow=EDataFlow.eRender.value,
                                            device_state=DEVICE_STATE.ACTIVE.value)

def get_active_input_devices():
    with warnings.catch_warnings():  # suppress COMError warnings
        warnings.simplefilter("ignore", UserWarning)
        return AudioUtilities.GetAllDevices(data_flow=EDataFlow.eCapture.value,
                                            device_state=DEVICE_STATE.ACTIVE.value)

def get_default_input_device():
    return AudioUtilities.GetMicrophone()

# def set_default_device(device: AudioDevice):
#     AudioUtilities.SetDefaultDevice(device.id)


# scene collection wrangling
def load_scene_collection(file: Path) -> dict:
    try:
        with open(file, 'r') as f:
            scene_collection = json.load(f)
        return scene_collection
    except FileNotFoundError:
        print(f"Error: Scene collection file not found: {file}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in scene collection file: {e}")
        raise

def save_scene_collection(file: Path, scene_collection: dict):
    try:
        if file.exists():
            bak_dir = file.parent / "bak"
            bak_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = bak_dir / f"{file.stem}_backup_{timestamp}{file.suffix}"
            shutil.copy2(file, backup_path)
            print(f"Created scene collection backup at: {backup_path}")

        # Save the new scene collection
        with open(file, 'w') as f:
            json.dump(scene_collection, f, indent=2)
        print(f"Updated scene collection saved to: {file}")

    except OSError as e:
        print(f"Error saving scene collection: {e}")
        raise

def find_source_by_name(sc: dict, sourcename: str) -> dict:
    for source in sc.get("sources", []):
        if source.get("name") == sourcename:
            return source
    print(f"Source '{sourcename}' not found in scene collection.")
    return None


# profile wrangling
def load_profile(file: Path) -> dict:
    try:
        # Disable interpolation to handle % characters in values
        config = configparser.ConfigParser(interpolation=None)
        # Explicitly open with UTF-8 encoding to handle BOM
        with open(file, 'r', encoding='utf-8-sig') as f:
            config.read_file(f)

        # Convert the config object to a dictionary
        profile_dict = {}
        for section in config.sections():
            profile_dict[section] = dict(config[section])

        return profile_dict
    except FileNotFoundError:
        print(f"Error: Profile file not found: {file}")
        raise
    except configparser.Error as e:
        print(f"Error: Invalid INI file format in profile: {e}")
        raise

def save_profile(file: Path, profile: dict):
    try:
        if file.exists():
            bak_dir = file.parent / "bak"
            bak_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = bak_dir / f"{file.stem}_backup_{timestamp}{file.suffix}"
            shutil.copy2(file, backup_path)
            print(f"Created profile backup at: {backup_path}")

        config = configparser.ConfigParser(interpolation=None)

        # Convert the dictionary to a config object
        for section, options in profile.items():
            config[section] = {}
            for option, value in options.items():
                config[section][option] = value

        # Save the profile to the file
        with open(file, 'w') as f:
            config.write(f)
        print(f"Profile saved to: {file}")

    except OSError as e:
        print(f"Error saving profile: {e}")
        raise



def go():
    parser = argparse.ArgumentParser(description="OBS Audio Device Configuration Tool")
    parser.add_argument("--dryrun", default=True, action="store_true", help="Show what would be done without making changes")
    args = parser.parse_args()

    print("OBS AUDIO DEVICE UN-FUCKER")

    sc_path = OBS_DIR / "config/obs-studio/basic/scenes" / f"{OBS_SCENE_COLLECTION}.json"
    prof_path = OBS_DIR / "config/obs-studio/basic/profiles" / OBS_PROFILE / "basic.ini"

    sc = load_scene_collection(sc_path)
    print(f"Loaded scene collection from {sc_path} with {len(sc.get('sources', []))} sources")

    prof = load_profile(prof_path)
    print(f"Loaded profile from {prof_path}")

    final_devices = []

    print("\nFINDING INPUTS:")
    active_input_devices = get_active_input_devices()

    for device in active_input_devices:
        if device.FriendlyName in INPUT_MAP:
            d = INPUT_MAP[device.FriendlyName]
            id = device.id
            # print(f"map {d} to {device.FriendlyName} (id: {id})")

            for sourcename in d:
                source = find_source_by_name(sc, sourcename)
                if source is not None:
                    print(f"INFO: Found {device.FriendlyName}: Assigning to source {source['name']}")
                    source["settings"]["device_id"] = device.id
                    final_devices.append(f"{source['name']} <- {device.FriendlyName} (id: {id})")
                    break
            else:
                print(f"WARNING: Found {device.FriendlyName}, but no matching sources found in OBS")
        else:
            print(f"DEBUG: Found {device.FriendlyName}: Not listed, ignoring")


    print("\nFINDING OUTPUTS:")
    active_output_devices = get_active_output_devices()

    # FIXME: DRY
    for device in active_output_devices:
        if device.FriendlyName in OUTPUT_CAPTURE_MAP:
            d = OUTPUT_CAPTURE_MAP[device.FriendlyName]
            id = device.id
            # print(f"map {d} to {device.FriendlyName} (id: {id})")

            for sourcename in d:
                source = find_source_by_name(sc, sourcename)
                if source is not None:
                    print(f"INFO: Found {device.FriendlyName}: Assigning to source {source['name']}")
                    source["settings"]["device_id"] = device.id
                    final_devices.append(f"{source['name']} <- (output catpure) {device.FriendlyName} (id: {id})")
                    break
            else:
                print(f"WARNING: Found {device.FriendlyName}, but no matching sources found in OBS")
        else:
            print(f"DEBUG: Found {device.FriendlyName}: Not listed, ignoring")


    print("\nFINDING MONITORING DEVICE:")

    # FIXME: DRY
    for device in active_output_devices:
        if device.FriendlyName in MONITOR_DEVICES:
            d = device.FriendlyName
            id = device.id
            print(f"INFO: Found {device.FriendlyName}: Assigning as monitoring device")

            prof["Audio"]["MonitoringDeviceId"] = id
            prof["Audio"]["MonitoringDeviceName"] = device.FriendlyName
            final_devices.append(f"MONITOR -> {device.FriendlyName} (id: {id})")
            break
        else:
            print(f"DEBUG: Found {device.FriendlyName}: Not listed, ignoring")
    else:
        print(f"WARNING: No monitoring device found")

    if args.dryrun:
        print("\nDRY RUN - No changes will be saved")
    else:
        save_scene_collection(sc_path, sc)
        save_profile(prof_path, prof)

    print("\n\n===== FINAL DEVICE ASSIGNMENTS =====")
    for assignment in final_devices:
        print(f"  {assignment}")

    # if device.id == default_input_device.id:
    #     print(f" * {device.FriendlyName}")
    # else:
    #     print(f"   {device.FriendlyName}")


if __name__ == "__main__":
    try:
        go()
    except Exception as e:
        print(f"An error occurred: {e}")

    print("\n\nPress any key to end...")
    msvcrt.getch()  # Waits for a keypress
