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

import configparser
import json
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

from pycaw.constants import DEVICE_STATE, EDataFlow
from pycaw.pycaw import AudioUtilities
from pycaw.utils import AudioDevice
from tdvutil import ppretty

INPUT_MAP = {
    "Steinberg UR44": ["UR44 Audio"],
    # don't use the VR-4HD for input, but may as well maintain it
    "VR-4HD(Audio)": ["VR4HD Audio [maybe]"],
    "Chat Mic (TC-Helicon GoXLR)": ["UR44 Audio"],
}

OUTPUT_CAPTURE_MAP = {
    "USB Audio CODEC": ["nightbot"],
    "System (TC-Helicon GoXLR)": ["nightbot"],
}

MONITOR_DEVICE = ["VR-4HD(Audio)", "Game (TC-Helicon GoXLR)"]


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

def get_default_output_device():
    return AudioUtilities.GetSpeakers()

def get_default_input_device():
    return AudioUtilities.GetMicrophone()

def set_default_device(device: AudioDevice):
    AudioUtilities.SetDefaultDevice(device.id)

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


def find_source_by_name(sc: dict, sourcename: str) -> dict:
    for source in sc.get("sources", []):
        if source.get("name") == sourcename:
            return source
    print(f"Source '{sourcename}' not found in scene collection.")
    return None


if __name__ == "__main__":
    # List devices
    print("List of available output devices (* = default): ")
    active_output_devices = get_active_output_devices()
    default_output_device = get_default_output_device()

    sc_name = Path("OngMain2024Rework.json")
    prof_name = Path("profile_basic.ini")

    sc = load_scene_collection(sc_name)
    prof = load_profile(prof_name)

    from tdvutil import ppretty

    # print(ppretty(prof))

    for device in active_output_devices:
        if device.FriendlyName in OUTPUT_CAPTURE_MAP:
            d = OUTPUT_CAPTURE_MAP[device.FriendlyName]
            id = device.id
            print(f"map {d} to {device.FriendlyName} (id: {id})")

            for sourcename in d:
                source = find_source_by_name(sc, sourcename)
                if source is not None:
                    print(f"Updating source {source['name']}: {source['settings']['device_id']} -> {id}")
                    source["settings"]["device_id"] = device.id
    # for device in active_output_devices:
    #     if device.id == default_output_device.id:
    #         print(f" * {device.FriendlyName}")
    #     else:
    #         print(f"   {device.FriendlyName}")


    print("\nList of available input devices (* = default): ")
    active_input_devices = get_active_input_devices()
    default_input_device = get_default_input_device()

    # print(dir(default_input_device))

    for device in active_input_devices:
        # print(f"{device.FriendlyName} (id: {device.id})")
        if device.FriendlyName in INPUT_MAP:
            d = INPUT_MAP[device.FriendlyName]
            id = device.id
            print(f"map {d} to {device.FriendlyName} (id: {id})")

            for sourcename in d:
                source = find_source_by_name(sc, sourcename)
                if source is not None:
                    print(f"Updating source {source['name']}: {source['settings']['device_id']} -> {id}")
                    source["settings"]["device_id"] = device.id

    # FIXME: DRY
    for device in active_output_devices:
        if device.FriendlyName in MONITOR_DEVICE:
            d = device.FriendlyName
            id = device.id
            print(f"map {d} to {device.FriendlyName} (id: {id})")

            print(f"Updating monitoring device to {id}")
            prof["Audio"]["MonitoringDeviceId"] = id
            prof["Audio"]["MonitoringDeviceName"] = device.FriendlyName

    save_scene_collection(sc_name, sc)
    save_profile(prof_name, prof)

        # if device.id == default_input_device.id:
        #     print(f" * {device.FriendlyName}")
        # else:
        #     print(f"   {device.FriendlyName}")

    sys.exit(0)

    other_device = None
    for device in active_output_devices:
        if device.id == default_device.id:
            print(f" * {device.FriendlyName}")
        else:
            print(f"   {device.FriendlyName}")
            other_device = device

    if other_device is not None:
        # Change default to other device
        print(f"Changing default device to {other_device.FriendlyName}...")
        # set_default_device(other_device)

        # List devices again
        print("Updated list of available output devices (* = default): ")
        active_output_devices = get_active_output_devices()
        default_device = get_default_output_device()
        for device in active_output_devices:
            if device.id == default_device.id:
                print(f"* {device.FriendlyName}")
            else:
                print(f"  {device.FriendlyName}")
