#!/usr/bin/env python
import argparse
import configparser
import ctypes
import json
import logging as lg
import msvcrt
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import toml
from pycaw.constants import DEVICE_STATE, EDataFlow  # type: ignore
from pycaw.pycaw import AudioUtilities  # type: ignore
from pycaw.utils import AudioDevice  # type: ignore
from tdvutil import ppretty


# return the directory the script is in, be it in a pyinstaller bundle, or
# as a normal Python script
def get_exedir() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    else:
        return Path(__file__).resolve().parent


# return the directory that pyinstaller extracted to, if it was used,
# otherwise behaves the same as get_exedir()
def get_extractdir() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)  # type: ignore
    else:
        return Path(__file__).resolve().parent


def get_default_output_device() -> AudioDevice:
    # with warnings.catch_warnings():  # suppress COMError warnings
    #     warnings.simplefilter("ignore", UserWarning)
    return AudioUtilities.GetSpeakers()


def get_active_output_devices() -> List[AudioDevice]:
    # with warnings.catch_warnings():  # suppress COMError warnings
    #     warnings.simplefilter("ignore", UserWarning)
    return AudioUtilities.GetAllDevices(data_flow=EDataFlow.eRender.value,
                                        device_state=DEVICE_STATE.ACTIVE.value)

def get_active_input_devices() -> List[AudioDevice]:
    # with warnings.catch_warnings():  # suppress COMError warnings
    #     warnings.simplefilter("ignore", UserWarning)
    return AudioUtilities.GetAllDevices(data_flow=EDataFlow.eCapture.value,
                                        device_state=DEVICE_STATE.ACTIVE.value)

def get_default_input_device() -> AudioDevice:
    return AudioUtilities.GetMicrophone()

# def set_default_device(device: AudioDevice):
#     AudioUtilities.SetDefaultDevice(device.id)


# This feels ugly, but it seems to be the best way to do this?
def show_alert(title: str, message: str) -> None:
    # This is MB_ICONEXCLAMATION + MB_TOPMOST
    flags = 0x30 + 0x1000
    ctypes.windll.user32.MessageBoxW(0, message, title, flags)


# Unfortunately, the API for setting the default device for an application is
# not public, so we can't just... call it. NirSoft has reverse engineered it,
# though, and provided us with the "svcl" command to do it. This sucks, but
# it's what we got.
#
# What sucks worse is that I don't think there's a way to know for certain
# if the command failed or not (it always has a 0 return code), and finding
# out if the configuration is *currently* correct seems pretty difficult
# as well. So, yeah, this is a hack.
def set_app_default_device(device_id: str, executable_name: str) -> bool:
    try:
        # Construct the svcl command
        cmd_path = get_extractdir() / "svcl.exe"
        cmd = [str(cmd_path), "/SetAppDefault", device_id, 'all', executable_name]

        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Log command output at DEBUG level
        lg.debug(f"Command executed: {' '.join(cmd)}")
        lg.debug(f"Command stdout: {result.stdout}")
        # lg.debug(f"Command stderr: {result.stderr}")
        # lg.debug(f"Command return code: {result.returncode}")

        print(f"  Successfully set {device_id} as default device for {executable_name}")
        return True

    except subprocess.CalledProcessError as e:
        lg.error(f"Failed to set default device for {executable_name}: {e}")
        lg.error(f"Command output: {e.stdout}")
        lg.error(f"Command error: {e.stderr}")
        return False
    except FileNotFoundError:
        lg.error("svcl executable not found in PATH")
        return False
    except Exception as e:
        lg.error(f"Unexpected error setting default device: {e}")
        return False


# scene collection wrangling
def load_scene_collection(file: Path) -> Dict[str, Any]:
    try:
        with file.open('r', encoding='utf-8') as f:
            scene_collection = json.load(f)
        return cast(Dict[str, Any], scene_collection)
    except FileNotFoundError:
        lg.error(f"Error: Scene collection file not found: {file}")
        raise
    except json.JSONDecodeError as e:
        lg.error(f"Error: Invalid JSON in scene collection file: {e}")
        raise


def save_scene_collection(file: Path, scene_collection: dict[str, Any]) -> None:
    try:
        if file.exists():
            bak_dir = file.parent / "bak"
            bak_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = bak_dir / f"{file.stem}_backup_{timestamp}{file.suffix}"
            shutil.copy2(file, backup_path)
            print(f"    Created scene collection backup at: {backup_path}")

        # Save the new scene collection
        with file.open('w', encoding='utf-8') as f:
            json.dump(scene_collection, f, indent=2)
        print(f"    Updated scene collection saved to: {file}")

    except OSError as e:
        lg.error(f"Error saving scene collection: {e}")
        raise


def find_source_by_name(sc: dict[str, Any], sourcename: str) -> Optional[dict[str, Any]]:
    for source in sc.get("sources", []):
        if source.get("name") == sourcename:
            return source
    lg.debug(f"Source '{sourcename}' not found in scene collection.")
    return None


# profile wrangling
# FIXME: Do we actually need to convert this to a dict?
def load_profile(file: Path) -> dict[str, Any]:
    try:
        # Disable interpolation to handle % characters in values
        config = configparser.ConfigParser(interpolation=None)
        config.optionxform = str  # don't lowercase keys
        # Explicitly open with UTF-8 encoding to handle BOM
        with file.open('r', encoding='utf-8-sig') as f:
            config.read_file(f)

        # Convert the config object to a dictionary
        profile_dict = {}
        for section in config.sections():
            profile_dict[section] = dict(config[section])

        return profile_dict
    except FileNotFoundError:
        lg.error(f"Error: Profile file not found: {file}")
        raise
    except configparser.Error as e:
        lg.error(f"Error: Invalid INI file format in profile: {e}")
        raise


def save_profile(file: Path, profile: dict[str, Any]) -> None:
    try:
        if file.exists():
            bak_dir = file.parent / "bak"
            bak_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = bak_dir / f"{file.stem}_backup_{timestamp}{file.suffix}"
            shutil.copy2(file, backup_path)
            print(f"    Created profile backup at: {backup_path}")

        config = configparser.ConfigParser(interpolation=None)
        config.optionxform = str  # don't lowercase keys

        # Convert the dictionary to a config object
        for section, options in profile.items():
            config[section] = {}
            for option, value in options.items():
                config[section][option] = value

        # Save the profile to the file. Make it look like it does when OBS
        # writes it -- no spaces around delimiters, only newlines for eol
        with file.open('w', newline="\n", encoding='utf-8-sig') as f:
            config.write(f, space_around_delimiters=False)
        print(f"    Profile saved to: {file}")

    except OSError as e:
        lg.error(f"Error saving profile: {e}")
        raise


def load_config(config_path: Path, hostname: Optional[str] = None) -> dict[str, Any]:
    try:
        with config_path.open('r', encoding='utf-8') as f:
            full_config = toml.load(f)
        lg.debug(f"Loaded configuration from: {config_path}")

        # If hostname is provided, try to get the specific hostname configuration
        if hostname and hostname in full_config:
            config = full_config[hostname]
            lg.debug(f"Using hostname-specific configuration: {hostname}")
        elif "default" in full_config:
            lg.warning("Couldn't find hostname-based config, using 'default'")
            config = full_config.get("default")
        else:
            raise ValueError(f"No configuration found for hostname: {hostname}")

        return config

    except toml.TomlDecodeError as e:
        lg.error(f"Error: Invalid TOML in configuration file: {e}")
        raise
    except Exception as e:
        lg.error(f"Error loading configuration: {e}")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OBS Audio Device Configuration Unfucker")

    parser.add_argument(
        "--dryrun",
        default=False,
        action="store_true",
        help="Show what would be done without making changes"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Configuration file to use (default: audiounfuck.conf in script directory)"
    )

    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List all available input and output devices on the system"
    )

    return parser.parse_args()


def list_devices() -> None:
    print("AUDIO DEVICES LISTING\n")

    try:
        input_devices = get_active_input_devices()
        output_devices = get_active_output_devices()
    except Exception as e:
        lg.error(f"Error getting device lists: {e}")
        return

    print("INPUT DEVICES:")
    print("==============")
    if input_devices:
        for device in input_devices:
            print(f"  {device.FriendlyName} (id: {device.id})")
    else:
        print("  No input devices found")

    print("\nOUTPUT DEVICES:")
    print("===============")
    if output_devices:
        for device in output_devices:
            print(f"  {device.FriendlyName} (id: {device.id})")
    else:
        print("  No output devices found")


def unfuck(args: argparse.Namespace) -> Tuple[int, int, List[str]]:
    # logformat = "%(name)s,%(module)s,%(funcName)s | %(levelname)s | %(message)s"
    logformat = "%(levelname)s | %(message)s"
    lg.basicConfig(
        stream=sys.stdout,
        format=logformat
    )

    if args.debug:
        loglevel = lg.DEBUG
    else:
        loglevel = lg.INFO

    lg.getLogger().setLevel(loglevel)

    # ignore COMError warnings when using pycaw, since we always get them
    lg.getLogger("comtypes").setLevel(lg.ERROR)

    print("OBS AUDIO DEVICE UN-FUCKER\n\n")

    # Get hostname for configuration selection
    hostname = socket.gethostname().lower()

    # Determine config file path
    if args.config is None:
        # Use audiounfuck.conf in script directory
        script_dir = get_exedir()
        config_path = script_dir / "audiounfuck.conf"
    else:
        config_path = Path(args.config)

    if not config_path.exists():
        lg.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    lg.info(f"Using config file: {config_path}")
    config = load_config(config_path, hostname)

    scene_changes = 0
    profile_changes = 0

    # Extract configuration values
    obs_dir = Path(config['obs_directory'])
    obs_profile = config['obs_profile']
    obs_scene_collection = config['obs_scene_collection']
    input_map = config['inputs'] or {}
    output_capture_map = config['output_captures'] or {}
    # monitor_devices = config['monitor_devices'] or []
    app_outputs = config['app_outputs'] or {}

    sc_path = obs_dir / "config/obs-studio/basic/scenes" / f"{obs_scene_collection}.json"
    prof_path = obs_dir / "config/obs-studio/basic/profiles" / obs_profile / "basic.ini"

    sc = load_scene_collection(sc_path)
    lg.info(f"Loaded scene collection from {sc_path} with {len(sc.get('sources', []))} sources")

    prof = load_profile(prof_path)
    lg.info(f"Loaded profile from {prof_path}")


    print("\n===== FINDING DEVICE ASSIGNMENTS =====")

    final_devices = []
    final_devices_changed = []

    print("\nFINDING INPUTS:")
    active_input_devices = get_active_input_devices()

    for device in active_input_devices:
        if device.FriendlyName in input_map:
            d = input_map[device.FriendlyName]
            devid = device.id
            # print(f"map {d} to {device.FriendlyName} (id: {id})")

            for sourcename in d:
                source = find_source_by_name(sc, sourcename)
                if source is not None:
                    if source["settings"]["device_id"] != device.id:
                        print(f"  Found {device.FriendlyName}: Assigning to source {source['name']}")
                        source["settings"]["device_id"] = device.id
                        final_devices.append(f"CHANGED: {source['name']} <- {device.FriendlyName} (id: {devid})")
                        final_devices_changed.append(f"  - CHANGED: input source '{source['name']}' = {device.FriendlyName}")
                        scene_changes += 1
                    else:
                        print(f"  Found {device.FriendlyName}: Already assigned to {source['name']}")
                        final_devices.append(f"UNCHANGED: {source['name']} <- {device.FriendlyName} (id: {devid})")
                    break
            else:
                lg.warning(f"Found input {device.FriendlyName}, but no matching sources found in OBS")
        else:
            lg.debug(f"Found {device.FriendlyName}: Not listed, ignoring")


    print("\nFINDING OUTPUTS:")
    active_output_devices = get_active_output_devices()

    # FIXME: DRY
    for device in active_output_devices:
        if device.FriendlyName in app_outputs:
            for executable_name in app_outputs[device.FriendlyName]:
                if executable_name != "MONITOR":
                    set_app_default_device(device.id, executable_name)
                else:
                    if prof["Audio"]["MonitoringDeviceId"] != device.id:
                        print(f"  Found {device.FriendlyName}: Assigning as monitoring device")

                        prof["Audio"]["MonitoringDeviceId"] = device.id
                        prof["Audio"]["MonitoringDeviceName"] = device.FriendlyName
                        final_devices.append(
                            f"CHANGED: MONITOR -> {device.FriendlyName} (id: {device.id})")
                        final_devices_changed.append(f"  - MONITORING = {device.FriendlyName}")
                        profile_changes += 1
                    else:
                        print(f"  Found {device.FriendlyName}: Already assigned as monitoring device")
                        final_devices.append(f"UNCHANGED: MONITOR -> {device.FriendlyName} (id: {device.id})")

        if device.FriendlyName in output_capture_map:
            d = output_capture_map[device.FriendlyName]
            devid = device.id
            # print(f"map {d} to {device.FriendlyName} (id: {id})")

            for sourcename in d:
                source = find_source_by_name(sc, sourcename)
                if source is not None:
                    if source["settings"]["device_id"] != device.id:
                        print(f"  Found {device.FriendlyName}: Assigning to source {source['name']}")
                        source["settings"]["device_id"] = device.id
                        final_devices.append(f"CHANGED: {source['name']} <- (output catpure) {device.FriendlyName} (id: {devid})")
                        final_devices_changed.append(f"  - CHANGED: output capture source '{source['name']}' = {device.FriendlyName}")
                        scene_changes += 1
                    else:
                        print(f"  Found {device.FriendlyName}: Already assigned to {source['name']}")
                        final_devices.append(f"UNCHANGED: {source['name']} <- (output catpure) {device.FriendlyName} (id: {devid})")
            else:
                lg.warning(f"Found output {device.FriendlyName}, but no matching sources found in OBS")
        else:
            lg.debug(f"Found {device.FriendlyName}: Not listed, ignoring")


    print("\nFINDING MONITORING DEVICE:")

    # FIXME: DRY -- we can probably just bundle this with the above
    # for device in active_output_devices:
    #     if device.FriendlyName in monitor_devices:
    #         if prof["Audio"]["MonitoringDeviceId"] != device.id:
    #             print(f"  Found {device.FriendlyName}: Assigning as monitoring device")

    #             prof["Audio"]["MonitoringDeviceId"] = device.id
    #             prof["Audio"]["MonitoringDeviceName"] = device.FriendlyName
    #             final_devices.append(f"CHANGED: MONITOR -> {device.FriendlyName} (id: {device.id})")
    #             final_devices_changed.append(f"  - MONITORING = {device.FriendlyName}")
    #             profile_changes += 1
    #         else:
    #             print(f"  Found {device.FriendlyName}: Already assigned as monitoring device")
    #             final_devices.append(f"UNCHANGED: MONITOR -> {device.FriendlyName} (id: {device.id})")
    #         break
    #     else:
    #         lg.debug(f"Found {device.FriendlyName}: Not listed, ignoring")
    # else:
    #     lg.warning("No monitoring device found")


    print("\n\n===== SAVING CHANGES =====")

    if args.dryrun:
        print("\nDRY RUN - No changes will be saved")
    else:
        if scene_changes > 0:
            print("\n  SAVING SCENE CHANGES...")
            save_scene_collection(sc_path, sc)
        else:
            print("\n  NO SCENE CHANGES TO SAVE")

        if profile_changes > 0:
            print("\n  SAVING PROFILE CHANGES...")
            save_profile(prof_path, prof)
        else:
            print("\n  NO PROFILE CHANGES TO SAVE")


    print("\n\n===== FINAL DEVICE ASSIGNMENTS =====")
    for assignment in final_devices:
        print(f"  {assignment}")

    return scene_changes, profile_changes, final_devices_changed


if __name__ == "__main__":
    args = parse_args()
    if args.list_devices:
        list_devices()
        sys.exit(0)

    scene_changes, profile_changes, final_devices_changed = unfuck(args)

    if not any([scene_changes, profile_changes]) and not args.debug:
        # print("\nNo changes were made. Exiting.")
        sys.exit(0)

    # Something changed and we're not debugging -- show popup and exit
    if final_devices_changed and not args.debug:
        popup_message = "Changed devices:\n\n" + "\n".join(final_devices_changed)
        popup_message += "\n\nClose dialog to continue..."

        show_alert("Autounfucker: OBS Devices Changed", popup_message)
        sys.exit(0)  # FIXME: Should this be non-zero?

    # Nothing changed and we're not debugging -- just exit
    if not final_devices_changed and not args.debug:
        sys.exit(0)

    # Otherwise, we're debugging
    # print("\n\nPress 'd' to debug, any other key to end...")
    # keypress = msvcrt.getch()  # Waits for a keypress

    # if keypress.decode('utf-8') == "d":
    #     args.debug = True
    #     unfuck(args)

    #     print("\n\nPress any key to end...")
    #     keypress = msvcrt.getch()  # Waits for a keypress

    print("\n\nPress any key to end...")
    keypress = msvcrt.getch()  # Waits for a keypress
