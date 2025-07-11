# AudioUnfuck

A python-based tool to automatically detect and configure OBS Studio audio devices when Windows decides to change your audio device GUIDs... which it seems to be doing _way_ too often these days.

## What it does

AudioUnfuck automatically:

- Scans your system for active audio input and output devices
- Maps these devices to your OBS Studio sources based on a configuration file
- Updates your scene collection with the correct device IDs
- Also figures out if one of those devices is assigned to be the "monitor" device
- Updates your profile with the correct monitoring device ID

Basically, it's a tool to make you never have to think about audio device mappings in OBS again. Not really needed if your only audio device is a desktop audio capture, but for a more complicated setup, it is (hopefully) a godsend.

Configurations are specified by the `audiounfuck.conf` config file, which is normally loaded from the same directory that the script exists in. This config file allows multiple different configurations (differentiated by hostname) so that the same config file can be used across multiple hosts if desired.

The script generates timestamped backups in a "bak" subdirectory in the OBS profile and scene collection directories.


## Limitations

In its current form, AudioUnfuck directly manipulates OBS' scene and profile config files. This is because changing a monitoring device while OBS is running is somewhat broken (see <https://github.com/obsproject/obs-studio/issues/11868>) and we really wanted something that just works without requiring interaction from the streamer. It would be pretty trivial to have AudioUnfuck work via obs-websocket, but for now it doesn't.

If you have multiple audio devices with the same name, this script won't work -- the name is the only part of the audio device that is stable (as far as Alinsa knows) if audio devices' GUIDs are changing.

The script does have several command-line options (dryrun, config file location, etc) that are not accessible if the script isn't run from an interactive terminal.

## Requirements

- Python 3.12 or higher
- Windows (uses pycaw for audio device management)
- OBS Studio installed

## Configuration

Create an `audiounfuck.conf` file in the same directory as the script. The configuration uses TOML format and is keyed by hostname. Suggestions for a better way to handle the configuration are welcome, the current config format feels a bit... rough.

You can mostly follow the format that exists in the config file provided, but here's an example configuration for Alinsa's desktop, named `jinokimijin`:

```toml
[jinokimijin]
obs_directory = "C:/obs/OBS 30.1.2-1"
obs_profile = "Untitled"
obs_scene_collection = "Main2024Rework"

monitor_devices = [
    "Game (TC-Helicon GoXLR)"
]

[jinokimijin.inputs]
"Chat Mic (TC-Helicon GoXLR)" = ["UR44 Audio"]

[jinokimijin.output_captures]
"System (TC-Helicon GoXLR)" = ["nightbot"]
```

### Configuration Options

- **`obs_directory`**: Path to your OBS Studio installation
- **`obs_profile`**: Name of the OBS profile to modify
- **`obs_scene_collection`**: Name of the scene collection to modify
- **`monitor_devices`**: List of output device names that could be used for monitoring (normally only one device would be listed)
- **`inputs`**: Mapping of Windows input device names to OBS source names
- **`output_captures`**: Mapping of Windows output device names to OBS source names

## Usage

### Basic usage

Normally AudioUnfuck is meant to be run without arguments. It will do its thing and get out of your way, only giving a popup if configs change. There are additional command line options that can be used, however:


- `--config PATH`: Use a custom configuration file
- `--dryrun`: Test configuration without making changes
- `--list-devices`: List all available input and output devices on the system
- `--debug`: Enable debug logging, useful for finding out the exact naming for your audio devices
- `--help`: Show help message
