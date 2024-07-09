#!/usr/bin/env python3
import sys

import ffmpeg
from tdvutil import ppretty

# Read the video file from the command line arguments
# video_file = "tmp/ytilamronllihcgnirob_v2181000284.mp4"
video_file = sys.argv[1]

# Use ffprobe to extract all possible metadata from the video file
metadata = ffmpeg.probe(video_file)

# Print the creation time if available
# print(ppretty(metadata))

fpsstring = None
timecode = None

for stream in metadata["streams"]:
    if stream["codec_type"] == "video":
        fpsstring = stream["r_frame_rate"]
        continue
    elif stream["codec_type"] == "data":
        if stream["codec_tag_string"] == "tmcd":
            timecode = stream["tags"]["timecode"]
            continue
        else:
            pass

if not all([fpsstring, timecode]):
    print("Error: could not find all required metadata")
    sys.exit(1)

assert fpsstring is not None

if "/" in fpsstring:
    num, den = fpsstring.split("/")
    if den != "1":
        print(f"Error: non-integer framerate ({fpsstring}) not currently supported")
        sys.exit(1)
    fps = int(num)
else:
    fps = int(fpsstring)
print(f"timecode: {timecode} @ {fps} fps")

# for stream in metadata['streams']:
#     if 'tags' in stream and 'creation_time' in stream['tags']:
#         print(f"Creation time: {stream['tags']['creation_time']}")
#     else:
#         print("Creation time not found in metadata.")
