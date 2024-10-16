#!/usr/bin/env python
import argparse
# import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import torch
from tdvutil import ppretty
from ultralytics import YOLO
from vidgear.gears import WriteGear

# If we're built into an executable via pyinstaller, use the bundled ffmpeg.
# Otherwise, set nothing here and resolve the ffmpeg path based on cli options
# later.
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    FFMPEG_BIN: str | None = os.path.join(sys._MEIPASS, "ffmpeg")
else:
    FFMPEG_BIN = None

if platform.system() == "Darwin":
    FFMPEG_ENCODER = [ "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-b:v", "0" ]
else:
    FFMPEG_ENCODER = ["-c:v", "h264_nvenc", "-preset", "p3", "-rc", "constqp","-qp", "16", "-b:v", "0" ]


def log(msg):
    print(msg)
    sys.stdout.flush()

def now():
    return time.time()

@dataclass
class SingleCenterpoint:
    detection_success: int
    detection_area: int
    detection_center: int

@dataclass
class SmoothedCenterpoint:
    detection_center: int
    smoothed_center_x: float
    smoothed_center_vel: float


# generate a temporary file with just the subset of the input video we want
def extract_partial_video(args: argparse.Namespace, video_path: Path, tmpdir: Path, time_offset: float, length: float) -> Optional[Path]:
    if not tmpdir.exists():
        log("ERROR: video extraction temp directory doesn't exist")
        return None

    fh, _tmpfile = tempfile.mkstemp(suffix=".mp4", prefix="tmp_autocrop_", dir=tmpdir)
    os.close(fh)
    tmpfile = Path(_tmpfile)

    ffmpeg_cmd = [
        FFMPEG_BIN if FFMPEG_BIN else args.ffmpeg_bin,
        "-hide_banner", "-hwaccel", "auto",
        "-ss", f"{time_offset}",
        "-t", str(length),
        "-i", str(video_path),
        "-vsync", "vfr",
        *FFMPEG_ENCODER,
        "-c:a", "alac",
        "-y",
        str(tmpfile)
    ]

    log(f"INFO: Extracting working file from {video_path}")
    if args.debug:
        log(f"DEBUG: Command: {' '.join(ffmpeg_cmd)}")

    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        log(f"INFO: Successfully extracted to working file {tmpfile}")
    except subprocess.CalledProcessError as e:
        log(f"ERROR: Couldn't extract working file: {e}")
        log(f"ERROR: ffmpeg stderr: {e.stderr}")
        tmpfile.unlink(missing_ok=True)
        return None

    return tmpfile


def centerpoints_from_video(args: argparse.Namespace, video_file: Path, cp_file: Path) -> List[SingleCenterpoint]:
    initial_centerpoints: Dict[int, SingleCenterpoint] = {}

    if cp_file.exists() and not args.force_detection:
        with cp_file.open("r") as f:
            for line in f:
                frame_num, detection_success, detection_area, detection_center = [
                    int(x) for x in line.strip().split(',')]
                initial_centerpoints[int(frame_num)] = SingleCenterpoint(
                    int(detection_success), int(detection_area), int(detection_center))
        return [initial_centerpoints[key] for key in sorted(initial_centerpoints.keys())]

    frame_num = 0
    stride = 1

    # try to figure out what torch backend to use
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    log(f"Using torch device '{device}' with model {args.model}")

    start_time = now()
    model = YOLO(args.model).to(device)

    results = model.track(video_file, show=args.show, stream=True, classes=[
                          0], vid_stride=stride, tracker="bytetrack.yaml", verbose=args.verbose)

    log(f"Model initialized in {(now() - start_time) * 1000:0.1f}ms")

    start_time = now()
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            bbox = result.boxes[0].xywh[0]
            cp = SingleCenterpoint(1, int(bbox[2] * bbox[3]), int(bbox[0]))
        else:
            cp = SingleCenterpoint(0, 0, 0)

        for i in range(stride):
            initial_centerpoints[frame_num + i] = cp

        frame_num += stride

    detect_time = now() - start_time
    log(f"Ran detections on {frame_num} frames in {detect_time:0.3f}s ({(detect_time / frame_num) * 1000:0.1f}ms/frame)")

    if args.keep_centerpoints:
        with cp_file.open("w") as f:
            for key in sorted(initial_centerpoints.keys()):
                cp = initial_centerpoints[key]
                f.write(f"{key},{cp.detection_success},{cp.detection_area},{cp.detection_center}\n")

    return [initial_centerpoints[key] for key in sorted(initial_centerpoints.keys())]


# This does the mass-spring-damper thing, courtesy of the amazing Penwywern
# (see https://en.wikipedia.org/wiki/Mass-spring-damper_model)
#
# freq should probably be adjusted to match the actual frame interval so that
# 'damp' actually ends up being a time in seconds, but what's here works for
# now.
def calcnewx(oldx: float, oldv: float, center: float, timestep: float = 0.0166,
             freq: float = 1.0, damp: float = 1.0) -> Tuple[float, float]:
    accel = -2 * damp * freq * oldv - freq**2 * (oldx - center)
    newv = oldv + accel * timestep
    newx = oldx + newv * timestep

    # print(f"{center}: {oldx},{oldv} --> {newx},{newv}")
    return newx, newv


# smooth out our framewise list of centerpoints to give us a responsive camera
# that isn't too "jumpy".
def smooth_centerpoints(centerpoints: List[SingleCenterpoint]) -> List[SmoothedCenterpoint]:
    smoothed_centerpoints = []

    new_x = new_v = None
    last_center = None

    # get a few frames into the spring so we can have a more stable beginning
    for cp in centerpoints[:15]:
        if cp.detection_success:
            current_center = cp.detection_center
        else:
            # If this frame didn't have a detection, use the last detection.
            # If there wasn't a last detection, use the center of the frame.
            current_center = last_center if last_center else (1920 // 2)

        last_center = current_center

        if new_x is None or new_v is None:
            new_x = float(current_center)
            new_v = 0.0

        new_x, new_v = calcnewx(new_x, new_v, current_center)

    assert new_x is not None
    assert new_v is not None

    # And now do it for real
    last_center = None
    for cp in centerpoints:
        if cp.detection_success:
            current_center = cp.detection_center
        else:
            current_center = last_center if last_center else (1920 // 2)

        last_center = current_center

        # newx/newv are guaranteed to already be set because of our warmup
        new_x, new_v = calcnewx(new_x, new_v, current_center)
        smoothed_centerpoints.append(SmoothedCenterpoint(current_center, new_x, new_v))

    return smoothed_centerpoints


# this function crops a video using ffmpeg's "sendcmd" filter, and *should* be
# identical to our existing crop code, only faster. However, it appears that it
# actually produces something different, and less smooth, and it's not obvious
# why. Keeping it here so we can revisit it, though.
#
# FIXME: revisit
def crop_video_new(args: argparse.Namespace, smoothed_centerpoints: List[SmoothedCenterpoint], input_path: Path, output_path: Path):
    print(f"cropping {input_path} -> {output_path}")
    frame_width = 1920
    frame_height = 1080

    crop_width = 608
    crop_height = 1080

    frametime = 1 / 60.0

    f = Path("commands.txt").open("w")

    for i, scp in enumerate(smoothed_centerpoints):
        left = int(max(0, min(scp.smoothed_center_x - crop_width // 2, frame_width - crop_width)))
        top = 0  # Always start from the top of the frame

        if i == 0:
            f.write(f"0 crop w {crop_width}, crop h {crop_height}, crop x {left}, crop y {top};\n")
        else:
            f.write(
                f"{i * frametime - 0.005} crop w {crop_width}, crop h {crop_height}, crop x {left}, crop y {top};\n")

    f.close()

    ffmpeg_cmd = [
        FFMPEG_BIN if FFMPEG_BIN else args.ffmpeg_bin,
        "-hide_banner", "-hwaccel", "auto",
        "-r", "60",
        "-i", str(input_path),   # Original video input
        "-filter_complex", "[0:v]sendcmd=f=commands.txt,crop",
        "-an",
        *FFMPEG_ENCODER,
        "-pix_fmt", "yuv420p",
        # "-shortest",           # Finish encoding when the shortest input ends
        "-y",
        str(output_path),
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Successfully cropped to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error cropping video: {e}")


# copy the video to a new file, cropping as we go based on previously generated
# and smoothed detection centerpoint info
def crop_video(args: argparse.Namespace, smoothed_centerpoints: List[SmoothedCenterpoint],
               input_path: Path, output_path: Path):
    cap = cv2.VideoCapture(str(input_path))

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if width != 1920:
        print(f"Autocropping currently supported only for 1080p videos (width was {width})")
        return

    # FIXME: see if we can use nvenc here
    output_params = {
        "-vcodec": "libx264", "-crf": 20, "-preset": "medium", "-pix_fmt": "yuv420p",
        "-input_framerate": 60,
        "-output_dimensions": (608, 1080)
    }
    out = WriteGear(output=str(output_path), logging=False, **output_params)

    frame_counter = 0
    prev = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Crop frame
        crop_width = 608
        crop_height = 1080
        # log(f"frame {frame_counter} of {len(smoothed_centerpoints)}")

        # Somewhere we have an off-by-one error where we can have more frames
        # than we have centerpoint data, so use the previous centerpoint if
        # that happens.
        try:
            scp = smoothed_centerpoints[frame_counter]
            left = int(max(0, min(scp.smoothed_center_x - crop_width // 2, frame.shape[1] - crop_width)))
            prev = left
        except IndexError:
            log(f"warning: off-by-one (no centerpoint for frame {frame_counter})")
            left = prev
        top = 0  # Always start from the top of the frame
        cropped_frame = frame[top:top + crop_height, left:left + crop_width]

        if args.debug:
            cropped_frame = cv2.putText(cropped_frame, f"frame {frame_counter}", (10, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            cropped_frame = cv2.putText(cropped_frame, f"c: {scp.detection_center}", (10, 135),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            cropped_frame = cv2.putText(cropped_frame, f"c_x: {scp.smoothed_center_x:0.3f}", (10, 170),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            cropped_frame = cv2.putText(cropped_frame, f"c_vel: {scp.smoothed_center_vel:0.3f}", (10, 205),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        out.write(cropped_frame)
        frame_counter += 1

    cap.release()
    out.close()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autocrop a video file to work better in vertical format"
    )

    parser.add_argument(
        "videofiles",
        metavar="VIDEOFILE",
        type=Path,
        nargs='+',
        help="The video file(s) to process",
    )

    # yolov8n.pt

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        nargs=1,
        help="Which YOLO model to use for Ong detection",
    )


    parser.add_argument(
        "--ffmpeg-bin",
        type=str,
        default="ffmpeg",
        help="ffmpeg binary name or full path to use",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Diplay object detection and centerpoint info",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Display YOLO 'verbose' output",
    )

    parser.add_argument(
        "--force-detection",
        action="store_true",
        default=False,
        help="Force (re)detection of Ong positions",
    )

    parser.add_argument(
        "--force-crop", "--force",
        action="store_true",
        default=False,
        help="Force (re)cropping of video",
    )

    parser.add_argument(
        "--keep-centerpoints",
        action="store_true",
        default=False,
        help="Keep the centerpoint analysis in .centerpoints file",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Visually show object detections",
    )

    parser_results = parser.parse_args()

    if parser_results.force_detection:
        parser_results.force_crop = True

    return parser.parse_args()


def main():
    args = parse_args()

    for vidfile in args.videofiles:
        if not vidfile.exists():
            print(f"video file '{vidfile}' does not exist", file=sys.stderr)
            continue

        print(f"Autocropping {vidfile}...")

        cropped_filename = f"tmp_crop_{vidfile.name}"
        cropped_path = vidfile.parent / cropped_filename
        final_filename = f"{vidfile.stem}_cropped{vidfile.suffix}"
        final_path = vidfile.parent / final_filename

        if final_path.exists() and not args.force_crop:
            print(f"Final file {final_path} already exists, skipping")
            continue

        workfile = extract_partial_video(args, vidfile, cropped_path.parent, 3, 60)
        if workfile is None:
            print("WARNING: Failed to extract working file, skipping")
            continue

        cpoint_file = vidfile.parent / (vidfile.name + ".centerpoints")

        log("Generating centerpoints...")
        centerpoints = centerpoints_from_video(args, workfile, cpoint_file)
        # print(centerpoints)
        log("Smoothing...")
        smoothed = smooth_centerpoints(centerpoints)

        log("Cropping...")
        crop_video(args, smoothed, workfile, cropped_path)

        log("Merging...")
        ffmpeg_cmd = [
            FFMPEG_BIN if FFMPEG_BIN else args.ffmpeg_bin,
            "-i", str(cropped_path),  # Video input (output from previous processing)
            "-i", str(workfile),      # Audio input (original file)
            "-c:v", "copy",          # Copy video codec
            "-c:a", "libfdk_aac", "-vbr", "5", "-cutoff", "18000",  # Use AAC for audio codec
            "-map", "0:v:0",         # Use video from first input
            "-map", "1:a:0",         # Use audio from second input
            # "-shortest",           # Finish encoding when the shortest input ends
            "-y",
            final_path
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True)
            print(f"Successfully combined video and audio into {final_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error combining video and audio: {e}")

        # Clean up intermediate file
        workfile.unlink(missing_ok=True)
        cropped_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
