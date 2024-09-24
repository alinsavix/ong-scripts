#!/usr/bin/env python
import argparse
import multiprocessing
import os
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import ObjectDetector, ObjectDetectorOptions
from tdvutil import ppretty
from vidgear.gears import WriteGear


def log(msg):
    print(msg)
    sys.stdout.flush()


@dataclass
class SingleCenterpoint:
    detection_success: int
    detection_area: int
    detection_center: int
    # smoothed_center: int
    # smoothed_target_point: int

@dataclass
class SmoothedCenterpoint:
    detection_center: int
    smoothed_center: int
    smoothed_target_point: int


class Consumer(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        pname = self.name

        baseopts = base_options.BaseOptions(model_asset_path='efficientdet_lite2.tflite')  # delegate=mp.tasks.BaseOptions.Delegate.GPU)
        detector_options = ObjectDetectorOptions(base_options=baseopts, score_threshold=0.5)
        detector = ObjectDetector.create_from_options(detector_options)

        log(f"Consumer {pname} started")

        # while not self.task_queue.empty():
        while True:
            temp_task = self.task_queue.get()
            if temp_task is None:
                log(f"Exit request for consumer {pname}")
                self.task_queue.task_done()
                break

            # log(f"Processing task: {pname}, {temp_task}")

            result = temp_task.process(detector)
            # log(f"Result from {pname}: {result}")

            self.result_queue.put(result)
            self.task_queue.task_done()


class Task():
    def __init__(self, args, frame_num, frame):
        self.args = args
        self.frame_num = frame_num
        self.frame = frame

    def process(self, detector):
        frame_num = self.frame_num
        frame = self.frame
        args = self.args

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Perform object detection
        detection_result = detector.detect(mp_image)

        # Calculate center point
        if detection_result.detections:
            detection_success = 1

            # Use only the first detection
            detection = detection_result.detections[0]
            bbox = detection.bounding_box

            # Calculate the area of the bounding box
            bbox_area = bbox.width * bbox.height
            if bbox_area < 100000:
                detection_success = 0
                current_center = (0, 540)
                # current_center = last_center if last_center else (frame.shape[1] // 2, 540)
                # print(f"bbox_area: {bbox_area}")
            else:
                # Calculate center point (only horizontal)
                center_x = bbox.origin_x + bbox.width // 2
                center_y = 540  # Fixed vertical component

                current_center = (center_x, center_y)

                # Draw bounding box
                if args.debug:
                    cv2.rectangle(frame, (bbox.origin_x, bbox.origin_y),
                                (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                                (0, 255, 0), 2)
                    cv2.circle(frame, current_center, 5, (0, 255, 0), -1)
        else:
            # If no detection, mark it as such, we'll come back to it
            detection_success = 0
            bbox_area = 0
            # current_center = last_center if last_center else (frame.shape[1] // 2, 540)
            current_center = (0, 540)

        if args.debug:
            return_frame = frame
        else:
            return_frame = None

        return frame_num, SingleCenterpoint(detection_success, bbox_area, current_center[0]), return_frame

    def __str__(self):
        return f"Processing object detections on frame_num={self.frame_num}"


def centerpoints_from_video(args: argparse.Namespace, video_file: Path, cp_file: Path) -> List[SingleCenterpoint]:
    initial_centerpoints: Dict[int, SingleCenterpoint] = {}

    if cp_file.exists():
        with open(cp_file, 'r') as f:
            for line in f:
                frame_num, detection_success, detection_area, detection_center = [int(x) for x in line.strip().split(',')]
                initial_centerpoints[int(frame_num)] = SingleCenterpoint(int(detection_success), int(detection_area), int(detection_center))
        return [initial_centerpoints[key] for key in sorted(initial_centerpoints.keys())]

    frame_num = 0
    stride = 1

    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

    results = model.track(video_file, show=False, stream=True, classes=[0], vid_stride=stride, tracker="bytetrack.yaml")

    for result in results:
        if len(result.boxes) > 0:
            bbox = result.boxes[0].xywh[0]
            cp = SingleCenterpoint(1, int(bbox[2] * bbox[3]), int(bbox[0]))
        else:
            cp = SingleCenterpoint(0, 0, 0)

        for i in range(stride):
            initial_centerpoints[frame_num + i] = cp

        frame_num += stride

    with open(cp_file, 'w') as f:
        for key in sorted(initial_centerpoints.keys()):
            cp = initial_centerpoints[key]
            f.write(f"{key},{cp.detection_success},{cp.detection_area},{cp.detection_center}\n")

    return [initial_centerpoints[key] for key in sorted(initial_centerpoints.keys())]


def centerpoints_from_video_old(args: argparse.Namespace, video_file: Path, cp_file: Path) -> List[SingleCenterpoint]:
    initial_centerpoints: Dict[int, SingleCenterpoint] = {}

    if cp_file.exists():
        with open(cp_file, 'r') as f:
            for line in f:
                frame_num, detection_success, detection_area, detection_center = line.strip().split(',')
                initial_centerpoints[int(frame_num)] = SingleCenterpoint(int(detection_success), int(detection_area), int(detection_center))
        return [initial_centerpoints[key] for key in sorted(initial_centerpoints.keys())]
    # Set up our detection model

    # detector = ObjectDetector.create_from_options(detector_options)


    # Initialize last_center as None
    last_center = None

    # Initialize variables for smoothing
    center_smoothing_factor = 0.05
    smoothed_center = None

    # Initialize variables for target point and frame counter
    target_point = None
    target_smoothing_factor = 0.05
    smoothed_target_point = None
    distance_threshold = 50

    smoothing_frame_counter = 0
    frame_num = 0

    num_consumers = multiprocessing.cpu_count()


    # Limit our queue size so we don't completely fill memory with frames
    tasks = multiprocessing.JoinableQueue(maxsize=num_consumers * 2)
    results = multiprocessing.Queue()

    log(f"Starting {num_consumers} consumers")

    consumers = [Consumer(tasks, results) for _ in range(num_consumers)]
    for consumer in consumers:
        consumer.daemon = True
        consumer.start()

    cap = cv2.VideoCapture(video_file)
    try:
        while True:

            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # if frame_num > 400:
            #     break

            # args, detector_options, frame_num, frame
            # log(f"Putting task {frame_num}")
            task = Task(args, frame_num, frame)
            tasks.put(task)

            while not results.empty():
                fn, cp, fr = results.get()
                # log(f"Got result {fn}, {cp}")
                initial_centerpoints[fn] = cp
                if fr is not None:
                    cv2.imshow('Cropped Object Detection', fr)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        # we've sent all the tasks, send them an exit command
        for i in range(num_consumers):
            # log(f"Putting exit task {i}")
            tasks.put(None)

        tasks.join()

        while not results.empty():
            fn, cp, fr = results.get()
            # print(f"Got result {fn}, {cp}")
            initial_centerpoints[fn] = cp
            if fr is not None:
                cv2.imshow('Cropped Object Detection', fr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        for consumer in consumers:
            consumer.terminate()

    # print("Done")
    # while len(pending) > 0:
    #     if pending[0].ready():
    #         fn, cp, fr = pending.popleft().get()
    #         centerpoints[fn] = cp

    cap.release()
    cv2.destroyAllWindows()

    log(f"Done (frames: {len(initial_centerpoints)})")

    with open(cp_file, 'w') as f:
        for key in sorted(initial_centerpoints.keys()):
            cp = initial_centerpoints[key]
            f.write(f"{key},{cp.detection_success},{cp.detection_area},{cp.detection_center}\n")

    return [initial_centerpoints[key] for key in sorted(initial_centerpoints.keys())]


# This does the mass-spring-damper thing, courtesy of the amazing Penwywern
# (see https://en.wikipedia.org/wiki/Mass-spring-damper_model)
def calcnewx(oldx, oldv, center, timestep=0.0166, freq=1, damp=1):
    accel = -2 * damp * freq * oldv - freq**2 * (oldx - center)
    newv = oldv + accel * timestep
    newx = oldx + newv * timestep

    print(f"{center}: {oldx},{oldv} --> {newx},{newv}")
    return newx, newv


def smooth_centerpoints(centerpoints: List[SingleCenterpoint]) -> List[SmoothedCenterpoint]:
    smoothed_centerpoints = []

    prevx = prevv = None

    for centerpoint in centerpoints:
        if centerpoint.detection_success:
            current_center = centerpoint.detection_center
        else:
            current_center = last_center if last_center else (1920 // 2)

        last_center = current_center

        if prevx is None or prevv is None:
            prevx = current_center
            prevv = 0

        # last_center = current_center
        newx, newv = calcnewx(prevx, prevv, current_center)

        smoothed_centerpoints.append(SmoothedCenterpoint(current_center, int(newx), int(newx)))
        prevx = newx
        prevv = newv

    return smoothed_centerpoints


def smooth_centerpoints_old(centerpoints: List[SingleCenterpoint]) -> List[SmoothedCenterpoint]:
    smoothed_centerpoints = []

    last_center = None

    # Initialize variables for smoothing
    center_smoothing_factor = 0.05
    smoothed_center = None

    # Initialize variables for target point and frame counter
    target_point = None
    target_smoothing_factor = 0.05
    smoothed_target_point = None
    distance_threshold = 50

    smoothing_frame_counter = 0
    frame_num = 0

    for centerpoint in centerpoints:
        if centerpoint.detection_success:
            current_center = centerpoint.detection_center
        else:
            current_center = last_center if last_center else (1920 // 2)

        last_center = current_center

        if smoothed_center is None:
            smoothed_center = current_center
        else:
            smoothed_center = int(smoothed_center * (1 - center_smoothing_factor) + current_center * center_smoothing_factor)

        # Initialize or update target point
        if target_point is None:
            target_point = smoothed_center
            smoothed_target_point = target_point
        else:
            distance = abs(smoothed_center - smoothed_target_point)
            if distance > distance_threshold:
                frame_counter += 1
                if frame_counter > 30:
                    target_point = smoothed_center
                    frame_counter = 0
            else:
                frame_counter = 0

        # Smooth the target point
        if smoothed_target_point is None:
            smoothed_target_point = target_point
        else:
            smoothed_target_point = int(smoothed_target_point * (1 - target_smoothing_factor) + target_point * target_smoothing_factor)

        smoothed_centerpoints.append(SmoothedCenterpoint(current_center, smoothed_center, smoothed_target_point))

    return smoothed_centerpoints


def crop_video(smoothed_centerpoints: List[SmoothedCenterpoint], input_path: Path, output_path: Path):
    cap = cv2.VideoCapture(str(input_path))

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if width != 1920:
        print(f"Autocropping currently supported only for 1080p videos (width was {width})")
        return

    output_params = {"-vcodec": "libx264", "-crf": 20, "-preset": "medium", "-pix_fmt": "yuv420p",
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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Crop frame
        crop_width = 608
        crop_height = 1080
        # log(f"frame {frame_counter} of {len(smoothed_centerpoints)}")

        # Somewhere we have an off-by-one error where we can have more frames
        # than we have centerpoint data, so use the previous centerpoint if
        # that happens.
        try:
            cp = smoothed_centerpoints[frame_counter].smoothed_target_point
            left = max(0, min(smoothed_centerpoints[frame_counter].smoothed_target_point - crop_width // 2, frame.shape[1] - crop_width))
            prev = left
        except IndexError:
            log(f"warning: off-by-one (no centerpoint for frame {frame_counter})")
            left = prev
        top = 0  # Always start from the top of the frame
        cropped_frame = frame[top:top+crop_height, left:left+crop_width]

        cropped_frame = cv2.putText(cropped_frame, str(frame_counter), (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cropped_frame = cv2.putText(cropped_frame, f"c_x: {cp}", (100, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
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
        help="The video file(s) to process"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Diplay object detection and centerpoint info"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    for vidfile in args.videofiles:
        if not vidfile.exists():
            print(f"video file '{vidfile}' does not exist", file=sys.stderr)
            continue

        print(f"Autocropping {vidfile}...")

        output_filename = f"cropped_video_{vidfile.name}"
        output_path = vidfile.parent / output_filename
        final_filename = f"cropped_{vidfile.name}"
        final_path = vidfile.parent / final_filename

        if final_path.exists():
            print(f"Final file {final_path} already exists, skipping")
            continue

        cpfile = vidfile.parent / (vidfile.name + ".centerpoints")
        # print(cpfile)
        log("Generating centerpoints...")
        centerpoints = centerpoints_from_video(args, vidfile, cpfile)
        # print(centerpoints)
        log("Smoothing...")
        smoothed = smooth_centerpoints(centerpoints)

        log("Cropping...")
        crop_video(smoothed, vidfile, output_path)

        ffmpeg_cmd = [
            "ffmpeg",
            "-i", str(output_path),  # Video input (output from previous processing)
            "-i", str(vidfile),      # Audio input (original file)
            "-c:v", "copy",          # Copy video codec
            "-c:a", "copy",          # Use AAC for audio codec
            "-map", "0:v:0",         # Use video from first input
            "-map", "1:a:0",         # Use audio from second input
            # "-shortest",           # Finish encoding when the shortest input ends
            final_path
        ]

        log("Merging...")
        try:
            subprocess.run(ffmpeg_cmd, check=True)
            print(f"Successfully combined video and audio into {final_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error combining video and audio: {e}")

        # Clean up intermediate file
        os.remove(output_path)


if __name__ == "__main__":
    main()
