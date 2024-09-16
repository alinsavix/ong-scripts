#!/usr/bin/env python3
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
    detection_success: bool
    detection_area: int
    detected_center: int
    # smoothed_center: int
    # smoothed_target_point: int


class Consumer(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        pname = self.name

        baseopts = base_options.BaseOptions(model_asset_path='efficientdet_lite0.tflite')  # delegate=mp.tasks.BaseOptions.Delegate.GPU)
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

        # # Set up our detection model
        # baseopts = base_options.BaseOptions(model_asset_path='efficientdet_lite0.tflite')  # delegate=mp.tasks.BaseOptions.Delegate.GPU)
        # options = ObjectDetectorOptions(base_options=baseopts, score_threshold=0.5)
        # detector = ObjectDetector.create_from_options(detector_options)

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Perform object detection
        detection_result = detector.detect(mp_image)

        # Calculate center point
        if detection_result.detections:
            detection_success = True

            # Use only the first detection
            detection = detection_result.detections[0]
            bbox = detection.bounding_box

            # Calculate the area of the bounding box
            bbox_area = bbox.width * bbox.height
            if bbox_area < 100000:
                detection_success = False
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
            # If no detection, use the last known center
            detection_success = False
            bbox_area = 0
            # current_center = last_center if last_center else (frame.shape[1] // 2, 540)
            current_center = (0, 540)

        return_frame = None
        return frame_num, SingleCenterpoint(detection_success, bbox_area, current_center[0]), return_frame

    def __str__(self):
        return f"Processing object detections on frame_num={self.frame_num}"


def centerpoints_from_video(args: argparse.Namespace, video_file: Path) -> Dict[int, SingleCenterpoint]:
    centerpoints: Dict[int, SingleCenterpoint] = {}

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
    print(f"Starting {num_consumers} consumers")

    tasks = multiprocessing.JoinableQueue(maxsize=num_consumers * 2)
    results = multiprocessing.Queue()

    consumers = [Consumer(tasks, results) for _ in range(num_consumers)]
    for consumer in consumers:
        consumer.start()

    threaded_mode = True

    cap = cv2.VideoCapture(video_file)
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
            centerpoints[fn] = cp
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
        centerpoints[fn] = cp
        if fr is not None:
            cv2.imshow('Cropped Object Detection', fr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # print("Done")
    # while len(pending) > 0:
    #     if pending[0].ready():
    #         fn, cp, fr = pending.popleft().get()
    #         centerpoints[fn] = cp

    log(f"Done (frames: {len(centerpoints)})")
        # # Update last_center
        # last_center = current_center

        # # Calculate smoothed center
        # if smoothed_center is None:
        #     smoothed_center = current_center
        # else:
        #     smoothed_center = (
        #         int(smoothed_center[0] * (1 - center_smoothing_factor) + current_center[0] * center_smoothing_factor),
        #         int(smoothed_center[1] * (1 - center_smoothing_factor) + current_center[1] * center_smoothing_factor)
        #     )

        # # Initialize or update target point
        # if target_point is None:
        #     target_point = smoothed_center
        #     smoothed_target_point = target_point
        # else:
        #     distance = abs(smoothed_center[0] - smoothed_target_point[0])
        #     if distance > distance_threshold:
        #         frame_counter += 1
        #         if frame_counter > 30:
        #             target_point = smoothed_center
        #             frame_counter = 0
        #     else:
        #         frame_counter = 0

        # # Smooth the target point
        # if smoothed_target_point is None:
        #     smoothed_target_point = target_point
        # else:
        #     smoothed_target_point = (
        #         int(smoothed_target_point[0] * (1 - target_smoothing_factor) + target_point[0] * target_smoothing_factor),
        #         int(smoothed_target_point[1] * (1 - target_smoothing_factor) + target_point[1] * target_smoothing_factor)
        #     )

        # # Crop frame
        # crop_width = 608
        # crop_height = 1080
        # left = max(0, min(smoothed_target_point[0] - crop_width // 2, frame.shape[1] - crop_width))
        # top = 0  # Always start from the top of the frame
        # # cropped_frame = frame[top:top+crop_height, left:left+crop_width]

        # centerpoints.append(Centerpoint(detection_success, current_center[0], smoothed_center[0], smoothed_target_point[0]))

        # # Draw a box around the cropped area
        # if args.debug:
        #     cv2.rectangle(frame, (left, top), (left + crop_width, top + crop_height), (255, 0, 0), 2)

        #     cv2.circle(frame, (smoothed_center[0], smoothed_center[1]), 5, (0, 255, 0), -1)
        #     cv2.circle(frame, (smoothed_target_point[0], smoothed_target_point[1]), 5, (0, 0, 255), -1)

        # cv2.imshow('Cropped Object Detection', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()

    return centerpoints


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

        whatever = centerpoints_from_video(args, vidfile)
        # print(ppretty(whatever))


if __name__ == "__main__":
    main()
