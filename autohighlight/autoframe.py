#!/usr/bin/env python3
# FILE = "X:/zTEMP/autohighlight_tmp/highlight_20_clean-2024-09-07 23h01m51s.mp4"
FILE = "X:/zTEMP/autohighlight_tmp/highlight_23_clean-2024-09-07 23h01m51s.mp4"
# FILE = "X:/zTEMP/autohighlight_tmp/highlight_24_clean-2024-09-07 23h01m51s.mp4"

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import ObjectDetector, ObjectDetectorOptions
from vidgear.gears import WriteGear

# Load the model
base_options = base_options.BaseOptions(model_asset_path='efficientdet_lite0.tflite')
options = ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = ObjectDetector.create_from_options(options)

# Initialize video capture
cap = cv2.VideoCapture(FILE)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_params = {"-vcodec": "libx264", "-crf": 23, "-preset": "fast",
                    "-input_framerate": 60,
                    "-output_dimensions": (608, 1080)
                    }

# Define writer with defined parameters and suitable output filename
out = WriteGear(output='test.mp4', logging=False, **output_params)


# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter("test.mp4", fourcc, fps, (608, 1080))

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

frame_counter = 0


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # Perform object detection
    detection_result = detector.detect(mp_image)

    # Calculate center point
    if detection_result.detections:
        # Use only the first detection
        detection = detection_result.detections[0]
        bbox = detection.bounding_box

        # Calculate center point (only horizontal)
        center_x = bbox.origin_x + bbox.width // 2
        center_y = 540  # Fixed vertical component

        current_center = (center_x, center_y)

        # Draw bounding box
        # cv2.rectangle(frame, (bbox.origin_x, bbox.origin_y),
        #               (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
        #               (0, 255, 0), 2)
    else:
        # If no detection, use the last known center
        current_center = last_center if last_center else (frame.shape[1] // 2, 540)

    # Update last_center
    last_center = current_center

    # Calculate smoothed center
    if smoothed_center is None:
        smoothed_center = current_center
    else:
        smoothed_center = (
            int(smoothed_center[0] * (1 - center_smoothing_factor) + current_center[0] * center_smoothing_factor),
            int(smoothed_center[1] * (1 - center_smoothing_factor) + current_center[1] * center_smoothing_factor)
        )

    # Initialize or update target point
    if target_point is None:
        target_point = smoothed_center
        smoothed_target_point = target_point
    else:
        distance = abs(smoothed_center[0] - smoothed_target_point[0])
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
        smoothed_target_point = (
            int(smoothed_target_point[0] * (1 - target_smoothing_factor) + target_point[0] * target_smoothing_factor),
            int(smoothed_target_point[1] * (1 - target_smoothing_factor) + target_point[1] * target_smoothing_factor)
        )

    # Crop frame
    crop_width = 608
    crop_height = 1080
    left = max(0, min(smoothed_target_point[0] - crop_width // 2, frame.shape[1] - crop_width))
    top = 0  # Always start from the top of the frame
    cropped_frame = frame[top:top+crop_height, left:left+crop_width]

    # Draw smoothed center point
    cv2.circle(cropped_frame, (smoothed_center[0] - left, smoothed_center[1]), 5, (0, 255, 0), -1)

    # Draw smoothed target point
    cv2.circle(cropped_frame, (smoothed_target_point[0] - left, smoothed_target_point[1]), 5, (0, 0, 255), -1)

    # Write the cropped frame to the output video
    out.write(cropped_frame)

    cv2.imshow('Cropped Object Detection', cropped_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.close()
cv2.destroyAllWindows()
