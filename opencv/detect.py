#!/usr/bin/env python3
import argparse
import colorsys
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

import cv2
import cv2.aruco as aruco
import numpy as np
import skimage
from feature_matching import FeatureMatching
from tdvutil import ppretty
from vidgear.gears import WriteGear

# FIXME: should these be int?
Point: TypeAlias = Tuple[int, int]

@dataclass
# UL, UR, BR, BL
class MarkerCorners:
    UL: Point
    UR: Point
    BR: Point
    BL: Point

    def as_array(self):
        return [self.UL, self.UR, self.BR, self.BL]


REFERENCE_IMAGE = "Looper Marker Reference.png"
REFERENCE_MASK1 = "Looper Channel 1 Mask.png"
REFERENCE_MASK2 = "Lopper Channel 2 Mask.png"

mask1_roi = ((660, 485), (755, 582))

reference_markers: Dict[int, MarkerCorners] = {
    1: MarkerCorners(
        BL=(432, 412),
        BR=(468, 384),
        UL=(403, 381),
        UR=(438, 355)
    ),
    2: MarkerCorners(
        BL=(744, 106),
        BR=(775, 129),
        UL=(771, 83),
        UR=(801, 107)
    ),
    9: MarkerCorners(
        BL=(539, 531),
        BR=(561, 553),
        UL=(564, 509),
        UR=(586, 533)
    )
}

reference_dims = (1280, 720)


def flatten(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))


def findAruco(args: argparse.Namespace, frame, detector: aruco.ArucoDetector) -> Optional[Dict[int, MarkerCorners]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # key = getattr(aruco, 'DICT_APRILTAG_16h5')
    # arucoDict = aruco.getPredefinedDictionary(key)
    # arucoParam = aruco.DetectorParameters()

    # # print(ppretty(arucoParam))
    # arucoParam.minMarkerDistanceRate = 0.03
    # arucoParam.adaptiveThreshConstant = 7
    # arucoParam.adaptiveThreshWinSizeMin = 20
    # arucoParam.adaptiveThreshWinSizeMax = 100
    # arucoParam.adaptiveThreshWinSizeStep = 10
    # # arucoParam.minMarkerPerimeterRate = 0.03
    # # arucoParam.maxMarkerPerimeterRate = 4.0
    # arucoParam.minMarkerPerimeterRate = 0.02
    # arucoParam.maxMarkerPerimeterRate = 0.2
    # arucoParam.polygonalApproxAccuracyRate = 0.03

    # detector = aruco.ArucoDetector(arucoDict, arucoParam)

    bboxes, ids, rejected = detector.detectMarkers(gray)

    # print(f"{ids=}")
    # print(f"{bboxes}")

    if len(bboxes) == 0:
        return None

    markers: MarkerCorners = {}

    ids = ids.flatten()
    for (markerCorner, markerID) in zip(bboxes, ids):
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners

        marker = MarkerCorners(
            UL=(int(topLeft[0]), int(topLeft[1])),
            UR=(int(topRight[0]), int(topRight[1])),
            BR=(int(bottomRight[0]), int(bottomRight[1])),
            BL=(int(bottomLeft[0]), int(bottomLeft[1])),
        )
        markers[markerID] = marker

    if not args.show_aruco:
        return markers

    for id, marker in markers.items():
        # print(f"{id=}")
        # print(f"{marker=}")
        # draw the bounding box of the ArUCo detection
        cv2.line(frame, marker.UL, marker.UR, (255, 255, 0), 4)
        cv2.line(frame, marker.UR, marker.BR, (0, 255, 0), 4)
        cv2.line(frame, marker.BR, marker.BL, (0, 255, 0), 4)
        cv2.line(frame, marker.BL, marker.UL, (0, 255, 0), 4)

        # compute and draw the center (x, y)-coordinates of the ArUco
        # marker
        cX = int((marker.UL[0] + marker.BR[0]) / 2.0)
        cY = int((marker.UL[1] + marker.BR[1]) / 2.0)
        cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

        # draw the ArUco marker ID on the image
        cv2.putText(frame, str(id),
                    (marker.UL[0], marker.UL[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    cv2.imshow("Image", frame)
    cv2.waitKey(0)

    return markers


def cluster(args: argparse.Namespace, frame: cv2.typing.MatLike):
    # k-means clustering attempt
    pixels = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    k = 3  # number of clusters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), k,
                                    None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    if args.show_clusters:
        print(f"centers={centers.astype(int)}")

    # remove the most dominant color, which is going to be black. If it's
    # NOT black, we have a problem.
    bincounts = np.bincount(labels.flatten())
    maxindex = np.argmax(bincounts)

    # Return early if black isn't the most dominant color
    if not np.array_equal(centers[maxindex], [0, 0, 0]):
        # print("Black is not the most common color, problem")
        return None

    bincounts = np.delete(bincounts, maxindex)
    centers = np.delete(centers, maxindex, axis=0)

    # What's left should have an actual color as the most common thing
    maxindex = np.argmax(bincounts)
    dominant_color = centers[maxindex]
    # print(f"{dominant_color=}")
    # dominant_color = centers[np.argmax(np.bincount(labels.flatten()))]
    # x = -np.sort(-np.bincount(labels.flatten()))
    # print(f"{x=}")
    # for c in centers:
    #     print(c)
    # print(dominant_color)

    if args.show_clusters:
        hist = np.zeros(k)
        for label in labels:
            hist[label] += 1

        import matplotlib.pyplot as plt

        plt.bar(range(k), hist, color=[tuple(c / 255.0 for c in center) for center in centers])
        plt.xlabel('Cluster')
        plt.ylabel('Pixel Count')
        plt.title('Color Clusters Histogram')
        plt.show()

    return dominant_color


def do_frame(args: argparse.Namespace, frame: cv2.typing.MatLike,
             detector: aruco.ArucoDetector, ref_markers: Dict[int, MarkerCorners],
             mask_img: cv2.typing.MatLike, mask_roi: Tuple[Point, Point]):
    # input_file = Path(sys.argv[1])
    # if not input_file.exists():
    #     print(f"ERROR: File {input_file} does not exist", file=sys.stderr)
    #     sys.exit(1)

    # print(f"{input_file}: ", end="")

    # frame = skimage.io.imread(input_file)
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    # frame = cv2.imread(sys.argv[1])

    # frame = cv2.resize(frame, reference_dims)
    # print(frame.shape)

    markers = findAruco(args, frame, detector)

    if not markers:
        print("No markers found.")
        # sys.exit(1)
        return None, None, None

    # if len(markers) != len(reference_markers):
    # print(f"Found {len(markers)}/{len(reference_markers)} markers")
    # sys.exit(1)
    # else:
    #     print(f"found correct number ({len(reference_markers)}) of markers")

    try:
        dst_pts = flatten([ref_markers[k].as_array()
                           for k in sorted(markers.keys())])
        src_pts = flatten([markers[k].as_array()
                           for k in sorted(markers.keys())])
    except KeyError as e:
        print(f"found invalid marker id: {e}")
        return None, None, None

    if len(markers) < 2:
        print("Not enough markers found")
        return None, None, None

    print(f"Found {len(markers)}/{len(reference_markers)} valid markers")

    matrix, _ = cv2.findHomography(np.array(src_pts), np.array(dst_pts))
    warped = cv2.warpPerspective(frame, matrix, reference_dims)

    if args.show_warped:
        cv2.imshow("warped", warped)
        cv2.waitKey(0)

    # channel_mask_img = skimage.io.imread("Looper Channel 1 Mask.png")
    # channel_mask_img = cv2.cvtColor(channel_mask_img, cv2.COLOR_RGBA2BGR)

    # print(warped.shape)
    # print(channel_mask_img.shape)

    masked_img = cv2.bitwise_and(warped, mask_img)
    # cv2.imshow("Masked", masked_img)
    # cv2.waitKey(0)

    roi = masked_img[mask_roi[0][1]:mask_roi[1][1], mask_roi[0][0]:mask_roi[1][0]]
    # hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    if args.show_roi:
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)

    # output_file = Path("frames_button") / input_file.name
    # cv2.imwrite(str(output_file), roi)

    # roi[:, :, 0] = 0
    # roi[:, :, 1] = 0
    # roi[:, :, 2] = 0

    # cv2.imshow("ROI", roi)
    # cv2.waitKey(0)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    threshold = 100
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    masked = cv2.bitwise_and(roi, roi, mask=mask)
    # cv2.imshow("roi", roi)
    # cv2.imshow("masked", masked)
    # cv2.waitKey(0)

    avgbright = np.mean(cv2.bitwise_and(gray, gray, mask=mask))
    dominant = cluster(args, masked)
    # print(f"{avgbright=}")
    # print(f"{dominant=}")

    return roi, avgbright, dominant

    asgrey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


    # k-means clustering attempt
    pixels = roi.reshape(-1, 3)
    k = 15  # number of clusters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), k,
                                    None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # remove the most dominant color, which is going to be black
    bincounts = np.bincount(labels.flatten())
    maxindex = np.argmax(bincounts)

    np.delete(bincounts, maxindex)
    np.delete(centers, maxindex)

    # What's left should have an actual color as the most common thing
    maxindex = np.argmax(bincounts)
    dominant_color = centers[maxindex]
    # dominant_color = centers[np.argmax(np.bincount(labels.flatten()))]
    # x = -np.sort(-np.bincount(labels.flatten()))
    # print(f"{x=}")
    for c in centers:
        print(c)
    # print(dominant_color)

    # histogram that
    hist = np.zeros(k)
    for label in labels:
        hist[label] += 1

    import matplotlib.pyplot as plt

    plt.bar(range(k), hist, color=[tuple(c / 255.0 for c in center) for center in centers])
    plt.xlabel('Cluster')
    plt.ylabel('Pixel Count')
    plt.title('Color Clusters Histogram')
    plt.show()


    # avg_color = np.mean(roi, axis=(0, 1))
    # print(avg_color)

    # cv2.waitKey(0)
    sys.exit(0)

    imgAug = cv2.imread("icon.png")
    capture = cv2.VideoCapture("looper 2024-06-15 08h06m12s-05.09.09.995-05.15.30.415.mp4")
    # looper 2024-05-23 21h19m26s.mp4
    assert capture.isOpened(), 'Cannot capture source'

    # inframe = cv2.imread('IMG_20240615_150916551_HDR.jpg')
    inframe = cv2.imread('looper 2024-06-15 08h06m12s-07.16.52.300.png')
    # fourcc = cv2.VideoWriter_fourcc(*'X264')
    # out = cv2.VideoWriter('output.mp4', fourcc, 29.97, (1280, 720))

    # define suitable (Codec,CRF,preset) FFmpeg parameters for writer
    # on windows, probably add logging false and -disable_ffmpeg_window
    output_params = {"-vcodec": "libx264", "-crf": 23, "-preset": "fast",
                     "-input_framerate": 60,
                     # "-output_dimensions": (1280, 720)
                     }


    # Define writer with defined parameters and suitable output filename
    out = WriteGear(output='output.mp4', logging=False, **output_params)

    # marked_frame = findAruco(inframe, imgAug)
    # cv2.imshow("img", marked_frame)
    # cv2.waitKey(10000)

    # loop over frames
    for success, frame in iter(capture.read, (False, None)):
        # cv2.imshow("frame", frame)
        marked_frame = findAruco(frame, imgAug)
        if marked_frame is not None:
            out.write(marked_frame)
            cv2.imshow('img', marked_frame)
        # else:
        #     out.write(frame)
        #     cv2.imshow('img', frame)

        # out.write(frame)

        global framenum
        framenum += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    out.close()
    cv2.destroyAllWindows()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Share to discord some of the changes ojbpm has detected")

    parser.add_argument(
        "filenames",
        type=Path,
        # action="append",
        nargs="+",
        metavar="filename",
        help="image file(s) to process"
    )

    parser.add_argument(
        "--show-aruco",
        default=False,
        action="store_true",
        help="show detected aruco markers"
    )

    parser.add_argument(
        "--show-warped",
        default=False,
        action="store_true",
        help="show warped image after marker detection"
    )

    parser.add_argument(
        "--show-roi",
        default=False,
        action="store_true",
        help="show masked and cropped region of interest"
    )

    parser.add_argument(
        "--show-clusters",
        default=False,
        action="store_true",
        help="show colors after clustering",
    )

    parser.add_argument(
        "--show-all",
        default=False,
        action="store_true",
        help="show all intermediate images",
    )

    # parser.add_argument(
    #     "--credentials-file", "-c",
    #     type=Path,
    #     default=None,
    #     action=CheckFile(must_exist=True),
    #     help="file with discord credentials"
    # )

    # parser.add_argument(
    #     "--watch-dir",
    #     type=Path,
    #     default=None,
    #     action=CheckFile(must_exist=True),
    #     required=True,
    #     help="ojbpm export path to watch for bpm changes",
    # )

    # parser.add_argument(
    #     "--host",
    #     type=str,
    #     default=None,  # 192.168.1.152
    #     help="address or hostname of host running OBS"
    # )

    # parser.add_argument(
    #     "--port",
    #     type=int,
    #     default=4455,
    #     help="port number for OBS websocket"
    # )

    parsed_args = parser.parse_args()

    if parsed_args.show_all:
        parsed_args.show_aruco = True
        parsed_args.show_warped = True
        parsed_args.show_roi = True
        parsed_args.show_clusters = True

    return parsed_args


def main():
    args = parse_args()

    key = aruco.DICT_APRILTAG_16h5
    arucoDict = aruco.getPredefinedDictionary(key)
    arucoParam = aruco.DetectorParameters()

    # print(ppretty(arucoParam))
    arucoParam.minMarkerDistanceRate = 0.03
    arucoParam.adaptiveThreshConstant = 7
    arucoParam.adaptiveThreshWinSizeMin = 20
    arucoParam.adaptiveThreshWinSizeMax = 100
    arucoParam.adaptiveThreshWinSizeStep = 10
    # arucoParam.minMarkerPerimeterRate = 0.03
    # arucoParam.maxMarkerPerimeterRate = 4.0
    arucoParam.minMarkerPerimeterRate = 0.02
    arucoParam.maxMarkerPerimeterRate = 0.2
    arucoParam.polygonalApproxAccuracyRate = 0.03

    # FIXME: Should we pass findAruco() the detector object, or the dict/params?
    detector = aruco.ArucoDetector(arucoDict, arucoParam)

    mask_img = skimage.io.imread("Looper Channel 1 Mask.png")
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGBA2BGR)

    # def do_frame(args: argparse.Namespace, detector: aruco.ArucoDetector,
    #             ref_markers: Dict[int, MarkerCorners], mask_img: cv2.typing.MatLike,
    #             mask_roi: Tuple[Point, Point]) -> None:
    if args.filenames[0].is_dir():
        args.filenames = args.filenames[0].glob("*.jpg")

    for file in args.filenames:
        print(f"{file}: ", end="")

        frame = skimage.io.imread(str(file))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        roi, avgbright, dominant = do_frame(args, frame, detector, reference_markers, mask_img, mask1_roi)

        if roi is None:
            continue

        print(f"{avgbright=}")
        print(f"{dominant=}")

        if dominant is not None:
            hsv = colorsys.rgb_to_hsv(dominant[0]/255, dominant[1]/255, dominant[2]/255)
            # hls = colorsys.rgb_to_hls(dominant[0]/255, dominant[1]/255, dominant[2]/255)
            print(f"{hsv=}")
            # print(f"{hls=}")

        # cv2.imshow("ROI", color_frame)
        # cv2.waitKey(1000)
        if roi is not None:
            if dominant is None:
                output_file = Path("frames_weird") / file.name
            elif avgbright < 35:
                output_file = Path("frames_dark") / str(str(int(avgbright)) + "_" + file.name)
            elif 0 < hsv[0] < (50/360):
                output_file = Path("frames_red") / str(str(int(hsv[0] * 360)) + "_" + file.name)
            elif (100/360) < hsv[0] < (180/360):
                output_file = Path("frames_green") / str(str(int(hsv[0] * 360)) + "_" + file.name)
            else:
                output_file = Path("frames_unknown") / str(str(int(hsv[0] * 360)) + "_" + file.name)
            cv2.imwrite(str(output_file), roi)
        # print(args)
        # do(args)


if __name__ == "__main__":
    main()
