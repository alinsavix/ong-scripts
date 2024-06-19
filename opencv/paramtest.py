#!/usr/bin/env python3
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

import cv2
import cv2.aruco as aruco
import numpy as np
from feature_matching import FeatureMatching
from skimage import io
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


def findAruco(frame, draw=True) -> Optional[Dict[int, MarkerCorners]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    key = getattr(aruco, 'DICT_APRILTAG_16h5')
    arucoDict = aruco.getPredefinedDictionary(key)
    arucoParam = aruco.DetectorParameters()
    # detector = aruco.ArucoDetector(arucoDict, arucoParam)

    arucoParam.minMarkerDistanceRate = 0.03
    arucoParam.adaptiveThreshWinSizeMin = 20
    arucoParam.adaptiveThreshWinSizeMax = 400
    arucoParam.adaptiveThreshWinSizeStep = 10
    arucoParam.minMarkerPerimeterRate = 0.02
    arucoParam.maxMarkerPerimeterRate = 0.2
    arucoParam.polygonalApproxAccuracyRate = 0.03
    arucoParam.adaptiveThreshWinSizeMax = 400

    for x in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
        arucoParam.adaptiveThreshConstant = x
        for y in [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 400, 500]:
            arucoParam.adaptiveThreshWinSizeMax = y

            detector = aruco.ArucoDetector(arucoDict, arucoParam)
            bboxes, ids, rejected = detector.detectMarkers(gray)
            print(f"{x},{y}: {len(bboxes)} found")

    return
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

    if not draw:
        return markers

    for id, marker in markers.items():
        print(f"{id=}")
        print(f"{marker=}")
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

    # print(rejected)
    for r in rejected:
        (topLeft, topRight, bottomRight, bottomLeft) = r[0]

        rm = MarkerCorners(
            UL=(int(topLeft[0]), int(topLeft[1])),
            UR=(int(topRight[0]), int(topRight[1])),
            BR=(int(bottomRight[0]), int(bottomRight[1])),
            BL=(int(bottomLeft[0]), int(bottomLeft[1])),
        )

        cv2.line(frame, rm.UL, rm.UR, (0, 0, 255), 2)
        cv2.line(frame, rm.UR, rm.BR, (0, 0, 255), 2)
        cv2.line(frame, rm.BR, rm.BL, (0, 0, 255), 2)
        cv2.line(frame, rm.BL, rm.UL, (0, 0, 255), 2)

        # cv2.line(frame, int(topLeft), int(topRigh), (0, 0, 255), 4)
        # cv2.line(frame, int(topRight), int(bottomRight), (0, 0, 255), 4)
        # cv2.line(frame, int(bottomRight), int(bottomLeft), (0, 0, 255), 4)
        # cv2.line(frame, int(bottomLeft), int(topLeft), (0, 0, 255), 4)
        # cv2.aruco.drawDetectedMarkers(frame, rejected[i], None, borderColor=(0, 0, 255))

    cv2.imshow("Image", frame)
    cv2.waitKey(5000)

    return markers

    # convert each of the (x, y)-coordinate pairs to integers
    topRight = (int(topRight[0]), int(topRight[1]))
    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
    topLeft = (int(topLeft[0]), int(topLeft[1]))

    global framenum
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # key = getattr(aruco, 'DICT_APRILTAG_25h9')
    key = getattr(aruco, 'DICT_APRILTAG_16h5')
    arucoDict = aruco.getPredefinedDictionary(key)
    arucoParam = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(arucoDict, arucoParam)

    bboxs, ids, rejected = detector.detectMarkers(gray)

    # Loop detected corners
    if len(bboxs) > 0:
        ids = ids.flatten()
        didit = False
        for (markerCorner, markerID) in zip(bboxs, ids):
            # if markerID != 12:
            #     continue
            didit = True

            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # cv2.line(frame, topLeft, topRight, (0, 255, 0), 4)
            print(f"top line: {topLeft} - {topRight} (frame {framenum})")
            # return frame

            # pts_src = np.array(bboxs[0][0], dtype=float)
            # pts_src = np.array([topLeft, topRight, bottomRight, bottomLeft], dtype=float)
            # pts_dst = np.array([[20, 20], [120, 20], [120, 120], [20, 120]], dtype=float)
            # h, _ = cv2.findHomography(pts_src, pts_dst)
            # warped = cv2.warpPerspective(frame, h, (1920, 1080))

            # cv2.putText(warped, str(framenum) + " - " + str(markerID),
            #             (0, 220), cv2.FONT_HERSHEY_SIMPLEX,
            #             1, (0, 255, 0), 2)

            # return frame

            # cv2.aruco.drawDetectedMarkers(frame, bboxs, ids)
            # return frame

            # extract corners, always UL, UR, BL, BR
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # draw the bounding box of the ArUCo detection
            # cv2.line(frame, topLeft, topRight, (0, 255, 0), 4)
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 4)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 4)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 4)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 4)

            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

            # draw the ArUco marker ID on the image
            cv2.putText(frame, str(markerID),
                        (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            # print(f"[INFO] ArUco marker ID: {markerID}")
        cv2.imwrite("output.png", frame)
        return frame
        # frame = augmentAruco(markerCorner, markerID, frame, imgAug)

        # cv2.imshow("Image", frame)
        # if didit:
        #     return frame

    # else
    return None


if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <filename>", file=sys.stderr)
    sys.exit(1)

frame = io.imread(sys.argv[1])
frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
# frame = cv2.imread(sys.argv[1])
findAruco(frame)

def findAruco_old(frame, imgAug, draw=True) -> Any:
    global framenum
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # key = getattr(aruco, 'DICT_APRILTAG_25h9')
    key = getattr(aruco, 'DICT_APRILTAG_16h5')
    arucoDict = aruco.getPredefinedDictionary(key)
    arucoParam = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(arucoDict, arucoParam)

    bboxs, ids, rejected = detector.detectMarkers(gray)

    # Loop detected corners
    if len(bboxs) > 0:
        ids = ids.flatten()
        didit = False
        for (markerCorner, markerID) in zip(bboxs, ids):
            # if markerID != 12:
            #     continue
            didit = True

            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # cv2.line(frame, topLeft, topRight, (0, 255, 0), 4)
            print(f"top line: {topLeft} - {topRight} (frame {framenum})")
            # return frame

            # pts_src = np.array(bboxs[0][0], dtype=float)
            # pts_src = np.array([topLeft, topRight, bottomRight, bottomLeft], dtype=float)
            # pts_dst = np.array([[20, 20], [120, 20], [120, 120], [20, 120]], dtype=float)
            # h, _ = cv2.findHomography(pts_src, pts_dst)
            # warped = cv2.warpPerspective(frame, h, (1920, 1080))

            # cv2.putText(warped, str(framenum) + " - " + str(markerID),
            #             (0, 220), cv2.FONT_HERSHEY_SIMPLEX,
            #             1, (0, 255, 0), 2)

            # return frame

            # cv2.aruco.drawDetectedMarkers(frame, bboxs, ids)
            # return frame

            # extract corners, always UL, UR, BL, BR
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # draw the bounding box of the ArUCo detection
            # cv2.line(frame, topLeft, topRight, (0, 255, 0), 4)
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 4)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 4)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 4)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 4)

            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

            # draw the ArUco marker ID on the image
            cv2.putText(frame, str(markerID),
                        (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            # print(f"[INFO] ArUco marker ID: {markerID}")
        cv2.imwrite("output.png", frame)
        return frame
        # frame = augmentAruco(markerCorner, markerID, frame, imgAug)

        # cv2.imshow("Image", frame)
        # if didit:
        #     return frame

    # else
    return None


def augmentAruco(bbox, id, img, imgAug, drawId=True):
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]
    h, w, c = imgAug.shape
    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix, _ = cv2.findHomography(pts2, pts1)
    imgout = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))
    imgout = img + imgout
    return imgout

    # sh_query = imgout.shape
    # matrix, _ = cv2.findHomography(pts1, pts2)
    # img_warped = cv2.warpPerspective(
    #     imgout, matrix, (sh_query[1], sh_query[0]))
    #
    # return img_warped


def main() -> None:
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


# if __name__ == "__main__":
#     main()
