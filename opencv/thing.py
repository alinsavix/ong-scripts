#!/usr/bin/env python3
from typing import Any

import cv2
import cv2.aruco as aruco
import numpy as np
from feature_matching import FeatureMatching
from vidgear.gears import WriteGear

# dict is aruco.DICT_APRILTAG_25h9

framenum = 0

def findAruco(frame, imgAug, draw=True) -> Any:
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
            # print(f"top line: {topLeft} - {topRight} (frame {framenum})")
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
    # capture = cv2.VideoCapture("looper 2024-06-15 08h06m12s-05.09.09.995-05.15.30.415.mp4")
    capture = cv2.VideoCapture("http://tenforward:8080/hls/looper/index.m3u8")
    # looper 2024-05-23 21h19m26s.mp4
    assert capture.isOpened(), 'Cannot capture source'

    # inframe = cv2.imread('IMG_20240615_150916551_HDR.jpg')
    # inframe = cv2.imread('looper 2024-06-15 08h06m12s-07.16.52.300.png')
    # inframe = cv2.imread("frames/img-005313.jpg")
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

    fps = capture.get(cv2.CAP_PROP_FPS)
    wait_ms = int(1000 / fps) - 1
    print('FPS:', fps)

    frame_count = 0

    # loop over frames
    for success, frame in iter(capture.read, (False, None)):
        frame_count += 1
        if (frame_count % 10) != 0:
            # cv2.waitKey(wait_ms)
            cv2.waitKey(wait_ms)
            continue

        # cv2.imshow("frame", frame)
        marked_frame = findAruco(frame, imgAug)
        if marked_frame is not None:
            # out.write(marked_frame)
            cv2.imshow('img', marked_frame)
        # else:
        #     out.write(frame)
        #     cv2.imshow('img', frame)

        # out.write(frame)

        # cv2.waitKey(wait_ms)
        cv2.waitKey(wait_ms)

    capture.release()
    out.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
