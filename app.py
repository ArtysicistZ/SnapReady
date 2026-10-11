"""SnapReady: a live webcam coach for passport photos.

    python app.py [--cam 0] [--hfov 70]      # q = quit, r = retake
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

from check import FACE_OVAL, Checker
from finish import Finisher
from guide import Coach

GREEN = (60, 200, 60)
ORANGE = (60, 160, 255)
RED = (40, 40, 230)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw(frame, measurements, checks, banner, all_passed, photo):
    """Face outline, instruction banner, per-rule status, and a thumbnail of the photo."""
    if measurements is not None:
        outline = measurements["points"][FACE_OVAL].astype(np.int32)
        cv2.polylines(frame, [outline], True, GREEN if all_passed else ORANGE, 2, cv2.LINE_AA)
    frame = cv2.flip(frame, 1)     # mirror, like a selfie camera
    height, width = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (width, 40), (40, 120, 40) if all_passed else (40, 40, 40), -1)
    cv2.putText(frame, banner, (12, 28), FONT, 0.8, WHITE, 2, cv2.LINE_AA)
    for row, (name, passed, _) in enumerate(checks):
        label = ("OK " if passed else "X  ") + name
        cv2.putText(frame, label, (12, 70 + 22 * row), FONT, 0.55, GREEN if passed else RED, 2, cv2.LINE_AA)

    if photo is not None:
        size = height // 3
        frame[height - size - 10:height - 10, width - size - 10:width - 10] = cv2.resize(photo, (size, size))
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--hfov", type=float, default=70, help="camera horizontal field of view (deg)")
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.cam)
    checker = Checker()
    finisher = Finisher()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")) as f:
        coach = Coach(json.load(f)["rules"])
    os.makedirs("captures", exist_ok=True)
    photo = None

    while True:
        ok, frame = camera.read()
        if not ok:
            break
        focal_px = frame.shape[1] / 2 / np.tan(np.radians(args.hfov / 2))   # pinhole model
        measurements = checker(frame, focal_px)
        checks, todo, capture = coach.update(frame, measurements)

        if capture is not None and photo is None:
            capture_frame, capture_measurements = capture
            photo = finisher(capture_frame, capture_measurements["points"])
            cv2.imwrite(time.strftime("captures/snap_%Y%m%d_%H%M%S.jpg"), photo)

        if photo is not None:
            banner = "Captured! r = retake, q = quit"
        elif todo is not None:
            banner = todo
        else:
            banner = f"Hold still... {(coach.HOLD - coach.streak) // 7 + 1}"

        view = draw(frame.copy(), measurements, checks, banner, todo is None, photo)
        cv2.imshow("SnapReady", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            photo = None
            coach.reset()

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
