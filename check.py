"""Module 1 - Check: measure the face in one video frame."""

import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/"
FACE_MODEL = "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

# MediaPipe face-mesh landmark ids. "Left" and "right" mean image left and right.
CHIN = 152
FOREHEAD = 10
NOSE_BRIDGE = 168
LEFT_EYE = (33, 133)      # (outer corner, inner corner)
RIGHT_EYE = (362, 263)    # (inner corner, outer corner)
LEFT_IRIS = [469, 470, 471, 472]
RIGHT_IRIS = [474, 475, 476, 477]
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

IRIS_DIAMETER_MM = 11.7   # almost the same for every adult


def download_model(path):
    """Return a local copy of a MediaPipe model, downloading it on first use."""
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    local_path = os.path.join(folder, os.path.basename(path))
    if not os.path.exists(local_path):
        os.makedirs(folder, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL + path, local_path)
    return local_path


class Checker:
    """Measures the face in a frame.

    Signed measurements: offset_x > 0 means the face sits right of center in the image,
    offset_y > 0 below center, head_turn > 0 facing image right, head_nod > 0 looking down,
    head_tilt > 0 the image-right eye lower.
    """

    def __init__(self):
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=download_model(FACE_MODEL)),
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def __call__(self, bgr, focal_px=None):
        """Return a dict of measurements, or None if there is no face.

        focal_px: camera focal length in pixels; without it the distance is not measured.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            return None

        height, width = bgr.shape[:2]
        points = np.array([[p.x * width, p.y * height] for p in result.face_landmarks[0]])
        blendshapes = {c.category_name: c.score for c in result.face_blendshapes[0]}
        rotation = np.array(result.facial_transformation_matrixes[0])[:3, :3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # face outline: its bounding box gives position and a crop for sharpness
        oval = points[FACE_OVAL]
        x_min, y_min = oval.min(axis=0).clip(0).astype(int)
        x_max, y_max = oval.max(axis=0).astype(int)
        face_mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(face_mask, [oval.astype(np.int32)], 1)
        face_mask = face_mask.astype(bool)

        # lighting: compare the two halves of the face, split at the nose
        left_half = face_mask.copy()
        left_half[:, int(points[NOSE_BRIDGE, 0]):] = False
        left_light = np.median(gray[left_half]) + 1
        right_light = np.median(gray[face_mask & ~left_half]) + 1

        # sharpness = variance of the Laplacian (a convolution, HW1) on the face only,
        # resized to a fixed width so the distance to the camera does not matter
        face = gray[y_min:y_max, x_min:x_max]
        face = cv2.resize(face, (256, max(1, round(256 * face.shape[0] / face.shape[1]))))

        # head angles from the rotation of MediaPipe's 3D face model
        turn = np.degrees(np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0])))
        nod = np.degrees(np.arctan2(rotation[2, 1], rotation[2, 2]))
        eye_line = points[RIGHT_EYE[1]] - points[LEFT_EYE[0]]

        # distance from the iris size, pinhole camera (lecture 1): focal * real size / image size
        distance_cm = None
        if focal_px is not None:
            iris_px = (np.ptp(points[LEFT_IRIS, 0]) + np.ptp(points[RIGHT_IRIS, 0])) / 2
            distance_cm = focal_px * IRIS_DIAMETER_MM / iris_px / 10

        return {
            "points": points,
            "distance_cm": distance_cm,
            "face_height": (points[CHIN, 1] - points[FOREHEAD, 1]) / height,
            "offset_x": ((x_min + x_max) / 2 - width / 2) / width,
            "offset_y": ((y_min + y_max) / 2 - height / 2) / width,
            "head_turn": turn,
            "head_nod": nod,
            "head_tilt": np.degrees(np.arctan2(eye_line[1], eye_line[0])),
            "eye_closure": max(blendshapes["eyeBlinkLeft"], blendshapes["eyeBlinkRight"]),
            "brightness": np.median(gray[face_mask]),
            "shadow_ratio": max(left_light, right_light) / min(left_light, right_light),
            "sharpness": cv2.Laplacian(face, cv2.CV_64F).var(),
        }
