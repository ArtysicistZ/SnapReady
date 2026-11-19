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
# Mirrored landmark pairs (image left, image right): eye corners, mouth corners, face sides, brows.
# The head tilt is the mean angle of the lines joining them.
LEFT_SIDE = [33, 133, 61, 234, 70]
RIGHT_SIDE = [263, 362, 291, 454, 300]
# Mirrored cheek landmarks: the same skin on both sides, compared for even lighting.
LEFT_CHEEK = [50, 205, 117, 123, 147, 187, 36, 101, 118]
RIGHT_CHEEK = [280, 425, 346, 352, 376, 411, 266, 330, 347]
FACE_WIDTH = 112   # sharpness is measured on the face at this width (about the benchmark faces),
                   # so its fitted limit holds at any distance from the camera


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

        # face outline: its bounding box gives a crop for sharpness, its inside a mask for light
        oval = points[FACE_OVAL]
        x_min, y_min = oval.min(axis=0).clip(0).astype(int)
        x_max, y_max = oval.max(axis=0).astype(int)
        face_mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(face_mask, [oval.astype(np.int32)], 1)
        face_mask = face_mask.astype(bool)

        # lighting: mean brightness around mirrored cheek points (a box filter, i.e. a convolution).
        # Comparing the same skin on both sides keeps hair, background and head turn out.
        patch = 2 * int(0.08 * (x_max - x_min)) + 1
        local = cv2.blur(gray.astype(np.float32), (patch, patch)) + 1
        x, y = points.clip(0, [width - 1, height - 1]).astype(int).T
        left_light = local[y[LEFT_CHEEK], x[LEFT_CHEEK]].sum()
        right_light = local[y[RIGHT_CHEEK], x[RIGHT_CHEEK]].sum()

        # sharpness = share of the edge strength that one more blur removes (re-blur): a sharp
        # face loses a lot, a blurry one little. Being a ratio, contrast cancels out.
        face = gray[y_min:y_max, x_min:x_max].astype(np.float32)   # no rounding of faint edges
        face = cv2.resize(face, (FACE_WIDTH, max(1, round(FACE_WIDTH * face.shape[0] / face.shape[1]))),
                          interpolation=cv2.INTER_AREA)
        reblurred = cv2.GaussianBlur(face, (0, 0), 2)

        # head angles from the rotation of MediaPipe's 3D face model
        turn = np.degrees(np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0])))
        nod = np.degrees(np.arctan2(rotation[2, 1], rotation[2, 2]))
        level_lines = points[RIGHT_SIDE] - points[LEFT_SIDE]
        eyes_center = (points[RIGHT_EYE[1]] + points[LEFT_EYE[0]]) / 2   # the spec places the eye line

        # distance from the iris size, pinhole camera (lecture 1): focal * real size / image size
        distance_cm = None
        if focal_px is not None:
            iris_px = (np.ptp(points[LEFT_IRIS, 0]) + np.ptp(points[RIGHT_IRIS, 0])) / 2
            distance_cm = focal_px * IRIS_DIAMETER_MM / iris_px / 10

        return {
            "points": points,
            "distance_cm": distance_cm,
            "face_height": (points[CHIN, 1] - points[FOREHEAD, 1]) / height,
            "offset_x": (eyes_center[0] - width / 2) / width,
            "offset_y": (eyes_center[1] - height / 2) / width,
            "head_turn": turn,
            "head_nod": nod,
            "head_tilt": np.degrees(np.arctan2(level_lines[:, 1], level_lines[:, 0])).mean(),
            "eye_closure": max(blendshapes["eyeBlinkLeft"], blendshapes["eyeBlinkRight"]),
            "brightness": np.percentile(gray[face_mask], 90),   # lit skin, not beard or brows
            "shadow_ratio": max(left_light, right_light) / min(left_light, right_light),
            "sharpness": 1 - edge_strength(reblurred) / edge_strength(face),
        }


def edge_strength(gray):
    """Mean gradient magnitude, from Sobel convolutions (HW2)."""
    return np.hypot(cv2.Sobel(gray, cv2.CV_64F, 1, 0), cv2.Sobel(gray, cv2.CV_64F, 0, 1)).mean()
