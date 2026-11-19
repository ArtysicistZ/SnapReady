"""Module 3 - Finish: level the eyes, crop to the US passport spec, white background."""

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from check import CHIN, FOREHEAD, LEFT_EYE, NOSE_BRIDGE, RIGHT_EYE, download_model

SEGMENTER_MODEL = "image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"

PHOTO_SIZE = 600         # 2 x 2 inches at 300 dpi
HEAD_HEIGHT = 0.60       # chin to top of head: 50-69% of the photo height
EYES_FROM_TOP = 0.38     # eye line: 56-69% of the height, measured from the bottom
CROWN_FACTOR = 1.22      # chin-to-crown / chin-to-forehead-landmark (the landmark is below the hair)
PYRAMID_LEVELS = 2       # deeper pyramids leave a gray halo around the head


class Finisher:
    def __init__(self):
        options = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=download_model(SEGMENTER_MODEL)),
            output_confidence_masks=True,
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def __call__(self, bgr, points):
        """Photo and its face landmarks -> PHOTO_SIZE x PHOTO_SIZE passport photo."""
        # 1. Person mask from the selfie segmenter, keeping only the region that contains
        #    our face, so people in the background are removed.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self.segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        confidence = np.squeeze(result.confidence_masks[-1].numpy_view())
        _, regions = cv2.connectedComponents((confidence > 0.5).astype(np.uint8))
        nose_x, nose_y = points[NOSE_BRIDGE].astype(int)
        person = (regions == regions[nose_y, nose_x]).astype(np.float32)

        # 2. Similarity transform (lecture 5): level the eyes, scale the head, place the eye line.
        left_eye = points[list(LEFT_EYE)].mean(axis=0)
        right_eye = points[list(RIGHT_EYE)].mean(axis=0)
        eyes_center = (left_eye + right_eye) / 2
        dx, dy = right_eye - left_eye
        angle = np.degrees(np.arctan2(dy, dx))
        head_px = CROWN_FACTOR * np.linalg.norm(points[CHIN] - points[FOREHEAD])
        scale = HEAD_HEIGHT * PHOTO_SIZE / head_px

        transform = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale)
        transform[:, 2] += np.array([PHOTO_SIZE / 2, EYES_FROM_TOP * PHOTO_SIZE]) - eyes_center

        size = (PHOTO_SIZE, PHOTO_SIZE)
        photo = cv2.warpAffine(bgr, transform, size, flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.warpAffine(person, transform, size)

        # 3. Laplacian-pyramid blending (Project 3) onto white. White has no detail at any
        #    scale, so only the coarsest level mixes with white; every finer level adds the
        #    photo's detail wherever the mask says "person".
        photos = [photo.astype(np.float32)]
        masks = [mask[..., None]]
        for _ in range(PYRAMID_LEVELS):
            photos.append(cv2.pyrDown(photos[-1]))
            masks.append(cv2.pyrDown(masks[-1])[..., None])

        result = photos[-1] * masks[-1] + 255 * (1 - masks[-1])
        for level in reversed(range(PYRAMID_LEVELS)):
            level_size = photos[level].shape[1::-1]
            detail = photos[level] - cv2.pyrUp(photos[level + 1], dstsize=level_size)
            result = cv2.pyrUp(result, dstsize=level_size) + detail * masks[level]
        return np.clip(result, 0, 255).astype(np.uint8)
