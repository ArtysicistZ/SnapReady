"""Make results/figures/guidance.jpg: the instruction v1 shows for 8 test cases (proposal Fig. 1).

    python figure_guidance.py      # needs data/ from benchmark.py

Every banner is the real output of check.py + guide.py with the limits in config.json.
(d) and (h) are failures on purpose: a soft side shadow passes, and a smiling face is told
to open its eyes.
"""

import io
import json
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from benchmark import crop as lfw_crop
from check import FACE_OVAL, Checker
from guide import evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 320
GREEN, ORANGE, RED = (60, 170, 60), (0, 140, 255), (40, 40, 220)

checker = Checker()
with open(os.path.join(HERE, "config.json")) as f:
    RULES = json.load(f)["rules"]


def decode(image_bytes):
    rgb = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def first_instruction(image):
    """(measurements, first failing instruction or None if every rule passes)."""
    m = checker(image)
    if m is None:
        return None, "Move into the frame"
    failing = [text for _, passed, text in evaluate(m, RULES) if not passed]
    return m, (failing[0] if failing else None)


def face_crop(image, shift=0.0):
    """Square crop around the face (2.2 face heights wide), shifted right by `shift` of its width.
    The window is kept inside the photo, so the face may sit a little off center."""
    m = checker(image)
    oval = m["points"][FACE_OVAL]
    (x0, y0), (x1, y1) = oval.min(axis=0), oval.max(axis=0)
    height, width = image.shape[:2]
    side = min(int(2.2 * (y1 - y0)), height, width)
    cx, cy = (x0 + x1) / 2 + shift * side, (y0 + y1) / 2
    x = int(np.clip(cx - side / 2, 0, width - side))
    y = int(np.clip(cy - side / 2, 0, height - side))
    return cv2.resize(image[y:y + side, x:x + side], (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)


def side_shadow(image, darkest):
    x = np.linspace(-1, 1, SIZE)
    ramp = darkest + (1 - darkest) * np.clip((x / 0.6 + 1) / 2, 0, 1)
    return np.clip(image * ramp[None, :, None], 0, 255).astype(np.uint8)


def rotate(image, degrees):
    rotation = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), degrees, 1)
    return cv2.warpAffine(image, rotation, (SIZE, SIZE), borderMode=cv2.BORDER_REFLECT)


# Pointing'04 rows picked by eye: the same person, in a copy with natural colors
# (the Hugging Face data has 5 color-shifted copies of every photo).
FRONTAL = (1239, (0, 0))
TURNED = (1143, (0, 30))


def pose_photo(pose, row):
    """The photo without the black border lines some Pointing'04 photos have at the edges."""
    index, (tilt, pan) = row
    assert tuple(pose["y"][index]) == (tilt, pan)
    image = decode(pose["x"][index]["bytes"])
    rows = np.flatnonzero(image.mean(axis=(1, 2)) > 60)
    cols = np.flatnonzero(image.mean(axis=(0, 2)) > 60)
    return image[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def panel(image, label, failure):
    m, text = first_instruction(image)
    view = image.copy()
    if m is not None:
        cv2.polylines(view, [m["points"][FACE_OVAL].astype(np.int32)], True,
                      GREEN if text is None else ORANGE, 2, cv2.LINE_AA)
    banner = np.full((34, SIZE, 3), (40, 40, 40), np.uint8)
    if text is None:
        banner[:] = (40, 120, 40)
    text_shown = text or "All checks pass - hold still"
    scale = 0.55
    while cv2.getTextSize(text_shown, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > SIZE - 12:
        scale -= 0.02                                    # shrink long instructions to fit
    cv2.putText(banner, text_shown, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 1, cv2.LINE_AA)
    caption = np.full((30, SIZE, 3), 255, np.uint8)
    cv2.putText(caption, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                RED if failure else (0, 0, 0), 1, cv2.LINE_AA)
    print(f"{label:40s} -> {text or 'all pass'}")
    return np.vstack([banner, view, caption])


def main():
    data = os.path.join(HERE, "data")
    pose = pd.read_parquet(os.path.join(data, "pose.parquet"))
    eyes = pd.read_parquet(os.path.join(data, "eyes_001.parquet"))
    lfw = pd.read_parquet(os.path.join(data, "lfw.parquet")).drop_duplicates("label")

    # one ID-like frontal photo, then damaged copies of it (same damages as benchmark.py)
    frontal = pose_photo(pose, FRONTAL)
    good = face_crop(frontal)
    shadow = side_shadow(good, 0.4)          # benchmark: 0.85 = good, 0.35 = bad
    print("shadow ratio of (d): %.2f, limit %.2f" % (checker(shadow)["shadow_ratio"],
                                                     RULES["shadow_ratio"]["max"]))
    turned = face_crop(pose_photo(pose, TURNED))

    closed = None
    for image_data in eyes["Image_data"]:
        image = cv2.resize(decode(image_data["file"]), (SIZE, SIZE))
        if first_instruction(image)[1] == "Open your eyes":
            closed = image
            break

    smiling = None                                       # a real LFW face, eyes open, told to open them
    for image in lfw.sample(200, random_state=1)["image"]:
        image = lfw_crop(decode(image["bytes"]))
        if first_instruction(image)[1] == "Open your eyes":
            smiling = image
            break

    panels = [
        panel(good, "(a) Good photo", False),
        panel(face_crop(frontal, 0.19), "(b) Off-center", False),
        panel(np.clip(good * 0.35, 0, 255).astype(np.uint8), "(c) Too dark", False),
        panel(shadow, "(d) Side shadow: should fail", True),
        panel(rotate(good, 15), "(e) Head tilted 15 deg", False),
        panel(turned, "(f) Head turned 30 deg", False),
        panel(closed, "(g) Eyes closed", False),
        panel(smiling, "(h) Smiling, eyes open: wrong", True),
    ]
    gap = np.full((panels[0].shape[0], 6, 3), 255, np.uint8)
    rows = [np.hstack(sum([[p, gap] for p in panels[r:r + 4]], [])[:-1]) for r in (0, 4)]
    figure = np.vstack([rows[0], np.full((6, rows[0].shape[1], 3), 255, np.uint8), rows[1]])
    out = os.path.join(HERE, "results", "figures", "guidance.jpg")
    cv2.imwrite(out, figure, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("saved", out, figure.shape)


if __name__ == "__main__":
    main()
