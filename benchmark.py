"""Compare the baseline with v1 on public data, and fit the rule limits in config.json.

    python benchmark.py        # the first run downloads ~900 MB of data into data/

Test data, each split in half: "calib" (fit the limits) and "test" (report accuracy).
  LFW faces, damaged on purpose  -> framing, blur, exposure, shadow, tilt
    (funneled LFW: aligned by congealing, independent of both analyzers, so the undamaged
     photos are level and centered; we also even out their lighting, so "good" means good)
  Pointing'04 head-pose photos   -> facing (bad if turned or nodded more than 15 degrees)
  Open/closed eyes (AI faces)    -> eyes, with the undamaged LFW photos as real open eyes
A frame where no face is found fails every check.
"""

import io
import itertools
import json
import os
import tarfile
import urllib.request

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from check import Checker
from guide import passes

HERE = os.path.dirname(os.path.abspath(__file__))
HUGGING_FACE = "https://huggingface.co/datasets/"
EYE_FILES = ["001", "050", "100", "400", "700", "800", "900", "950", "990"]
LFW_URL = "https://ndownloader.figshare.com/files/5976015"   # lfw-funneled.tgz
DATA_URLS = {
    "pose": HUGGING_FACE + "StevenLe456/head-pose/resolve/main/data/train-00000-of-00003.parquet",
}
for number in EYE_FILES:
    DATA_URLS[f"eyes_{number}"] = (
        HUGGING_FACE + f"MichalMlodawski/closed-open-eyes/resolve/main/data/dataset_{number}.parquet")

N_PEOPLE = 600     # LFW people (one photo each)
CROP = 180         # LFW crop around the face = test image size, in pixels (no resizing blur)

# Each check is scored with these rule limits (measurement, limit type); the app uses the same rules.
CHECK_RULES = {
    "framing": [("offset_x", "max_abs"), ("offset_y", "min"), ("offset_y", "max")],
    "blur": [("sharpness", "min")],
    "exposure": [("brightness", "min"), ("brightness", "max")],
    "shadow": [("shadow_ratio", "max")],
    "tilt": [("head_tilt", "max_abs")],
    "facing": [("head_turn", "max_abs"), ("head_nod", "min"), ("head_nod", "max")],
    "eyes": [("eye_closure", "max")],
}
# Candidate limits per bound: fine for one bound, coarser when several are searched together.
GRID_POINTS = {1: 400, 2: 150, 3: 40}
MEASUREMENTS = sorted({measurement for rules in CHECK_RULES.values() for measurement, _ in rules})


def fetch(url, filename):
    """Local path of a data file, downloaded on first use."""
    path = os.path.join(HERE, "data", filename)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def download(name):
    return pd.read_parquet(fetch(DATA_URLS[name], name + ".parquet"))


def lfw_photos():
    """One funneled LFW photo per person, with even lighting, for N_PEOPLE people."""
    with tarfile.open(fetch(LFW_URL, "lfw-funneled.tgz")) as tar:
        names = sorted(m.name for m in tar.getmembers() if m.name.endswith("_0001.jpg"))
        for name in np.random.default_rng(0).choice(names, N_PEOPLE, replace=False):
            yield even_light(decode(tar.extractfile(name).read()))


def even_light(image):
    """Make the lighting left-right symmetric. Image = detail x light, where the light is a
    heavy blur; the light becomes the geometric mean of itself and its mirror image."""
    light = cv2.GaussianBlur(image.astype(np.float32) + 1, (0, 0), 20)
    return np.clip(image * np.sqrt(light[:, ::-1] / light), 0, 255).astype(np.uint8)


def decode(image_bytes):
    rgb = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def crop(image, dx=0, dy=0):
    """CROP x CROP window around the image center, shifted by dx, dy."""
    x = (image.shape[1] - CROP) // 2 + dx
    y = (image.shape[0] - CROP) // 2 + dy
    return image[y:y + CROP, x:x + CROP]


def damaged_versions(image, rng):
    """Yield (check, is_bad, image). The undamaged crop (check "*") is good for every LFW check."""
    base = crop(image)
    yield "*", 0, base

    angle = rng.uniform(0, 2 * np.pi)
    for fraction, bad in [(0.05, 0), (0.19, 1)]:
        shift = fraction * CROP
        yield "framing", bad, crop(image, int(shift * np.cos(angle)), int(shift * np.sin(angle)))

    for sigma, bad in [(0.8, 0), (2.0, 1), (3.5, 1)]:
        yield "blur", bad, cv2.GaussianBlur(base, (0, 0), sigma)
    kernel = np.zeros((11, 11), np.float32)
    kernel[5, :] = 1 / 11
    rotation = cv2.getRotationMatrix2D((5.5, 5.5), rng.uniform(0, 180), 1)
    yield "blur", 1, cv2.filter2D(base, -1, cv2.warpAffine(kernel, rotation, (11, 11)))   # motion blur

    for gain, bad in [(0.8, 0), (1.15, 0), (0.35, 1), (2.0, 1)]:
        yield "exposure", bad, np.clip(base * gain, 0, 255).astype(np.uint8)

    x = np.linspace(-1, 1, CROP)
    for darkest, bad in [(0.85, 0), (0.35, 1)]:
        ramp = darkest + (1 - darkest) * np.clip((x / 0.6 + 1) / 2, 0, 1)   # across the face
        if rng.random() < 0.5:
            ramp = ramp[::-1]
        yield "shadow", bad, np.clip(base * ramp[None, :, None], 0, 255).astype(np.uint8)

    sign = rng.choice([-1, 1])
    center = (image.shape[1] / 2, image.shape[0] / 2)
    for degrees, bad in [(3, 0), (12, 1), (20, 1)]:
        rotation = cv2.getRotationMatrix2D(center, sign * degrees, 1)
        rotated = cv2.warpAffine(image, rotation, image.shape[1::-1], borderMode=cv2.BORDER_REFLECT)
        yield "tilt", bad, crop(rotated)


def test_cases():
    rng = np.random.default_rng(0)
    for i, image in enumerate(lfw_photos()):
        split = "calib" if i < N_PEOPLE // 2 else "test"
        for check, bad, damaged in damaged_versions(image, rng):
            yield {"source": "lfw", "split": split, "check": check, "bad": bad, "image": damaged}

    pose = download("pose")
    for i, (image, (tilt, pan)) in enumerate(zip(pose["x"], pose["y"])):
        split = "calib" if i < len(pose) // 2 else "test"
        bad = int(abs(pan) > 15 or abs(tilt) > 15)
        yield {"source": "pose", "split": split, "check": "facing", "bad": bad,
               "image": decode(image["bytes"])}

    for number in EYE_FILES:
        eyes = download(f"eyes_{number}")
        for i, (image, label) in enumerate(zip(eyes["Image_data"], eyes["Label"])):
            split = "calib" if i % 2 == 0 else "test"
            bad = int(label == "closed_eyes")
            yield {"source": "eyes", "split": split, "check": "eyes", "bad": bad,
                   "image": cv2.resize(decode(image["file"]), (CROP, CROP))}


class Baseline:
    """Viola-Jones face and eye detectors with whole-image statistics (the first version)."""

    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    def __call__(self, bgr):
        height, width = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 5, minSize=(width // 8, height // 8))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        face = gray[y:y + h, x:x + w]

        eyes = self.eye_detector.detectMultiScale(face[:h * 2 // 3], 1.1, 4, minSize=(w // 10, w // 10))
        eyes = sorted(eyes, key=lambda box: box[2] * box[3])[-2:]          # the two largest
        tilt = np.nan                                                       # needs two eyes
        if len(eyes) == 2:
            (x1, y1), (x2, y2) = sorted((ex + ew / 2, ey + eh / 2) for ex, ey, ew, eh in eyes)
            tilt = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        left_light = face[:, :w // 2].mean() + 1
        right_light = face[:, w // 2:].mean() + 1
        return {
            "offset_x": (x + w / 2 - width / 2) / width,
            "offset_y": (y + h / 2 - height / 2) / width,
            "head_turn": 0.0,        # the detector only finds frontal faces
            "head_nod": 0.0,
            "head_tilt": tilt,
            "eye_closure": 1 - len(eyes) / 2,
            "brightness": gray.mean(),
            "shadow_ratio": max(left_light, right_light) / min(left_light, right_light),
            "sharpness": cv2.Laplacian(gray, cv2.CV_64F).var(),
        }


def flagged(rows, rules):
    """True where a row breaks at least one rule."""
    ok = np.ones(len(rows), bool)
    for measurement, rule in rules.items():
        ok &= passes(rows[measurement], rule)
    return ~ok


def balanced_accuracy(is_flagged, is_bad):
    """Mean of the catch rate on bad frames and the pass rate on good frames (50% = guessing)."""
    return 0.5 * (is_flagged[is_bad == 1].mean() + 1 - is_flagged[is_bad == 0].mean())


def fit_rules(rows, bounds):
    """Grid-search the limits for `bounds` that give the best balanced accuracy on `rows`."""
    grids = []
    for measurement, limit_type in bounds:
        values = rows[measurement].abs() if limit_type == "max_abs" else rows[measurement]
        candidates = np.nanquantile(values, np.linspace(0, 1, GRID_POINTS[len(bounds)]))
        grids.append(np.unique(candidates))

    best_accuracy, best_rules = -1, None
    for limits in itertools.product(*grids):
        rules = {}
        for (measurement, limit_type), limit in zip(bounds, limits):
            rules.setdefault(measurement, {})[limit_type] = float(limit)
        accuracy = balanced_accuracy(flagged(rows, rules), rows["bad"].values)
        if accuracy > best_accuracy:
            best_accuracy, best_rules = accuracy, rules
    return best_rules


def main():
    analyzers = {"baseline": Baseline(), "v1": Checker()}
    records = []
    for case in test_cases():
        image = case.pop("image")
        for name, analyze in analyzers.items():
            measurements = analyze(image) or {}            # no face -> all NaN -> every check fails
            values = {key: measurements.get(key, np.nan) for key in MEASUREMENTS}
            records.append({**case, "analyzer": name, **values})
    data = pd.DataFrame(records)

    accuracy = {}
    v1_rules = {"distance_cm": {"min": 50}, "face_height": {"min": 0.20}}   # set by hand
    for name, rows in data.groupby("analyzer"):
        for check, bounds in CHECK_RULES.items():
            if check == "facing":
                selected = rows[rows["check"] == check]
            else:   # undamaged LFW photos are good examples for every other check, eyes included
                selected = rows[rows["check"].isin([check, "*"])]
            calib = selected[selected["split"] == "calib"]
            test = selected[selected["split"] == "test"]

            rules = fit_rules(calib, bounds)
            accuracy[(check, name)] = balanced_accuracy(flagged(test, rules), test["bad"].values)
            if name == "v1":
                v1_rules.update(rules)

    with open(os.path.join(HERE, "config.json"), "w") as f:
        json.dump({"rules": v1_rules}, f, indent=2)

    table = pd.Series(accuracy).unstack().reindex(list(CHECK_RULES)) * 100
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    table.to_csv(os.path.join(HERE, "results", "metrics.csv"))
    print("Balanced accuracy (%) on the test split\n", table.round(1).to_string())


if __name__ == "__main__":
    main()
