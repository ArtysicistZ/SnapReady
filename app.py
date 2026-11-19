"""SnapReady demo in the browser: live webcam coaching, or check an uploaded photo.

    python app.py      # open the printed link and allow the camera
"""

import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from check import FACE_OVAL, Checker
from finish import Finisher
from guide import Coach, evaluate

HFOV = 70   # assumed horizontal field of view of a laptop webcam, degrees
RULES = json.loads(Path(__file__).with_name("config.json").read_text())["rules"]
GREEN, ORANGE = (60, 200, 60), (0, 140, 255)

checker = Checker()
finisher = Finisher()


def measure(rgb, mirrored):
    """Browser RGB image -> (BGR image the right way round, measurements)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if mirrored:
        bgr = cv2.flip(bgr, 1)
    focal_px = bgr.shape[1] / 2 / np.tan(np.radians(HFOV / 2))   # pinhole model
    return bgr, checker(bgr, focal_px)


def draw(bgr, measurements, banner, progress, mirrored):
    """Face outline, instruction banner and hold-still progress bar, as RGB."""
    view = bgr.copy()
    if measurements is not None:
        outline = measurements["points"][FACE_OVAL].astype(np.int32)
        cv2.polylines(view, [outline], True, GREEN if progress else ORANGE, 2, cv2.LINE_AA)
    if mirrored:
        view = cv2.flip(view, 1)
    width = view.shape[1]
    cv2.rectangle(view, (0, 0), (width, 44), (40, 40, 40), -1)
    cv2.rectangle(view, (0, 38), (int(width * progress), 44), GREEN, -1)
    cv2.putText(view, banner, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(view, cv2.COLOR_BGR2RGB)


def checklist(measurements, checks):
    """Markdown table: every rule with its current value and allowed range."""
    if measurements is None:
        return "No face found"
    rows = ["| | Check | Value | Allowed |", "|---|---|---|---|"]
    for name, passed, _ in checks:
        limits = ", ".join(f"{kind} {limit:.3g}" for kind, limit in RULES[name].items())
        rows.append(f"| {'✅' if passed else '❌'} | {name} | {measurements[name]:.3g} | {limits} |")
    return "\n".join(rows)


def live(rgb, coach, photo):
    """One webcam frame: coach the user, and take the photo once every rule holds."""
    if rgb is None:
        return None, "", photo, coach
    bgr, measurements = measure(rgb, mirrored=True)
    checks, todo, capture = coach.update(bgr, measurements)
    if capture is not None and photo is None:
        frame, frame_measurements = capture
        photo = cv2.cvtColor(finisher(frame, frame_measurements["points"]), cv2.COLOR_BGR2RGB)
    if photo is not None:
        banner, progress = "Captured! Press Retake for another", 1.0
    elif todo is not None:
        banner, progress = todo, 0.0
    else:
        banner, progress = "Hold still...", coach.streak / coach.HOLD
    view = draw(bgr, measurements, banner, progress, mirrored=True)
    return view, checklist(measurements, checks), photo, coach


def single(rgb):
    """An uploaded photo: check every rule once and finish it."""
    if rgb is None:
        return None, "", None
    bgr, measurements = measure(rgb, mirrored=False)
    checks = evaluate(measurements, RULES)
    failing = [text for _, passed, text in checks if not passed]
    view = draw(bgr, measurements, failing[0] if failing else "All checks pass", 0.0 if failing else 1.0,
                mirrored=False)
    photo = None
    if measurements is not None:
        photo = cv2.cvtColor(finisher(bgr, measurements["points"]), cv2.COLOR_BGR2RGB)
    return view, checklist(measurements, checks), photo


with gr.Blocks(title="SnapReady") as demo:
    gr.Markdown("# SnapReady\nA webcam coach for US passport photos (2 x 2 in, white background).")
    with gr.Tab("Live"):
        coach = gr.State(Coach(RULES))
        with gr.Row():
            webcam = gr.Image(sources="webcam", streaming=True, label="Camera")
            view = gr.Image(label="Coach")
        with gr.Row():
            table = gr.Markdown()
            with gr.Column():
                photo = gr.Image(label="Passport photo (600 x 600)", format="jpeg")
                retake = gr.Button("Retake")
        webcam.stream(live, [webcam, coach, photo], [view, table, photo, coach], stream_every=0.1)
        retake.click(lambda: (Coach(RULES), None), None, [coach, photo])
    with gr.Tab("Photo"):
        with gr.Row():
            upload = gr.Image(sources="upload", label="Your photo")
            upload_view = gr.Image(label="Checks")
        with gr.Row():
            upload_table = gr.Markdown()
            upload_photo = gr.Image(label="Passport photo (600 x 600)", format="jpeg")
        upload.change(single, upload, [upload_view, upload_table, upload_photo])

if __name__ == "__main__":
    demo.launch()
