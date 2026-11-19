# SnapReady

A live webcam coach for passport photos. It measures the face in every frame, shows the one
most important fix ("Turn your head slightly left"), takes the photo automatically once every
rule passes, and outputs a 2 x 2 inch photo with a white background.

| File | Role |
|---|---|
| `check.py` | Module 1: MediaPipe face landmarks -> measurements (position, head angles, eyes, light, sharpness, distance) |
| `guide.py` | Module 2: rules -> pass/fail -> one instruction -> automatic capture |
| `finish.py` | Module 3: level the eyes, crop to the US spec, Laplacian-blend onto white |
| `app.py` | Browser demo: live webcam coaching, or check an uploaded photo |
| `benchmark.py` | Baseline vs v1 on public data; fits the rule limits in `config.json` |
| `config.json` | The rules, e.g. `"head_turn": {"max_abs": 19}` = head turned at most 19 degrees |

```bash
pip install -r requirements.txt
python app.py            # open the printed link and allow the camera
python benchmark.py      # optional: re-fit config.json (downloads ~900 MB of data)
```

Models (MediaPipe Face Landmarker, Selfie Segmenter) download on first run.
Data: LFW (funneled, i.e. aligned), Pointing'04 head pose (Hugging Face mirror), open/closed eyes (AI-generated faces).
