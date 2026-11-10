"""GeoGuessr model demo: upload a screenshot, see where each model thinks it is.

    pip install -r demo_app/requirements.txt
    python demo_app/app.py            # then open http://localhost:5000

Expects the files made by export_demo.py in demo_models/ (next to
demo_predict.py), or in the folder named by the DEMO_MODELS environment variable.
Either model may be missing; the page shows whichever ones are there.
"""

import io
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import demo_predict  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from country_centers import CENTERS  # noqa: E402

MODEL_DIR = Path(os.environ.get("DEMO_MODELS", ROOT / "demo_models"))
DEVICE = "cuda" if os.environ.get("DEMO_GPU") else "cpu"


def load(cls, *filenames):
    """Load the first of `filenames` that exists in MODEL_DIR; None if none do."""
    for filename in filenames:
        path = MODEL_DIR / filename
        if path.exists():
            print(f"[demo] loading {path} ...")
            return cls(path, device=DEVICE)
    print(f"[demo] none of {', '.join(filenames)} found in {MODEL_DIR}; skipping {cls.__name__}")
    return None


clip_model = load(demo_predict.ClipCell, "clip_cell.pt")
# train_model_sample_1.pt is the raw head saved by resnet_model.ipynb
resnet_model = load(demo_predict.ResNetCountry, "resnet_country.pt", "train_model_sample_1.pt")
if not (clip_model or resnet_model):
    sys.exit(f"No models in {MODEL_DIR}. Run export_demo.py and download demo_models/ first.")

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB upload limit


@app.get("/")
def index():
    return send_from_directory(Path(__file__).parent, "index.html")


@app.get("/api/models")
def models():
    return jsonify({
        "clip": None if clip_model is None else {"fusion_weight": clip_model.b["fusion_weight"],
                                                 "cells": len(clip_model.b["cells"])},
        "resnet": None if resnet_model is None else {"countries": len(resnet_model.classes)},
    })


def crop_box(image):
    """Optional crop sent by the page as fractions of width/height (x0, y0, x1, y1)."""
    try:
        x0, y0, x1, y1 = (float(request.form[k]) for k in ("x0", "y0", "x1", "y1"))
    except (KeyError, ValueError):
        return image
    x0, x1 = sorted((min(max(x0, 0), 1), min(max(x1, 0), 1)))
    y0, y1 = sorted((min(max(y0, 0), 1), min(max(y1, 0), 1)))
    w, h = image.size
    box = (round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h))
    if box[2] - box[0] < 32 or box[3] - box[1] < 32:
        return image  # too small to be a real selection
    return image.crop(box)


@app.post("/api/predict")
def predict():
    file = request.files.get("image")
    if file is None:
        return jsonify(error="No image was uploaded."), 400
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except UnidentifiedImageError:
        return jsonify(error="That file isn't an image this demo can read. Try a PNG or JPEG."), 400
    image = crop_box(image)
    choice = request.form.get("model", "clip")
    out = {"size": list(image.size), "model": choice}
    if choice == "clip":
        if clip_model is None:
            return jsonify(error="The CLIP model isn't loaded on the server."), 400
        expected = request.form.get("expected", "1") != "0"   # rank by expected points unless the page sends expected=0
        out["clip"] = clip_model.predict(image, k=5, expected=expected)
        plain = clip_model.predict(image, k=1, use_driving_side=False, expected=expected)["guesses"][0]
        out["clip"]["plain_top"] = plain
    elif choice == "resnet":
        if resnet_model is None:
            return jsonify(error="The ResNet model isn't loaded on the server."), 400
        guesses = []
        for g in resnet_model.predict(image, k=3):
            lat, lon = CENTERS.get(g["country"], (None, None))
            guesses.append({**g, "lat": lat, "lon": lon})
        out["resnet"] = guesses
    else:
        return jsonify(error=f"Unknown model '{choice}'."), 400
    return jsonify(out)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="That image is over 20 MB. Try a smaller screenshot."), 413


if __name__ == "__main__":
    # Locally this only listens on your own machine; the Space's Dockerfile sets HOST=0.0.0.0.
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=False)
