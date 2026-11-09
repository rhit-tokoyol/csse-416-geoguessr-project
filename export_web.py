"""Convert demo_models/clip_cell.pt into files the GitHub Pages demo loads in the browser.

    python export_web.py            # writes docs/model/head.bin and docs/model/meta.json

head.bin holds the trained head's weights as little-endian float32 arrays, one
after another; meta.json says where each one starts and holds the cell list,
cell centres, driving side per cell and the fusion weight. The CLIP backbone is
not included: the page loads it from the Hugging Face CDN.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "demo_models" / "clip_cell.pt"
OUT = ROOT / "docs" / "model"

# The browser runs the same layers as driving_side.MultiTaskHead (eval mode, so no dropout).
TENSORS = ["trunk.0.weight", "trunk.0.bias", "cell_out.weight", "cell_out.bias",
           "side_net.0.weight", "side_net.0.bias", "side_net.3.weight", "side_net.3.bias"]


def main():
    if not SRC.exists():
        sys.exit(f"{SRC} not found. Run export_demo.py on the cluster and download demo_models/ first.")
    b = torch.load(SRC, map_location="cpu", weights_only=False)
    state = b["head"]
    missing = [k for k in TENSORS if k not in state]
    if missing:
        sys.exit(f"Head is missing {missing}; expected a mirror-based driving-side head.")

    OUT.mkdir(parents=True, exist_ok=True)
    index, offset = [], 0
    with open(OUT / "head.bin", "wb") as f:
        for name in TENSORS:
            arr = state[name].detach().cpu().numpy().astype("<f4")
            f.write(arr.tobytes())
            index.append({"name": name, "shape": list(arr.shape), "offset": offset})
            offset += arr.size
    t = b["transform"]
    meta = {
        "clip_model": "Xenova/clip-vit-base-patch32",
        "image_size": t["resize"],          # [224, 224]: whole image squashed, no crop (as in training)
        "mean": t["mean"], "std": t["std"],
        "fusion_weight": b["fusion_weight"],
        "tensors": index,
        "cells": b["cells"],
        "cell_lat": [round(v, 5) for v in b["cell_lat"]],
        "cell_lon": [round(v, 5) for v in b["cell_lon"]],
        "cell_country": b["cell_country"],
        "cell_left": [round(v, 4) for v in b["cell_left"]],
    }
    (OUT / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    mb = (OUT / "head.bin").stat().st_size / 1e6
    print(f"Wrote {OUT / 'head.bin'} ({mb:.1f} MB) and meta.json: {len(b['cells'])} cells, "
          f"fusion weight {b['fusion_weight']}")


if __name__ == "__main__":
    main()
