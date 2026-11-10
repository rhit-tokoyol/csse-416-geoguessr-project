"""Convert an exported CLIP model (made by export_demo.py) into files the GitHub Pages demo loads.

    python export_web.py                                   # built-in model: docs/model/head.bin + meta.json
    python export_web.py demo_models/clip_cell.pt --bundle run3.geomodel --name "Run 3: more dropout"
                                                           # one file to load with "Load a model" on the site

The weights are little-endian float32 arrays, one after another; the metadata says where each one
starts and holds the cell list, cell centres, driving side per cell, fusion weight and how photos
were resized. A .geomodel bundle is the same two parts in one file:
    b"GEOMODEL1\\n" + uint32 (little-endian) metadata length + metadata JSON + padding to 4 bytes + weights
The CLIP backbone is never included: the page loads it from the Hugging Face CDN.

Supported heads (the page works out which from the weights it finds):
    mirror  trunk.0 + cell_out + side_net.0/side_net.3  (current notebook: mirror-based driving side)
    shared  trunk.0 + cell_out + side_out               (run 1: side output on the shared layer)
    plain   trunk.0 + cell_out                          (no driving side; also MLPHead's net.0/net.3)
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
MAGIC = b"GEOMODEL1\n"

ORDER = ["trunk.0.weight", "trunk.0.bias", "cell_out.weight", "cell_out.bias",
         "side_net.0.weight", "side_net.0.bias", "side_net.3.weight", "side_net.3.bias",
         "side_out.weight", "side_out.bias"]
RENAME = {"net.0.weight": "trunk.0.weight", "net.0.bias": "trunk.0.bias",      # plain MLPHead
          "net.3.weight": "cell_out.weight", "net.3.bias": "cell_out.bias"}


def head_kind(state):
    if "side_net.0.weight" in state:
        return "mirror"
    if "side_out.weight" in state:
        return "shared"
    return "plain"


def build(src, name):
    b = torch.load(src, map_location="cpu", weights_only=False)
    state = {RENAME.get(k, k): v for k, v in b["head"].items()}
    if "trunk.0.weight" not in state or "cell_out.weight" not in state:
        sys.exit(f"{src}: the head has no trunk/cell_out layers; found {sorted(state)}")
    kind = head_kind(state)
    index, chunks, offset = [], [], 0
    for key in ORDER:
        if key in state:
            arr = state[key].detach().cpu().float().numpy().astype("<f4")
            chunks.append(arr.tobytes())
            index.append({"name": key, "shape": list(arr.shape), "offset": offset})
            offset += arr.size
    t = b["transform"]
    size = b.get("image_size", 224)
    meta = {
        "name": name,
        "head_kind": kind,
        "clip_model": b.get("web_model", "Xenova/clip-vit-base-patch32"),   # ONNX build the page loads
        "clip_dim": state["trunk.0.weight"].shape[1],                       # features the head expects
        "resize": t["resize"],              # [size, size] = squash whole photo; size = shorter side, then...
        "center_crop": t.get("center_crop"),  # ...a size x size center crop (run 1)
        "image_size": [size, size],
        "mean": t["mean"], "std": t["std"],
        "fusion_weight": b["fusion_weight"] if kind != "plain" else 0.0,
        "tensors": index,
        "cells": b["cells"],
        "cell_lat": [round(v, 5) for v in b["cell_lat"]],
        "cell_lon": [round(v, 5) for v in b["cell_lon"]],
        "cell_country": b["cell_country"],
        "cell_left": [round(v, 4) for v in b["cell_left"]],
    }
    return meta, b"".join(chunks)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", nargs="?", type=Path, default=ROOT / "demo_models" / "clip_cell.pt")
    p.add_argument("--bundle", type=Path, help="write one .geomodel file instead of docs/model/")
    p.add_argument("--name", help="name the site shows for this model (default: the file name)")
    p.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "model")
    a = p.parse_args()
    if not a.src.exists():
        sys.exit(f"{a.src} not found. Run export_demo.py on the cluster and download it first.")
    name = a.name or (a.bundle.stem if a.bundle else "Built-in model")
    meta, weights = build(a.src, name)
    summary = f"{len(meta['cells'])} cells, {meta['head_kind']} head, fusion weight {meta['fusion_weight']}"
    if a.bundle:
        header = json.dumps(meta, separators=(",", ":")).encode()
        pad = (-(len(MAGIC) + 4 + len(header))) % 4
        a.bundle.write_bytes(MAGIC + struct.pack("<I", len(header) + pad) + header + b" " * pad + weights)
        print(f"Wrote {a.bundle} ({a.bundle.stat().st_size / 1e6:.1f} MB): {summary}")
    else:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        (a.out_dir / "head.bin").write_bytes(weights)
        (a.out_dir / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
        print(f"Wrote {a.out_dir}/head.bin ({len(weights) / 1e6:.1f} MB) and meta.json: {summary}")


if __name__ == "__main__":
    main()
