"""Package the trained models into small files for the demo site.

Both models use a frozen, unmodified pretrained backbone (torchvision ResNet-101
and OpenAI CLIP ViT-B/32), so only the trained head is exported; the demo
downloads the backbone itself. The training runs saved only the heads' weights,
so this script rebuilds the rest (class lists, cell centres, driving side per
cell) from the dataset CSVs, exactly as the notebooks built them.

Run on the cluster, from the project folder:

    python export_demo.py \
        --resnet-head output/model_sampler_1/train_model_sampler_1.pt \
        --clip-head   output/<your run_name>/train_<your run_name>.pt

It writes demo_models/resnet_country.pt and demo_models/clip_cell.pt.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from driving_side import side_label

DATA = Path("/work/cssema416/202710/11/datasets/OpenWorld")

RESNET_TRANSFORM = {"resize": 256, "center_crop": 224,
                    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
DEFAULT_CLIP = "openai/clip-vit-base-patch32"
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]     # shared by OpenAI's CLIP variants
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def clip_transform(size, center_crop):
    """How photos were resized for training: squash the whole photo to size x size, or
    shorter side to `size` then a size x size center crop."""
    if center_crop:
        return {"resize": size, "center_crop": size, "mean": CLIP_MEAN, "std": CLIP_STD}
    return {"resize": [size, size], "center_crop": None, "mean": CLIP_MEAN, "std": CLIP_STD}


def out_dim(state, key):
    if key not in state:
        raise SystemExit(f"Expected '{key}' in the head's weights; found {list(state)}")
    return state[key].shape[0]


def export_resnet(head_path, csv_path, out_path):
    state = torch.load(head_path, map_location="cpu")
    n = out_dim(state, "fc.weight")
    # The ResNet notebook (Optimized branch) trained on test.csv with
    # LabelEncoder().fit(df["unique_country"]), i.e. the sorted unique codes.
    df = pd.read_csv(csv_path, usecols=["unique_country"])
    classes = sorted(df["unique_country"].dropna().astype(str).unique())
    if len(classes) != n:
        raise SystemExit(f"ResNet head has {n} outputs but {csv_path} gives {len(classes)} "
                         "countries. Pass the CSV the model was trained on with --resnet-csv.")
    torch.save({
        "kind": "resnet_country",
        "backbone": "torchvision resnet101, ResNet101_Weights.DEFAULT, fc replaced by Identity",
        "head": state,                      # nn.Linear(2048, n) stored as fc.*
        "classes": classes,                 # ISO country codes, index = model output
        "transform": RESNET_TRANSFORM,
    }, out_path)
    print(f"ResNet: {n} countries -> {out_path}")


PLAIN_RENAME = {"net.0.weight": "trunk.0.weight", "net.0.bias": "trunk.0.bias",
                "net.3.weight": "cell_out.weight", "net.3.bias": "cell_out.bias"}


def export_clip(head_path, csv_path, out_path, fusion_weight, center_crop=False,
                clip_model=DEFAULT_CLIP, web_model=None, image_size=224):
    # Heads: "mirror" (current driving_side.MultiTaskHead), "shared" (run 1: side_out on the
    # shared layer) or "plain" (no driving side; the older MLPHead's net.0/net.3 are renamed).
    # The backbone is any CLIP vision model: the head's input size (feat_dim) is read from its weights.
    state = {PLAIN_RENAME.get(k, k): v for k, v in torch.load(head_path, map_location="cpu").items()}
    if "fc.0.weight" in state and "fc.10.weight" in state:
        # "residual": the BatchNorm + residual-block MLP (Head in the notebook), no driving side.
        kind, n, feat_dim = "residual", state["fc.10.weight"].shape[0], state["fc.0.weight"].shape[1]
    else:
        n = out_dim(state, "cell_out.weight")
        kind = "mirror" if "side_net.0.weight" in state else "shared" if "side_out.weight" in state else "plain"
        feat_dim = state["trunk.0.weight"].shape[1]
        if kind == "shared":
            print(f"Note: {head_path} is a 'shared' head. The GitHub Pages site can run it; "
                  "the Flask demo (demo_predict.py) runs mirror, plain and residual heads.")
    # Name of the same model for the browser (transformers.js needs an ONNX build). The default is a
    # guess that follows the Xenova naming; override it with --web-model if the repo is named differently.
    web_model = web_model or "Xenova/" + clip_model.split("/")[-1]
    print(f"Backbone {clip_model}: head expects {feat_dim}-d features, {image_size}px images. "
          f"Browser model: {web_model}")
    # main.ipynb: LabelEncoder().fit(df["cell"]) over every row of train.csv.
    df = pd.read_csv(csv_path, usecols=["latitude", "longitude", "cell", "country"])
    df["cell"] = df["cell"].astype(str)
    classes = sorted(df["cell"].unique())
    if len(classes) != n:
        raise SystemExit(f"CLIP head has {n} outputs but {csv_path} gives {len(classes)} cells.")
    df["side"] = df["country"].map(side_label)
    g = df.groupby("cell")
    stats = pd.DataFrame({
        "lat": g["latitude"].mean(),
        "lon": g["longitude"].mean(),
        "country": g["country"].agg(lambda s: s.mode().iat[0] if s.notna().any() else ""),
        "left": g["side"].mean(),
    }).reindex(classes)
    head_kwargs = {"n_cells": n, "in_dim": feat_dim}
    if kind != "residual":
        head_kwargs["hidden"] = state["trunk.0.weight"].shape[0]
    if kind == "mirror":
        head_kwargs["side_hidden"] = state["side_net.0.weight"].shape[0]
    torch.save({
        "kind": "clip_cell",
        "backbone": f"{clip_model} (vision_model + visual_projection)",
        "clip_model": clip_model,           # Hugging Face name the Python demo loads
        "web_model": web_model,             # ONNX build the browser loads (export_web.py)
        "feat_dim": feat_dim,
        "image_size": image_size,
        "head": state,                      # driving_side.MultiTaskHead when head_kind is "mirror"
        "head_kind": kind,
        "head_kwargs": head_kwargs if kind in ("mirror", "plain", "residual") else None,
        "cells": classes,
        "cell_lat": stats["lat"].tolist(),
        "cell_lon": stats["lon"].tolist(),
        "cell_country": stats["country"].tolist(),
        "cell_left": stats["left"].fillna(0.5).tolist(),
        "fusion_weight": fusion_weight,
        "transform": clip_transform(image_size, center_crop),
    }, out_path)
    print(f"CLIP: {n} cells, {kind} head, fusion weight {fusion_weight} -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resnet-head", type=Path)
    p.add_argument("--resnet-csv", type=Path, default=DATA / "test.csv")
    p.add_argument("--clip-head", type=Path)
    p.add_argument("--clip-csv", type=Path, default=DATA / "train.csv")
    p.add_argument("--fusion-weight", type=float, default=1.0,
                   help="'Chosen fusion weight' printed by cell 14 of the CLIP run")
    p.add_argument("--center-crop", action="store_true",
                   help="the CLIP run used Resize(size) + CenterCrop(size) (run 1), not the whole photo")
    p.add_argument("--clip-model", default=DEFAULT_CLIP,
                   help="Hugging Face name of the CLIP backbone the head was trained on, "
                        "e.g. openai/clip-vit-large-patch14")
    p.add_argument("--web-model", default=None,
                   help="the same backbone as an ONNX build for the browser (default: Xenova/<model name>)")
    p.add_argument("--image-size", type=int, default=224,
                   help="input size the backbone was trained with (224 for most CLIPs, 336 for the -336 variants)")
    p.add_argument("--clip-out", default="clip_cell.pt",
                   help="file name for the CLIP export, e.g. clip_run3.pt to keep the current one")
    p.add_argument("--out", type=Path, default=Path("demo_models"))
    a = p.parse_args()
    if not (a.resnet_head or a.clip_head):
        p.error("pass --resnet-head and/or --clip-head")
    a.out.mkdir(exist_ok=True)
    if a.resnet_head:
        export_resnet(a.resnet_head, a.resnet_csv, a.out / "resnet_country.pt")
    if a.clip_head:
        export_clip(a.clip_head, a.clip_csv, a.out / a.clip_out, a.fusion_weight, a.center_crop,
                    a.clip_model, a.web_model, a.image_size)
