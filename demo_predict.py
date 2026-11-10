"""Run the exported models on a single image (e.g. a GeoGuessr screenshot).

    python demo_predict.py screenshot.png

Needs demo_models/resnet_country.pt and/or demo_models/clip_cell.pt (made by
export_demo.py) plus torch, torchvision, transformers, pillow and numpy.
The first run downloads the pretrained ResNet-101 and CLIP backbones.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from driving_side import MultiTaskHead, fuse

MODEL_DIR = Path(__file__).parent / "demo_models"


def make_transform(t):
    steps = [T.Resize(t["resize"] if isinstance(t["resize"], int) else tuple(t["resize"]))]
    if t.get("center_crop"):
        steps.append(T.CenterCrop(t["center_crop"]))
    return T.Compose(steps + [T.ToTensor(), T.Normalize(t["mean"], t["std"])])


class ResNetCountry:
    """ResNet-101 features -> linear head over countries."""

    def __init__(self, path=MODEL_DIR / "resnet_country.pt", device="cpu"):
        import torchvision.models as models
        b = torch.load(path, map_location="cpu", weights_only=False)
        self.classes = b["classes"]
        self.transform = make_transform(b["transform"])
        self.backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(2048, len(self.classes))
        self.head.load_state_dict({k.removeprefix("fc."): v for k, v in b["head"].items()})
        self.device = device
        self.backbone.to(device).eval()
        self.head.to(device).eval()

    @torch.no_grad()
    def predict(self, image, k=5):
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = self.head(self.backbone(x)).softmax(1)[0].cpu()
        p, i = probs.topk(min(k, len(self.classes)))
        return [{"country": self.classes[j], "prob": float(q)} for q, j in zip(p, i)]


class ClipCell:
    """CLIP features -> geographic cell, adjusted by the mirror-based driving-side detector."""

    def __init__(self, path=MODEL_DIR / "clip_cell.pt", device="cpu"):
        from transformers import CLIPModel
        b = torch.load(path, map_location="cpu", weights_only=False)
        if b.get("head_kind", "mirror") != "mirror":
            raise ValueError(f"{path} has a '{b['head_kind']}' head; this demo runs only mirror heads. "
                             "Load it on the GitHub Pages site instead (export_web.py --bundle).")
        self.b = b
        self.transform = make_transform(b["transform"])
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision, self.proj = clip.vision_model, clip.visual_projection
        self.head = MultiTaskHead(**b["head_kwargs"])
        self.head.load_state_dict(b["head"])
        self.cell_left = np.asarray(b["cell_left"], dtype=np.float32)
        self.device = device
        for m in (self.vision, self.proj, self.head):
            m.to(device).eval()

    def _features(self, x):
        return self.proj(self.vision(pixel_values=x).pooler_output)

    @torch.no_grad()
    def predict(self, image, k=5, use_driving_side=True):
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        cell_logits, side_logit = self.head(self._features(x), self._features(torch.flip(x, dims=[3])))
        logp = cell_logits.log_softmax(1).cpu().numpy()
        p_left = torch.sigmoid(side_logit).cpu().numpy()
        w = self.b["fusion_weight"] if use_driving_side else 0.0
        scores = fuse(logp, p_left, self.cell_left, w)[0]
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        top = np.argsort(-probs)[:k]
        b = self.b
        return {
            "p_left_driving": float(p_left[0]),
            "guesses": [{"cell": b["cells"][j], "country": b["cell_country"][j],
                         "lat": b["cell_lat"][j], "lon": b["cell_lon"][j],
                         "prob": float(probs[j])} for j in top],
        }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    img = Image.open(sys.argv[1])
    if (MODEL_DIR / "resnet_country.pt").exists():
        print("ResNet (country):", ResNetCountry().predict(img))
    if (MODEL_DIR / "clip_cell.pt").exists():
        print("CLIP (cell):", ClipCell().predict(img))
