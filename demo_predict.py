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


# Country order the notebook's LabelEncoder produced (sorted unique_country, 222 codes).
# Raw checkpoints from resnet_model.ipynb are only a state_dict, so the labels live here.
RESNET_CLASSES = [
    "AD", "AE", "AF", "AG", "AI", "AL", "AM", "AO", "AR", "AT", "AU", "AW", "AX", "AZ", "BA",
    "BB", "BD", "BE", "BF", "BG", "BH", "BI", "BJ", "BM", "BN", "BO", "BQ", "BR", "BS", "BT",
    "BW", "BY", "BZ", "CA", "CD", "CF", "CG", "CH", "CI", "CL", "CM", "CN", "CO", "CR", "CU",
    "CV", "CW", "CX", "CY", "CZ", "DE", "DJ", "DK", "DM", "DO", "DZ", "EC", "EE", "EG", "ER",
    "ES", "ET", "FI", "FJ", "FK", "FM", "FO", "FR", "GA", "GB", "GD", "GE", "GF", "GG", "GH",
    "GI", "GL", "GM", "GN", "GP", "GQ", "GR", "GT", "GW", "GY", "HK", "HN", "HR", "HT", "HU",
    "ID", "IE", "IL", "IM", "IN", "IQ", "IR", "IS", "IT", "JE", "JM", "JO", "JP", "KE", "KG",
    "KH", "KI", "KM", "KN", "KP", "KR", "KW", "KZ", "LA", "LB", "LC", "LI", "LK", "LR", "LS",
    "LT", "LU", "LV", "LY", "MA", "MC", "MD", "ME", "MF", "MG", "MK", "ML", "MM", "MN", "MO",
    "MQ", "MR", "MS", "MT", "MU", "MV", "MW", "MX", "MY", "MZ", "NC", "NE", "NG", "NI", "NL",
    "NO", "NP", "NZ", "OM", "PA", "PE", "PF", "PG", "PH", "PK", "PL", "PR", "PS", "PT", "PW",
    "PY", "QA", "RE", "RO", "RS", "RU", "RW", "SA", "SB", "SD", "SE", "SG", "SI", "SJ", "SK",
    "SL", "SM", "SN", "SO", "SR", "SS", "ST", "SV", "SX", "SY", "SZ", "TC", "TD", "TG", "TH",
    "TJ", "TL", "TM", "TN", "TR", "TT", "TW", "TZ", "UA", "UG", "US", "UY", "UZ", "VA", "VC",
    "VE", "VG", "VI", "VN", "VU", "WS", "XK", "YE", "YT", "ZA", "ZM", "ZW"
]
RESNET_TRANSFORM = {"resize": 256, "center_crop": 224,
                    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


class ResidualBlock(nn.Module):
    def __init__(self, features, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(features, features), nn.BatchNorm1d(features), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(features, features), nn.BatchNorm1d(features),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class ResNetHead(nn.Module):
    """The residual MLP head trained in resnet_model.ipynb (layer names must match its state_dict)."""

    def __init__(self, n, in_features=2048):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.3),
            ResidualBlock(1024, dropout=0.3),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            ResidualBlock(512, dropout=0.2),
            nn.Linear(512, n),
        )

    def forward(self, x):
        return self.fc(x)


class ResNetCountry:
    """ResNet-101 features -> head over countries.

    Accepts either the raw state_dict saved by resnet_model.ipynb (train_model_sample_1.pt, an MLP
    head) or a bundle from export_demo.py ({"classes", "transform", "head"}, a linear head).
    """

    def __init__(self, path=MODEL_DIR / "resnet_country.pt", device="cpu"):
        import torchvision.models as models
        b = torch.load(path, map_location="cpu", weights_only=False)
        bundle = isinstance(b, dict) and "head" in b
        state = b["head"] if bundle else b
        self.classes = b["classes"] if bundle else RESNET_CLASSES
        self.transform = make_transform(b["transform"] if bundle else RESNET_TRANSFORM)
        if "fc.10.weight" in state:  # notebook MLP head
            n_out = state["fc.10.weight"].shape[0]
            self.head = ResNetHead(n_out)
        else:  # single linear layer
            state = {k.removeprefix("fc."): v for k, v in state.items()}
            n_out = state["weight"].shape[0]
            self.head = nn.Linear(2048, n_out)
        if n_out != len(self.classes):
            raise ValueError(f"{path} predicts {n_out} classes but {len(self.classes)} country labels are known.")
        self.head.load_state_dict(state)
        self.backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.device = device
        self.backbone.to(device).eval()
        self.head.to(device).eval()  # eval() matters: the MLP has BatchNorm and Dropout

    @torch.no_grad()
    def predict(self, image, k=3):
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
