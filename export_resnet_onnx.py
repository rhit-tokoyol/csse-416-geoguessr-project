"""Export the ResNet country model as an ONNX file the GitHub Pages site can run in the browser.

    python export_resnet_onnx.py demo_models/resnet.pt
    python export_resnet_onnx.py demo_models/resnet.pt --fp16 --url https://huggingface.co/USER/REPO/resolve/main/resnet_country.onnx

The input is the head saved by resnet_model.ipynb (a raw state_dict), or a bundle made by
export_demo.py. The ONNX file holds the whole model: torchvision ResNet-101 (ImageNet weights) with
the head on top, so the page only has to resize the photo and read back 222 country scores.

It writes two files into --out-dir (default docs/model):
    resnet_country.onnx   the model (about 170 MB in fp32, about 85 MB with --fp16)
    resnet_meta.json      country codes, map pins, photo resize/normalisation, and where the page
                          should download the .onnx from ("url")

GitHub rejects files over 100 MB, so the fp32 file can't be committed. Either use --fp16 (and commit
it), or upload the .onnx to a Hugging Face model repo and pass its ".../resolve/main/<file>" link
as --url (the page can read it from there). If you only change --url later, edit it in
resnet_meta.json; the model needn't be exported again.

Needs torch and torchvision (plus onnx; onnxruntime to check the export; onnxconverter-common for --fp16).
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo_app"))
from country_centers import CENTERS          # noqa: E402
from demo_predict import RESNET_TRANSFORM, ResNetCountry  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("head", type=Path, help="resnet.pt (raw head from resnet_model.ipynb or an export_demo.py bundle)")
    p.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "model")
    p.add_argument("--url", default="model/resnet_country.onnx",
                   help="where the page downloads the model from: a path relative to docs/ or a full https link")
    p.add_argument("--fp16", action="store_true", help="store the weights as float16 (half the size)")
    p.add_argument("--name", default="ResNet-101 + MLP head (countries)")
    a = p.parse_args()

    rc = ResNetCountry(a.head)                       # builds backbone + head exactly as the demo does
    model = nn.Sequential(rc.backbone, rc.head).eval()
    raw = torch.load(a.head, map_location="cpu", weights_only=False)
    t = raw["transform"] if isinstance(raw, dict) and "head" in raw else RESNET_TRANSFORM
    if not isinstance(t["resize"], int) or not t.get("center_crop"):
        sys.exit("The page only supports 'shorter side to N, then a center crop' resizing for ResNet.")
    size = t["center_crop"]

    a.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = a.out_dir / "resnet_country.onnx"
    dummy = torch.zeros(1, 3, size, size)
    kw = dict(input_names=["pixels"], output_names=["logits"], opset_version=17)
    try:
        torch.onnx.export(model, dummy, str(onnx_path), dynamo=False, **kw)   # newer torch: use the classic exporter
    except TypeError:
        torch.onnx.export(model, dummy, str(onnx_path), **kw)

    if a.fp16:
        try:
            import onnx
            from onnxconverter_common import float16
        except ImportError:
            sys.exit("--fp16 needs: pip install onnx onnxconverter-common")
        m16 = float16.convert_float_to_float16(onnx.load(str(onnx_path)), keep_io_types=True)
        onnx.save(m16, str(onnx_path))

    # Check the exported file against PyTorch on random input and, if given, report the agreement.
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime isn't installed, so the export wasn't checked (pip install onnxruntime).")
    else:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        agree, worst = 0, 0.0
        for seed in range(8):
            torch.manual_seed(seed)
            x = torch.randn(1, 3, size, size)
            with torch.no_grad():
                ref = model(x)[0]
            out = torch.from_numpy(sess.run(None, {"pixels": x.numpy()})[0])[0]
            worst = max(worst, float((ref.softmax(0) - out.softmax(0)).abs().max()))
            agree += int(ref.argmax() == out.argmax())
        print(f"ONNX vs PyTorch on 8 random inputs: same top class {agree}/8, "
              f"largest probability difference {worst:.4f}")

    classes = rc.classes
    missing = [c for c in classes if c not in CENTERS]
    if missing:
        print("No map position in demo_app/country_centers.py for:", ", ".join(missing))
    size_mb = round(onnx_path.stat().st_size / 1e6)
    meta = {
        "name": a.name,
        "url": a.url,
        "size_mb": size_mb,
        "input_size": size,
        "resize": t["resize"],
        "mean": t["mean"], "std": t["std"],
        "classes": classes,
        "centers": {c: list(CENTERS[c]) for c in classes if c in CENTERS},
    }
    (a.out_dir / "resnet_meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(f"Wrote {onnx_path} ({size_mb} MB) and {a.out_dir / 'resnet_meta.json'} ({len(classes)} countries).")
    if size_mb > 95 and not a.url.startswith("http"):
        print("WARNING: that file is over GitHub's 100 MB limit. Use --fp16, or upload it to a Hugging Face "
              "model repo and re-run with --url (or edit 'url' in resnet_meta.json).")


if __name__ == "__main__":
    main()
