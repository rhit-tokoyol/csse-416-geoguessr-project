"""Pack the ResNet ONNX model and its metadata into ONE file the website can load.

    python pack_resnet_bundle.py docs/model/resnet_meta.json docs/model/resnet_country.onnx -o resnet.resnetmodel

Then on the site, click "Load ResNet…" and choose the .resnetmodel file. The file is read in the
browser and never uploaded anywhere; reloading the page forgets it.

Both inputs come from export_resnet_onnx.py. This script needs only the standard library, so you
can run it on your own computer after downloading the two files from the cluster.

File layout:  b"RESNET1\\n" + uint32 (little-endian) metadata length + metadata JSON + the .onnx bytes
"""

import argparse
import json
import struct
from pathlib import Path

MAGIC = b"RESNET1\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("meta", type=Path, help="resnet_meta.json from export_resnet_onnx.py")
    p.add_argument("onnx", type=Path, help="resnet_country.onnx from export_resnet_onnx.py")
    p.add_argument("-o", "--out", type=Path, default=Path("resnet.resnetmodel"))
    p.add_argument("--name", help="name the site shows for this model")
    a = p.parse_args()

    meta = json.loads(a.meta.read_text(encoding="utf-8"))
    onnx = a.onnx.read_bytes()
    meta.pop("url", None)                      # the model travels inside this file
    meta["size_mb"] = round(len(onnx) / 1e6)
    if a.name:
        meta["name"] = a.name
    for key in ("classes", "centers", "input_size", "resize", "mean", "std"):
        if key not in meta:
            raise SystemExit(f"{a.meta} has no '{key}'. Was it made by export_resnet_onnx.py?")
    header = json.dumps(meta, separators=(",", ":")).encode()
    a.out.write_bytes(MAGIC + struct.pack("<I", len(header)) + header + onnx)
    print(f"Wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB): {len(meta['classes'])} countries, "
          f"{meta['input_size']}px input.")


if __name__ == "__main__":
    main()
