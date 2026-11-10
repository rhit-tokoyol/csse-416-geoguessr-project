"""Turn a .resnetmodel file back into the two files a built-in ResNet needs (resnet_meta.json + resnet_country.onnx).

    python unpack_resnetmodel.py resnet.resnetmodel docs/model

Use this to ship a ResNet you already have as a .resnetmodel inside the site, so visitors don't have to
upload it. The page shows the ResNet option by itself when docs/model/resnet_meta.json exists.
Only the standard library is needed.

File layout:  b"RESNET1\\n" + uint32 (little-endian) metadata length + metadata JSON + the .onnx bytes
"""

import json
import struct
import sys
from pathlib import Path

MAGIC = b"RESNET1\n"


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    raw = src.read_bytes()
    if raw[:len(MAGIC)] != MAGIC:
        sys.exit(f"{src} isn't a .resnetmodel file.")
    (n,) = struct.unpack("<I", raw[len(MAGIC):len(MAGIC) + 4])
    start = len(MAGIC) + 4 + n
    meta = json.loads(raw[len(MAGIC) + 4:start])
    onnx = raw[start:]
    out.mkdir(parents=True, exist_ok=True)
    (out / "resnet_country.onnx").write_bytes(onnx)
    meta["url"] = "model/resnet_country.onnx"          # where the page downloads the model from (relative to docs/)
    meta["size_mb"] = round(len(onnx) / 1e6)
    (out / "resnet_meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(f"Wrote {out / 'resnet_country.onnx'} ({len(onnx) / 1e6:.1f} MB) and {out / 'resnet_meta.json'}: "
          f"{len(meta['classes'])} countries.")


if __name__ == "__main__":
    main()
