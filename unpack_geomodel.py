"""Turn a .geomodel file back into the two files a built-in model needs (meta.json + head.bin).

    python unpack_geomodel.py run12.geomodel docs/model/run12

Use this to ship a model you already have as a .geomodel inside the site (see docs/model/models.json),
so visitors don't have to upload it. Only the standard library is needed.

File layout:  b"GEOMODEL1\\n" + uint32 (little-endian) metadata length + metadata JSON (+ padding) + float32 weights
"""

import json
import struct
import sys
from pathlib import Path

MAGIC = b"GEOMODEL1\n"


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    raw = src.read_bytes()
    if raw[:len(MAGIC)] != MAGIC:
        sys.exit(f"{src} isn't a .geomodel file.")
    (n,) = struct.unpack("<I", raw[len(MAGIC):len(MAGIC) + 4])
    start = len(MAGIC) + 4 + n
    meta = json.loads(raw[len(MAGIC) + 4:start])
    weights = raw[start:]
    if len(weights) % 4:
        sys.exit(f"{src} looks damaged: the weights aren't a whole number of float32 values.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "head.bin").write_bytes(weights)
    (out / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(f"Wrote {out / 'head.bin'} ({len(weights) / 1e6:.1f} MB) and {out / 'meta.json'}: "
          f"{len(meta['cells'])} cells, {meta.get('head_kind', '?')} head, name '{meta.get('name', '?')}'.")


if __name__ == "__main__":
    main()
