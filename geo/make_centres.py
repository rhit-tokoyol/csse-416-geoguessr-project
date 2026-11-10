"""Build centres.npz (lat, lon, classes) for server.py from train.csv.

    python geo/make_centres.py /work/.../OpenWorld/train.csv geo/centres.npz [head.pt]

`classes` is the sorted list of cell labels, which is the order sklearn's LabelEncoder gave the model
in main.ipynb (that holds when every cell of train.csv was in the label space, i.e. K = None).
Each cell's lat/lon is the mean of all its rows in train.csv, so no cell is left without a centre.
Pass the trained head as a third argument to check that the number of cells matches its output size.
"""
import sys

import numpy as np
import pandas as pd

if len(sys.argv) not in (3, 4):
    sys.exit(__doc__)
csv, out = sys.argv[1], sys.argv[2]

df = pd.read_csv(csv, usecols=["latitude", "longitude", "cell"])
df["cell"] = df["cell"].astype(str)
g = df.groupby("cell")
classes = sorted(g.groups)
lat = g["latitude"].mean().reindex(classes).to_numpy()
lon = g["longitude"].mean().reindex(classes).to_numpy()

if len(sys.argv) == 4:
    import torch
    sd = torch.load(sys.argv[3], map_location="cpu")
    n = sd["fc.10.weight"].shape[0] if "fc.10.weight" in sd else sd["net.3.weight"].shape[0] if "net.3.weight" in sd else None
    if n is None:
        sys.exit(f"Couldn't find the output layer in {sys.argv[3]} (looked for fc.10.weight and net.3.weight).")
    if n != len(classes):
        sys.exit(f"The head predicts {n} cells but {csv} has {len(classes)}: they are not from the same run.")
    print(f"Head and CSV agree: {n} cells.")

np.savez(out, lat=lat, lon=lon, classes=np.array(classes))
print(f"Wrote {out}: {len(classes)} cells, {int(np.isnan(lat).sum())} without a centre.")
