"""Check train_model_sample_1.pt against the cached features from resnet_model.ipynb.

    python check_resnet_head.py output/feats_resnet_K20_n200000.pt demo_models/train_model_sample_1.pt

Run from the project root. It should print the "Accuracy" your notebook reported for the best epoch
(about 0.575 for the 15-epoch run, about 0.515 for the 5-epoch run).
The demo itself does not need the features file: it computes features from the uploaded screenshot.
"""
import sys

import torch

from demo_predict import ResNetHead

if len(sys.argv) != 3:
    sys.exit(__doc__)
feats_path, head_path = sys.argv[1:]
_, _, test_x, test_y = torch.load(feats_path, map_location="cpu")  # (train_x, train_y, test_x, test_y)
head = ResNetHead(222)
head.load_state_dict(torch.load(head_path, map_location="cpu"))
head.eval()
with torch.no_grad():
    pred = head(test_x).argmax(1)
print(f"test accuracy: {(pred == test_y).float().mean():.4f} on {len(test_y):,} images")
