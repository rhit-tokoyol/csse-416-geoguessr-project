"""Driving-side recognition for the GeoGuessr cell classifier.

Idea: which side of the road traffic drives on is one of the strongest visual
clues in GeoGuessr and it splits the world cleanly (~75 territories drive on
the left). We use it two ways:

1. Multi-task training: the MLP head gets a second output that predicts
   left vs. right driving from the (frozen) CLIP features. Labels come for free
   from the existing `country` column, so no hand labelling is needed.
2. Inference-time fusion: each cell gets a "fraction of training images that
   drive on the left". The predicted P(left) re-weights the cell
   probabilities, pushing down cells on the wrong side of the road:

       log p'(cell) = log p(cell) + w * log P(side of cell | image)

   `w` is tuned on a validation split (w = 0 reproduces the plain model).

Mirroring: a mirrored left-driving scene looks like a right-driving one. The
side detector exploits this (see MultiTaskHead): it compares each image's CLIP
features with the features of its mirror image. Never add plain random
horizontal-flip augmentation without also swapping the side label.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ISO 3166-1 alpha-2 codes of countries/territories with left-hand traffic.
# (OSV-5M's `country` column uses alpha-2 codes, e.g. "US", "GB".)
LEFT_HAND_TRAFFIC = frozenset({
    # Europe
    "GB", "IE", "IM", "JE", "GG", "MT", "CY",
    # Asia
    "JP", "IN", "PK", "BD", "LK", "NP", "BT", "MV", "TH", "MY", "SG", "BN",
    "ID", "TL", "HK", "MO",
    # Oceania
    "AU", "NZ", "PG", "FJ", "SB", "TO", "WS", "KI", "TV", "NR", "CK", "NU",
    "NF", "PN", "CX", "CC",
    # Africa
    "ZA", "LS", "SZ", "BW", "NA", "ZW", "ZM", "MW", "MZ", "TZ", "KE", "UG",
    "MU", "SC", "SH",
    # Americas / Caribbean / Atlantic
    "JM", "BS", "BB", "TT", "GY", "SR", "AG", "DM", "GD", "KN", "LC", "VC",
    "AI", "BM", "KY", "MS", "TC", "VG", "VI", "FK",
})


def side_label(country):
    """1.0 if the country drives on the left, 0.0 if on the right."""
    if not isinstance(country, str):
        return 0.0
    return 1.0 if country.strip().upper() in LEFT_HAND_TRAFFIC else 0.0


def cell_left_fraction(frame, num_cells, cell_col="cell_label", side_col="side_label"):
    """Fraction of training images in each cell that drive on the left.

    Cells with no training images get 0.5 (neutral: fusion leaves them alone).
    Using a fraction instead of a hard 0/1 handles cells that straddle a
    border (e.g. Thailand/Laos, Ireland/Northern Ireland is fine, Hong Kong/
    Shenzhen is not).
    """
    frac = frame.groupby(cell_col)[side_col].mean().reindex(range(num_cells))
    return frac.fillna(0.5).to_numpy(dtype=np.float32)


class MultiTaskHead(nn.Module):
    """(features, mirrored features) -> (cell logits, driving-side logit).

    The side logit is mirror-antisymmetric:

        side(x) = g(f(x)) - g(f(mirror(x)))

    so mirroring an image exactly flips the prediction (left <-> right). This
    forces the side detector to rely on left/right asymmetry in the scene
    (which side traffic, shoulders and signs are on) rather than on "this looks
    like Japan", which the cell classifier already knows. A scene with no
    asymmetric cues gives a logit near 0 (P = 0.5), and fusion then leaves the
    cell probabilities alone. It also means every training image is
    effectively used twice: once as-is and once mirrored with the opposite label.
    """

    def __init__(self, n_cells, in_dim, hidden=1024, p=0.3, side_hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p))
        self.cell_out = nn.Linear(hidden, n_cells)
        self.side_net = nn.Sequential(nn.Linear(in_dim, side_hidden), nn.ReLU(), nn.Dropout(p),
                                      nn.Linear(side_hidden, 1))

    def forward(self, x, x_mirror):
        cell_logits = self.cell_out(self.trunk(x))
        side_logit = (self.side_net(x) - self.side_net(x_mirror)).squeeze(-1)
        return cell_logits, side_logit


def train_multitask(head, train_loader, test_loader, optimizer, epochs, device,
                    side_weight=0.5):
    """Trains on loaders yielding (features, mirrored features, cell_label, side_label).

    loss = CE(cell) + side_weight * BCE(side)
    Returns history dict and the final state_dict.
    """
    hist = {k: [] for k in ("train_loss", "test_loss", "train_acc", "test_acc",
                            "train_side_acc", "test_side_acc")}

    def run(loader, train):
        head.train(train)
        tot_loss = correct = side_correct = n = 0
        with torch.set_grad_enabled(train):
            for x, xm, y, s in loader:
                x, xm, y, s = x.to(device), xm.to(device), y.to(device), s.to(device).float()
                cell_logits, side_logit = head(x, xm)
                loss = F.cross_entropy(cell_logits, y) \
                    + side_weight * F.binary_cross_entropy_with_logits(side_logit, s)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                tot_loss += loss.item()
                correct += (cell_logits.argmax(1) == y).sum().item()
                side_correct += ((side_logit > 0).float() == s).sum().item()
                n += len(y)
        return tot_loss / len(loader), correct / n, side_correct / n

    for epoch in range(epochs):
        l, a, sa = run(train_loader, True)
        hist["train_loss"].append(l); hist["train_acc"].append(a); hist["train_side_acc"].append(sa)
        l, a, sa = run(test_loader, False)
        hist["test_loss"].append(l); hist["test_acc"].append(a); hist["test_side_acc"].append(sa)
        print(f"epoch {epoch+1}: train acc {hist['train_acc'][-1]:.3f}, "
              f"test acc {hist['test_acc'][-1]:.3f}, "
              f"test side acc {hist['test_side_acc'][-1]:.3f}")
    return hist, head.state_dict()


@torch.no_grad()
def predict(head, loader, device):
    """Returns (cell log-probs [N, C], P(left) [N], true cells [N], true sides [N]) as numpy."""
    head.eval()
    logps, pls, ys, ss = [], [], [], []
    for x, xm, y, s in loader:
        cell_logits, side_logit = head(x.to(device), xm.to(device))
        logps.append(F.log_softmax(cell_logits, dim=1).cpu())
        pls.append(torch.sigmoid(side_logit).cpu())
        ys.append(y)
        ss.append(s)
    return (torch.cat(logps).numpy(), torch.cat(pls).numpy(),
            torch.cat(ys).numpy(), torch.cat(ss).numpy())


def fuse(cell_logp, p_left, cell_left, weight, eps=1e-4):
    """Re-weights cell log-probs by agreement with the predicted driving side.

    P(side of cell | image) = P(left)*frac_left(cell) + P(right)*(1 - frac_left(cell))
    """
    p_left = np.clip(p_left, eps, 1 - eps)[:, None]
    agree = p_left * cell_left[None, :] + (1 - p_left) * (1 - cell_left[None, :])
    return cell_logp + weight * np.log(np.clip(agree, eps, 1.0))


def geoscore(pred_cells, lat, lon, cell_lat, cell_lon):
    """Mean GeoGuessr-style score (5000 * exp(-d / 1492.7 km))."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat, lon, cell_lat[pred_cells], cell_lon[pred_cells]])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    d = 2 * 6371 * np.arcsin(np.sqrt(a))
    return float(np.nanmean(5000 * np.exp(-d / 1492.7)))


def tune_weight(cell_logp, p_left, cell_left, y_true,
                weights=(0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)):
    """Picks the fusion weight with the best cell accuracy (use a validation split!)."""
    results = {w: float((fuse(cell_logp, p_left, cell_left, w).argmax(1) == y_true).mean())
               for w in weights}
    best = max(results, key=results.get)
    return best, results
