"""Shared world-model core: ONE definition of the architecture, imported by both
the pretrainer (pretrain_ckpt.py) and the agent (wm_pretrained_agent.py).

Keeping a single definition is deliberate. This project has been bitten before
by two copies of the "same" net drifting apart (E10: an EPOCHS mismatch across
machines looked like a diversity effect until the scratch baseline exposed it).
A checkpoint saved by the pretrainer must load into the agent's net with the
exact same shapes, so the class lives in one place.

Architecture matches exp_pretrain.py::WM exactly (RF=9, WIDTH=96 is the measured
peak, E11). Input channels: NCOL one-hot frame + 7 action planes + 2 click-coord
planes.
"""
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NCOL = 16
DEFAULT_RF = 9
DEFAULT_WIDTH = 96


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.n = nn.GroupNorm(8, w)
        self.c = nn.Conv2d(w, w, 3, padding=1)

    def forward(self, x):
        return x + self.c(F.relu(self.n(x)))


class WM(nn.Module):
    def __init__(self, rf=DEFAULT_RF, width=DEFAULT_WIDTH):
        super().__init__()
        self.rf = rf
        self.width = width
        n = max(1, (rf - 1) // 2)
        self.stem = nn.Conv2d(NCOL + 7 + 2, width, 3, padding=1)
        self.body = nn.Sequential(*[Block(width) for _ in range(n - 1)])
        self.head = nn.Conv2d(width, NCOL, 1)

    def forward(self, g, a, xy):
        x = F.one_hot(g, NCOL).permute(0, 3, 1, 2).float()
        ap = F.one_hot(a - 1, 7).float()[:, :, None, None].expand(-1, -1, g.shape[1], g.shape[2])
        B, H, W = g.shape
        cp = torch.zeros(B, 2, H, W, device=g.device)
        ok = xy[:, 0] >= 0
        if ok.any():
            i = torch.nonzero(ok).squeeze(1)
            cx = xy[i, 0].clamp(0, W - 1)
            cy = xy[i, 1].clamp(0, H - 1)
            cp[i, 0, cy, cx] = 1.0
            cp[i, 1, cx, cy] = 1.0
        return self.head(self.body(self.stem(torch.cat([x, ap, cp], 1))))


def save_ckpt(path, model, meta):
    torch.save({"state_dict": model.state_dict(),
                "rf": model.rf, "width": model.width, "meta": meta}, path)


def load_ckpt(path, device):
    """Rebuild a WM with the saved geometry and load its weights."""
    blob = torch.load(path, map_location=device, weights_only=False)
    m = WM(rf=blob["rf"], width=blob["width"]).to(device)
    m.load_state_dict(blob["state_dict"])
    return m, blob.get("meta", {})


def small_component_cells(g, max_size=40):
    """Cells belonging to connected same-colour blobs of size <= max_size.
    Same routine the guided/wm agents use to aim clicks. Returns list of (y, x).
    """
    H, W = g.shape
    seen = np.zeros((H, W), bool)
    out = []
    for y in range(H):
        for x in range(W):
            if seen[y, x]:
                continue
            col = g[y, x]
            q = deque([(y, x)])
            seen[y, x] = True
            comp = []
            while q:
                cy, cx = q.popleft()
                comp.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == col:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if len(comp) <= max_size:
                out.extend(comp)
    return out


def load_pool_from_cache(cache_dir, game_prefixes, per_game=3000, seed=11):
    """Concatenate cached (G0,A,G1,XY) transitions for the given game prefixes.
    Files are written by exp_pretrain.collect() as {prefix}_{steps}_{seed}.npz.
    Skips any prefix whose cache file is missing (caller checks coverage)."""
    G0, A, G1, XY = [], [], [], []
    missing = []
    for pfx in game_prefixes:
        path = os.path.join(cache_dir, f"{pfx}_{per_game}_{seed}.npz")
        if not os.path.exists(path):
            missing.append(pfx)
            continue
        z = np.load(path)
        G0.append(z["G0"]); A.append(z["A"]); G1.append(z["G1"]); XY.append(z["XY"])
    if not G0:
        raise RuntimeError(f"no cache files found in {cache_dir} for {game_prefixes}")
    return (np.concatenate(G0), np.concatenate(A),
            np.concatenate(G1), np.concatenate(XY), missing)
