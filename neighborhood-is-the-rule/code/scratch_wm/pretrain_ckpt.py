"""Produce leave-one-game-out pretrained world-model checkpoints.

For each held-out game we train a WM on the OTHER 24 games' transitions and save
the weights. The agent later loads this as an INITIALISATION and finetunes it on
the held-out game's own data collected online (that is the whole thesis from E7:
zero-shot transfer is ~0, but the pretrained weights are a much better starting
point than random -- finetune 0.45 vs scratch 0.35 exact frame).

The held-out game is NEVER in the pool, so a checkpoint has never seen a single
frame of the game it will be evaluated on. That is the private-set proxy.

Pool data is read from pt_cache (exp_pretrain.collect wrote 3000 transitions per
game at seed 11 for all 25 games), so this does zero env stepping.

Env:  PT_HELD=all | comma-sep prefixes    PT_EPOCHS_PRE=6   PT_RF=9  PT_WIDTH=96
"""
import os, sys, time, json, glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from wm_core import WM, save_ckpt, load_pool_from_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.environ.get("PT_CACHE", os.path.join(ROOT, "pt_cache"))
CKPT_DIR = os.environ.get("PT_CKPT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts"))
RF = int(os.environ.get("PT_RF", "9"))
WIDTH = int(os.environ.get("PT_WIDTH", "96"))
EPOCHS_PRE = int(os.environ.get("PT_EPOCHS_PRE", "6"))
PER_GAME = int(os.environ.get("PT_PER_GAME", "3000"))
POOL_SEED = int(os.environ.get("PT_POOL_SEED", "11"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CKPT_DIR, exist_ok=True)

ALL = sorted(json.load(open(m))["game_id"] for m in
             glob.glob(os.path.join(ROOT, "environment_files", "*", "*", "metadata.json")))
PFX = {g.split("-")[0]: g for g in ALL}
ALLPFX = sorted(PFX)

held_env = os.environ.get("PT_HELD", "all")
HELD = ALLPFX if held_env in ("", "all") else [h for h in held_env.split(",") if h]


def train(model, G0, A, G1, XY, epochs, lr=1e-3, bs=32):
    g0 = torch.from_numpy(G0); a = torch.from_numpy(A)
    g1 = torch.from_numpy(G1); xy = torch.from_numpy(XY)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    n = len(g0)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            b0 = g0[idx].to(DEV, non_blocking=True); ba = a[idx].to(DEV, non_blocking=True)
            b1 = g1[idx].to(DEV, non_blocking=True); bxy = xy[idx].to(DEV, non_blocking=True)
            loss = F.cross_entropy(model(b0, ba, bxy), b1)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


print(f"CONFIG rf={RF} width={WIDTH} epochs_pre={EPOCHS_PRE} per_game={PER_GAME} "
      f"pool_seed={POOL_SEED} dev={DEV} n_held={len(HELD)}", flush=True)

for held in HELD:
    out = os.path.join(CKPT_DIR, f"{held}.pt")
    if os.path.exists(out) and os.path.getsize(out) > 0 and os.environ.get("PT_FORCE", "0") != "1":
        print(f"{held}: exists, skip", flush=True)
        continue
    t0 = time.time()
    others = [p for p in ALLPFX if p != held]
    G0, A, G1, XY, missing = load_pool_from_cache(CACHE, others, PER_GAME, POOL_SEED)
    if missing:
        print(f"{held}: WARNING missing cache for {missing}", flush=True)
    torch.manual_seed(0)
    m = WM(rf=RF, width=WIDTH).to(DEV)
    train(m, G0, A, G1, XY, EPOCHS_PRE)
    meta = dict(held=held, rf=RF, width=WIDTH, epochs_pre=EPOCHS_PRE,
                per_game=PER_GAME, pool_seed=POOL_SEED, n_pool=len(G0),
                n_games=len(others))
    save_ckpt(out, m, meta)
    print(f"{held}: pool={len(G0)} trans, {EPOCHS_PRE} epochs, saved -> {out} "
          f"({time.time()-t0:.0f}s)", flush=True)

print("PRETRAIN_CKPT DONE", flush=True)
