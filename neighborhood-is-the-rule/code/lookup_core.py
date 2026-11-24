"""Window memoriser shared by exp_lookup.py and exp_hybrid.py.
(k x k window of colours + click marker, action) -> next colour of the centre cell, hashed exactly
with two random integer convolutions in float64. Unseen keys predict 'no change'."""
import numpy as np
import torch
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WindowLookup:
    def __init__(self, k, seed=0):
        self.k = k
        g = torch.Generator().manual_seed(1234 + seed)
        self.W = torch.randint(1, 1 << 20, (2, 2, k, k), generator=g).double().to(DEV)

    def keys(self, G, A, XY):
        out = []
        for i in range(0, len(A), 64):
            s = torch.as_tensor(np.asarray(G[i:i + 64]), device=DEV).double() + 1.0
            B = s.shape[0]
            click = torch.zeros_like(s)
            xy = torch.as_tensor(np.asarray(XY[i:i + 64]), device=DEV).long()
            m = xy[:, 0] >= 0
            if m.any():
                bi = torch.nonzero(m).squeeze(1)
                click[bi, xy[m, 1].clamp(0, 63), xy[m, 0].clamp(0, 63)] = 1.0
            h = F.conv2d(torch.stack([s, click], 1), self.W, padding=self.k // 2)
            h1, h2 = h[:, 0].long() % (1 << 27), h[:, 1].long() % (1 << 28)
            act = torch.as_tensor(np.asarray(A[i:i + 64]), device=DEV).long().view(B, 1, 1).expand(B, 64, 64)
            out.append(((h1 << 31) | (h2 << 3) | (act & 7)).cpu().numpy().reshape(B, -1))
        return np.concatenate(out)

    def fit(self, G0, A, G1, XY):
        k = self.keys(G0, A, XY).ravel()
        c = np.asarray(G1).reshape(-1).astype(np.int64)
        u, cnt = np.unique((k << 4) | c, return_counts=True)
        uk, uc = u >> 4, u & 15
        order = np.lexsort((-cnt, uk))
        uk, uc = uk[order], uc[order]
        first = np.ones(len(uk), bool); first[1:] = uk[1:] != uk[:-1]
        self.kk, self.cc = uk[first], uc[first]
        return self

    def predict(self, G0, A, XY):
        G0 = np.asarray(G0)
        key = self.keys(G0, A, XY)
        pos = np.clip(np.searchsorted(self.kk, key), 0, len(self.kk) - 1)
        seen = self.kk[pos] == key
        p = G0.reshape(len(G0), -1).copy()
        p[seen] = self.cc[pos[seen]]
        return p.reshape(G0.shape), seen.reshape(G0.shape)
