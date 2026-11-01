"""Developmental timetable, step 2: per-image behavioural scores and per-class / per-superclass representational scores per checkpoint.

Per image (from canonical logits):
  fine   : argmax over 100 == true class
  sc     : argmax over 20 summed-softmax superclass probs == true superclass
  ws     : argmax over the 5 classes of the true superclass == true class  (within-superclass restricted)
  pt     : softmax prob of true class
  psc    : summed softmax prob of true superclass
  pws    : restricted (5-way) softmax prob of true class
Representational (pooled layer, z-scored per checkpoint, split-half nearest centroid; centroids from even images, test odd, and v.v.):
  nc_sc  : per-image correct under 20-way superclass nearest-centroid
  nc_ws  : per-image correct under 5-way nearest-class-centroid restricted to true superclass
Output: cache/<lane>_img.npz with arrays (n_ckpt, 10000) and epochs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timetable_lib import CACHE, CLS_OF_IMG, SC_OF_CLASS, SC_OF_IMG, lanes  # noqa: E402

import numpy as np  # noqa: E402

KEYS = ["fine", "sc", "ws", "pt", "psc", "pws", "nc_sc", "nc_ws"]
MEMBERS = np.stack([np.where(SC_OF_CLASS == s)[0] for s in range(20)])  # (20,5) canonical classes


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def behaviour(Z):
    y, s = CLS_OF_IMG, SC_OF_IMG
    P = softmax(Z.astype(np.float64))
    Psc = P[:, MEMBERS].sum(2)  # (N,20)
    own = MEMBERS[s]  # (N,5)
    Zr = np.take_along_axis(Z, own, 1)
    Pr = softmax(Zr)
    pos = (own == y[:, None]).argmax(1)
    return dict(fine=(Z.argmax(1) == y), sc=(Psc.argmax(1) == s), ws=(Zr.argmax(1) == pos),
                pt=P[np.arange(len(y)), y], psc=Psc[np.arange(len(y)), s], pws=Pr[np.arange(len(y)), pos])


def nearest_centroid(X):
    X = X.astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    j = np.tile(np.arange(100), 100)
    half = j % 2
    nc_sc = np.zeros(len(X), bool)
    nc_ws = np.zeros(len(X), bool)
    for h in (0, 1):
        tr, te = half == h, half != h
        Cc = np.stack([X[tr & (CLS_OF_IMG == c)].mean(0) for c in range(100)])
        Cs = np.stack([X[tr & (SC_OF_IMG == s)].mean(0) for s in range(20)])
        Xt = X[te]
        d_s = ((Xt[:, None, :] - Cs[None]) ** 2).sum(2)
        nc_sc[te] = d_s.argmin(1) == SC_OF_IMG[te]
        own = MEMBERS[SC_OF_IMG[te]]
        d_c = ((Xt[:, None, :] - Cc[own]) ** 2).sum(2)
        nc_ws[te] = own[np.arange(len(own)), d_c.argmin(1)] == CLS_OF_IMG[te]
    return nc_sc, nc_ws


def main():
    for key, lane in lanes().items():
        f = CACHE / f"{key}_img.npz"
        if f.exists():
            continue
        if not all((CACHE / f"{key}_m{l}_logits.npy").exists() for l in lane["labels"]):
            print("skip (missing)", key)
            continue
        out = {k: [] for k in KEYS}
        for lab in lane["labels"]:
            Z = np.load(CACHE / f"{key}_m{lab}_logits.npy")
            b = behaviour(Z)
            X = np.load(CACHE / f"{key}_m{lab}_pooled.npy")
            b["nc_sc"], b["nc_ws"] = nearest_centroid(X)
            for k in KEYS:
                out[k].append(b[k].astype(np.float32))
        np.savez_compressed(f, epochs=np.array(lane["epochs"]), **{k: np.stack(v) for k, v in out.items()})
        print(key, {k: np.round(np.stack(out[k]).mean(1), 3).tolist() for k in ["fine", "sc", "ws", "nc_sc", "nc_ws"]},
              flush=True)


if __name__ == "__main__":
    main()
