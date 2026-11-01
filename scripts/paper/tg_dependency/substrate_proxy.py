"""Read-out proxy of the task-general transplant: subspaces are removed from frozen early stages and a fresh linear
read-out is fitted, without retraining later layers (unlike the transplant runs tgtransplant_*).

Per clean network (critical_sham sham_e00, seeds 7, 42, 137, 256, 314) and epoch (0-15): the frozen stem + layer1 +
layer2 of that epoch. A subspace is projected out of the layer1 and the layer2 output at every spatial position
(mean-preserving, x -> x - Q Q^T (x - mu), mu = channel mean of the unprojected spatially pooled outputs on the training
images), and a fresh linear read-out (StandardScaler + RidgeClassifier, alpha 10, as curriculum_probe) is fitted on the full
layer2 output (128 x 4 x 4 = 2,048 dims) of 200 training images per class and scored on the 10,000 test images.
Subspaces per layer (built as run_tg_transplant.build_spec, from the per-epoch SAE of sham_perepoch, stored per-run
thresholds, tg_nesting.cats; covariance S of the spatially pooled outputs):
  tg     span of the unit decoder directions of the task-general features (rank k)
  rand   random rank-k subspace matched in captured variance (tg_nesting.var_matched, 64 draws, closest)
  pc     the top-k principal directions of S (equal rank, maximal variance)
  ot     span of k decoder directions of alive features that are not task-general, the subset (of 64 random k-subsets)
         closest in captured variance to tg
  none   nothing removed
Read-outs: fine-class accuracy (100-way), superclass accuracy (a separate 20-way ridge), and accuracy among the five
classes of the true superclass (argmax of the fine scores restricted to them). Rows also record rank and captured
variance per layer and subspace.
Writes analysis_outputs/paper/TG_dependency/proxy_long.csv (appends; resumable).
Usage: python scripts/paper/tg_dependency/substrate_proxy.py [--seeds 7,42] [--epochs 1,5]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

torch.set_num_threads(6)
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
import transfer_lib as T  # noqa: E402
from curriculum_probe import SUPER_OF  # noqa: E402
from tg_nesting import cats, orth, var_matched, var_share  # noqa: E402

E = Path(RDT_DATA + "/extended/experiments")
OUT = HERE.parents[3] / "analysis_outputs/paper/TG_dependency"
LAYERS = ["layer1", "layer2"]
N_TR = 200
# 'pcv': the leading principal directions up to the task-general variance share (variance-matched, lower rank);
# '<subspace>@<layer>': removal in that layer only
CONDS = ("none", "tg", "rand", "pc", "ot", "pcv", "tg@layer1", "rand@layer1", "tg@layer2", "rand@layer2")
MEMBERS = np.array([np.where(SUPER_OF == SUPER_OF[c])[0] for c in range(100)])


def cifar_train(n_per_class):
    with open(T.CIFAR / "train", "rb") as f:
        d = pickle.load(f, encoding="latin1")
    X = d["data"].reshape(-1, 3, 32, 32)
    y = np.array(d["fine_labels"])
    idx = np.concatenate([np.where(y == c)[0][:n_per_class] for c in range(100)])
    return X[idx], y[idx]


def checkpoint(s, e):
    w = E / f"critical_sham/lanes/sham_e00_seed{s}"
    snaps = json.load(open(w / "dev_snapshots/metrics/devtrain_snapshots.json"))
    if e == 0:
        ms = [x["milestone_index"] for x in snaps if x.get("snapshot_type") == "t0_resume"][0]
    else:
        ms = {int(x["epoch"]): x["milestone_index"] for x in snaps if x.get("snapshot_type") == "intermediate"}[e]
    sd = torch.load(w / f"dev_snapshots/milestone_{ms}/checkpoint.pt", map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) and "model_state_dict" in sd else sd
    net = T.build_model("ResNet-18", int(sd["fc.weight"].shape[0]))
    net.load_state_dict(sd, strict=True)
    return net.eval()


@torch.no_grad()
def forward(net, X, P=None, bs=1000):
    """Layer1/layer2 spatially pooled outputs (unprojected path if P is None) and the flattened layer2 output."""
    pooled, flat = {L: [] for L in LAYERS}, []
    for i in range(0, len(X), bs):
        h = net.maxpool(net.relu(net.bn1(net.conv1(T.cifar_batch(X[i:i + bs])))))
        for L in LAYERS:
            h = getattr(net, L)(h)
            if P is not None and P.get(L) is not None:
                Q, mu = P[L]
                z = h.permute(0, 2, 3, 1) - mu
                h = (h.permute(0, 2, 3, 1) - (z @ Q) @ Q.T).permute(0, 3, 1, 2)
            pooled[L].append(h.mean(dim=(2, 3)).double())
        flat.append(h.flatten(1).float())
    return {L: torch.cat(v).numpy() for L, v in pooled.items()}, torch.cat(flat).numpy()


def bases(s, e, Spool, rng):
    sd = E / f"sham_perepoch/lanes/sham_e00_seed{s}/sae_analysis"
    thr = json.load(open(sd / "sae_results.json"))["metadata"]["analysis_thresholds"]
    t = {"sai": thr["sai_task_general_thresh"], "ssi": thr["ssi_abstraction_thresh"], "csi": thr["csi_differentiation_thresh"]}
    out = {}
    for L in LAYERS:
        H = np.load(sd / "activations" / str(e) / f"{L}.npy").astype(np.float64)
        alive, tg = cats(H, t)[:2]
        D = torch.load(sd / "saes" / str(e) / f"{L}.pt", map_location="cpu", weights_only=True)["decoder.weight"].double().numpy()
        D = D / np.maximum(np.linalg.norm(D, axis=0, keepdims=True), 1e-12)
        X = Spool[L] - Spool[L].mean(0)
        S = X.T @ X / len(X)
        k = int(tg.sum())
        if k == 0:
            out[L] = {}
            continue
        Qtg = orth(D[:, tg])
        k = Qtg.shape[1]
        v = var_share(Qtg, S)
        Qs, _ = var_matched(S, k, v, rng, 64)
        Qr = Qs[int(np.argmin([abs(var_share(q, S) - v) for q in Qs]))]
        w, V = np.linalg.eigh(S)
        Qpc = V[:, ::-1][:, :k]
        ot = np.where(alive & ~tg)[0]
        Qot = None
        if len(ot) >= k:
            cand = [orth(D[:, rng.choice(ot, k, replace=False)]) for _ in range(64)]
            Qot = cand[int(np.argmin([abs(var_share(q, S) - v) for q in cand]))]
        cum = np.cumsum(w[::-1]) / w.sum()
        Qpcv = V[:, ::-1][:, :int(np.argmin(np.abs(cum - v))) + 1]
        out[L] = {"tg": Qtg, "rand": Qr, "pc": Qpc, "pcv": Qpcv, "ot": Qot, "_S": S, "_k": k, "_ntg": int(tg.sum())}
    return out


def readout(Ftr, ytr, Fte, yte):
    sc = StandardScaler().fit(Ftr)
    A, B = sc.transform(Ftr), sc.transform(Fte)
    m = RidgeClassifier(alpha=10.0).fit(A, ytr)
    D = m.decision_function(B)
    fine = float((D.argmax(1) == yte).mean() * 100)
    within = float((MEMBERS[yte][np.arange(len(yte)), D[np.arange(len(yte))[:, None], MEMBERS[yte]].argmax(1)]
                    == yte).mean() * 100)
    m2 = RidgeClassifier(alpha=10.0).fit(A, SUPER_OF[ytr])
    sup = float((m2.predict(B) == SUPER_OF[yte]).mean() * 100)
    return fine, sup, within


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,42,137,256,314")
    ap.add_argument("--epochs", default="0,1,2,3,4,5,6,8,10,12,15")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "proxy_long.csv"
    done = set()
    if f.exists():
        old = pd.read_csv(f)
        done = set(zip(old.seed, old.epoch, old.cond))
    Xtr, ytr = cifar_train(N_TR)
    te = T.cifar_test()
    Xte, yte = te["X"], te["y"]
    for s in map(int, a.seeds.split(",")):
        for e in map(int, a.epochs.split(",")):
            if all((s, e, c) in done for c in CONDS):
                continue
            net = checkpoint(s, e)
            pool_tr, Ftr0 = forward(net, Xtr)
            _, Fte0 = forward(net, Xte)
            B = bases(s, e, pool_tr, np.random.default_rng(500 + s))
            mu = {L: torch.from_numpy(pool_tr[L].mean(0)).float() for L in LAYERS}
            for cond in CONDS:
                if (s, e, cond) in done:
                    continue
                row = dict(seed=s, epoch=e, cond=cond)
                if cond == "none":
                    Ftr, Fte = Ftr0, Fte0
                else:
                    base, _, only = cond.partition("@")
                    P = {}
                    for L in LAYERS:
                        if only and L != only:
                            continue
                        Q = B[L].get(base)
                        if Q is None:
                            continue
                        P[L] = (torch.from_numpy(np.ascontiguousarray(Q)).float(), mu[L])
                        row[f"{L}_rank"] = Q.shape[1]
                        row[f"{L}_var"] = var_share(Q, B[L]["_S"])
                        row[f"{L}_ntg"] = B[L]["_ntg"]
                    if not P:
                        continue
                    _, Ftr = forward(net, Xtr, P)
                    _, Fte = forward(net, Xte, P)
                row["fine"], row["superclass"], row["within"] = readout(Ftr, ytr, Fte, yte)
                print(row, flush=True)
                cols = ["seed", "epoch", "cond", "fine", "superclass", "within"] + [f"{L}_{k}" for L in LAYERS
                                                                                   for k in ("rank", "var", "ntg")]
                pd.DataFrame([row]).reindex(columns=cols).to_csv(f, mode="a", header=not f.exists(), index=False)


if __name__ == "__main__":
    main()
