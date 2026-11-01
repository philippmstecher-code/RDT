#!/usr/bin/env python3
"""Fixed-label noise (Extended Data Fig. 8): do the extra class-selective features under between-superclass noise fit the wrong
labels? A forward pass of the 50,000 training images through every stored checkpoint (epochs 5-30) of the clean,
within- and between-superclass runs (seeds 7, 42, 137, 256, 314), encoded with the checkpoint's own pooled-layer SAE.

Noisy labels: reconstructed exactly as packages/ml/src/training.py NoisyLabelDataset(fixed=True) draws them
(random.Random(seed * 1_000_003 + idx), noise_prob 0.3, the run seed), on the training set in file order; verified by
the final checkpoint's agreement with the reconstructed noisy labels on the flipped images (memorised far above chance).
Class-selective features: as noise_routing_nesting.py (validation activations stored by the pipeline, one common
threshold set per seed = the clean run's stored thresholds, exclusive rule), dominant class = argmax of the feature's
class-mean activation on the validation images.
Noise-fit index per class-selective feature f with dominant class c: mean activation on the training images of true
class c whose label was flipped, over the mean on those whose label was kept. Null: the clean run of the same seed with
the between-superclass flip pattern as pseudo-mask (no label was flipped). A feature is 'bent' (fits the wrong labels
by firing less on the mislabelled images of its own class) when its index falls below the 5th percentile of the clean
null at the same epoch (pooled over seeds); this lower-tail rule is applied downstream (noise_fit_lineage.py, Extended Data
Fig. 8h). The 'fits_noise' column written here is the upper tail (index above the 95th percentile); it is not used
in the paper.
Readout check: class with the largest logit contribution of the feature's decoder direction (checkpoint fc weights).
Writes analysis_outputs/paper/RDT_noise_fit/{features.csv, summary.csv, verify.json}. Read-only on data; CPU.
Usage: python scripts/paper/noise_fit_features.py
"""
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(4)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "packages" / "ml" / "src"))
import noise_routing_nesting as N  # noqa: E402
from dependency_lib import selectivity  # noqa: E402
from sae import SparseAutoencoder  # noqa: E402
from cifar100_hierarchy import build_superclass_map  # noqa: E402

CIFAR = N.E / "_cifar_src" / "datasets" / "cifar100" / "cifar-100-python"
OUT = REPO / "analysis_outputs" / "paper" / "RDT_noise_fit"
OUT.mkdir(parents=True, exist_ok=True)
MEAN, STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
P = 0.3
FINE = pickle.load(open(CIFAR / "meta", "rb"), encoding="latin1")["fine_label_names"]
SCM = build_superclass_map(FINE, {c: i for i, c in enumerate(FINE)})
MEMBERS = {}
for ci, sc in SCM.items():
    MEMBERS.setdefault(sc, []).append(ci)


def train_set():
    d = pickle.load(open(CIFAR / "train", "rb"), encoding="latin1")
    return d["data"].reshape(-1, 3, 32, 32), np.array(d["fine_labels"])


def noisy_labels(y, kind, seed):
    out = y.copy()
    flipped = np.zeros(len(y), bool)
    for idx, lab in enumerate(y):
        rng = random.Random(seed * 1_000_003 + idx)
        if rng.random() < P:
            if kind == "within":
                sib = [c for c in MEMBERS[SCM[lab]] if c != lab]
                out[idx] = rng.choice(sib)
            else:
                others = sorted(set(range(100)) - set(MEMBERS[SCM[lab]]))
                out[idx] = rng.choice(others)
            flipped[idx] = True
    return out, flipped


def load_net(lane, m):
    import torchvision.models as tvm
    sd = torch.load(lane / "dev_snapshots" / f"milestone_{m}" / "checkpoint.pt", map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    net = tvm.resnet18(weights=None)
    net.fc = nn.Linear(512, int(sd["fc.weight"].shape[0]))
    net.load_state_dict(sd, strict=True)
    return net.eval()


@torch.no_grad()
def pooled(net, X, bs=1000):
    feats, logits = [], []
    cap = []
    h = net.avgpool.register_forward_hook(lambda mod, i, o: cap.append(o.flatten(1)))
    try:
        for b in range(0, len(X), bs):
            x = torch.from_numpy(X[b:b + bs]).float() / 255.0
            x = (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
            logits.append(net(x))
    finally:
        h.remove()
    return torch.cat(cap), torch.cat(logits)


def main():
    X, y = train_set()
    verify, rows = {}, []
    for s in N.SEEDS:
        lab = {"clean": (y, noisy_labels(y, "between", s)[1])}
        for kind in ("within", "between"):
            lab[kind] = noisy_labels(y, kind, s)
        ssi_t, csi_t, sai_t = N.thresholds(s)
        for arm in ("clean", "within", "between"):
            lane = N.lane_dir(arm, s)
            yn, fl = lab[arm]
            res = json.load(open(lane / "sae_analysis" / "sae_results.json"))
            k = int(res["metadata"]["k_sparse"])
            for m in range(1, 7):
                net = load_net(lane, m)
                A, Z = pooled(net, X)
                if m == 6:
                    pred = Z.argmax(1).numpy()
                    verify[f"{arm}|{s}"] = {"flipped": int(fl.sum()), "pred_eq_noisy_on_flipped": float((pred[fl] == yn[fl]).mean()),
                                            "pred_eq_true_on_flipped": float((pred[fl] == y[fl]).mean()),
                                            "pred_eq_label_on_kept": float((pred[~fl] == y[~fl]).mean())}
                sd = torch.load(lane / "sae_analysis" / "saes" / str(m) / "avgpool.pt", map_location="cpu")
                d_in, d_hid = sd["encoder.weight"].shape[1], sd["encoder.weight"].shape[0]
                sae = SparseAutoencoder(d_in, d_hid // d_in, k_sparse=k)
                sae.load_state_dict(sd)
                sae.eval()
                with torch.no_grad():
                    Htr = np.concatenate([sae.encode(A[b:b + 5000]).numpy() for b in range(0, len(A), 5000)])
                Hva = np.load(lane / "sae_analysis" / "activations" / str(m) / "avgpool.npy")
                sel = selectivity(Hva)
                alive = (Hva != 0).mean(0) >= 0.01
                csi_idx = np.where((sel["csi"] > csi_t) & alive & ~(sel["sai"] > sai_t))[0]
                M = sel["M"]
                Dn = torch.nn.functional.normalize(sae.decoder.weight.detach().double(), dim=0).numpy()
                W = net.fc.weight.detach().double().numpy()
                for f in csi_idx:
                    c = int(np.argmax(M[f]))
                    inc = (y == c)
                    a_fl, a_kp = Htr[inc & fl, f].mean(), Htr[inc & ~fl, f].mean()
                    rd = int(np.argmax(W @ Dn[:, f]))
                    rows.append(dict(arm=arm, seed=s, epoch=5 * m, feature=int(f), cls=c, act_flipped=float(a_fl),
                                     act_kept=float(a_kp), nfi=float((a_fl + 1e-8) / (a_kp + 1e-8)), readout=rd,
                                     readout_is_cls=bool(rd == c),
                                     readout_same_sc=bool(SCM[rd] == SCM[c])))
                print(arm, s, m, len(csi_idx), flush=True)
    json.dump(verify, open(OUT / "verify.json", "w"), indent=1)
    F = pd.DataFrame(rows)
    thr = F[F.arm == "clean"].groupby("epoch").nfi.quantile(0.95)
    F["fits_noise"] = F.nfi > F.epoch.map(thr)
    F.to_csv(OUT / "features.csv", index=False)
    S = F.groupby(["arm", "seed", "epoch"]).agg(n_cls=("feature", "size"), n_fit=("fits_noise", "sum"),
                                                readout_is_cls=("readout_is_cls", "mean")).reset_index()
    S["n_right"] = S.n_cls - S.n_fit
    S.to_csv(OUT / "summary.csv", index=False)
    print(json.dumps(verify, indent=1))
    print(thr.round(2).to_dict())
    print(S.groupby(["arm", "epoch"])[["n_cls", "n_right", "n_fit", "readout_is_cls"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
