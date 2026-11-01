#!/usr/bin/env python3
"""Clean networks only, no training: are some task-general (Tg) features 'dominant', i.e. do they carry
the large shared data patterns that span all superclasses, and does the network rely on them more than on unspecific Tg
features?

Networks: the five clean 30-epoch ResNet-18 lanes (static_noise_s{S}/standard_resnet18), epochs 5, 10, 30, with the
per-checkpoint top-k SAEs of the main pipeline (layer1-layer4). Activations are recomputed on all 10,000 CIFAR-100 test
images (block output, spatial mean = the SAE input) and checked against the stored SAE activations.

Per layer and checkpoint:
  Tg feature      alive (active on >= 1% of images) and SAI above the lane's null-calibrated threshold (pipeline)
  dominance       label-free share of the layer's centred activation variance carried by the feature:
                  Var(h_f) * ||d_f||^2 / tr Cov(x)
Stage 'describe' writes the dominance distribution of Tg features (concentration, overlap with the mean response and
with the top centred class-agnostic PCs, i.e. after removing the between-class scatter).
Stage 'ablate' projects out (mean-preserving: x -> x - P(x - mu)) the decoder directions of
  tg_dom   the n most dominant Tg features          tg_uns  the n least dominant alive Tg features
  rnd_n    n random directions                      rnd_var random directions until the removed variance matches tg_dom
  sc_dom   the n most dominant superclass-selective (SSI > 0.3, not Tg) features (reference)
at all four stages simultaneously and per stage, and reports the drop in fine accuracy, superclass accuracy (head
argmax mapped to its superclass) and superclass nearest-centroid accuracy of the avgpool features (fit on even, test on
odd test images). Writes analysis_outputs/paper/TG_dominant/.
Usage: tg_dominant.py check|describe|ablate [--n N]
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)
torch.backends.mps.is_available = lambda: False
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import transfer_lib as T  # noqa: E402

sys.path.insert(0, str(T.REPO / "packages" / "ml" / "src"))
from sae import SparseAutoencoder  # noqa: E402
from cifar100_hierarchy import build_superclass_map  # noqa: E402

OUT = T.REPO / "analysis_outputs" / "paper" / "TG_dominant"
OUT.mkdir(parents=True, exist_ok=True)
STAGES = ["layer1", "layer2", "layer3", "layer4"]
EPOCHS = (5, 10, 30)


def lane_of(s):
    d = T.RUNS_EXTENDED / f"static_noise_s{s}" / "lanes" / "standard_resnet18"
    return {"dir": str(d), "arch": "ResNet-18", "group": "extended"}


def canon_sc():
    """Superclass index of each canonical CIFAR-100 class."""
    names = T.FINE_NAMES
    classes = [f"cifar100_{n}" for n in names]
    m = build_superclass_map(classes, {c: i for i, c in enumerate(classes)})
    scn = sorted(set(m.values()))
    return np.array([scn.index(m[i]) for i in range(100)])


class MeanPreservingProjection:
    """Forward hook: removes span(Q) from the channel vector at every position, keeping the population mean mu."""

    def __init__(self, Q, mu):
        Q, _ = torch.linalg.qr(Q)
        self.Q, self.mu = Q, mu

    def __call__(self, mod, inp, out):
        x = out - self.mu.view(1, -1, 1, 1)
        c = torch.einsum("bchw,ck->bkhw", x, self.Q)
        return out - torch.einsum("bkhw,ck->bchw", c, self.Q)


@torch.no_grad()
def forward(net, X, hooks=None):
    """Pooled stage outputs, avgpool features and logits for all images, with optional projection hooks."""
    feats = {s: [] for s in STAGES}
    hs = []
    for s, hk in (hooks or {}).items():
        hs.append(getattr(net, s).register_forward_hook(hk))
    hs += [getattr(net, s).register_forward_hook(lambda m, i, o, s=s: feats[s].append(o.mean(dim=(2, 3))))
           for s in STAGES]
    logits = []
    for b in range(0, len(X), T.BATCH):
        logits.append(net(T.cifar_batch(X[b:b + T.BATCH])))
    for h in hs:
        h.remove()
    F = {s: torch.cat(v) for s, v in feats.items()}
    return F, torch.cat(logits)


def sai_thr(lane):
    """The lane's null-calibrated Tg threshold used by the main pipeline (adaptive SAI threshold)."""
    md = json.load(open(Path(lane["dir"]) / "sae_analysis" / "sae_results.json"))["metadata"]
    return float(md["adaptive_thresholds"]["sai_adaptive_thresh"])


def load_sae(lane, label, stage):
    sd = torch.load(Path(lane["dir"]) / "sae_analysis" / "saes" / str(label) / f"{stage}.pt", map_location="cpu",
                    weights_only=False)
    d_in = sd["decoder.weight"].shape[0]
    sae = SparseAutoencoder(d_in, sd["encoder.weight"].shape[0] // d_in, 32)
    sae.load_state_dict(sd)
    sae.eval()
    return sae


def feature_table(x, y, sae, scv):
    """Per-feature SAI, SSI, alive flag, dominance; decoder directions (unit) and code h."""
    with torch.no_grad():
        h = sae.encode(x)
    D = sae.decoder.weight.detach()  # C x F
    dn = D.norm(dim=0).clamp(min=1e-8)
    n = x.shape[0]
    alive = ((h > T.ACT_EPS).float().mean(0) >= T.ALIVE_FRAC).numpy()
    M = torch.stack([h[y == c].mean(0) for c in range(100)], 1).numpy()  # F x 100 class means
    tot = M.sum(1) + 1e-12
    p = M / tot[:, None]
    sai = -(np.where(p > 0, p * np.log(p + 1e-12), 0)).sum(1) / np.log(100)
    S = np.stack([M[:, scv == k].sum(1) for k in range(20)], 1) / tot[:, None]
    ssi = np.clip((S.max(1) - 1 / 20) / (1 - 1 / 20), 0, None)
    xc = x - x.mean(0)
    trc = float((xc ** 2).sum() / n)
    dom = (h.var(0) * dn ** 2).numpy() / trc
    return {"h": h, "dir": (D / dn), "alive": alive, "sai": sai, "ssi": ssi, "dom": dom, "trc": trc}


def class_agnostic_pcs(x, y, k):
    """Top-k centred PCs after removing the between-class scatter (99%): the centred analogue of the Tg arm."""
    grand = x.mean(0)
    B = torch.zeros(x.shape[1], x.shape[1])
    for c in range(100):
        mu = x[y == c].mean(0) - grand
        B += (y == c).float().mean() * torch.outer(mu, mu)
    eb, Ub = torch.linalg.eigh(B)
    eb, Ub = eb.flip(0).clamp(min=0), Ub.flip(1)
    kk = int((torch.cumsum(eb, 0) / eb.sum() < 0.99).sum()) + 1
    U = Ub[:, :kk]
    R = (x - grand) - ((x - grand) @ U) @ U.T
    em, V = torch.linalg.eigh(R.T @ R / len(x))
    return V.flip(1)[:, :k]


def removed_var(x, Q):
    Q, _ = torch.linalg.qr(Q)
    xc = x - x.mean(0)
    return float(((xc @ Q) ** 2).sum() / len(x)) / float((xc ** 2).sum() / len(x))


def describe(cf, scv):
    rows = []
    for s in T.SEEDS:
        lane = lane_of(s)
        lab_of = {c["epoch"]: c["label"] for c in T.real_ckpts(T.checkpoint_info(Path(lane["dir"]), T._snap_labels(Path(lane["dir"]))))}
        for e in EPOCHS:
            net = T.load_net(lane, lab_of[e])
            F, _ = forward(net, cf["X"])
            for st in STAGES:
                x = F[st].float()
                ft = feature_table(x, torch.tensor(cf["y"]), load_sae(lane, lab_of[e], st), scv)
                tg = ft["alive"] & (ft["sai"] > sai_thr(lane))
                d = np.sort(ft["dom"][tg])[::-1]
                mean_dir = x.mean(0) / x.mean(0).norm()
                cos_mean = (ft["dir"][:, tg].T @ mean_dir).abs().numpy() if tg.any() else np.array([])
                order = np.argsort(-ft["dom"] * tg)
                top = order[:min(5, tg.sum())]
                pcs = class_agnostic_pcs(x, torch.tensor(cf["y"]), 5)
                ovl = (ft["dir"][:, top].T @ pcs).pow(2).sum(1).numpy() if len(top) else np.array([])
                rows.append({"seed": s, "epoch": e, "stage": st, "n_alive": int(ft["alive"].sum()), "n_tg": int(tg.sum()),
                             "tg_var_share_total": float(d.sum()),
                             "all_alive_var_share_total": float(ft["dom"][ft["alive"]].sum()),
                             "tg_top5_share_of_tg": float(d[:5].sum() / max(d.sum(), 1e-12)),
                             "tg_top10pct_share_of_tg": float(d[:max(1, len(d) // 10)].sum() / max(d.sum(), 1e-12)),
                             "tg_dom_max": float(d[0]) if len(d) else np.nan,
                             "tg_dom_median": float(np.median(d)) if len(d) else np.nan,
                             "tg_cos_mean_top5": float(np.mean(cos_mean[np.argsort(-ft['dom'][tg])[:5]])) if len(d) else np.nan,
                             "tg_cos_mean_all": float(np.mean(cos_mean)) if len(d) else np.nan,
                             "top5_tg_in_top5_agnostic_pcs": float(np.mean(ovl)) if len(ovl) else np.nan,
                             "chance_overlap": 5 / x.shape[1]})
            print(pd.DataFrame(rows[-4:])[["seed", "epoch", "stage", "n_alive", "n_tg", "tg_var_share_total",
                                            "tg_top5_share_of_tg", "tg_cos_mean_top5", "top5_tg_in_top5_agnostic_pcs"]].round(3).to_string(), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "describe.csv", index=False)
    print(df.groupby(["epoch", "stage"]).mean(numeric_only=True).drop(columns="seed").round(3).to_string())


def sc_centroid_acc(f, ysc):
    f = f - f.mean(0)
    tr, te = np.arange(0, len(f), 2), np.arange(1, len(f), 2)
    C = np.stack([f[tr][ysc[tr] == k].mean(0) for k in range(20)])
    d = ((f[te][:, None, :] - C[None]) ** 2).sum(-1)
    return 100.0 * float((d.argmin(1) == ysc[te]).mean())


def ablate(cf, scv, n):
    y = cf["y"]
    ysc = scv[y]
    rows = []
    for s in T.SEEDS:
        lane = lane_of(s)
        lab_of = {c["epoch"]: c["label"] for c in T.real_ckpts(T.checkpoint_info(Path(lane["dir"]), T._snap_labels(Path(lane["dir"]))))}
        c2c = T.lane_class_to_canonical({**lane, "label": lab_of[30]})
        for e in EPOCHS:
            net = T.load_net(lane, lab_of[e])
            F, logits = forward(net, cf["X"])
            base_pred = c2c[logits.argmax(1).numpy()]
            base = {"fine": 100.0 * float((base_pred == y).mean()), "sc": 100.0 * float((scv[base_pred] == ysc).mean())}
            sets, info = {}, {}
            g = torch.Generator().manual_seed(1000 * s + e)
            for st in STAGES:
                x = F[st].float()
                ft = feature_table(x, torch.tensor(y), load_sae(lane, lab_of[e], st), scv)
                tg = np.flatnonzero(ft["alive"] & (ft["sai"] > sai_thr(lane)))
                scf = np.flatnonzero(ft["alive"] & (ft["ssi"] > 0.3) & ~(ft["sai"] > sai_thr(lane)))
                k = min(n, len(tg) // 2)
                tg_sorted = tg[np.argsort(-ft["dom"][tg])]
                cand = {"tg_dom": ft["dir"][:, tg_sorted[:k]], "tg_uns": ft["dir"][:, tg_sorted[-k:]],
                        "sc_dom": ft["dir"][:, scf[np.argsort(-ft["dom"][scf])[:k]]],
                        "rnd_n": torch.randn(x.shape[1], k, generator=g),
                        "agn_pc": class_agnostic_pcs(x, torch.tensor(y), k)}
                target = removed_var(x, cand["tg_dom"]) if k else 0.0
                R = torch.randn(x.shape[1], x.shape[1], generator=g)
                m = 1
                while m < int(0.7 * x.shape[1]) and removed_var(x, R[:, :m]) < target:
                    m += 1
                cand["rnd_var"] = R[:, :m]
                mu = x.mean(0)
                for kind, Q in cand.items():
                    if Q.shape[1] > 0:
                        sets.setdefault(kind, {})[st] = (Q, mu)
                info[st] = {"k": k, "n_tg": len(tg), "n_sc": len(scf), "rnd_var_m": m,
                            **{f"var_{kd}": removed_var(x, Q) for kd, Q in cand.items() if Q.shape[1] > 0}}
            for kind, per in sets.items():
                for site, stages in [("all", STAGES)] + [(st, [st]) for st in STAGES]:
                    hooks = {st: MeanPreservingProjection(*per[st]) for st in stages if st in per}
                    if not hooks:
                        continue
                    Fa, la = forward(net, cf["X"], hooks)
                    pred = c2c[la.argmax(1).numpy()]
                    f_avg = Fa["layer4"].numpy()
                    rows.append({"seed": s, "epoch": e, "site": site, "kind": kind,
                                 "d_fine": base["fine"] - 100.0 * float((pred == y).mean()),
                                 "d_sc": base["sc"] - 100.0 * float((scv[pred] == ysc).mean()),
                                 "sc_centroid": sc_centroid_acc(f_avg, ysc),
                                 "sc_centroid_base": sc_centroid_acc(F["layer4"].numpy(), ysc),
                                 "base_fine": base["fine"],
                                 "var_removed": np.mean([info[st][f"var_{kind}"] for st in stages if f"var_{kind}" in info[st]]),
                                 "k": json.dumps({st: info[st]["k"] for st in stages}),
                                 "rnd_var_m": json.dumps({st: info[st]["rnd_var_m"] for st in stages})})
            print(s, e, round(base["fine"], 1), pd.DataFrame([r for r in rows if r["seed"] == s and r["epoch"] == e and r["site"] == "all"])[
                ["kind", "d_fine", "d_sc", "sc_centroid", "var_removed"]].round(2).to_string(index=False), flush=True)
    df = pd.DataFrame(rows)
    df["d_sc_centroid"] = df["sc_centroid_base"] - df["sc_centroid"]
    df.to_csv(OUT / f"ablate_n{n}.csv", index=False)
    print(df.groupby(["site", "epoch", "kind"])[["d_fine", "d_sc", "d_sc_centroid", "var_removed"]].mean().round(2).to_string())


def check_inputs(cf, scv):
    """Recomputed pooled activations vs the stored SAE activations (order and pooling)."""
    lane = lane_of(7)
    lab_of = {c["epoch"]: c["label"] for c in T.real_ckpts(T.checkpoint_info(Path(lane["dir"]), T._snap_labels(Path(lane["dir"]))))}
    net = T.load_net(lane, lab_of[5])
    F, _ = forward(net, cf["X"])
    for st in STAGES:
        a = np.load(Path(lane["dir"]) / "sae_analysis" / "activations" / str(lab_of[5]) / f"{st}.npy")
        with torch.no_grad():
            b = load_sae(lane, lab_of[5], st).encode(F[st].float()).numpy()
        print(st, a.shape, b.shape, "max abs diff", float(np.abs(a - b).max()), "corr", round(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]), 5),
              "same active set", round(float(((a > 0) == (b > 0)).mean()), 5))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["check", "describe", "ablate"])
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    cf = T.cifar_test()
    scv = canon_sc()
    {"check": lambda: check_inputs(cf, scv), "describe": lambda: describe(cf, scv),
     "ablate": lambda: ablate(cf, scv, a.n)}[a.stage]()
