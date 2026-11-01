#!/usr/bin/env python3
"""Mean-shift diagnostic for the suppression results: how much of the cost of removing the superclass code comes
from shifting the activation mean? Analysis only, CPU only (2 threads), read-only on all data.

Per seed S in {7, 42, 137, 256, 314} and epoch e:
  window 1-10  (e in 1,2,3,5,10):   sham network critical_sham/sham_e00_seed{S}; coarse-arm ranks from
                                     prevention lesion_scaffold_e00_seed{S}_rp65_Llayer4-avgpool (seed 42: _s42_copy)
  window 11-20 (e in 11,12,13,15,20): sham_e10_seed{S}; ranks from critical lesion_scaffold_e10_seed{S}_..._Llayer4-avgpool
Ranks: per-stage total_rank of the last `reproject` report with epoch <= e (the projection in force when the epoch-e
checkpoint and validation accuracy were taken).
Subspaces on the sham network at epoch e:
  coarse  : superclass directions estimated iteratively on the projected pooled activations of CIFAR-100 test images
            0-19 of each class (between-superclass scatter, evr_floor 0.02, q_max 19 per step, accumulate) until the
            target rank is reached or no direction passes the floor (achieved rank reported);
  random  : random orthonormal directions of the target ranks; also random directions of the achieved coarse ranks
            ('random_ach', descriptive).
Removal at the layer4 output (per spatial position) and avgpool:
  uncentred x - QQ^T x            (= lesion.py ProjectionLesionHook)
  centred   x - QQ^T (x - mu)     (mu = per-channel mean of the unprojected activations of the estimation images,
                                   spatially pooled; all hooks off)
Outcome: fine-class accuracy of the network's own head on the 10,000 test images. xlayer (all five stages) is
evaluated for coarse lanes marked .lane_done. Writes analysis_outputs/paper/MEAN_shift/.

Usage: python scripts/paper/mean_shift.py [--seeds ...] [--part verify|lastblock|xlayer|all|run]
       python scripts/paper/mean_shift.py analyze
"""
import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

torch.set_num_threads(2)
torch.backends.mps.is_available = lambda: False
DEVICE = torch.device("cpu")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "packages" / "ml" / "src"))
from cifar100_hierarchy import build_superclass_map  # noqa: E402

# lesion.py of the training code (read only): ProjectionLesionHook = the hook semantics of prevention/critical/xlayer
_LESION_SRC = Path(RDT_DATA + "/checkouts/lesion/packages/ml/src/lesion.py")
_spec = importlib.util.spec_from_file_location("es_lesion_readonly", _LESION_SRC)
lesion = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesion)

EXP = Path(RDT_DATA + "/extended/experiments")
CIFAR = EXP / "_cifar_src" / "datasets" / "cifar100" / "cifar-100-python"
OUT = REPO / "analysis_outputs" / "paper" / "MEAN_shift"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [7, 42, 137, 256, 314]
WINDOWS = {"w01_10": {"epochs": [1, 2, 3, 5, 10], "sham": "sham_e00"},
           "w11_20": {"epochs": [11, 12, 13, 15, 20], "sham": "sham_e10"}}
LAST = ["layer4", "avgpool"]
ALL = ["layer1", "layer2", "layer3", "layer4", "avgpool"]
MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)
Q_MAX, EVR_FLOOR = 19, 0.02
NUM_ZERO = 1e-10  # numerical-zero guard: an eigenvalue must also exceed 1e-10 x the stage's initial between-SC variance
MAX_PASS = 300
BATCH = 500


# data
def cifar_test():
    with open(CIFAR / "test", "rb") as f:
        d = pickle.load(f, encoding="latin1")
    X = d["data"].reshape(-1, 3, 32, 32)
    y = np.array(d["fine_labels"])
    o = np.argsort(y, kind="stable")  # class-major, file order within class (= stored sample_predictions order)
    return X[o], y[o]


def to_input(Xu8):
    x = torch.from_numpy(Xu8).float() / 255.0
    return (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)


FINE = pickle.load(open(CIFAR / "meta", "rb"), encoding="latin1")["fine_label_names"]
_scm = build_superclass_map(FINE, {c: i for i, c in enumerate(FINE)})
_scn = sorted(set(_scm.values()))
assert len(_scn) == 20 and "unknown" not in _scn
CLS2SC = np.array([_scn.index(_scm[i]) for i in range(100)])


# lanes
def sham_dir(win, seed):
    return EXP / "critical_sham" / "lanes" / f"{WINDOWS[win]['sham']}_seed{seed}"


def coarse_dir(win, seed, xlayer=False):
    if xlayer:
        return EXP / "xlayer" / "lanes" / f"lesion_scaffold_e00_seed{seed}_rp65_Llayer1-layer2-layer3-layer4-avgpool"
    if win == "w01_10":
        root = "prevention_s42_copy" if seed == 42 else "prevention"
        return EXP / root / "lanes" / f"lesion_scaffold_e00_seed{seed}_rp65_Llayer4-avgpool"
    return EXP / "critical" / "lanes" / f"lesion_scaffold_e10_seed{seed}_rp65_Llayer4-avgpool"


def snapshots(lane):
    return json.load(open(lane / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"))


def epoch_label(lane, e):
    s = [x for x in snapshots(lane) if x["snapshot_type"] == "intermediate" and int(x["epoch"]) == e]
    assert len(s) == 1, (lane, e, len(s))
    return str(s[0]["milestone_index"]), float(s[0]["val_accuracy"])


def ranks_at(lane, e, stages):
    rows = [json.loads(l) for l in open(lane / "dev_snapshots" / "metrics" / "lesion_log.jsonl")]
    rp = [r for r in rows if r["event"] == "reproject" and int(r["epoch"]) <= e]
    assert rp, (lane, e)
    last = rp[-1]
    return {s: int(last["report"][s]["total_rank"]) for s in stages}, int(last["weight_updates"]), int(last["epoch"])


def load_net(lane, label):
    import torchvision.models as tvm
    sd = torch.load(lane / "dev_snapshots" / f"milestone_{label}" / "checkpoint.pt", map_location="cpu",
                    weights_only=False)
    sd = sd.get("model_state_dict", sd)
    m = tvm.resnet18(weights=None)
    m.fc = nn.Linear(512, int(sd["fc.weight"].shape[0]))
    m.load_state_dict(sd, strict=True)
    return m.eval()


def stored_predictions(lane, label):
    p = torch.load(lane / "dev_snapshots" / f"milestone_{label}" / "sample_predictions.pt", map_location="cpu",
                   weights_only=False)
    return np.concatenate([p[c]["predictions"].numpy() for c in range(100)])


# hooks
class CentredHook:
    """x <- x - QQ^T (x - mu), per spatial position."""

    def __init__(self, Q, mu):
        self.QQ = Q @ Q.T
        self.off = self.QQ @ mu
        self.P = torch.eye(Q.shape[0]) - self.QQ

    def __call__(self, module, inputs, output):
        if output.dim() == 4:
            return torch.einsum("dc,bchw->bdhw", self.P, output) + self.off.view(1, -1, 1, 1)
        return output @ self.P.T + self.off


def module_of(net, s):
    return getattr(net, s)


@torch.no_grad()
def forward(net, X, spec=None, centred=False, mus=None, capture=None, capture_idx=None):
    """spec: {stage: Q}. Returns logits (N,100) and, if capture, {stage: pooled post-hook activations of capture_idx}."""
    spec = spec or {}
    hs = []
    for s, Q in spec.items():
        if Q is None or Q.shape[1] == 0:
            continue
        if centred:
            hk = CentredHook(Q, mus[s])
        else:
            hk = lesion.ProjectionLesionHook(torch.eye(Q.shape[0]) - Q @ Q.T, s)
        hs.append(module_of(net, s).register_forward_hook(hk))
    cap = {s: [] for s in (capture or [])}
    for s in cap:  # registered AFTER the projection hooks: sees the projected output
        hs.append(module_of(net, s).register_forward_hook(
            lambda m, i, o, s=s: cap[s].append(o.mean(dim=(2, 3)) if o.dim() == 4 else o)))
    Xs = X if capture_idx is None else X[capture_idx]
    out = []
    try:
        for b in range(0, len(Xs), BATCH):
            out.append(net(to_input(Xs[b:b + BATCH])))
    finally:
        for h in hs:
            h.remove()
    return torch.cat(out), {s: torch.cat(v) for s, v in cap.items()}


@torch.no_grad()
def layer4_raw(net, X):
    """Base logits and raw (unprojected) layer4 output for all images."""
    h4 = []
    hk = net.layer4.register_forward_hook(lambda m, i, o: h4.append(o.clone()))
    logits = []
    try:
        for b in range(0, len(X), BATCH):
            logits.append(net(to_input(X[b:b + BATCH])))
    finally:
        hk.remove()
    return torch.cat(logits), torch.cat(h4)


def proj_apply(x, Q, mu, centred):
    """x (N,C,H,W) or (N,C); per-position channel projection."""
    if Q is None or Q.shape[1] == 0:
        return x
    QQ = Q @ Q.T
    if x.dim() == 4:
        if centred:
            return x - torch.einsum("dc,bchw->bdhw", QQ, x - mu.view(1, -1, 1, 1))
        return x - torch.einsum("dc,bchw->bdhw", QQ, x)
    if centred:
        return x - (x - mu) @ QQ.T
    return x - x @ QQ.T


@torch.no_grad()
def lastblock_logits(net, h4, spec, centred, mus):
    z = proj_apply(h4, spec.get("layer4"), mus.get("layer4"), centred)
    z = net.avgpool(z).flatten(1)
    z = proj_apply(z, spec.get("avgpool"), mus.get("avgpool"), centred)
    return net.fc(z)


# estimation
def between_sc(X, ysc):
    X = X.double()
    grand = X.mean(0)
    B = torch.zeros(X.shape[1], X.shape[1], dtype=torch.float64)
    n = X.shape[0]
    for s in ysc.unique():
        m = ysc == s
        mu = X[m].mean(0) - grand
        B += (m.sum().item() / n) * torch.outer(mu, mu)
    ev, V = torch.linalg.eigh(B)
    return ev.flip(0).clamp(min=0), V.flip(1)


def extend_basis(Q_old, new_dirs, max_rank):
    """Copy of lesion.LesionController._extend_basis with rank_policy='accumulate'."""
    room = max_rank - Q_old.shape[1]
    if room <= 0 or new_dirs.numel() == 0:
        return Q_old, 0
    norms = new_dirs.norm(dim=0)
    new_dirs = new_dirs[:, norms > 1e-8] / norms[norms > 1e-8]
    if new_dirs.numel() == 0:
        return Q_old, 0
    R = new_dirs - Q_old @ (Q_old.T @ new_dirs)
    U, S, _ = torch.linalg.svd(R, full_matrices=False)
    U = U[:, S > 1e-3][:, :room]
    if U.shape[1] == 0:
        return Q_old, 0
    return torch.cat([Q_old, U], 1), int(U.shape[1])


def estimate_coarse(get_acts, stages, target, ysc, C):
    Q = {s: torch.zeros(C[s], 0) for s in stages}
    b0 = {}
    trace = []
    for p in range(MAX_PASS):
        A = get_acts(Q)
        added, rec = 0, {}
        for s in stages:
            if Q[s].shape[1] >= target[s]:
                continue
            ev, V = between_sc(A[s], ysc)
            tot = float(ev.sum())
            if s not in b0:
                b0[s] = tot
            evr = ev / max(tot, 1e-300)
            n_floor = int(((evr >= EVR_FLOOR) & (ev >= NUM_ZERO * b0[s])).sum())
            n_floor_raw = int((evr >= EVR_FLOOR).sum())  # what the floor alone would pass (incl. numerical zeros)
            q = min(Q_MAX, n_floor)
            Q[s], n = extend_basis(Q[s], V[:, :q].float().contiguous(), target[s])
            added += n
            rec[s] = {"rank": int(Q[s].shape[1]), "n_added": n, "q": q, "q_floor_only": min(Q_MAX, n_floor_raw),
                      "bsc_rel": tot / b0[s]}
        trace.append(rec)
        if added == 0:
            break
    return Q, trace


def random_basis(C, r, seed):
    if r == 0:
        return torch.zeros(C, 0)
    g = torch.Generator().manual_seed(int(seed))
    Q, _ = torch.linalg.qr(torch.randn(C, r, generator=g))
    return Q


def acc(logits, y):
    return 100.0 * float((logits.argmax(1).numpy() == y).mean())


# main per network
def run_one(net, X, y, idx, target, stages, rseed, lastblock):
    ysc = torch.as_tensor(CLS2SC[y[idx]])
    if lastblock:
        base_logits, h4 = layer4_raw(net, X)
        h4e = h4[idx]
        C = {"layer4": h4.shape[1], "avgpool": h4.shape[1]}
        p4 = h4e.mean(dim=(2, 3))
        mus = {"layer4": p4.mean(0), "avgpool": p4.mean(0)}

        def get_acts(Q):
            z4 = proj_apply(h4e, Q["layer4"], None, False)
            za = proj_apply(z4.mean(dim=(2, 3)), Q["avgpool"], None, False)
            return {"layer4": z4.mean(dim=(2, 3)), "avgpool": za}

        def logits_of(spec, centred):
            return lastblock_logits(net, h4, spec, centred, mus)
    else:
        base_logits, _ = forward(net, X)
        _, A0 = forward(net, X, capture=stages, capture_idx=idx)
        C = {s: A0[s].shape[1] for s in stages}
        mus = {s: A0[s].mean(0) for s in stages}

        def get_acts(Q):
            return forward(net, X, spec=Q, capture=stages, capture_idx=idx)[1]

        def logits_of(spec, centred):
            return forward(net, X, spec=spec, centred=centred, mus=mus)[0]

    Qc, trace = estimate_coarse(get_acts, stages, target, ysc, C)
    ach = {s: int(Qc[s].shape[1]) for s in stages}
    Qr = {s: random_basis(C[s], target[s], rseed * 100 + k) for k, s in enumerate(stages)}
    Qra = {s: random_basis(C[s], ach[s], rseed * 100 + 50 + k) for k, s in enumerate(stages)}
    res = {"A_sham": acc(base_logits, y), "base_logits": base_logits, "achieved": ach, "trace": trace,
           "passes": len(trace), "Qc": Qc, "Qr": Qr, "mus": mus}
    for name, Q in (("coarse", Qc), ("random", Qr), ("random_ach", Qra)):
        for ver, cen in (("unc", False), ("cen", True)):
            res[f"A_{name}_{ver}"] = acc(logits_of(Q, cen), y)
    # residual between-SC variance (relative) after the coarse estimate, per stage
    Af = get_acts(Qc)
    A0 = get_acts({s: torch.zeros(C[s], 0) for s in stages})
    res["bsc_resid"] = {s: float(between_sc(Af[s], ysc)[0].sum() / between_sc(A0[s], ysc)[0].sum()) for s in stages}
    return res


def verify_fastpath(net, X, y, res):
    """Fast last-block path vs full forward with ProjectionLesionHook / CentredHook at layer4 + avgpool."""
    out = {}
    for name in ("coarse", "random"):
        Q = res["Qc"] if name == "coarse" else res["Qr"]
        for ver, cen in (("unc", False), ("cen", True)):
            lg, _ = forward(net, X, spec=Q, centred=cen, mus=res["mus"])
            out[f"{name}_{ver}"] = {"full": acc(lg, y), "fast": res[f"A_{name}_{ver}"]}
    return out


def tci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 2:
        return {"mean": float(v.mean()) if n else np.nan, "lo": np.nan, "hi": np.nan, "n": n}
    from scipy import stats
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return {"mean": float(v.mean()), "lo": float(v.mean() - h), "hi": float(v.mean() + h), "n": n,
            "n_pos": int((v > 0).sum()), "n_neg": int((v < 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--part", choices=["verify", "lastblock", "xlayer", "all", "run"], default="run")
    a = ap.parse_args()
    X, y = cifar_test()
    idx = np.concatenate([np.flatnonzero(y == c)[:20] for c in range(100)])
    log = open(OUT / "run_log.txt", "a")

    def P(*m):
        s = " ".join(str(x) for x in m)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    P(f"=== run {time.strftime('%Y-%m-%d %H:%M:%S')} part={a.part} seeds={a.seeds}")

    # 1. normalisation check against stored predictions (all sham checkpoints used)
    if a.part in ("verify", "all"):
        vrows = []
        for win, W in WINDOWS.items():
            for s in a.seeds:
                lane = sham_dir(win, s)
                for e in W["epochs"]:
                    lab, va = epoch_label(lane, e)
                    net = load_net(lane, lab)
                    lg, _ = forward(net, X)
                    pr = lg.argmax(1).numpy()
                    sp = stored_predictions(lane, lab)
                    vrows.append({"window": win, "seed": s, "epoch": e, "label": lab,
                                  "agreement": float((pr == sp).mean()), "acc_recomputed": acc(lg, y),
                                  "acc_stored_preds": 100.0 * float((sp == y).mean()), "val_acc_logged": va})
                    P("verify", vrows[-1])
        vdf = pd.DataFrame(vrows)
        vdf.to_csv(OUT / "verify_predictions.csv", index=False)
        P("min agreement", vdf.agreement.min(), "PASS" if vdf.agreement.min() >= 0.999 else "FAIL")
        if vdf.agreement.min() < 0.999:
            P("ABORT: normalisation check failed")
            return

    # 2. last block, both windows
    if a.part in ("lastblock", "all", "run"):
        rows, vf = [], []
        for win, W in WINDOWS.items():
            for s in a.seeds:
                sl, cl = sham_dir(win, s), coarse_dir(win, s)
                for e in W["epochs"]:
                    t0 = time.time()
                    lab, va_sham = epoch_label(sl, e)
                    _, va_coarse = epoch_label(cl, e)
                    target, wu, rep_ep = ranks_at(cl, e, LAST)
                    net = load_net(sl, lab)
                    r = run_one(net, X, y, idx, target, LAST, rseed=s * 1000 + e, lastblock=True)
                    # normalisation check on the same forward pass (stored predictions of the sham checkpoint)
                    agree = float((r["base_logits"].argmax(1).numpy() == stored_predictions(sl, lab)).mean())
                    P("verify", {"window": win, "seed": s, "epoch": e, "label": lab, "agreement": agree,
                                 "acc_recomputed": r["A_sham"], "val_acc_logged": va_sham})
                    if agree < 0.999:
                        P("ABORT: normalisation check failed", win, s, e, agree)
                        return
                    if s == a.seeds[0] and e in (1, 11):
                        v = verify_fastpath(net, X, y, r)
                        vf.append({"window": win, "seed": s, "epoch": e, **{k: json.dumps(x) for k, x in v.items()}})
                        P("fastpath check", win, s, e, v)
                    row = {"config": "lastblock", "window": win, "seed": s, "epoch": e, "sham_label": lab,
                           "pred_agreement": agree,
                           "rank_report_epoch": rep_ep, "rank_report_wu": wu,
                           **{f"target_{k}": v for k, v in target.items()},
                           **{f"achieved_{k}": v for k, v in r["achieved"].items()},
                           **{f"bsc_resid_{k}": v for k, v in r["bsc_resid"].items()},
                           "passes": r["passes"], "A_sham": r["A_sham"], "val_sham_logged": va_sham,
                           "val_coarse_logged": va_coarse,
                           **{k: v for k, v in r.items() if k.startswith("A_") and k != "A_sham"}}
                    rows.append(row)
                    json.dump(r["trace"], open(OUT / f"trace_lastblock_{win}_s{s}_e{e}.json", "w"))
                    P(win, s, e, "target", target, "ach", r["achieved"], "A_sham %.2f" % r["A_sham"],
                      "coarse unc/cen %.2f/%.2f" % (r["A_coarse_unc"], r["A_coarse_cen"]),
                      "random unc/cen %.2f/%.2f" % (r["A_random_unc"], r["A_random_cen"]),
                      "rand_ach unc/cen %.2f/%.2f" % (r["A_random_ach_unc"], r["A_random_ach_cen"]),
                      "%.0fs" % (time.time() - t0))
        df = pd.DataFrame(rows)
        df.to_csv(OUT / "lastblock_raw.csv", index=False)
        pd.DataFrame(vf).to_csv(OUT / "fastpath_check.csv", index=False)

    # 3. xlayer, completed coarse lanes only
    if a.part in ("xlayer", "all", "run"):
        rows, status = [], {}
        for s in SEEDS:
            cl = coarse_dir("w01_10", s, xlayer=True)
            done = (cl / ".lane_done").exists()
            status[s] = "DONE" if done else "NOT DONE"
            if not done or s not in a.seeds:
                continue
            sl = sham_dir("w01_10", s)
            for e in WINDOWS["w01_10"]["epochs"]:
                t0 = time.time()
                lab, _ = epoch_label(sl, e)
                target, wu, rep_ep = ranks_at(cl, e, ALL)
                net = load_net(sl, lab)
                r = run_one(net, X, y, idx, target, ALL, rseed=s * 1000 + e + 500, lastblock=False)
                rows.append({"config": "xlayer", "window": "w01_10", "seed": s, "epoch": e, "sham_label": lab,
                             "rank_report_epoch": rep_ep, "rank_report_wu": wu,
                             **{f"target_{k}": v for k, v in target.items()},
                             **{f"achieved_{k}": v for k, v in r["achieved"].items()},
                             **{f"bsc_resid_{k}": v for k, v in r["bsc_resid"].items()},
                             "passes": r["passes"], "A_sham": r["A_sham"],
                             **{k: v for k, v in r.items() if k.startswith("A_") and k != "A_sham"}})
                json.dump(r["trace"], open(OUT / f"trace_xlayer_s{s}_e{e}.json", "w"))
                P("xlayer", s, e, "target", target, "ach", r["achieved"], {k: round(v, 2) for k, v in rows[-1].items()
                                                                            if k.startswith("A_")},
                  "%.0fs" % (time.time() - t0))
        json.dump(status, open(OUT / "xlayer_status.json", "w"))
        P("xlayer status", status)
        if rows:
            pd.DataFrame(rows).to_csv(OUT / "xlayer_raw.csv", index=False)


def derive(df):
    d = df.copy()
    for k in ("coarse", "random", "random_ach"):
        for v in ("unc", "cen"):
            d[f"cost_{k}_{v}"] = d["A_sham"] - d[f"A_{k}_{v}"]
    d["F_coarse"] = (d.cost_coarse_unc - d.cost_coarse_cen) / d.cost_coarse_unc
    d["F_random"] = (d.cost_random_unc - d.cost_random_cen) / d.cost_random_unc
    d["info_cen"] = d.cost_coarse_cen - d.cost_random_cen          # information component (random @ target rank)
    d["info_cen_ach"] = d.cost_coarse_cen - d.cost_random_ach_cen  # descriptive (random @ achieved coarse rank)
    d["info_unc"] = d.cost_coarse_unc - d.cost_random_unc
    return d


def analyze():
    df = derive(pd.read_csv(OUT / "lastblock_raw.csv"))
    df.to_csv(OUT / "lastblock_derived.csv", index=False)
    summ = {}
    for win in WINDOWS:
        w = df[df.window == win]
        first3 = sorted(w.epoch.unique())[:3]
        ws = w[w.epoch.isin(first3)].groupby("seed")
        per_seed = pd.DataFrame({
            "F_mean_e": ws.F_coarse.mean(),
            "F_ratio_of_means": (ws.cost_coarse_unc.mean() - ws.cost_coarse_cen.mean()) / ws.cost_coarse_unc.mean(),
            "cost_coarse_unc": ws.cost_coarse_unc.mean(), "cost_coarse_cen": ws.cost_coarse_cen.mean(),
            "cost_random_unc": ws.cost_random_unc.mean(), "cost_random_cen": ws.cost_random_cen.mean(),
            "cost_random_ach_cen": ws.cost_random_ach_cen.mean(),
            "info_cen": ws.info_cen.mean(), "info_cen_ach": ws.info_cen_ach.mean(), "info_unc": ws.info_unc.mean(),
            "F_random": ws.F_random.mean()})
        summ[win] = {"epochs": [int(x) for x in first3], "per_seed": per_seed.round(4).to_dict(orient="index"),
                     **{c: tci(per_seed[c].values) for c in per_seed.columns}}
        per_ep = {}
        for e, g in w.groupby("epoch"):
            per_ep[int(e)] = {c: tci(g[c].values) for c in
                              ["F_coarse", "cost_coarse_unc", "cost_coarse_cen", "cost_random_unc", "cost_random_cen",
                               "cost_random_ach_unc", "cost_random_ach_cen", "info_cen", "info_cen_ach", "info_unc",
                               "A_sham", "achieved_layer4", "achieved_avgpool", "target_layer4", "target_avgpool"]}
        summ[win]["per_epoch"] = per_ep
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1, default=float)
    if (OUT / "xlayer_raw.csv").exists():
        derive(pd.read_csv(OUT / "xlayer_raw.csv")).to_csv(OUT / "xlayer_derived.csv", index=False)
    return df, summ


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze()
    else:
        main()
