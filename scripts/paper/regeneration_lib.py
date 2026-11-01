"""Regeneration of the coarse code after suppression: shared helpers.

Read-only on all data volumes. CPU only. Writes only to analysis_outputs/paper/REGENERATION/.
"""
import json
import os
import pickle
from pathlib import Path
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "2")

import numpy as np
import torch

torch.set_num_threads(2)

RUNS_EXTENDED = Path(RDT_DATA + "/extended/experiments")
S42_ORIG = Path(RDT_DATA + "/core/experiments/scaffold_lesion/lanes")
OUT = Path(__file__).resolve().parents[2] / "analysis_outputs" / "paper" / "REGENERATION"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [7, 42, 137, 256, 314]
WINDOWS = {"1-10": 0, "11-20": 10, "21-30": 20}  # window -> branch epoch
CIFAR = RUNS_EXTENDED / "_cifar_src" / "datasets" / "cifar100" / "cifar-100-python"


def lane_dir(window, arm, seed):
    e = WINDOWS[window]
    if arm == "sham":
        # no sham exists for 21-30: sham_e10 (resumed at epoch 10, no projection) is used for epochs 21-30
        return RUNS_EXTENDED / "critical_sham" / "lanes" / f"sham_e{0 if e == 0 else 10:02d}_seed{seed}"
    if arm == "taskgeneral":  # task-general suppression lanes, window 1-10 only, seeds 7/42/137/256
        assert e == 0
        return RUNS_EXTENDED / "tg_suppression" / "lanes" / f"lesion_taskgeneral_e00_seed{seed}_rp65_Llayer4-avgpool"
    cond = {"coarse": "scaffold", "random": "random_variance"}[arm]
    name = f"lesion_{cond}_e{e:02d}_seed{seed}_rp65_Llayer4-avgpool"
    if e == 0:
        if seed == 42:
            return RUNS_EXTENDED / "prevention_s42_copy" / "lanes" / name
        return RUNS_EXTENDED / "prevention" / "lanes" / name
    return RUNS_EXTENDED / "critical" / "lanes" / name


def milestones(lane):
    """{global epoch: milestone dir}; t0 snapshots and duplicate terminal captures dropped."""
    snaps = json.load(open(lane / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"))
    out = {}
    for s in snaps:
        if s["weight_updates"] == 0:
            continue
        ep = s["epoch"]
        if ep not in out:
            out[ep] = lane / "dev_snapshots" / f"milestone_{s['milestone_index']}"
    return out


def ckpt_path(mdir):
    p = mdir / "checkpoint.pt"
    if p.exists():
        return p
    alt = S42_ORIG / mdir.parent.parent.name / "dev_snapshots" / mdir.name / "checkpoint.pt"
    return alt if alt.exists() else None


def load_acts(mdir, layer="avgpool"):
    a = torch.load(mdir / "individual_multilayer_activations.pt", map_location="cpu", weights_only=False)
    X = torch.cat([a[layer][c] for c in sorted(a[layer])]).double().numpy()
    y = np.concatenate([np.full(a[layer][c].shape[0], c) for c in sorted(a[layer])])
    return X, y


_test = None


def cifar_test():
    global _test
    if _test is None:
        d = pickle.load(open(CIFAR / "test", "rb"), encoding="latin1")
        x = np.asarray(d["data"], dtype=np.float32).reshape(-1, 3, 32, 32) / 255.0
        mean = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)[None, :, None, None]
        std = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)[None, :, None, None]
        x = (x - mean) / std  # CIFAR-100 statistics (verified against the stored activations)
        fine = np.asarray(d["fine_labels"])
        coarse = np.asarray(d["coarse_labels"])
        order = np.concatenate([np.where(fine == c)[0] for c in range(100)])  # class-major, dataset order
        sc_of = np.array([coarse[fine == c][0] for c in range(100)])
        assert all(len(set(coarse[fine == c])) == 1 for c in range(100))
        _test = (torch.from_numpy(x[order]), fine[order], sc_of)
    return _test


def sc_of_class():
    return cifar_test()[2]


def build_model():
    import torch.nn as nn
    import torchvision.models as models
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(512, 100)
    return m


@torch.no_grad()
def raw_avgpool(ckpt, bs=500):
    """Un-projected avgpool features (no lesion hooks) of the 10k test images, class-major order."""
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) and "model_state_dict" in sd else sd
    m = build_model()
    m.load_state_dict(sd)
    m.eval()
    x = cifar_test()[0]
    feats = []
    body = torch.nn.Sequential(*list(m.children())[:-1])
    for i in range(0, x.shape[0], bs):
        feats.append(body(x[i:i + bs]).flatten(1))
    W, b = m.fc.weight.double().numpy(), m.fc.bias.double().numpy()
    return torch.cat(feats).double().numpy(), W, b


def fc_weights(ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    return sd["fc.weight"].double().numpy(), sd["fc.bias"].double().numpy()


def null_basis(X, tol=1e-6):
    """Orthonormal basis (d x k) of the null space of rows of X (the projected-out subspace)."""
    _, S, Vt = np.linalg.svd(X, full_matrices=False)  # X is n x 512 with n > 512: Vt is complete
    keep = S > tol * S[0]
    r = int(keep.sum())
    return Vt[r:].T, r


def scatter(X, labels):
    mu = X.mean(0)
    T = (X - mu).T @ (X - mu) / len(X)
    B = np.zeros_like(T)
    for g in np.unique(labels):
        m = labels == g
        d = X[m].mean(0) - mu
        B += m.mean() * np.outer(d, d)
    return T, B


def part_ratio(X):
    ev = np.clip(np.linalg.eigvalsh(np.cov(X.T)), 0, None)
    return ev.sum() ** 2 / (ev ** 2).sum()


def split_idx(y, est=20, train_end=60):
    """Per class: images 0..est-1 were used by the run to estimate the removed directions -> excluded.
    Probe train on est..train_end-1, test on train_end..99."""
    pos = np.concatenate([np.arange((y == c).sum()) for c in np.unique(y)])
    return (pos >= est) & (pos < train_end), pos >= train_end


def ridge_acc(X, lab, tr, te, alpha=1.0):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X[tr])
    clf = RidgeClassifier(alpha=alpha).fit(sc.transform(X[tr]), lab[tr])
    return float((clf.predict(sc.transform(X[te])) == lab[te]).mean()), clf, sc


def logreg_acc(X, lab, tr, te, C=0.1):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(C=C, max_iter=300).fit(sc.transform(X[tr]), lab[tr])
    return float((clf.predict(sc.transform(X[te])) == lab[te]).mean())


def knn_acc(X, lab, tr, te, k=10):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = Xn[te] @ Xn[tr].T
    nn_idx = np.argpartition(-S, k, axis=1)[:, :k]
    lt = lab[tr][nn_idx]
    pred = np.array([np.bincount(r).argmax() for r in lt])
    return float((pred == lab[te]).mean())


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y) ** 2
    return float(hsic / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def sc_rsm(X, y):
    sc = sc_of_class()[y]
    C = np.stack([X[sc == s].mean(0) for s in range(20)])
    C = C - C.mean(0)
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    return C @ C.T


def rsm_corr(A, B):
    iu = np.triu_indices(A.shape[0], 1)
    return float(np.corrcoef(A[iu], B[iu])[0, 1])
