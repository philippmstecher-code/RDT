"""Shared helpers for the transfer analysis (lane registry, CIFAR-100 / Tiny ImageNet data, network loading and
forward passes, ridge read-out), also imported by other analyses in scripts/paper.

Target task: Tiny ImageNet-200, 64 -> 32 px (area interpolation), CIFAR-100 normalisation; training set
100 images/class from the train split (seed 0, 20,000 images); test set = official val split (10,000).
Read-out: frozen pooled features of the last real checkpoint, standardised on the training set; ridge
classifier (alpha = 10, one-vs-rest least squares on {-1, +1} targets, intercept) = sklearn RidgeClassifier.

CPU only (CUDA/MPS hidden), <= 3 torch threads. Read-only on all data. Writes only to
analysis_outputs/paper/TRANSFER and /data/rdt/extended/transfer_cache.
"""
from __future__ import annotations

import os
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "3"

import json  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(3)
DEVICE = torch.device("cpu")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "packages" / "ml" / "src"))

from sae_outcomes import checkpoint_info, paired, real_ckpts, verdict_dir  # noqa: E402,F401
from cifar100_hierarchy import CIFAR100_SUPERCLASSES  # noqa: E402

# lesion.py helpers, loaded from the lesion checkout of the training code (variants/lesion)
_LESION_SRC = Path(RDT_DATA + "/checkouts/lesion/packages/ml/src")
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("es_lesion_readonly", _LESION_SRC / "lesion.py")
lesion = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesion)
orthonormal_basis = lesion.orthonormal_basis
build_projector = lesion.build_projector
_exact_subspace_variance_fraction = lesion._exact_subspace_variance_fraction
_greedy_variance_match = lesion._greedy_variance_match

OUT = REPO / "analysis_outputs" / "paper" / "TRANSFER"
CACHE = OUT / "cache"
SSD_CACHE = Path(RDT_DATA + "/extended/transfer_cache")
RUNS_EXTENDED = Path(RDT_DATA + "/extended/experiments")
CORE_EXP = Path(RDT_DATA + "/core/experiments")  # READ-ONLY
S42_COPY = RUNS_EXTENDED / "prevention_s42_copy" / "lanes"
CIFAR = RUNS_EXTENDED / "_cifar_src" / "datasets" / "cifar100" / "cifar-100-python"
TI = REPO / "data" / "datasets_cache" / "tiny-imagenet-200"
CORE_LANES = REPO / "analysis_outputs" / "paper" / "CORE_lanes" / "lanes.json"
SAE_outcomes_EN = REPO / "analysis_outputs" / "paper" / "SAE_outcomes" / "cache" / "EN"

SEEDS = [7, 42, 137, 256, 314]
CORE_SEEDS = [42, 137]
MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)
ALPHA = 10.0
N_TRAIN_PER_CLASS = 100
N_FEW = 10
FEW_DRAWS = 5
SAI_T = 0.9
ALIVE_FRAC = 0.01
ACT_EPS = 1e-6
VAR_TOL = 0.10
BATCH = 250

FINE_NAMES: List[str] = pickle.load(open(CIFAR / "meta", "rb"), encoding="latin1")["fine_label_names"]


# Lane registry

def _snap_labels(lane_dir: Path) -> List[str]:
    snaps = json.load(open(lane_dir / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"))
    return [str(i) for i in sorted(int(s["milestone_index"]) for s in snaps)]


def lane_status(lane_dir: Path) -> Dict[str, Any]:
    """complete = a 'final' snapshot exists and the last real checkpoint has checkpoint.pt."""
    f = lane_dir / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"
    if not f.exists():
        return {"complete": False, "reason": "no devtrain_snapshots.json"}
    snaps = json.load(open(f))
    if not any(s.get("snapshot_type") == "final" for s in snaps):
        return {"complete": False, "reason": "no final snapshot yet (training running)"}
    ci = real_ckpts(checkpoint_info(lane_dir, _snap_labels(lane_dir)))
    last = ci[-1]
    if not (lane_dir / "dev_snapshots" / f"milestone_{last['label']}" / "checkpoint.pt").exists():
        return {"complete": False, "reason": "checkpoint.pt missing"}
    return {"complete": True, "label": last["label"], "epoch": last["epoch"], "stored_val_acc": last["val_accuracy"],
            "n_real_ckpts": len(ci)}


def registry() -> List[Dict[str, Any]]:
    """All lanes of the transfer analysis. Incomplete lanes are kept and marked by lane_status."""
    L: List[Dict[str, Any]] = []

    def add(group, arm, seed, d, arch="ResNet-18", head="fine100"):
        L.append({"key": f"{group}__{arm}__s{seed}", "group": group, "arm": arm, "seed": seed, "dir": str(d),
                  "arch": arch, "head": head})

    sn = {"clean": "standard_resnet18", "between": "between_sc_resnet18_throughout",
          "within": "within_sc_resnet18_throughout", "matched": "between_sc_matched_resnet18_throughout"}
    for a, nm in sn.items():
        for s in SEEDS:
            add("static_noise", a, s, RUNS_EXTENDED / f"static_noise_s{s}" / "lanes" / nm)
    for a, nm in {"superclass": "lesion_scaffold_e00_seed{}_rp65_Llayer4-avgpool",
                  "random": "lesion_random_variance_e00_seed{}_rp65_Llayer4-avgpool"}.items():
        for s in SEEDS:
            # seed 42: prevention_s42_copy holds no checkpoint.pt (activations / predictions only); its
            # source lane (scaffold_lesion on the read-only drive) has them, with byte-identical
            # sample_predictions.pt at the last real checkpoint (checked).
            root = (CORE_EXP / "scaffold_lesion" / "lanes") if s == 42 else (RUNS_EXTENDED / "prevention" / "lanes")
            add("suppression", a, s, root / nm.format(s))
    for s in SEEDS:
        add("sham", "sham", s, RUNS_EXTENDED / "critical_sham" / "lanes" / f"sham_e00_seed{s}")
    for a in ("taskgeneral", "random"):
        for s in SEEDS:  # tg_suppression lanes (lane_status marks incomplete ones)
            add("tg", a, s, RUNS_EXTENDED / "tg_suppression" / "lanes" / f"{a}_e00_seed{s}")
    for a in ("sc_true_K20", "sc_rand_K20", "scratch_E30"):
        for s in SEEDS:
            add("curriculum", a, s, RUNS_EXTENDED / "curriculum" / "lanes" / f"{a}_seed{s}")
    for s in SEEDS:
        add("taskirr", "taskirr", s, RUNS_EXTENDED / "taskirr" / "lanes" / f"taskirr_within_index_seed{s}", head="within5")
    core = {ln["key"]: ln for ln in json.load(open(CORE_LANES))["lanes"] if ln["group"] == "core"}
    for arch in ("ResNet-18", "ViT-Small", "CCT-7"):
        for s in CORE_SEEDS:
            ln = core[f"core_{arch}_s{s}"]
            add("core", arch, s, ln["dir"], arch=arch)
    for ln in L:
        st = lane_status(Path(ln["dir"]))
        ln.update(st)
    return L


def core_thresholds(arch: str, seed: int) -> Dict[str, float]:
    core = {ln["key"]: ln for ln in json.load(open(CORE_LANES))["lanes"]}
    t = core[f"core_{arch}_s{seed}"]["thr_null"]
    return {"ssi": float(t["ssi"]), "csi": float(t["csi"]), "sai": SAI_T,
            "src": core[f"core_{arch}_s{seed}"]["thr_null_src"]}


def clean_thresholds(seed: int) -> Dict[str, float]:
    """Run-specific alive-only null thresholds, cached: key 'pooled' -> ssi/csi -> p95."""
    nl = json.load(open(SAE_outcomes_EN / f"clean__s{seed}_null.json"))
    return {"ssi": float(nl["pooled"]["ssi"]["p95"]), "csi": float(nl["pooled"]["csi"]["p95"]), "sai": SAI_T,
            "src": f"SAE_outcomes/cache/EN/clean__s{seed}_null.json pooled p95"}


# Data

def normalise(x01: torch.Tensor) -> torch.Tensor:
    m = torch.tensor(MEAN).view(1, 3, 1, 1)
    s = torch.tensor(STD).view(1, 3, 1, 1)
    return (x01 - m) / s


def cifar_test() -> Dict[str, np.ndarray]:
    """CIFAR-100 test set (10,000), canonical order (class-major, file order within class)."""
    with open(CIFAR / "test", "rb") as f:
        d = pickle.load(f, encoding="latin1")
    X = d["data"].reshape(-1, 3, 32, 32)
    y = np.array(d["fine_labels"])
    o = np.argsort(y, kind="stable")
    return {"X": X[o], "y": y[o]}


def cifar_batch(Xu8: np.ndarray) -> torch.Tensor:
    return normalise(torch.from_numpy(Xu8).float() / 255.0)


def _load_rgb64(p: Path) -> np.ndarray:
    from PIL import Image
    with Image.open(p) as im:
        a = np.asarray(im.convert("RGB"), dtype=np.uint8)
    assert a.shape == (64, 64, 3), (p, a.shape)
    return a


def tiny_imagenet() -> Dict[str, Any]:
    """Tiny ImageNet at 32 px, CIFAR-normalised float32 (N,3,32,32). Cached on the RUNS_EXTENDED."""
    f = SSD_CACHE / "tinyimagenet_32area_cifarnorm.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        return {k: z[k] for k in z.files}
    SSD_CACHE.mkdir(parents=True, exist_ok=True)
    wnids = sorted(p.name for p in (TI / "train").iterdir() if p.is_dir())
    assert len(wnids) == 200
    w2i = {w: i for i, w in enumerate(wnids)}
    rng = np.random.default_rng(0)
    tr_paths, tr_y = [], []
    for w in wnids:
        files = sorted((TI / "train" / w / "images").glob("*.JPEG"))
        assert len(files) == 500, (w, len(files))
        pick = np.sort(rng.choice(len(files), N_TRAIN_PER_CLASS, replace=False))
        tr_paths += [files[i] for i in pick]
        tr_y += [w2i[w]] * N_TRAIN_PER_CLASS
    te_paths, te_y = [], []
    for line in open(TI / "val" / "val_annotations.txt"):
        parts = line.strip().split("\t")
        te_paths.append(TI / "val" / "images" / parts[0])
        te_y.append(w2i[parts[1]])
    assert len(te_paths) == 10000

    def prep(paths):
        out = np.empty((len(paths), 3, 32, 32), dtype=np.float32)
        for s in range(0, len(paths), 1000):
            a = np.stack([_load_rgb64(p) for p in paths[s:s + 1000]])
            t = torch.from_numpy(a).permute(0, 3, 1, 2).float() / 255.0
            t = F.interpolate(t, size=(32, 32), mode="area")  # area interpolation, 64 -> 32
            out[s:s + len(a)] = normalise(t).numpy()
        return out

    d = {"Xtr": prep(tr_paths), "ytr": np.array(tr_y), "Xte": prep(te_paths), "yte": np.array(te_y),
         "wnids": np.array(wnids), "tr_files": np.array([str(p.relative_to(TI)) for p in tr_paths])}
    np.savez(f, **d)
    return d


# Models: site activations (where the SAE was trained / the lesion acts) and head

def build_model(arch: str, n_out: int) -> nn.Module:
    import torchvision.models as tvm
    if arch == "ResNet-18":
        m = tvm.resnet18(weights=None)
        m.fc = nn.Linear(512, n_out)
        return m
    if arch == "ViT-Small":
        # The core 'vit_small' lanes were trained with the 32-px config (patch 4, dim 192, 12 layers, 3 heads,
        # MLP 768; checkpoint shapes conv_proj (192,3,4,4), pos_embedding (1,65,192)) -- the config that
        # packages/ml/src/models.py calls 'vit_tiny'. Verified by reproducing the stored val accuracy.
        from torchvision.models.vision_transformer import VisionTransformer
        m = VisionTransformer(image_size=32, patch_size=4, num_layers=12, num_heads=3, hidden_dim=192, mlp_dim=768)
        m.heads.head = nn.Linear(192, n_out)
        return m
    if arch == "CCT-7":
        from models import create_network
        return create_network("cct_7", num_classes=n_out)
    raise ValueError(arch)


SITE = {"ResNet-18": "avgpool", "ViT-Small": "encoder.ln", "CCT-7": "norm"}


def load_net(lane: Dict[str, Any], label: Optional[str] = None) -> nn.Module:
    label = label or lane["label"]
    sd = torch.load(Path(lane["dir"]) / "dev_snapshots" / f"milestone_{label}" / "checkpoint.pt",
                    map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    head_key = {"ResNet-18": "fc.weight", "ViT-Small": "heads.head.weight", "CCT-7": "fc.weight"}[lane["arch"]]
    net = build_model(lane["arch"], int(sd[head_key].shape[0]))
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net


@torch.no_grad()
def forward_pooled(net: nn.Module, arch: str, x: torch.Tensor,
                   projectors: Optional[Dict[str, Optional[torch.Tensor]]] = None) -> Dict[str, Dict[str, np.ndarray]]:
    """Returns {name: {'feat': classifier-input features, 'logits': head output}} for each projector
    (None = unablated). The projector acts at the SAE site: ResNet avgpool vector; ViT encoder.ln output
    (all tokens; only CLS is read); CCT norm output, per token, before attention pooling (lesion-hook semantics)."""
    projectors = projectors or {"none": None}
    out = {}
    if arch == "ResNet-18":
        h = net.conv1(x); h = net.bn1(h); h = net.relu(h); h = net.maxpool(h)
        h = net.layer4(net.layer3(net.layer2(net.layer1(h))))
        f0 = torch.flatten(net.avgpool(h), 1)
        for k, P in projectors.items():
            f = f0 if P is None else f0 @ P.T
            out[k] = {"feat": f.numpy(), "logits": net.fc(f).numpy()}
    elif arch == "ViT-Small":
        h = net._process_input(x)
        h = torch.cat([net.class_token.expand(h.shape[0], -1, -1), h], dim=1)
        h = net.encoder(h)  # includes encoder.ln
        f0 = h[:, 0]
        for k, P in projectors.items():
            f = f0 if P is None else f0 @ P.T
            out[k] = {"feat": f.numpy(), "logits": net.heads(f).numpy()}
    elif arch == "CCT-7":
        t = net.norm(net.transformer(net.tokenizer(x)))
        for k, P in projectors.items():
            tt = t if P is None else t @ P.T
            w = torch.softmax(net.attention_pool(tt), dim=1)
            f = (tt * w).sum(dim=1)
            out[k] = {"feat": f.numpy(), "logits": net.fc(f).numpy()}
    else:
        raise ValueError(arch)
    return out


def run_images(net, arch, X, projectors=None, to_tensor=None) -> Dict[str, Dict[str, np.ndarray]]:
    acc: Dict[str, Dict[str, list]] = {}
    for s in range(0, len(X), BATCH):
        xb = X[s:s + BATCH]
        xb = to_tensor(xb) if to_tensor else torch.from_numpy(np.ascontiguousarray(xb))
        r = forward_pooled(net, arch, xb, projectors)
        for k, v in r.items():
            for kk, vv in v.items():
                acc.setdefault(k, {}).setdefault(kk, []).append(vv)
    return {k: {kk: np.concatenate(vv) for kk, vv in v.items()} for k, v in acc.items()}


def lane_class_to_canonical(lane: Dict[str, Any]) -> np.ndarray:
    """lane class index -> canonical CIFAR-100 fine index. RUNS_EXTENDED lanes use the canonical (torchvision) order;
    core lanes use their sae_results selected_classes order ('cifar100_<name>')."""
    if lane["group"] != "core":
        return np.arange(100)
    meta = json.load(open(Path(lane["dir"]) / "sae_analysis" / "sae_results.json"))["metadata"]
    sc = meta["selected_classes"]
    if isinstance(sc, str):
        import ast
        sc = ast.literal_eval(sc)
    names = [c.split("_", 1)[1] if c.startswith("cifar100_") else c for c in sc]
    return np.array([FINE_NAMES.index(n) for n in names])


def within_sc_index_of_canonical() -> np.ndarray:
    """taskirr label: index of a fine class within its true superclass (alphabetical), as task_irrelevant devtrain."""
    m = {}
    for fines in CIFAR100_SUPERCLASSES.values():
        for j, f in enumerate(sorted(fines)):
            m[f] = j
    return np.array([m[n] for n in FINE_NAMES])


def task_head_logits(lane: Dict[str, Any], feat: np.ndarray) -> np.ndarray:
    """taskirr lanes: checkpoint.pt carries the (swapped-in) 100-way fine head used for snapshot capture;
    the 5-way task head is milestone_k/task_head.pt. Returns task logits from avgpool features."""
    h = torch.load(Path(lane["dir"]) / "dev_snapshots" / f"milestone_{lane['label']}" / "task_head.pt",
                   map_location="cpu", weights_only=False)
    assert h["label_mode"] == "within_superclass_index"
    W, b = h["state_dict"]["weight"].numpy(), h["state_dict"]["bias"].numpy()
    return feat @ W.T + b


def in_task_accuracy(lane: Dict[str, Any], logits: np.ndarray, y_canon: np.ndarray,
                     feat: Optional[np.ndarray] = None) -> float:
    if lane["head"] == "within5":
        lg = task_head_logits(lane, feat)
        return float((lg.argmax(1) == within_sc_index_of_canonical()[y_canon]).mean() * 100)
    perm = lane_class_to_canonical(lane)
    return float((perm[logits.argmax(1)] == y_canon).mean() * 100)


# Ridge read-out (sklearn RidgeClassifier(alpha=10) equivalent, closed form)

def ridge_accuracy(Ftr: np.ndarray, ytr: np.ndarray, Fte: np.ndarray, yte: np.ndarray, alpha: float = ALPHA,
                   n_classes: int = 200) -> float:
    Ftr = Ftr.astype(np.float64)
    Fte = Fte.astype(np.float64)
    mu = Ftr.mean(0)
    sd = Ftr.std(0)
    sd[sd < 1e-12] = 1.0
    Ztr = (Ftr - mu) / sd
    Zte = (Fte - mu) / sd
    Y = -np.ones((len(ytr), n_classes))
    Y[np.arange(len(ytr)), ytr] = 1.0
    ym = Y.mean(0)
    W = np.linalg.solve(Ztr.T @ Ztr + alpha * np.eye(Ztr.shape[1]), Ztr.T @ (Y - ym))
    pred = (Zte @ W + ym).argmax(1)
    return float((pred == yte).mean() * 100)


def few_shot_draws(ytr: np.ndarray) -> List[np.ndarray]:
    out = []
    for d in range(FEW_DRAWS):
        rng = np.random.default_rng(1000 + d)
        idx = []
        for c in range(200):
            ii = np.flatnonzero(ytr == c)
            idx.append(np.sort(rng.choice(ii, N_FEW, replace=False)))
        out.append(np.concatenate(idx))
    return out


def transfer_scores(Ftr, ytr, Fte, yte, draws) -> Dict[str, float]:
    full = ridge_accuracy(Ftr, ytr, Fte, yte)
    few = [ridge_accuracy(Ftr[i], ytr[i], Fte, yte) for i in draws]
    return {"transfer_full": full, "transfer_few": float(np.mean(few)), "transfer_few_sd": float(np.std(few, ddof=1))}


# Stats helpers

def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not math.isfinite(v) else v
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o
