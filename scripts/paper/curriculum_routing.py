#!/usr/bin/env python3
"""Routing analyses of the controlled-curriculum head start (stored checkpoints only; no training; CPU, 2 threads).

Fine accuracy factorises as P(fine) = P(superclass right) x P(fine right | superclass right). Does coarse pre-training
give fine-class learning its head start through superclass assignment (routing), and does it shorten the stage in
which the classes' learning signal is shared with their superclass siblings?

Design (Methods, 'Controlled curriculum experiment'; lanes curriculum, seeds 7, 42, 137, 256, 314):
  primary   K20: sc_true_K20 / sc_rand_K20 (20 pre-training epochs on the 20 true superclasses / on the seed's fixed
            random partition into 20 groups of five, then 30 fine-class epochs with a warm restart and a fresh 30-epoch
            cosine anneal) vs scratch_E30 (30 fine-class epochs, same 30-epoch cosine anneal, same initialisation).
  secondary K30: sc_true_K30 / sc_rand_K30 (30 + 20) vs scratch_E20 (20 fine-class epochs).
Alignment. Fine-class epoch e = global epoch K + e for the pre-trained arms and epoch e for scratch, exactly as
  Fig. 6b / Extended Data Fig. 9 (curriculum_lib.fine_val); checked at run time: the fine phase starts at K + 1 in
  lane.json, the learning rate at fine epoch e is identical across the three arms, our validation curve equals
  L.fine_val, and the stored per-image test predictions reproduce each snapshot's validation accuracy. The fine-class
  head is freshly initialised at the switch, so the pre-training phase (20-way task) is excluded throughout.
  Stored fine-phase checkpoints (per-image test predictions, pooled-layer activations, weights): pre-trained K20 fine
  epochs 1, 5, 10, 15, 20, 25, 30; scratch_E30 epochs 5, 10, ..., 30; K30 fine epochs 1, 5, 10, 15, 20; scratch_E20
  epochs 5, 10, 15, 20. Fine epochs 2-4 are not stored in any arm, and scratch has no fine epoch 1 checkpoint.
  Dense scratch reference (proxy): critical_sham sham_e00_seed{s}, plain fine-class training with the same 30-epoch
  cosine schedule and per-epoch checkpoints (epochs 1-15, 20, 25, 30), but from a different initialisation than the
  curriculum lanes of the same seed number (not paired by initialisation). Used only for pre-trained fine epoch 1 and
  for matching accuracies below the first scratch_E30 checkpoint; always reported separately.

Definitions (stored test predictions, 100 images per class, pooled over classes):
  P(fine)          fraction of images predicted as their class
  P(sc)            fraction predicted inside the right superclass (scored against the true superclasses for every arm);
                   P(group) is the same scored against the seed's random partition (the grouping the sc_rand arm was
                   pre-trained on, reconstructed from lane.json random_partition_seed exactly as devtrain.py does)
  P(fine | sc)     = P(fine) / P(sc) (exact factorisation)
  dlog             log(pre / scratch) of each factor; dlog_fine = dlog_sc + dlog_cond; share_via_sc = dlog_sc / dlog_fine
  matched epoch    pre-trained minus scratch at the same fine-class epoch, paired over seeds; share_via_sc is set
                   against share_via_sc_expected_on_scratch_curve, the share that scratch training itself puts
                   through superclass assignment over the same accuracy gain (its own curve from its fine epoch e to
                   the pre-trained network's P(fine)), so that a routing-specific head start shows as share > expected
  matched accuracy scratch's factor linearly interpolated (over its own checkpoints, ordered by P(fine)) at the
                   pre-trained checkpoint's P(fine); NaN outside scratch's range. At matched P(fine), d P(sc) > 0 means
                   more of the fine accuracy is carried by superclass assignment (d log P(sc) = -d log P(fine | sc)).
Class-level learning signal (as class_routing_signal.py): per stored test image at the pooled layer the fine-class
  cross-entropy gradient g = W'(softmax(Wx + b) - onehot(y)) with the checkpoint's fc; error type correct / within
  (wrong class, right superclass) / misrouted (another superclass). Per class c: G_c = class-mean gradient;
  excess = cos^2(G_c, mean G of c's four superclass siblings) minus the same for c's group-mates in 20 random groupings
  (RandomState(0..19); identical null to class_routing_signal.py). misrouted rate per class. The '_own' variants use the
  seed's random partition instead of the true superclasses (siblings and misrouting).
Statistics: per-seed differences, mean with 95% t interval (n = 5), number of seeds in each direction.

Outputs
  analysis_outputs/paper/CURRICULUM_routing/
    per_class_long.csv          per lane, fine epoch and class: counts and class-signal measures (cache)
    factors_per_checkpoint.csv  part 1, per arm, seed, fine epoch: P(fine), P(sc), P(fine|sc), P(group), P(fine|group)
    contrasts_part1.csv         part 1 paired contrasts (matched epoch and matched accuracy)
    signal_per_checkpoint.csv   part 2, per arm, seed, fine epoch: class means of misrouting and excess sharing
    contrasts_part2.csv         part 2 paired contrasts
    alignment_checks.json, summary.json
  Source Data analysis_outputs/paper/figures/final/source_data/curriculum_routing_{factors,signal,contrasts}.csv
Usage: python scripts/paper/curriculum_routing.py [--recompute]
"""
import json
import os
import sys
from pathlib import Path
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

torch.set_num_threads(2)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True
from class_routing_signal import RG, excess_of, unit  # noqa: E402
from gradient_routing import classifier  # noqa: E402
from curriculum_probe import SUPER_OF, load  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "analysis_outputs/paper"
OUT = RESULTS / "CURRICULUM_routing"
SD = RESULTS / "figures/final/source_data"
EXP = Path(RDT_DATA + "/extended/experiments")
CUR = EXP / "curriculum/lanes"
SHAM = EXP / "critical_sham/lanes"
SEEDS = [7, 42, 137, 256, 314]
DESIGNS = {"K20": {"true": "sc_true_K20", "rand": "sc_rand_K20", "scratch": "scratch_E30", "K": 20, "n_fine": 30},
           "K30": {"true": "sc_true_K30", "rand": "sc_rand_K30", "scratch": "scratch_E20", "K": 30, "n_fine": 20}}
PROXY = "scratch_proxy_sham"


# lanes, alignment
def lane_dir(name, s):
    return SHAM / f"sham_e00_seed{s}" if name == PROXY else CUR / f"{name}_seed{s}"


def fine_offset(name, s):
    """Global epoch of fine epoch 0 (K for the curricula, 0 for scratch), from lane.json."""
    if name == PROXY:
        return 0
    cfg = json.load(open(lane_dir(name, s) / "_config/lane.json"))
    cur = cfg.get("curriculum")
    if not cur:
        return 0
    ph = cur["phases"]
    assert [p["label_mode"] for p in ph] == ["superclass", "fine"], ph
    assert ph[1]["start_epoch"] == ph[0]["end_epoch"] + 1
    return ph[0]["end_epoch"]


def partition(s):
    """The seed's random partition (devtrain.py curriculum block): permutation of the true label vector."""
    ps = {json.load(open(lane_dir(f"sc_rand_{k}", s) / "_config/lane.json"))["curriculum"]["random_partition_seed"]
          for k in ("K20", "K30")}
    assert len(ps) == 1, ps
    ps = ps.pop()
    g = np.array(np.random.RandomState(int(ps)).permutation(list(SUPER_OF)), dtype=int)
    assert np.bincount(g, minlength=20).tolist() == [5] * 20
    return g, int(ps)


def fine_checkpoints(name, s):
    """{fine epoch: milestone dir} for the fine-class phase (intermediate snapshots after the switch)."""
    ld, off = lane_dir(name, s), fine_offset(name, s)
    snaps = json.load(open(ld / "dev_snapshots/metrics/devtrain_snapshots.json"))
    out = {}
    for x in snaps:
        if x.get("snapshot_type") == "intermediate" and int(x["epoch"]) > off:
            out[int(x["epoch"]) - off] = (ld / "dev_snapshots" / f"milestone_{x['milestone_index']}",
                                          float(x["val_accuracy"]), int(x["epoch"]))
    return out


def alignment_checks():
    import importlib
    res = {"fine_epoch_definition": "fine epoch e = global epoch K + e (pre-trained), e (scratch)", "checks": []}
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "figures"))
        L = importlib.import_module("curriculum_lib")
    except Exception as exc:  # pragma: no cover
        L = None
        res["checks"].append(f"curriculum_lib not importable ({exc}); figure alignment not cross-checked")
    for dname, D in DESIGNS.items():
        for s in SEEDS:
            curves = {}
            for arm in ("true", "rand", "scratch"):
                name = D[arm]
                off = fine_offset(name, s)
                assert off == (0 if arm == "scratch" else D["K"]), (name, s, off)
                m = {int(e["epoch"]): e for e in json.load(open(lane_dir(name, s) / "dev_snapshots/metrics/devtrain_metrics.json"))}
                curves[arm] = (np.array([m[off + e]["learning_rate"] for e in range(1, D["n_fine"] + 1)]),
                               np.array([m[off + e]["val_accuracy"] for e in range(1, D["n_fine"] + 1)]))
                if L is not None and dname == "K20":
                    ref = L.fine_val({"true": "sc_true", "rand": "sc_rand", "scratch": "scratch"}[arm], s)
                    assert np.allclose(ref, curves[arm][1]), (name, s)
            for arm in ("true", "rand"):
                assert np.allclose(curves[arm][0], curves["scratch"][0]), (dname, arm, s, "learning rate misaligned")
    res["checks"].append("fine phase starts at K + 1 in every curriculum lane.json (K20: 21, K30: 31)")
    res["checks"].append("learning rate at fine epoch e identical across pre-trained and scratch arms (both designs)")
    if L is not None:
        res["checks"].append("K20/E30 fine-class validation curves equal curriculum_lib.fine_val (Fig. 6b, ED 9)")
    return res


# per-checkpoint measures
def excess_groups(V, Gc, groups):
    """excess_of (class_routing_signal.py) generalised to any grouping of the 100 classes."""
    def share(gr):
        out = np.zeros(100)
        for c in range(100):
            mates = np.where((gr == gr[c]) & (np.arange(100) != c))[0]
            u = unit(Gc[mates].mean(0))
            v = V[c]
            nv = v @ v
            out[c] = (v @ u) ** 2 / nv if nv > 0 else np.nan
        return out
    return share(groups) - np.nanmean(np.stack([share(g) for g in RG]), 0)


def checkpoint_rows(ms, own, with_signal=True):
    sp = torch.load(ms / "sample_predictions.pt", map_location="cpu", weights_only=False)
    rows = {}
    for c in range(100):
        pr = sp[c]["predictions"].numpy()
        rows[c] = {"cls": c, "n": len(pr), "correct": int((pr == c).sum()),
                   "sc_correct": int((SUPER_OF[pr] == SUPER_OF[c]).sum()),
                   "grp_correct": int((own[pr] == own[c]).sum())}
    if with_signal:
        X, y = load(ms, "avgpool")
        X = X.astype(np.float64)
        W, b = classifier(ms)
        assert W.shape == (100, 512), W.shape
        Z = X @ W.T + b
        pred = Z.argmax(1)
        P = np.exp(Z - Z.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        P[np.arange(len(y)), y] -= 1.0
        G = P @ W
        Gc = np.stack([G[y == c].mean(0) for c in range(100)])
        ex_true = excess_groups(Gc, Gc, SUPER_OF)
        ex_own = excess_groups(Gc, Gc, own)
        mis_t = SUPER_OF[pred] != SUPER_OF[y]
        mis_o = own[pred] != own[y]
        for c in range(100):
            m = y == c
            rows[c].update({"acc_act": float((pred[m] == c).mean()), "misrouted_rate": float(mis_t[m].mean()),
                            "within_rate": float(((pred[m] != c) & ~mis_t[m]).mean()),
                            "misrouted_rate_own": float(mis_o[m].mean()),
                            "excess_all": float(ex_true[c]), "excess_own": float(ex_own[c])})
    return list(rows.values())


def build(recompute=False):
    cache = OUT / "per_class_long.csv"
    if cache.exists() and not recompute:
        return pd.read_csv(cache)
    parts, check_done = [], False
    for s in SEEDS:
        own, _ = partition(s)
        names = sorted({D[a] for D in DESIGNS.values() for a in ("true", "rand", "scratch")}) + [PROXY]
        for name in names:
            ck = fine_checkpoints(name, s)
            for fe, (ms, val, ge) in sorted(ck.items()):
                rows = checkpoint_rows(ms, own)
                t = pd.DataFrame(rows)
                t["lane"], t["seed"], t["fine_epoch"], t["global_epoch"], t["snapshot_val_acc"] = name, s, fe, ge, val
                # the per-image predictions reproduce the snapshot's validation accuracy
                assert abs(100 * t.correct.sum() / t.n.sum() - val) < 0.015, (name, s, fe, t.correct.sum(), val)
                # stored activations + checkpoint head reproduce the stored predictions (per class)
                assert np.abs(t.acc_act - t.correct / t.n).max() < 0.05, (name, s, fe)
                if not check_done:   # our excess equals class_routing_signal.excess_of for the true superclasses
                    X, y = load(ms, "avgpool")
                    W, b = classifier(ms)
                    Z = X.astype(np.float64) @ W.T + b
                    P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
                    P[np.arange(len(y)), y] -= 1.0
                    G = P @ W
                    Gc = np.stack([G[y == c].mean(0) for c in range(100)])
                    assert np.allclose(excess_of(Gc, Gc), t.excess_all.values, equal_nan=True)
                    check_done = True
                parts.append(t)
            print(name, s, sorted(ck), flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(cache, index=False)
    return df


# statistics
def tci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return [float(x.mean()) if len(x) else np.nan, np.nan, np.nan]
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def summarise(d, label):
    d = np.asarray(d, float)
    m, lo, hi = tci(d)
    ok = d[np.isfinite(d)]
    return {"contrast": label, "n": len(ok), "mean": m, "ci_lo": lo, "ci_hi": hi,
            "n_pos": int((ok > 0).sum()), "n_neg": int((ok < 0).sum()),
            "per_seed": json.dumps([None if not np.isfinite(v) else round(float(v), 4) for v in d])}


def interp_at(xq, x, y):
    """Linear interpolation of y(x) at xq over checkpoints ordered by x; NaN outside the range."""
    o = np.argsort(x)
    x, y = np.asarray(x)[o], np.asarray(y)[o]
    if not (x[0] <= xq <= x[-1]):
        return np.nan
    return float(np.interp(xq, x, y))


ARM_OF = {"sc_true_K20": ("K20", "true"), "sc_rand_K20": ("K20", "rand"), "scratch_E30": ("K20", "scratch"),
          "sc_true_K30": ("K30", "true"), "sc_rand_K30": ("K30", "rand"), "scratch_E20": ("K30", "scratch"),
          PROXY: ("proxy", "scratch_proxy")}


def factors_table(df):
    g = df.groupby(["lane", "seed", "fine_epoch"])[["n", "correct", "sc_correct", "grp_correct"]].sum().reset_index()
    g["p_fine"] = g.correct / g.n
    g["p_sc"] = g.sc_correct / g.n
    g["p_cond"] = g.correct / g.sc_correct
    g["p_grp"] = g.grp_correct / g.n
    g["p_cond_grp"] = g.correct / g.grp_correct
    g["design"] = g.lane.map(lambda x: ARM_OF[x][0])
    g["arm"] = g.lane.map(lambda x: ARM_OF[x][1])
    return g[["design", "arm", "lane", "seed", "fine_epoch", "n", "p_fine", "p_sc", "p_cond", "p_grp", "p_cond_grp"]]


def part1(F):
    rows = []
    for dname, D in DESIGNS.items():
        for arm in ("true", "rand"):
            pre = F[F.lane == D[arm]].set_index(["seed", "fine_epoch"])
            for ref_name, ref_lane in (("scratch", D["scratch"]), ("proxy", PROXY)):
                if ref_name == "proxy" and dname != "K20":
                    continue
                ref = F[F.lane == ref_lane].set_index(["seed", "fine_epoch"])
                eps = sorted(set(pre.index.get_level_values(1)))
                # matched fine epoch
                for e in eps:
                    if not all((s, e) in ref.index for s in SEEDS):
                        continue
                    if ref_name == "proxy" and e != 1:
                        continue   # proxy only where the paired scratch lane has no checkpoint
                    a = pre.loc[[(s, e) for s in SEEDS]]
                    r = ref.loc[[(s, e) for s in SEEDS]]
                    tag = f"{dname}|{arm}-{ref_name}|epoch {e}"
                    for k in ("p_fine", "p_sc", "p_cond", "p_grp", "p_cond_grp"):
                        rows.append({"design": dname, "arm": arm, "ref": ref_name, "match": "epoch", "fine_epoch": e,
                                     "measure": f"d_{k}_pp", **summarise(100 * (a[k].values - r[k].values), tag)})
                    dl = {k: np.log(a[k].values / r[k].values) for k in ("p_fine", "p_sc", "p_cond", "p_grp", "p_cond_grp")}
                    for k, v in dl.items():
                        rows.append({"design": dname, "arm": arm, "ref": ref_name, "match": "epoch", "fine_epoch": e,
                                     "measure": f"dlog_{k}", **summarise(v, tag)})
                    rows.append({"design": dname, "arm": arm, "ref": ref_name, "match": "epoch", "fine_epoch": e,
                                 "measure": "share_via_sc", **summarise(dl["p_sc"] / dl["p_fine"], tag)})
                    rows.append({"design": dname, "arm": arm, "ref": ref_name, "match": "epoch", "fine_epoch": e,
                                 "measure": "share_via_grp", **summarise(dl["p_grp"] / dl["p_fine"], tag)})
                    # the share scratch training itself puts through superclass assignment over the same accuracy
                    # gain (reference curve, from its fine epoch e to the pre-trained network's accuracy)
                    exp = []
                    for s in SEEDS:
                        rr, pa, r0 = ref.loc[s], pre.loc[(s, e)].p_fine, ref.loc[(s, e)]
                        psc = interp_at(pa, rr.p_fine.values, rr.p_sc.values)
                        exp.append(np.log(psc / r0.p_sc) / np.log(pa / r0.p_fine))
                    exp = np.array(exp)
                    rows.append({"design": dname, "arm": arm, "ref": ref_name, "match": "epoch", "fine_epoch": e,
                                 "measure": "share_via_sc_expected_on_scratch_curve", **summarise(exp, tag)})
                    rows.append({"design": dname, "arm": arm, "ref": ref_name, "match": "epoch", "fine_epoch": e,
                                 "measure": "share_via_sc_minus_expected",
                                 **summarise(dl["p_sc"] / dl["p_fine"] - exp, tag)})
                # matched fine accuracy (interpolated on the reference's checkpoints)
                for e in eps:
                    vals = {k: [] for k in ("p_sc", "p_cond", "p_grp", "p_cond_grp", "p_fine")}
                    for s in SEEDS:
                        a = pre.loc[(s, e)]
                        rr = ref.loc[s]
                        vals["p_fine"].append(100 * a.p_fine)
                        for k in ("p_sc", "p_cond", "p_grp", "p_cond_grp"):
                            vals[k].append(100 * (a[k] - interp_at(a.p_fine, rr.p_fine.values, rr[k].values)))
                    tag = f"{dname}|{arm}-{ref_name}|matched acc at fine epoch {e}"
                    for k in ("p_sc", "p_cond", "p_grp", "p_cond_grp"):
                        rows.append({"design": dname, "arm": arm, "ref": ref_name, "match": "accuracy",
                                     "fine_epoch": e, "measure": f"d_{k}_pp", **summarise(vals[k], tag),
                                     "pre_p_fine_pct": json.dumps([round(v, 2) for v in vals["p_fine"]])})
        # true - random at matched fine epoch
        a, r = (F[F.lane == D[x]].set_index(["seed", "fine_epoch"]) for x in ("true", "rand"))
        for e in sorted(set(a.index.get_level_values(1))):
            tag = f"{dname}|true-rand|epoch {e}"
            A, R = a.loc[[(s, e) for s in SEEDS]], r.loc[[(s, e) for s in SEEDS]]
            for k in ("p_fine", "p_sc", "p_cond", "p_grp"):
                rows.append({"design": dname, "arm": "true", "ref": "rand", "match": "epoch", "fine_epoch": e,
                             "measure": f"d_{k}_pp", **summarise(100 * (A[k].values - R[k].values), tag)})
    return pd.DataFrame(rows)


def signal_table(df):
    g = df.groupby(["lane", "seed", "fine_epoch"])[["acc_act", "misrouted_rate", "within_rate", "misrouted_rate_own",
                                                     "excess_all", "excess_own"]].mean().reset_index()
    g["design"] = g.lane.map(lambda x: ARM_OF[x][0])
    g["arm"] = g.lane.map(lambda x: ARM_OF[x][1])
    return g[["design", "arm", "lane", "seed", "fine_epoch", "acc_act", "misrouted_rate", "within_rate",
              "misrouted_rate_own", "excess_all", "excess_own"]]


def part2(S):
    keys = ("misrouted_rate", "within_rate", "excess_all", "misrouted_rate_own", "excess_own")
    rows = []
    for dname, D in DESIGNS.items():
        for arm in ("true", "rand"):
            pre = S[S.lane == D[arm]].set_index(["seed", "fine_epoch"])
            for ref_name, ref_lane in (("scratch", D["scratch"]), ("proxy", PROXY)):
                if ref_name == "proxy" and dname != "K20":
                    continue
                ref = S[S.lane == ref_lane].set_index(["seed", "fine_epoch"])
                eps = sorted(set(pre.index.get_level_values(1)))
                for e in eps:
                    if not all((s, e) in ref.index for s in SEEDS) or (ref_name == "proxy" and e != 1):
                        continue
                    a, r = pre.loc[[(s, e) for s in SEEDS]], ref.loc[[(s, e) for s in SEEDS]]
                    for k in ("acc_act",) + keys:
                        rows.append({"design": dname, "arm": arm, "ref": ref_name, "match": "epoch", "fine_epoch": e,
                                     "measure": f"d_{k}", **summarise(a[k].values - r[k].values,
                                                                      f"{dname}|{arm}-{ref_name}|epoch {e}")})
                for e in eps:
                    vals = {k: [] for k in keys}
                    accs = []
                    for s in SEEDS:
                        a, rr = pre.loc[(s, e)], ref.loc[s]
                        accs.append(round(100 * a.acc_act, 2))
                        for k in keys:
                            vals[k].append(a[k] - interp_at(a.acc_act, rr.acc_act.values, rr[k].values))
                    for k in keys:
                        rows.append({"design": dname, "arm": arm, "ref": ref_name, "match": "accuracy",
                                     "fine_epoch": e, "measure": f"d_{k}",
                                     **summarise(vals[k], f"{dname}|{arm}-{ref_name}|matched acc at fine epoch {e}"),
                                     "pre_acc_pct": json.dumps(accs)})
    return pd.DataFrame(rows)


def main():
    recompute = "--recompute" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    SD.mkdir(parents=True, exist_ok=True)
    al = alignment_checks()
    al["partition_seeds"] = {s: partition(s)[1] for s in SEEDS}
    try:  # independent reconstruction of the random partition (imposed_grouping_lib, torchvision class order)
        import imposed_grouping_lib as H
        cls = H.canonical_classes()
        for s in SEEDS:
            g, ps = partition(s)
            assert np.array_equal(H.label_vector(H.imposed_label_map(cls, ps)), g)
        al["checks"].append("random partition equals imposed_grouping_lib.imposed_label_map (devtrain.py replica)")
    except Exception as exc:  # pragma: no cover
        al["checks"].append(f"partition cross-check skipped: {exc}")
    df = build(recompute)
    al["checks"] += ["per-image test predictions reproduce every snapshot's validation accuracy (|diff| < 0.015 pp)",
                     "activations + checkpoint head reproduce the per-class stored predictions (|diff| < 0.05)",
                     "excess (true superclasses) equals class_routing_signal.excess_of"]
    al["fine_epochs_available"] = {f"{ln}": sorted(df[(df.lane == ln) & (df.seed == 7)].fine_epoch.unique().tolist())
                                   for ln in sorted(df.lane.unique())}
    al["seeds_available"] = {ln: sorted(df[df.lane == ln].seed.unique().tolist()) for ln in sorted(df.lane.unique())}
    json.dump(al, open(OUT / "alignment_checks.json", "w"), indent=1, default=int)

    F = factors_table(df)
    F.to_csv(OUT / "factors_per_checkpoint.csv", index=False)
    C1 = part1(F)
    C1.to_csv(OUT / "contrasts_part1.csv", index=False)
    S = signal_table(df)
    S.to_csv(OUT / "signal_per_checkpoint.csv", index=False)
    C2 = part2(S)
    C2.to_csv(OUT / "contrasts_part2.csv", index=False)

    F.to_csv(SD / "curriculum_routing_factors.csv", index=False)
    S.to_csv(SD / "curriculum_routing_signal.csv", index=False)
    pd.concat([C1.assign(part="1_factorisation"), C2.assign(part="2_class_signal")], ignore_index=True).to_csv(
        SD / "curriculum_routing_contrasts.csv", index=False)

    def pick(C, **kw):
        m = np.ones(len(C), bool)
        for k, v in kw.items():
            m &= (C[k] == v).values
        return {r.contrast + " :: " + r.measure: [round(r.mean, 3), round(r.ci_lo, 3), round(r.ci_hi, 3),
                                                  f"{r.n_pos}+/{r.n_neg}-", r.n] for r in C[m].itertuples()}
    summ = {"alignment": al,
            "part1_K20_epoch": pick(C1, design="K20", match="epoch"),
            "part1_K20_accuracy": pick(C1, design="K20", match="accuracy"),
            "part1_K30": pick(C1, design="K30"),
            "part2_K20_epoch": pick(C2, design="K20", match="epoch"),
            "part2_K20_accuracy": pick(C2, design="K20", match="accuracy"),
            "part2_K30": pick(C2, design="K30")}
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    print(F[F.design != "K30"].groupby(["arm", "fine_epoch"])[["p_fine", "p_sc", "p_cond", "p_grp", "p_cond_grp"]]
          .mean().round(4).to_string())
    print(S[S.design != "K30"].groupby(["arm", "fine_epoch"])[["acc_act", "misrouted_rate", "excess_all",
                                                               "misrouted_rate_own", "excess_own"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
