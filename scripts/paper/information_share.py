"""information_share: the superclass share of label information is fixed by the data, tested on held-out networks.

Every measure is computed by the functions in scripts/paper/timetable_*.py, called unchanged; only the lane list
and the output and cache paths are re-pointed (module globals `lanes`, `CACHE`, `OUT`). The lane list is built here
with the code of timetable_lib.lanes() (open networks) and the selection of held-out networks below.

Networks (7): core s137 ResNet-18 / ViT-Small / CCT-7; clean ResNet-18 s256, s314; label-noise "standard" s256, s314.
Noise ordering: label-noise within_sc_p03 / standard / between_sc_p03, seeds 256 and 314.

Steps
  1. logits: ResNet / ViT = head(stored pooled activation)  (timetable_extract.main);
             CCT-7 = CPU forward pass (body of timetable_cct_forward.main with the lane key as a parameter; norm tag 'cifar').
     Every logit set must agree with the stored sample_predictions (argmax agreement >= 0.999), else abort.
  2. info share + 50 random 20x5 groupings (rng seed 0) + flat model: timetable_infoshare.main.
  3. criteria and further measures (rule text stored in the output JSON).
Usage: python scripts/paper/information_share.py [extract|cct|infoshare|decide|all]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timetable_lib  # noqa: E402  (sets OMP/MKL/OPENBLAS/VECLIB threads = 2)
from timetable_lib import CLS_OF_IMG, NOISE_EXP, REPO, SC_OF_CLASS, test_canonical, to_canonical_perm  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

DATA_ORDER = REPO / "analysis_outputs" / "paper" / "DATA_order"
CACHE_ORDER = DATA_ORDER / "cache"
OUT_IS = DATA_ORDER / "information_share"
TIMETABLE = REPO / "analysis_outputs" / "paper" / "DICTFREE" / "B_timetable"

HELDOUT_SEEDS = {256, 314}
CORE_KEYS = ["core_ResNet-18_s137", "core_ViT-Small_s137", "core_CCT-7_s137"]
CLEAN_KEYS = ["clean_ResNet-18_s256", "clean_ResNet-18_s314"]
NOISE_CONDS = ["standard", "within_sc_p03", "between_sc_p03"]
LO, HI = 0.65, 0.71
METHOD_NOTES = [
    "CCT-7 logits: body of timetable_cct_forward.main copied into step_cct with the lane key parametrised "
    "(timetable_cct_forward.main fixes core_CCT-7_s42) and the n_check branch dropped; normalisation tag 'cifar', as recorded in the extract log "
    "(the script's default argument is 'half').",
    "Noise-condition shares: the share computation of the open seeds 7/42/137 is applied to "
    "seeds 256/314; logits come from the timetable_extract cache (identical fc(avgpool) computation) and are "
    "verified against stored predictions.",
    "'Peak test information' is operationalised as the argmax over checkpoints of test I_fine (inclusive); 'trained' = "
    "ckpt > 0. The argmax rule reproduces the "
    "open statistic exactly (58 open checkpoints, 0.673 +- 0.008, range 0.660-0.705).",
    "Open consensus profile = mean over the three architectures of per-architecture mean shares (stored DICTFREE "
    "B_timetable outputs; the post-peak core ResNet s42 checkpoint 10 is kept).",
    "Noise ordering is evaluated on the mean share over trained milestones (as for the open seeds); per-checkpoint ordering "
    "is also reported.",
    "Pooled-activation caches written by step_cct are not used by information_share or mode_strength.",
]


def _lane(dir_, arch, seed, group, labels, epochs, val_acc):
    return dict(dir=dir_, arch=arch, seed=seed, group=group, labels=labels, epochs=epochs, val_acc=val_acc)


def heldout_lanes(include_noise_p03=False):
    """Held-out lanes; same construction as timetable_lib.lanes(), with the held-out selection."""
    L = json.load(open(REPO / "analysis_outputs/paper/CORE_lanes/lanes.json"))["lanes"]
    out = {}
    for l in L:
        if l["key"] not in CORE_KEYS + CLEAN_KEYS:
            continue
        out[l["key"]] = _lane(l["dir"], l["arch"], l["seed"], l["group"], [c["label"] for c in l["ckpts"]],
                              [float(c["epoch"]) for c in l["ckpts"]], [c["val_acc"] for c in l["ckpts"]])
    m = pd.read_csv(REPO / "analysis_outputs/paper/NOISE_scaffold/per_lane_metrics.csv")
    conds = NOISE_CONDS if include_noise_p03 else ["standard"]
    m = m[(m.condition.isin(conds)) & (m.seed.isin(sorted(HELDOUT_SEEDS)))]
    for (lid, cond, seed), g in m.groupby(["lane_id", "condition", "seed"]):
        g = g.sort_values("milestone")
        key = f"noise_std_ResNet-18_s{seed}" if cond == "standard" else f"noise_{cond}_ResNet-18_s{seed}"
        out[key] = _lane(str(NOISE_EXP / lid), "ResNet-18", int(seed), "noise_std" if cond == "standard" else f"noise_{cond}",
                         [str(x) for x in g.milestone], [float(x) for x in g.epoch],
                         [float(x) * 100 for x in g.val_acc_logged])
    for k, v in out.items():
        assert (v["seed"] in HELDOUT_SEEDS) or (v["group"] == "core" and v["seed"] == 137), k
    return out


def seven():
    L = heldout_lanes()
    keys = CORE_KEYS + CLEAN_KEYS + ["noise_std_ResNet-18_s256", "noise_std_ResNet-18_s314"]
    assert set(keys) == set(L), (sorted(L), keys)
    return {k: L[k] for k in keys}


# step 1: logits
def step_extract():
    import timetable_extract
    CACHE_ORDER.mkdir(parents=True, exist_ok=True)
    timetable_extract.CACHE = CACHE_ORDER
    timetable_extract.lanes = lambda: heldout_lanes(include_noise_p03=True)
    timetable_extract.main()


def step_cct(key="core_CCT-7_s137", tag="cifar"):
    """Body of timetable_cct_forward.main() with the lane key as a parameter (timetable_cct_forward fixes core_CCT-7_s42;
    norm tag 'cifar' as in cache/extract_log.txt)."""
    import torch
    from timetable_cct_forward import NORMS
    from models import create_network
    torch.set_num_threads(2)
    CACHE = CACHE_ORDER
    CACHE.mkdir(parents=True, exist_ok=True)
    lane = heldout_lanes()[key]
    perm = to_canonical_perm(lane)  # lane class -> canonical
    inv = np.empty(100, int)
    inv[perm] = np.arange(100)
    X, y = test_canonical()
    mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in NORMS[tag])
    Xt = (torch.from_numpy(X).permute(0, 3, 1, 2).float() / 255.0 - mean) / std
    net = create_network("cct_7", num_classes=100)
    pooled = {}
    n_check = 0
    for lab in lane["labels"]:
        f_log = CACHE / f"{key}_m{lab}_logits.npy"
        if f_log.exists() and not n_check:
            continue
        d = Path(lane["dir"]) / "dev_snapshots" / f"milestone_{lab}"
        sd = torch.load(d / "checkpoint.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(sd.get("model_state_dict", sd))
        net.eval()
        h = net.fc.register_forward_pre_hook(lambda m, inp: pooled.__setitem__("x", inp[0].detach()))
        idx = np.arange(10000)
        Z, P = [], []
        with torch.no_grad():
            for s in range(0, len(idx), 250):
                z = net(Xt[idx[s:s + 250]])
                Z.append(z.numpy())
                P.append(pooled["x"].numpy())
        h.remove()
        Z = np.concatenate(Z)
        Zc = np.empty_like(Z)
        Zc[:, perm] = Z
        sp = torch.load(d / "sample_predictions.pt", map_location="cpu", weights_only=False)
        pred_lane = np.concatenate([sp[i]["predictions"].numpy() for i in range(100)])
        cor_lane = np.concatenate([sp[i]["correct"].numpy() for i in range(100)]).astype(bool)
        rows = (inv[:, None] * 100 + np.arange(100)[None, :]).ravel()
        pred_c = perm[pred_lane[rows]]  # canonical predicted class
        agree = float((Zc.argmax(1) == pred_c[idx]).mean())
        msg = f"{key} m{lab} norm={tag} n={len(idx)} argmax-agree={agree:.4f} acc={(Zc.argmax(1) == y[idx]).mean():.4f}"
        print(msg, flush=True)
        np.save(f_log, Zc.astype(np.float32))
        np.save(CACHE / f"{key}_m{lab}_pooled.npy", np.concatenate(P).astype(np.float16))
        np.save(CACHE / f"{key}_m{lab}_storedcorrect.npy", cor_lane[rows])
        with open(CACHE / "extract_log.txt", "a") as f:
            f.write(msg + "\n")


def verify_logits(min_agree=0.999):
    """Every logit file needed must be present and logged with argmax agreement >= min_agree."""
    log = (CACHE_ORDER / "extract_log.txt").read_text().splitlines()
    agree = {}
    for line in log:
        m = re.match(r"(\S+) m(\S+) .*argmax-agree=([0-9.]+)", line)
        if m:
            agree[(m.group(1), m.group(2))] = float(m.group(3))
    rows = []
    for key, lane in heldout_lanes(include_noise_p03=True).items():
        for lab in lane["labels"]:
            a = agree.get((key, lab), np.nan)
            ok = (CACHE_ORDER / f"{key}_m{lab}_logits.npy").exists() and np.isfinite(a) and a >= min_agree
            rows.append(dict(lane=key, milestone=lab, argmax_agree=a, ok=bool(ok)))
    d = pd.DataFrame(rows)
    d.to_csv(DATA_ORDER / "logit_verification.csv", index=False)
    bad = d[~d.ok]
    assert bad.empty, f"logit verification failed:\n{bad}"
    return d


# step 2: info share
def step_infoshare():
    import timetable_infoshare
    OUT_IS.mkdir(parents=True, exist_ok=True)
    timetable_infoshare.CACHE = CACHE_ORDER
    timetable_infoshare.OUT = OUT_IS
    timetable_infoshare.lanes = seven
    timetable_infoshare.main()


def noise_condition_shares():
    """Share per condition/seed/milestone (clean test labels), computed as for the open seeds, here seeds 256/314,
    conditions standard / within_sc_p03 / between_sc_p03; logits from the verified cache of step 1."""
    from timetable_infoshare import group_info, logsoftmax
    rows = []
    for key, lane in heldout_lanes(include_noise_p03=True).items():
        if not key.startswith("noise_"):
            continue
        cond = "standard" if lane["group"] == "noise_std" else lane["group"].replace("noise_", "")
        for lab, ep in zip(lane["labels"], lane["epochs"]):
            Z = np.load(CACHE_ORDER / f"{key}_m{lab}_logits.npy").astype(np.float64)
            lp = logsoftmax(Z)
            If = (lp[np.arange(10000), CLS_OF_IMG] + np.log(100)).mean()
            Is = (group_info(lp, SC_OF_CLASS) + np.log(20)).mean()
            rows.append(dict(condition=cond, seed=lane["seed"], milestone=int(lab), epoch=ep,
                             acc=(Z.argmax(1) == CLS_OF_IMG).mean(), I_fine=If, I_sc=Is, share=Is / If))
    d = pd.DataFrame(rows)
    d.to_csv(OUT_IS / "infoshare_noise_conditions.csv", index=False)
    return d


# step 3: criteria
def eligible(N):
    """Trained checkpoints (ckpt > 0) up to and including the checkpoint of peak test information (argmax I_fine).
    Reproduces the exclusion of the open analysis (58 checkpoints; ViT s42 ckpt >= 7 and CCT s42 ckpt >= 8)."""
    out = []
    for lane, g in N.groupby("lane", sort=False):
        peak = int(g.loc[g.I_fine.idxmax(), "ckpt"])
        out.append(g[(g.ckpt > 0) & (g.ckpt <= peak)].assign(peak_ckpt=peak))
    return pd.concat(out)


def open_consensus():
    """Per-superclass share profile of the open networks
    (stored DICTFREE B_timetable outputs; ckpt > 0, ViT s42 ckpt >= 7 and CCT s42 ckpt >= 8 excluded; mean per architecture, then mean)."""
    S = pd.read_csv(TIMETABLE / "infoshare_superclass.csv")
    assert set(S.lane) == {
        "core_ResNet-18_s42", "core_ViT-Small_s42", "core_CCT-7_s42", "clean_ResNet-18_s7", "clean_ResNet-18_s42",
        "clean_ResNet-18_s137", "noise_std_ResNet-18_s7", "noise_std_ResNet-18_s42", "noise_std_ResNet-18_s137"}
    S = S[(S.ckpt > 0) & ~((S.lane == "core_ViT-Small_s42") & (S.ckpt >= 7)) & ~((S.lane == "core_CCT-7_s42") & (S.ckpt >= 8))]
    S["share"] = S.I_sc / S.I_fine
    piv = S.groupby(["sc", "arch"]).share.mean().unstack()
    return piv.mean(1)


def decide():
    N = pd.read_csv(OUT_IS / "infoshare_network.csv")
    Sc = pd.read_csv(OUT_IS / "infoshare_superclass.csv")
    E = eligible(N)
    E.to_csv(OUT_IS / "eligible_checkpoints.csv", index=False)
    cons = open_consensus()
    rows = []
    for lane, g in E.groupby("lane", sort=False):
        s = Sc[(Sc.lane == lane) & Sc.ckpt.isin(g.ckpt)].copy()
        s["share"] = s.I_sc / s.I_fine
        prof = s.groupby("sc").share.mean()
        rows.append(dict(
            lane=lane, arch=g.arch.iloc[0], n_ckpt=len(g), peak_ckpt=int(g.peak_ckpt.iloc[0]),
            peak_epoch=float(N[(N.lane == lane) & (N.ckpt == g.peak_ckpt.iloc[0])].epoch.iloc[0]),
            share_min=g.share_true.min(), share_max=g.share_true.max(), share_mean=g.share_true.mean(),
            rand_max_max=g.share_rand_max.max(), min_margin_over_rand_max=(g.share_true - g.share_rand_max).min(),
            crit_i_in_band=bool(((g.share_true >= LO) & (g.share_true <= HI)).all()),
            crit_ii_above_rand_max=bool((g.share_true > g.share_rand_max).all()),
            rho_share_acc=spearmanr(g.share_true, g.acc)[0], rho_randmean_acc=spearmanr(g.share_rand_mean, g.acc)[0],
            rho_profile_vs_open=spearmanr(prof.loc[cons.index], cons)[0],
        ))
    R = pd.DataFrame(rows)
    R["passes"] = R.crit_i_in_band & R.crit_ii_above_rand_max
    R["sec_acc_flat"] = (R.rho_share_acc.abs() < 0.3) & (R.rho_randmean_acc > 0.9)
    R["sec_profile"] = R.rho_profile_vs_open >= 0.7
    R.to_csv(OUT_IS / "information_share_per_network.csv", index=False)
    # noise ordering (mean share over trained milestones per condition and seed, as for the open seeds)
    Nn = noise_condition_shares()
    t = Nn[Nn.milestone > 0]
    agg = t.groupby(["seed", "condition"]).share.mean().unstack()
    per_ck = t.pivot_table(index=["seed", "milestone"], columns="condition", values="share")
    per_ck["ordered"] = (per_ck.within_sc_p03 > per_ck.standard) & (per_ck.standard > per_ck.between_sc_p03)
    per_ck.to_csv(OUT_IS / "noise_ordering_per_checkpoint.csv")
    order = {int(s): bool(agg.loc[s, "within_sc_p03"] > agg.loc[s, "standard"] > agg.loc[s, "between_sc_p03"])
             for s in agg.index}
    verdict = "criteria met" if bool(R.passes.all()) else "criteria not met"
    res = dict(
        test="information_share", verdict=verdict,
        rule="in every one of the 7 networks, at every trained checkpoint up to peak test information: (i) share in "
             "[0.65, 0.71]; (ii) share > max over 50 random 20x5 groupings",
        open_estimate="0.673 (SD 0.008, range 0.660-0.705) over 58 checkpoints in 9 open networks; "
                             "best of 50 random groupings never above 0.555",
        n_networks_passing=int(R.passes.sum()),
        per_network=R.round(4).to_dict(orient="records"),
        pooled_eligible=dict(n_ckpt=int(len(E)), share_mean=float(E.share_true.mean()), share_sd=float(E.share_true.std()),
                             share_min=float(E.share_true.min()), share_max=float(E.share_true.max()),
                             rand_max_max=float(E.share_rand_max.max())),
        secondary=dict(
            accuracy_flatness=dict(rule="|rho(share, acc)| < 0.3 within each network while rho(random-mean, acc) > 0.9",
                                   holds_in=int(R.sec_acc_flat.sum()), of=len(R)),
            profile_vs_open=dict(rule="rho(per-superclass share profile, open consensus) >= 0.7",
                                      holds_in=int(R.sec_profile.sum()), of=len(R),
                                      consensus_definition="mean over architectures "
                                                           "of per-architecture mean share (open networks only)"),
            noise_ordering=dict(rule="within_sc_p03 > standard > between_sc_p03 (mean share over trained milestones)",
                                mean_share=agg.round(4).to_dict(orient="index"), holds_per_seed=order,
                                checkpoints_ordered=f"{int(per_ck.ordered.sum())}/{len(per_ck)}"),
        ),
        multiplicity="no correction (a single test with its own criteria)",
        logit_verification=dict(min_argmax_agree=float(verify_logits().argmax_agree.min())),
    )
    json.dump(res, open(OUT_IS / "information_share_results.json", "w"), indent=2, default=float)
    with pd.option_context("display.width", 250):
        print(R.round(4).to_string())
        print(agg.round(4))
        print(per_ck.round(4))
    print("information_share:", verdict)
    return res


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("extract", "all"):
        step_extract()
    if step in ("cct", "all"):
        step_cct()
    if step in ("infoshare", "all"):
        verify_logits()
        step_infoshare()
    if step in ("decide", "all"):
        verify_logits()
        decide()
