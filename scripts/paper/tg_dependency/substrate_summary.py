"""Contrasts of the read-out proxy (substrate_proxy.py) and, if present, of the content and maturation runs
(content_long.csv, maturation_long.csv). Paired over seeds (mean, t-based 95% CI, seeds below 0).
Proxy contrasts per epoch and read-out: tg - rand, tg - pc, tg - pcv, tg - ot, each subspace - none; layer-specific
tg@L - rand@L. Validation: proxy tg - rand against the transplant C - Crand (transplant_summary.csv; epochs 1 and 5
transplants, fine-class accuracy at fine epoch 1 and 15) for seeds 7, 42, 137.
Writes analysis_outputs/paper/TG_dependency/proxy_contrasts.csv, proxy_validation.csv, content_contrasts.csv.
Usage: python scripts/paper/tg_dependency/substrate_summary.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parents[3] / "analysis_outputs/paper/TG_dependency"


def ci(z):
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    if len(z) < 2:
        return dict(mean=z.mean() if len(z) else np.nan, lo=np.nan, hi=np.nan, n=len(z), n_neg=int((z < 0).sum()))
    h = stats.t.ppf(.975, len(z) - 1) * z.std(ddof=1) / np.sqrt(len(z))
    return dict(mean=z.mean(), lo=z.mean() - h, hi=z.mean() + h, n=len(z), n_neg=int((z < 0).sum()))


def contrasts(d, key, pairs, measures):
    out = []
    for e in sorted(d.epoch.unique()):
        x = d[d.epoch == e]
        for a, b in pairs:
            A = x[x[key] == a].set_index("seed")
            B = x[x[key] == b].set_index("seed")
            for m in measures:
                if m not in A or m not in B:
                    continue
                z = (A[m] - B[m]).dropna()
                out.append(dict(epoch=e, contrast=f"{a} - {b}", measure=m, **ci(z),
                                per_seed=";".join(f"{k}:{v:.2f}" for k, v in z.items())))
    return pd.DataFrame(out)


p = pd.read_csv(OUT / "proxy_long.csv")
P = contrasts(p, "cond", [("tg", "rand"), ("tg", "pc"), ("tg", "pcv"), ("tg", "ot"), ("tg", "none"),
                          ("rand", "none"), ("pc", "none"), ("pcv", "none"), ("ot", "none"),
                          ("tg@layer1", "rand@layer1"), ("tg@layer2", "rand@layer2")],
              ["fine", "superclass", "within"])
P.to_csv(OUT / "proxy_contrasts.csv", index=False)
show = P[P.measure == "fine"].pivot(index="epoch", columns="contrast", values="mean")
print("proxy, fine-class accuracy (points), mean over seeds\n", show.round(2).to_string())
key = P[(P.measure == "fine") & P.contrast.isin(["tg - rand", "tg - pcv", "tg - pc", "tg - ot"])]
print(key[["epoch", "contrast", "mean", "lo", "hi", "n", "n_neg"]].round(2).to_string())
print("variance removed, layer1 (mean):\n", p.groupby(["epoch", "cond"]).layer1_var.mean().unstack().round(3).to_string())
print("rank, layer1 (mean):\n", p.groupby(["epoch", "cond"]).layer1_rank.mean().unstack().round(1).to_string())

tr = pd.read_csv(OUT / "transplant_summary.csv")
val = []
for run, te in (("tgtransplant", 1), ("tgtransplant_e5", 5)):
    t = tr[(tr.run == run) & (tr.contrast == "C - Crand")]
    for s in (7, 42, 137):
        q = p[(p.seed == s) & (p.epoch == te)].set_index("cond").fine
        if {"tg", "rand"} <= set(q.index):
            ps = {int(k): float(v) for k, v in (x.split(":") for x in t[t.epoch == 1].per_seed.iloc[0].split(";"))}
            ps15 = {int(k): float(v) for k, v in (x.split(":") for x in t[t.epoch == 15].per_seed.iloc[0].split(";"))}
            val.append(dict(transplant_epoch=te, seed=s, proxy_tg_minus_rand=q["tg"] - q["rand"],
                            transplant_C_minus_Crand_ep1=ps[s], transplant_C_minus_Crand_ep15=ps15[s]))
V = pd.DataFrame(val)
print("validation against the transplant:\n", V.round(2).to_string())
V.to_csv(OUT / "proxy_validation.csv", index=False)

cpath = OUT / "content_long.csv"
if cpath.exists():
    c = pd.read_csv(cpath)
    C = contrasts(c, "cond", [("tg", "rand"), ("tg", "pcv"), ("tg", "pc"), ("tg", "ot")],
                  ["comp_fine", "comp_superclass", "comp_within", "pool_fine", "stats_r2", "rm_fine", "rm_superclass",
                   "rm_within"])
    C.to_csv(OUT / "content_contrasts.csv", index=False)
    print("content, means:\n", c.groupby(["epoch", "cond"])[["rank", "var", "comp_fine", "comp_superclass",
                                                              "comp_within", "pool_fine", "stats_r2", "rm_fine"]]
          .mean().round(2).to_string())
    print(C[C.contrast == "tg - rand"].pivot(index="epoch", columns="measure", values="mean").round(2).to_string())
    m = OUT / "maturation_long.csv"
    if m.exists():
        print("maturation:\n", pd.read_csv(m).groupby("epoch").mean(numeric_only=True).round(3).to_string())
