"""Head start against pre-training events: across the three pre-training arms (superclasses, random grouping,
superclass-blind task) and five seeds, does the head start track the task-general or the superclass events of
pre-training? Events: common threshold set per seed (scratch arm's), last block (layer4 + pooled), raw counts, summed
over the pre-training transitions (end epoch <= 20), as Fig. 6 (curriculum_lib.events_common, blind_arm).
Head start: accuracy lead after one fine-class epoch against the per-epoch sham (BLIND_PRETRAIN/long.csv,
table accuracy_lead). Also the 20-way true-superclass probe at the switch.
Fits (OLS, n = 15): lead ~ log TG events, ~ log Ab events, ~ both, ~ probe, ~ probe + log TG. Within-arm Spearman of the
lead with TG events, Ab events and the probe (n = 5 per arm).
Writes analysis_outputs/paper/TG_dependency/tg_pretrain_{rows,fits}.csv.
Usage: python scripts/paper/tg_dependency/tg_pretrain_events.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "figures"))
sys.path.insert(0, str(HERE.parents[1] / "figures" / "interventions"))
import curriculum_lib as L  # noqa: E402
import blind_arm as B  # noqa: E402

RESULTS = HERE.parents[3] / "analysis_outputs/paper"
OUT = RESULTS / "TG_dependency"
ARMS = {"sc_true_K20": "true", "sc_rand_K20": "rand", B.LANE_NAME: "blind"}

ev = pd.concat([L.events_common(), B.events_rows()], ignore_index=True)
pre = ev[ev.arm.isin(ARMS) & (ev.epoch_b <= 20) & (ev.epoch_b > ev.epoch_a)]
E = pre.groupby(["arm", "seed"])[["tg_h", "ab_h", "di_h"]].sum().reset_index()
E["arm"] = E.arm.map(ARMS)
lg = pd.read_csv(RESULTS / "BLIND_PRETRAIN/long.csv")
lead = lg[(lg.table == "accuracy_lead") & (lg.fine_epoch == 1.0)][["arm", "seed", "lead_p_fine"]]
probe = lg.dropna(subset=["sc_probe_switch"])[["arm", "seed", "sc_probe_switch"]]
d = E.merge(lead, on=["arm", "seed"]).merge(probe, on=["arm", "seed"], how="left")
OUT.mkdir(parents=True, exist_ok=True)
d.to_csv(OUT / "tg_pretrain_rows.csv", index=False)
print(d.round(3).to_string())
print(d.groupby("arm")[["tg_h", "ab_h", "di_h", "lead_p_fine", "sc_probe_switch"]].mean().round(2).to_string())

fits = []
y = d.lead_p_fine.values
for name, cols in (("log TG", ["tg_h"]), ("log Ab", ["ab_h"]), ("log TG + log Ab", ["tg_h", "ab_h"]),
                   ("probe", ["sc_probe_switch"]), ("probe + log TG", ["sc_probe_switch", "tg_h"])):
    Xm = np.column_stack([np.ones(len(d))] + [np.log(d[c].values) if c.endswith("_h") else d[c].values
                                               for c in cols])
    beta, res, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    yhat = Xm @ beta
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    n, k = Xm.shape
    s2 = ((y - yhat) ** 2).sum() / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(Xm.T @ Xm)))
    for c, b, e in zip(["const"] + cols, beta, se):
        fits.append(dict(model=name, term=c, beta=b, se=e, t=b / e, p=2 * stats.t.sf(abs(b / e), n - k), r2=r2,
                         n=n))
for arm in ("true", "rand", "blind"):
    q = d[d.arm == arm]
    for c in ("tg_h", "ab_h", "sc_probe_switch"):
        r = stats.spearmanr(q[c], q.lead_p_fine)
        fits.append(dict(model=f"within {arm} (Spearman)", term=c, beta=r.correlation, p=r.pvalue, n=len(q)))
F = pd.DataFrame(fits)
F.to_csv(OUT / "tg_pretrain_fits.csv", index=False)
print(F.round(3).to_string())
