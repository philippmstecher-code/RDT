#!/usr/bin/env python3
"""Generate Nature-required Source Data Excel files.

For each main-text figure, creates one .xlsx file with one sheet per panel
containing the plotted numerical values.  Output goes to output/source_data/.

Reads from:
  - data/consolidated_findings.json  (Figures 1-4)
  - data/feature_survival_tg_expanded.json  (Figure 4)
  - data/targeted_label_noise_summary_5seeds.json  (Figure 5)
  - Hardcoded data from plot_fig8.py  (Figure 6 — curriculum switch)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "consolidated_findings.json"
OUT = ROOT / "output" / "source_data"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load master data ──
with open(DATA) as f:
    data = json.load(f)

# Helper: standard CIFAR-100 lanes (9 runs: 3 arch x 3 seeds, 50 ep, 4x)
CIFAR_LANES = sorted([
    k for k in data["lanes"]
    if data["lanes"][k]["metadata"]["dataset"] == "CIFAR-100"
    and data["lanes"][k]["metadata"]["epochs"] == 50
    and data["lanes"][k]["metadata"]["expansion"] == "4x"
])

TINYIMAGENET_LANES = sorted([
    k for k in data["lanes"]
    if data["lanes"][k]["metadata"]["dataset"] == "TinyImageNet"
])


def _cr_ratio(r):
    """(Tg-E + Ab-E) / Di-E from a ratio record."""
    return (r.get("tg_h", 0) + r["ab_h"]) / max(r["di_h"], 1)


# ════════════════════════════════════════════════════════════════════════════
# Figure 1: Panel (b) process fractions, Panel (c) C/R ratios
# ════════════════════════════════════════════════════════════════════════════
def generate_fig1():
    n_trans = min(len(data["lanes"][ln]["abh_dih_ratios"]) for ln in CIFAR_LANES)

    # Panel b: process fractions (mean across 9 lanes)
    tg_all = np.zeros((len(CIFAR_LANES), n_trans))
    ab_all = np.zeros((len(CIFAR_LANES), n_trans))
    di_all = np.zeros((len(CIFAR_LANES), n_trans))

    for i, ln in enumerate(CIFAR_LANES):
        for j, r in enumerate(data["lanes"][ln]["abh_dih_ratios"][:n_trans]):
            total = r["ab_h"] + r["tg_h"] + r["di_h"]
            if total > 0:
                tg_all[i, j] = r["tg_h"] / total * 100
                ab_all[i, j] = r["ab_h"] / total * 100
                di_all[i, j] = r["di_h"] / total * 100

    df_b = pd.DataFrame({
        "transition": list(range(n_trans)),
        "Tg-E_mean_pct": tg_all.mean(0),
        "Ab-E_mean_pct": ab_all.mean(0),
        "Di-E_mean_pct": di_all.mean(0),
    })

    # Panel c: C/R ratios per lane, median, IQR
    all_ratios = []
    for ln in CIFAR_LANES:
        ratios = data["lanes"][ln]["abh_dih_ratios"]
        vals = [_cr_ratio(r) for r in ratios[:n_trans]]
        vals = [v if np.isfinite(v) else 1000 for v in vals]
        all_ratios.append(vals)

    padded = np.array(all_ratios)
    rows_c = []
    for j in range(n_trans):
        row = {"transition": j, "median_CR": np.median(padded[:, j]),
               "Q25_CR": np.percentile(padded[:, j], 25),
               "Q75_CR": np.percentile(padded[:, j], 75)}
        for i, ln in enumerate(CIFAR_LANES):
            row[ln] = padded[i, j]
        rows_c.append(row)
    df_c = pd.DataFrame(rows_c)

    with pd.ExcelWriter(OUT / "Source Data Fig. 1.xlsx", engine="openpyxl") as w:
        df_b.to_excel(w, sheet_name="b", index=False)
        df_c.to_excel(w, sheet_name="c", index=False)
    print("  Source Data Fig. 1.xlsx")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2: (a) C/R initial vs final, (b) process per SC, (c) SSI, (d) CSI
# ════════════════════════════════════════════════════════════════════════════
def generate_fig2():
    cross = data["cross_lane_statistics"]

    # Panel a: initial vs final C/R ratio per architecture
    rows_a = []
    for ln in CIFAR_LANES + TINYIMAGENET_LANES:
        lane = data["lanes"][ln]
        ratios = lane["abh_dih_ratios"]
        if not ratios:
            continue
        rows_a.append({
            "lane": ln,
            "architecture": lane["metadata"]["architecture"],
            "dataset": lane["metadata"]["dataset"],
            "CR_initial": _cr_ratio(ratios[0]),
            "CR_final": _cr_ratio(ratios[-1]),
        })
    df_a = pd.DataFrame(rows_a)

    # Panel b: per-superclass process fractions (CIFAR-100 + TinyImageNet)
    inv_cifar = cross.get("f4_superclass_invariance", {})
    inv_tin = cross.get("f4_superclass_invariance_tinyimagenet", {})
    rows_b = []
    for proc_key in ["as_h", "de_h", "ab_h", "di_h", "tg_h"]:
        # CIFAR-100
        if proc_key == "tg_h":
            sc_vals = {}
            for ln in CIFAR_LANES:
                ss = data["lanes"][ln].get("superclass_summary", {})
                for sc, info in ss.items():
                    frac = info.get("process_fractions", {}).get("tg_h", 0)
                    sc_vals.setdefault(sc, []).append(frac)
            cifar_sc = {sc: np.mean(vals) for sc, vals in sc_vals.items()}
        else:
            cifar_sc = inv_cifar.get(proc_key, {}).get("per_superclass_mean", {})
        for sc, val in cifar_sc.items():
            rows_b.append({"dataset": "CIFAR-100", "process": proc_key,
                           "superclass": sc, "fraction_pct": val * 100})

        # TinyImageNet
        if proc_key == "tg_h":
            sc_vals = {}
            for ln in TINYIMAGENET_LANES:
                ss = data["lanes"][ln].get("superclass_summary", {})
                for sc, info in ss.items():
                    frac = info.get("process_fractions", {}).get("tg_h", 0)
                    sc_vals.setdefault(sc, []).append(frac)
            tin_sc = {sc: np.mean(vals) for sc, vals in sc_vals.items()}
        else:
            tin_sc = inv_tin.get(proc_key, {}).get("per_superclass_mean", {})
        for sc, val in tin_sc.items():
            rows_b.append({"dataset": "TinyImageNet", "process": proc_key,
                           "superclass": sc, "fraction_pct": val * 100})
    df_b = pd.DataFrame(rows_b)

    # Panel c: SSI trajectories
    N_CP = 10
    seed_lanes = ["ResNet18-CIFAR100-seed42", "ResNet18-CIFAR100-seed137",
                  "ResNet18-CIFAR100-seed256"]
    rows_c = []
    for ln in seed_lanes + TINYIMAGENET_LANES:
        if ln not in data["lanes"]:
            continue
        sel = data["lanes"][ln]["selectivity_evolution"][:N_CP]
        cp_epochs = data["lanes"][ln]["epoch_info"].get("checkpoint_epochs", [])
        for i, se in enumerate(sel):
            rows_c.append({"lane": ln, "checkpoint": i,
                           "epoch": cp_epochs[i] if i < len(cp_epochs) else i,
                           "mean_SSI": se["mean_ssi"]})
    df_c = pd.DataFrame(rows_c)

    # Panel d: CSI trajectories
    rows_d = []
    for ln in seed_lanes + TINYIMAGENET_LANES:
        if ln not in data["lanes"]:
            continue
        sel = data["lanes"][ln]["selectivity_evolution"][:N_CP]
        cp_epochs = data["lanes"][ln]["epoch_info"].get("checkpoint_epochs", [])
        for i, se in enumerate(sel):
            rows_d.append({"lane": ln, "checkpoint": i,
                           "epoch": cp_epochs[i] if i < len(cp_epochs) else i,
                           "mean_CSI": se["mean_csi"]})
    df_d = pd.DataFrame(rows_d)

    with pd.ExcelWriter(OUT / "Source Data Fig. 2.xlsx", engine="openpyxl") as w:
        df_a.to_excel(w, sheet_name="a", index=False)
        df_b.to_excel(w, sheet_name="b", index=False)
        df_c.to_excel(w, sheet_name="c", index=False)
        df_d.to_excel(w, sheet_name="d", index=False)
    print("  Source Data Fig. 2.xlsx")


# ════════════════════════════════════════════════════════════════════════════
# Figure 3: (a) churn rate, (b) birth/death events
# ════════════════════════════════════════════════════════════════════════════
def generate_fig3():
    N_TR = 9

    # Panel a: churn rate per 1k updates
    rows_a = []
    for ln in CIFAR_LANES:
        lane = data["lanes"][ln]
        pi = lane["process_intensity"]
        wu_gaps = lane["epoch_info"].get("transition_weight_update_gaps", [])
        if wu_gaps and len(wu_gaps) >= len(pi):
            churn = [p["churn"] * 100 / max(wu / 1000, 0.001) for p, wu in zip(pi, wu_gaps)]
        else:
            churn = [p["churn"] * 100 for p in pi]
        for j in range(min(N_TR, len(churn))):
            rows_a.append({"lane": ln, "transition": j,
                           "arch": lane["metadata"]["architecture"],
                           "churn_per_1k_updates": churn[j]})
    df_a = pd.DataFrame(rows_a)

    # Panel b: born/died per 1k updates
    rows_b = []
    for ln in CIFAR_LANES:
        fm = data["lanes"][ln]["feature_matching"]
        trans_keys = sorted(fm.keys(), key=lambda x: int(x.split("->")[0]))
        wu_gaps = data["lanes"][ln]["epoch_info"].get("transition_weight_update_gaps", [])
        for j, tk in enumerate(trans_keys[:N_TR]):
            born_raw = fm[tk]["n_born"]
            died_raw = fm[tk]["n_died"]
            if wu_gaps and j < len(wu_gaps):
                scale = max(wu_gaps[j] / 1000, 0.001)
                born = born_raw / scale
                died = died_raw / scale
            else:
                born = born_raw
                died = died_raw
            rows_b.append({"lane": ln, "transition": j,
                           "born_per_1k_updates": born,
                           "died_per_1k_updates": died})
    df_b = pd.DataFrame(rows_b)

    with pd.ExcelWriter(OUT / "Source Data Fig. 3.xlsx", engine="openpyxl") as w:
        df_a.to_excel(w, sheet_name="a", index=False)
        df_b.to_excel(w, sheet_name="b", index=False)
    print("  Source Data Fig. 3.xlsx")


# ════════════════════════════════════════════════════════════════════════════
# Figure 4: (a) lifespan ratio, (b) correlations, (c) persistence, (d) layer lifespan
# ════════════════════════════════════════════════════════════════════════════
def generate_fig4():
    tg_path = ROOT / "data" / "feature_survival_tg_expanded.json"
    with open(tg_path) as f:
        tg_expanded = json.load(f)

    surv = data.get("feature_survival", {})

    # Panel a: lifespan ratio (SAI+SSI) / CSI per lane
    standard_keys = sorted([
        k for k in tg_expanded
        if "CIFAR100-seed" in k and "200ep" not in k and "8x" not in k
    ])
    rows_a = []
    for k in standard_keys:
        sai_c = tg_expanded[k]["cohorts"]["tg"]
        ssi_c = tg_expanded[k]["cohorts"]["high_ssi"]
        csi_c = tg_expanded[k]["cohorts"]["high_csi"]
        combined_ml = (sai_c["mean_lifespan"] * sai_c["n"] +
                       ssi_c["mean_lifespan"] * ssi_c["n"]) / max(sai_c["n"] + ssi_c["n"], 1)
        ratio = combined_ml / max(csi_c["mean_lifespan"], 0.001)
        rows_a.append({
            "lane": k,
            "SAI_mean_lifespan": sai_c["mean_lifespan"],
            "SSI_mean_lifespan": ssi_c["mean_lifespan"],
            "CSI_mean_lifespan": csi_c["mean_lifespan"],
            "combined_SAI_SSI_lifespan": combined_ml,
            "lifespan_ratio": ratio,
        })
    df_a = pd.DataFrame(rows_a)

    # Panel b: Spearman correlations (SSI and CSI vs survival)
    standard_labels = sorted([
        l for l in surv if "CIFAR100" in l and "200ep" not in l and "8x" not in l
    ])
    rows_b = []
    for l in standard_labels:
        rows_b.append({
            "lane": l,
            "SSI_survival_rho": surv[l]["ssi_survival_corr"],
            "SSI_survival_p": surv[l].get("ssi_survival_p", None),
            "CSI_survival_rho": surv[l]["csi_survival_corr"],
            "CSI_survival_p": surv[l].get("csi_survival_p", None),
            "n_tracked": surv[l].get("n_total_tracked", None),
        })
    df_b = pd.DataFrame(rows_b)

    # Panel c: survival summary per cohort per lane (persistence)
    rows_c = []
    for k in standard_keys:
        for cohort_name in ["tg", "non_tg", "high_ssi", "low_ssi", "high_csi", "low_csi"]:
            c = tg_expanded[k]["cohorts"].get(cohort_name, {})
            rows_c.append({
                "lane": k, "cohort": cohort_name,
                "n": c.get("n", 0),
                "survival_rate": c.get("survival_rate", None),
                "mean_lifespan": c.get("mean_lifespan", None),
                "max_lifespan": c.get("max_lifespan", None),
            })
    df_c = pd.DataFrame(rows_c)

    # Panel d: per-layer lifespan
    rows_d = []
    for l in standard_labels:
        per_layer = surv[l].get("per_layer", {})
        for layer, info in per_layer.items():
            rows_d.append({
                "lane": l, "layer": layer,
                "n_tracked": info.get("n_tracked", None),
                "survival_rate": info.get("survival_rate", None),
                "mean_lifespan": info.get("mean_lifespan", None),
            })
    df_d = pd.DataFrame(rows_d)

    with pd.ExcelWriter(OUT / "Source Data Fig. 4.xlsx", engine="openpyxl") as w:
        df_a.to_excel(w, sheet_name="a", index=False)
        df_b.to_excel(w, sheet_name="b", index=False)
        df_c.to_excel(w, sheet_name="c", index=False)
        df_d.to_excel(w, sheet_name="d", index=False)
    print("  Source Data Fig. 4.xlsx")


# ════════════════════════════════════════════════════════════════════════════
# Figure 5 (label noise): (b) process fractions, (c) selective deficit,
#                          (d) trajectories
# ════════════════════════════════════════════════════════════════════════════
def generate_fig5():
    noise_path = ROOT / "data" / "targeted_label_noise_summary_5seeds.json"
    with open(noise_path) as f:
        noise = json.load(f)

    SEEDS = [str(s) for s in noise["seeds"]]
    CONDITION_KEYS = ["standard", "within_sc_p01", "within_sc_p03",
                      "between_sc_p01", "between_sc_p03", "random_p01", "random_p03"]

    # Panel b: process fractions per condition (mean +/- SEM)
    rows_b = []
    for cond in CONDITION_KEYS:
        s = noise["conditions"][cond]["summary"]
        rows_b.append({
            "condition": cond,
            "Ab-E_frac_mean_pct": s["ab_frac_mean"] * 100,
            "Ab-E_frac_sem_pct": s["ab_frac_sem"] * 100,
            "Di-E_frac_mean_pct": s["di_frac_mean"] * 100,
            "Di-E_frac_sem_pct": s["di_frac_sem"] * 100,
            "Tg-E_frac_mean_pct": s["tg_frac_mean"] * 100,
            "Tg-E_frac_sem_pct": s["tg_frac_sem"] * 100,
        })
    df_b = pd.DataFrame(rows_b)

    # Panel c: selective deficit (between_sc minus random at matching dose)
    # Panel c: Di-E shows a selective deficit of -7.8 pp (boundary noise
    # hurts differentiation more than random noise does)
    rows_c = []
    for dose in ["p01", "p03"]:
        bsc_cond = f"between_sc_{dose}"
        rnd_cond = f"random_{dose}"
        bsc_di = [noise["conditions"][bsc_cond]["per_seed"][s]["process_fractions"]["di_frac"] * 100 for s in SEEDS]
        rnd_di = [noise["conditions"][rnd_cond]["per_seed"][s]["process_fractions"]["di_frac"] * 100 for s in SEEDS]
        bsc_ab = [noise["conditions"][bsc_cond]["per_seed"][s]["process_fractions"]["ab_frac"] * 100 for s in SEEDS]
        rnd_ab = [noise["conditions"][rnd_cond]["per_seed"][s]["process_fractions"]["ab_frac"] * 100 for s in SEEDS]
        di_deltas = [bsc_di[i] - rnd_di[i] for i in range(len(SEEDS))]
        ab_deltas = [bsc_ab[i] - rnd_ab[i] for i in range(len(SEEDS))]
        rows_c.append({
            "condition": f"between_sc_{dose}_minus_random_{dose}",
            "Di-E_delta_mean_pp": np.mean(di_deltas),
            "Di-E_delta_sem_pp": np.std(di_deltas, ddof=1) / np.sqrt(len(SEEDS)),
            "Ab-E_delta_mean_pp": np.mean(ab_deltas),
            "Ab-E_delta_sem_pp": np.std(ab_deltas, ddof=1) / np.sqrt(len(SEEDS)),
        })
    df_c = pd.DataFrame(rows_c)

    # Panel d: per-transition trajectories for 3 conditions
    transitions = ["0->1", "1->2", "2->3", "3->4", "4->5", "5->6", "6->7"]
    traj_conds = ["standard", "between_sc_p03", "random_p03"]
    rows_d = []
    for cond in traj_conds:
        for frac_key in ["di_frac", "ab_frac"]:
            all_seeds = []
            for s in SEEDS:
                seed_vals = [noise["conditions"][cond]["per_seed"][s]["process_events_per_transition"][t][frac_key] * 100
                             for t in transitions]
                all_seeds.append(seed_vals)
            all_seeds = np.array(all_seeds)
            mean = np.mean(all_seeds, axis=0)
            sem = np.std(all_seeds, axis=0, ddof=1) / np.sqrt(len(SEEDS))
            process = "Di-E" if "di" in frac_key else "Ab-E"
            for j, t in enumerate(transitions):
                rows_d.append({
                    "condition": cond, "process": process,
                    "transition": t,
                    "fraction_mean_pct": mean[j],
                    "fraction_sem_pct": sem[j],
                })
    df_d = pd.DataFrame(rows_d)

    with pd.ExcelWriter(OUT / "Source Data Fig. 5.xlsx", engine="openpyxl") as w:
        df_b.to_excel(w, sheet_name="b", index=False)
        df_c.to_excel(w, sheet_name="c", index=False)
        df_d.to_excel(w, sheet_name="d", index=False)
    print("  Source Data Fig. 5.xlsx")


# ════════════════════════════════════════════════════════════════════════════
# Figure 6 (curriculum switch):
#   (a) accuracy, (b) Di-E events, (c) composition, (d) overfit gap
# ════════════════════════════════════════════════════════════════════════════
def generate_fig6():
    CONDITIONS = ["standard", "switch_e05", "switch_e10", "switch_e15",
                  "switch_e20", "switch_e25", "switch_e30"]
    SC_EPOCHS = [0, 5, 10, 15, 20, 25, 30]

    FINE_TRAJECTORIES = {
        "standard":   {"epochs": [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                       "acc":    [13.0, 29.8, 35.5, 43.9, 44.5, 49.3, 51.5, 54.2, 57.1, 58.8, 59.1]},
        "switch_e05": {"epochs": [10, 15, 20, 25, 30, 35, 40, 45, 50],
                       "acc":    [27.4, 35.5, 41.6, 42.7, 47.4, 51.1, 54.3, 56.7, 57.9]},
        "switch_e10": {"epochs": [15, 20, 25, 30, 35, 40, 45, 50],
                       "acc":    [30.7, 39.2, 42.6, 48.0, 48.9, 54.7, 57.2, 58.2]},
        "switch_e15": {"epochs": [20, 25, 30, 35, 40, 45, 50],
                       "acc":    [33.5, 40.4, 43.4, 48.0, 51.8, 55.4, 57.1]},
        "switch_e20": {"epochs": [25, 30, 35, 40, 45, 50],
                       "acc":    [38.4, 42.9, 46.4, 51.9, 55.7, 57.9]},
        "switch_e25": {"epochs": [30, 35, 40, 45, 50],
                       "acc":    [39.5, 44.8, 49.4, 55.1, 58.2]},
        "switch_e30": {"epochs": [35, 40, 45, 50],
                       "acc":    [40.1, 47.6, 52.5, 58.2]},
    }
    SC_TRAJECTORIES = {
        "switch_e05": {"epochs": [1, 5],                      "acc": [14.5, 31.2]},
        "switch_e10": {"epochs": [1, 5, 10],                  "acc": [19.3, 36.2, 46.4]},
        "switch_e15": {"epochs": [15],                        "acc": [43.9]},
        "switch_e20": {"epochs": [1, 5, 10, 15, 20],         "acc": [17.8, 34.5, 44.1, 53.0, 58.4]},
        "switch_e25": {"epochs": [1, 5, 10, 15, 20, 25],     "acc": [16.2, 34.7, 42.3, 50.7, 57.8, 62.2]},
        "switch_e30": {"epochs": [1, 5, 10, 15, 20, 25, 30], "acc": [15.9, 34.1, 41.5, 50.8, 57.2, 61.7, 64.5]},
    }

    DIH_EVENTS = [358816, 99974, 96357, 53872, 91430, 55315, 77578]
    ABH_EVENTS = [72727, 28744, 34326, 9214, 41686, 30872, 25424]
    TGH_EVENTS = [218519, 175589, 288343, 285110, 247141, 244427, 207218]
    TRAIN_ACC = [96.19, 90.89, 89.83, 84.91, 83.06, 79.83, 76.69]
    VAL_ACC   = [59.13, 57.86, 58.21, 57.07, 57.85, 58.24, 58.21]

    # Panel a: accuracy trajectories
    rows_a = []
    for cond in CONDITIONS:
        ft = FINE_TRAJECTORIES[cond]
        for ep, acc in zip(ft["epochs"], ft["acc"]):
            rows_a.append({"condition": cond, "phase": "fine", "epoch": ep, "val_acc_pct": acc})
        if cond in SC_TRAJECTORIES:
            st = SC_TRAJECTORIES[cond]
            for ep, acc in zip(st["epochs"], st["acc"]):
                rows_a.append({"condition": cond, "phase": "superclass", "epoch": ep, "val_acc_pct": acc})
    df_a = pd.DataFrame(rows_a)

    # Panel b: Di-E events
    df_b = pd.DataFrame({
        "condition": CONDITIONS,
        "SC_epochs": SC_EPOCHS,
        "Di-E_events_total": DIH_EVENTS,
    })

    # Panel c: process composition
    total_3 = [a + t + d for a, t, d in zip(ABH_EVENTS, TGH_EVENTS, DIH_EVENTS)]
    df_c = pd.DataFrame({
        "condition": CONDITIONS,
        "Ab-E_events": ABH_EVENTS,
        "Tg-E_events": TGH_EVENTS,
        "Di-E_events": DIH_EVENTS,
        "Ab-E_frac_pct": [a / s * 100 for a, s in zip(ABH_EVENTS, total_3)],
        "Tg-E_frac_pct": [t / s * 100 for t, s in zip(TGH_EVENTS, total_3)],
        "Di-E_frac_pct": [d / s * 100 for d, s in zip(DIH_EVENTS, total_3)],
    })

    # Panel d: overfit gap
    df_d = pd.DataFrame({
        "condition": CONDITIONS,
        "SC_epochs": SC_EPOCHS,
        "train_acc_pct": TRAIN_ACC,
        "val_acc_pct": VAL_ACC,
        "overfit_gap_pp": [t - v for t, v in zip(TRAIN_ACC, VAL_ACC)],
    })

    with pd.ExcelWriter(OUT / "Source Data Fig. 6.xlsx", engine="openpyxl") as w:
        df_a.to_excel(w, sheet_name="a", index=False)
        df_b.to_excel(w, sheet_name="b", index=False)
        df_c.to_excel(w, sheet_name="c", index=False)
        df_d.to_excel(w, sheet_name="d", index=False)
    print("  Source Data Fig. 6.xlsx")


# ════════════════════════════════════════════════════════════════════════════
# ED Figure 1: Methodology validation (6 panels)
# ════════════════════════════════════════════════════════════════════════════
def generate_ed_fig1():
    try:
        lane = data["lanes"]["ResNet18-CIFAR100-seed42"]
        meta = lane["metadata"]
        wcc = lane["within_checkpoint_control"]["per_checkpoint"]
        fm = lane["feature_matching"]
        ratios = lane["abh_dih_ratios"]
        nb = lane["null_baseline"]["per_checkpoint"]
        at = meta["adaptive_thresholds"]

        layers = sorted(meta["layers"])

        # Panel a: false-positive rates per checkpoint (avg over layers)
        rows_a = []
        for entry in wcc:
            fb = np.mean([l["false_birth_rate"] for l in entry["per_layer"]])
            fd = np.mean([l["false_death_rate"] for l in entry["per_layer"]])
            rows_a.append({
                "checkpoint": entry["checkpoint"],
                "false_birth_rate_pct": fb * 100,
                "false_death_rate_pct": fd * 100,
            })
        df_a = pd.DataFrame(rows_a)

        # Panel b: within-checkpoint stability per layer (avg over checkpoints)
        rows_b = []
        for layer in layers:
            stable_vals = []
            for entry in wcc:
                for ldata in entry["per_layer"]:
                    if ldata["layer"] == layer:
                        stable_vals.append(ldata["stable_rate"])
            rows_b.append({
                "layer": layer,
                "stable_rate_mean_pct": np.mean(stable_vals) * 100,
                "stable_rate_std_pct": np.std(stable_vals) * 100,
            })
        df_b = pd.DataFrame(rows_b)

        # Panel c: independent init C/R ratio trajectory
        rows_c = []
        for i, r in enumerate(ratios):
            rows_c.append({
                "transition": i,
                "ab_di_ratio": r["ratio"],
                "ab_h": r["ab_h"],
                "di_h": r["di_h"],
                "tg_h": r.get("tg_h", 0),
            })
        df_c = pd.DataFrame(rows_c)

        # Panel d: observed vs null SSI
        rows_d = []
        for entry in nb:
            rows_d.append({
                "checkpoint": entry["checkpoint"],
                "observed_ssi": entry["observed_ssi"],
                "null_ssi": entry["null_ssi"],
                "p_value": entry["p_value"],
            })
        df_d = pd.DataFrame(rows_d)

        # Panel e: adaptive thresholds
        rows_e = []
        for idx_name in ["ssi", "csi", "sai"]:
            rows_e.append({
                "index": idx_name.upper(),
                "adaptive_threshold": at.get(f"{idx_name}_adaptive_thresh", None),
                "floor": at.get(f"{idx_name}_floor", None) or {"ssi": 0.1, "csi": 0.15, "sai": 0.5}.get(idx_name),
                "null_mean": at.get(f"{idx_name}_null_mean", None),
                "null_std": at.get(f"{idx_name}_null_std", None),
                "n_null_samples": at.get(f"n_null_{idx_name}_samples", None),
            })
        df_e = pd.DataFrame(rows_e)

        # Panel f: p-values across checkpoints (same as d, kept as separate sheet for clarity)
        df_f = pd.DataFrame([{"checkpoint": e["checkpoint"], "p_value": e["p_value"]} for e in nb])

        with pd.ExcelWriter(OUT / "Source Data ED Fig. 1.xlsx", engine="openpyxl") as w:
            df_a.to_excel(w, sheet_name="a_false_positive_rates", index=False)
            df_b.to_excel(w, sheet_name="b_stability", index=False)
            df_c.to_excel(w, sheet_name="c_CR_ratio", index=False)
            df_d.to_excel(w, sheet_name="d_observed_vs_null_SSI", index=False)
            df_e.to_excel(w, sheet_name="e_adaptive_thresholds", index=False)
            df_f.to_excel(w, sheet_name="f_p_values", index=False)
        print("  Source Data ED Fig. 1.xlsx")
    except Exception as e:
        print(f"  SKIP ED Fig. 1: {e}")


# ════════════════════════════════════════════════════════════════════════════
# ED Figure 2: Layer-wise stability heatmaps (3 arch x 3 metrics)
# ════════════════════════════════════════════════════════════════════════════
def generate_ed_fig2():
    try:
        import re

        def _layer_sort_key(name):
            m = re.search(r"\.?(\d+)$", name)
            if m:
                return (0, int(m.group(1)))
            return (1, 0)

        ARCH_CONFIGS = {
            "ResNet-18": "ResNet18-CIFAR100-seed42",
            "ViT-Small": "ViTSmall-CIFAR100-seed42",
            "CCT-7": "CCT7-CIFAR100-seed42",
        }

        rows = []
        for arch_name, lane_key in ARCH_CONFIGS.items():
            if lane_key not in data["lanes"]:
                continue
            lane_d = data["lanes"][lane_key]
            layers = sorted(lane_d["metadata"]["layers"], key=_layer_sort_key)
            fm = lane_d["feature_matching"]
            fl = lane_d["feature_landscape"]

            # stable_rate from feature_matching (per transition)
            transitions = sorted(fm.keys(), key=lambda t: int(t.split("->")[0]))
            for ti, t in enumerate(transitions):
                pl = fm[t].get("per_layer", {})
                for layer in layers:
                    if layer in pl:
                        total = (pl[layer]["n_stable"] + pl[layer]["n_born"]
                                 + pl[layer]["n_died"] + pl[layer]["n_transformed"])
                        sr = pl[layer]["n_stable"] / max(total, 1)
                        rows.append({
                            "architecture": arch_name, "layer": layer,
                            "transition": t, "metric": "stable_rate", "value": sr,
                        })

            # mean_ssi and mean_csi from feature_landscape (per checkpoint)
            checkpoints = sorted(fl.keys(), key=lambda x: int(x) if x.isdigit() else 999)
            for ci, ckpt in enumerate(checkpoints):
                for layer in layers:
                    if layer in fl[ckpt]:
                        for metric in ["mean_ssi", "mean_csi"]:
                            rows.append({
                                "architecture": arch_name, "layer": layer,
                                "checkpoint": ckpt, "metric": metric,
                                "value": fl[ckpt][layer].get(metric, 0),
                            })

        df = pd.DataFrame(rows)

        with pd.ExcelWriter(OUT / "Source Data ED Fig. 2.xlsx", engine="openpyxl") as w:
            for metric in ["stable_rate", "mean_ssi", "mean_csi"]:
                sub = df[df["metric"] == metric].copy()
                sub.to_excel(w, sheet_name=metric, index=False)
        print("  Source Data ED Fig. 2.xlsx")
    except Exception as e:
        print(f"  SKIP ED Fig. 2: {e}")


# ════════════════════════════════════════════════════════════════════════════
# ED Figure 3: Architecture temporal profiles (3 arch x 3 processes, 3 seeds)
# ════════════════════════════════════════════════════════════════════════════
def generate_ed_fig3():
    try:
        ARCH_LANES = {
            "ResNet-18": ["ResNet18-CIFAR100-seed42", "ResNet18-CIFAR100-seed137",
                          "ResNet18-CIFAR100-seed256"],
            "ViT-Small": ["ViTSmall-CIFAR100-seed42", "ViTSmall-CIFAR100-seed137",
                          "ViTSmall-CIFAR100-seed256"],
            "CCT-7": ["CCT7-CIFAR100-seed42", "CCT7-CIFAR100-seed137",
                      "CCT7-CIFAR100-seed256"],
        }
        rows = []
        for arch_name, lanes in ARCH_LANES.items():
            for lane_key in lanes:
                if lane_key not in data["lanes"]:
                    continue
                lane_d = data["lanes"][lane_key]
                pi = lane_d["process_intensity"]
                seed = lane_d["metadata"]["seed"]
                wu_gaps = lane_d["epoch_info"].get("transition_weight_update_gaps", [])
                for i, entry in enumerate(pi):
                    gap = wu_gaps[i] if i < len(wu_gaps) else (wu_gaps[-1] if wu_gaps else 1)
                    for proc in ["ab_h", "di_h", "tg_h"]:
                        rows.append({
                            "architecture": arch_name, "lane": lane_key,
                            "seed": seed, "transition": i,
                            "process": proc,
                            "raw_events": entry[proc],
                            "events_per_1k_updates": entry[proc] / gap * 1000 if gap > 0 else 0,
                        })
        df = pd.DataFrame(rows)

        with pd.ExcelWriter(OUT / "Source Data ED Fig. 3.xlsx", engine="openpyxl") as w:
            for proc in ["ab_h", "di_h", "tg_h"]:
                sub = df[df["process"] == proc].copy()
                sub.to_excel(w, sheet_name=proc.replace("_h", "_E"), index=False)
        print("  Source Data ED Fig. 3.xlsx")
    except Exception as e:
        print(f"  SKIP ED Fig. 3: {e}")


# ════════════════════════════════════════════════════════════════════════════
# ED Figure 4: Robustness — expansion factor (4x vs 8x) & training duration
# ════════════════════════════════════════════════════════════════════════════
def generate_ed_fig4():
    try:
        COMPARE_LANES = {
            "standard_4x": "ResNet18-CIFAR100-seed42",
            "expansion_8x": "ResNet18-CIFAR100-8x",
            "duration_200ep": "ResNet18-CIFAR100-200ep",
        }

        # Panels a-c: 4x vs 8x (ratio, process intensity, churn)
        # Panels d-f: 50ep vs 200ep (ratio, selectivity, churn)
        rows_ratio = []
        rows_proc = []
        rows_churn = []
        rows_sel = []

        for cond_label, lane_key in COMPARE_LANES.items():
            if lane_key not in data["lanes"]:
                continue
            lane_d = data["lanes"][lane_key]

            # Ratio
            for i, r in enumerate(lane_d["abh_dih_ratios"]):
                rows_ratio.append({
                    "condition": cond_label, "transition": i,
                    "ab_di_ratio": r["ratio"],
                })

            # Process intensity
            pi = lane_d["process_intensity"]
            for i, entry in enumerate(pi):
                rows_proc.append({
                    "condition": cond_label, "transition": i,
                    "ab_h": entry["ab_h"], "di_h": entry["di_h"],
                    "tg_h": entry.get("tg_h", 0),
                })

            # Churn
            for i, entry in enumerate(pi):
                rows_churn.append({
                    "condition": cond_label, "transition": i,
                    "churn_pct": entry["churn"] * 100,
                })

            # Selectivity evolution
            for i, se in enumerate(lane_d["selectivity_evolution"]):
                rows_sel.append({
                    "condition": cond_label, "checkpoint": i,
                    "mean_ssi": se["mean_ssi"], "mean_csi": se["mean_csi"],
                })

        with pd.ExcelWriter(OUT / "Source Data ED Fig. 4.xlsx", engine="openpyxl") as w:
            pd.DataFrame(rows_ratio).to_excel(w, sheet_name="a_d_ratio", index=False)
            pd.DataFrame(rows_proc).to_excel(w, sheet_name="b_e_process_intensity", index=False)
            pd.DataFrame(rows_churn).to_excel(w, sheet_name="c_f_churn", index=False)
            pd.DataFrame(rows_sel).to_excel(w, sheet_name="e_selectivity", index=False)
        print("  Source Data ED Fig. 4.xlsx")
    except Exception as e:
        print(f"  SKIP ED Fig. 4: {e}")


# ════════════════════════════════════════════════════════════════════════════
# ED Figure 5: Granger causality (6 panels)
# ════════════════════════════════════════════════════════════════════════════
def generate_ed_fig5():
    try:
        granger_path = ROOT / "data" / "granger_causality_results.json"
        sc_series_path = ROOT / "data" / "superclass_transition_series.json"
        tg_path = ROOT / "data" / "granger_causality_tg_results.json"
        tg_crit_path = ROOT / "data" / "granger_causality_tg_critical_results.json"
        onset_path = ROOT / "data" / "granger_causality_onset_results.json"

        sheets = {}

        # Panel a: Pooled Ab->Di Granger (from granger_causality_results.json)
        if granger_path.exists():
            with open(granger_path) as f:
                gc = json.load(f)
            rows_pooled = [{
                "test": "Ab-E -> Di-E (pooled)",
                "F": gc["pooled"]["granger_f"],
                "p": gc["pooled"]["granger_p"],
                "delta_r2": gc["pooled"]["delta_r2"],
                "coef": gc["pooled"]["ab_h_coef"],
            }, {
                "test": "Reverse: Di-E -> Ab-E",
                "F": gc["reverse"]["granger_f"],
                "p": gc["reverse"]["granger_p"],
                "delta_r2": gc["reverse"].get("delta_r2", None),
                "coef": gc["reverse"].get("coef", None),
            }]
            sheets["a_pooled_granger"] = pd.DataFrame(rows_pooled)

            # Panel b: Per-superclass Granger
            rows_sc = []
            for sc, vals in gc["per_superclass"].items():
                rows_sc.append({
                    "superclass": sc, "F": vals["F"], "p": vals["p"],
                    "p_adj": vals["p_adj"], "coef": vals["coef"], "sig": vals["sig"],
                })
            sheets["b_per_superclass"] = pd.DataFrame(rows_sc)

        # Panel c: Tg-E Granger (tg_to_ab, tg_to_di)
        if tg_path.exists():
            with open(tg_path) as f:
                tg = json.load(f)
            rows_tg = []
            for direction in ["tg_to_ab", "tg_to_di"]:
                if direction in tg:
                    pooled = tg[direction]["pooled"]
                    rows_tg.append({
                        "direction": direction,
                        "F": pooled["granger_f"], "p": pooled["granger_p"],
                        "delta_r2": pooled["delta_r2"], "coef": pooled["coef"],
                    })
                    rev = tg[direction].get("reverse", {})
                    rows_tg.append({
                        "direction": f"{direction}_reverse",
                        "F": rev.get("granger_f"), "p": rev.get("granger_p"),
                        "delta_r2": rev.get("delta_r2"), "coef": rev.get("coef"),
                    })
            sheets["c_tg_granger"] = pd.DataFrame(rows_tg)

            # Per-superclass for Tg
            rows_tg_sc = []
            for direction in ["tg_to_ab", "tg_to_di"]:
                if direction in tg and "per_superclass" in tg[direction]:
                    for sc, vals in tg[direction]["per_superclass"].items():
                        rows_tg_sc.append({
                            "direction": direction, "superclass": sc,
                            "F": vals["F"], "p": vals["p"],
                            "p_adj": vals["p_adj"], "coef": vals["coef"],
                            "sig": vals["sig"],
                        })
            if rows_tg_sc:
                sheets["d_tg_per_superclass"] = pd.DataFrame(rows_tg_sc)

        # Panel e: Critical period Granger
        if tg_crit_path.exists():
            with open(tg_crit_path) as f:
                tg_crit = json.load(f)
            rows_crit = []
            gcp = tg_crit.get("granger_critical_period", {})
            for test_name, vals in gcp.items():
                rows_crit.append({
                    "test": test_name,
                    "F": vals.get("granger_f"), "p": vals.get("granger_p"),
                    "delta_r2": vals.get("delta_r2"), "coef": vals.get("coef"),
                    "sig": vals.get("sig"),
                })
            sheets["e_critical_period"] = pd.DataFrame(rows_crit)

        # Panel f: Onset results
        if onset_path.exists():
            with open(onset_path) as f:
                onset = json.load(f)
            rows_onset = []
            for test_key in ["ab_to_di", "ab_to_di_reverse"]:
                if test_key in onset:
                    v = onset[test_key]
                    rows_onset.append({
                        "test": test_key,
                        "n_obs": v.get("n_obs"),
                        "F": v.get("granger_f"), "p": v.get("granger_p"),
                        "delta_r2": v.get("delta_r2"), "coef": v.get("coef"),
                        "sig": v.get("sig"),
                    })
            sheets["f_onset_granger"] = pd.DataFrame(rows_onset)

        if sheets:
            with pd.ExcelWriter(OUT / "Source Data ED Fig. 5.xlsx", engine="openpyxl") as w:
                for name, df in sheets.items():
                    df.to_excel(w, sheet_name=name[:31], index=False)
            print("  Source Data ED Fig. 5.xlsx")
        else:
            print("  SKIP ED Fig. 5: no Granger data files found")
    except Exception as e:
        print(f"  SKIP ED Fig. 5: {e}")


# ════════════════════════════════════════════════════════════════════════════
# ED Figure 6: Cross-architecture superclass invariance (4 panels)
# ════════════════════════════════════════════════════════════════════════════
def generate_ed_fig6():
    try:
        inv = data["cross_lane_statistics"]["f4_superclass_invariance"]

        # Identify CIFAR-100 standard lanes per architecture
        arch_lanes = {}
        for label in data["lanes"]:
            meta = data["lanes"][label]["metadata"]
            if (meta["dataset"] == "CIFAR-100" and meta["epochs"] == 50
                    and meta["expansion"] == "4x"):
                arch_lanes.setdefault(meta["architecture"], []).append(label)

        arch_names = ["ResNet-18", "ViT-Small", "CCT-7"]
        superclasses = sorted(
            data["lanes"][arch_lanes["ResNet-18"][0]]["superclass_summary"].keys()
        )

        # Panel a+b: mean fractions (dominant + minor processes)
        rows_mean = []
        for sc in superclasses:
            row = {"superclass": sc}
            for proc in ["as_h", "de_h", "tg_h", "ab_h", "di_h"]:
                # Mean across 3 arch, each averaged across 3 seeds
                arch_vals = []
                for arch in arch_names:
                    seed_vals = []
                    for lb in arch_lanes.get(arch, []):
                        pf = data["lanes"][lb]["superclass_summary"][sc]["process_fractions"]
                        seed_vals.append(pf.get(proc, 0) * 100)
                    if seed_vals:
                        arch_vals.append(np.mean(seed_vals))
                row[f"{proc}_mean_pct"] = np.mean(arch_vals) if arch_vals else 0
                row[f"{proc}_sd_pct"] = np.std(arch_vals) if arch_vals else 0
            rows_mean.append(row)
        df_mean = pd.DataFrame(rows_mean)

        # Panel c: per-architecture breakdowns
        rows_arch = []
        for arch in arch_names:
            for sc in superclasses:
                row = {"architecture": arch, "superclass": sc}
                seed_vals = {p: [] for p in ["as_h", "de_h", "tg_h", "ab_h", "di_h"]}
                for lb in arch_lanes.get(arch, []):
                    pf = data["lanes"][lb]["superclass_summary"][sc]["process_fractions"]
                    for p in seed_vals:
                        seed_vals[p].append(pf.get(p, 0) * 100)
                for p in seed_vals:
                    row[f"{p}_mean_pct"] = np.mean(seed_vals[p]) if seed_vals[p] else 0
                rows_arch.append(row)
        df_arch = pd.DataFrame(rows_arch)

        # Panel d: cross_lane_statistics summary
        rows_inv = []
        for proc in ["ab_h", "di_h", "as_h", "de_h"]:
            entry = inv[proc]
            rows_inv.append({
                "process": proc,
                "overall_mean": entry["overall_mean"],
                "range_pp": entry["range_pp"],
            })
        df_inv = pd.DataFrame(rows_inv)

        with pd.ExcelWriter(OUT / "Source Data ED Fig. 6.xlsx", engine="openpyxl") as w:
            df_mean.to_excel(w, sheet_name="a_b_cross_arch_mean_sd", index=False)
            df_arch.to_excel(w, sheet_name="c_per_architecture", index=False)
            df_inv.to_excel(w, sheet_name="d_invariance_summary", index=False)
        print("  Source Data ED Fig. 6.xlsx")
    except Exception as e:
        print(f"  SKIP ED Fig. 6: {e}")


# ════════════════════════════════════════════════════════════════════════════
# ED Figure 7: Compensatory dynamics under label noise (2 panels)
# ════════════════════════════════════════════════════════════════════════════
def generate_ed_fig7():
    try:
        noise_path = ROOT / "data" / "targeted_label_noise_summary_5seeds.json"
        with open(noise_path) as f:
            noise = json.load(f)

        SEEDS = [str(s) for s in noise["seeds"]]
        conditions = noise["conditions"]
        cond_keys = ["standard", "within_sc_p01", "between_sc_p01", "random_p01",
                     "within_sc_p03", "between_sc_p03", "random_p03"]

        # Panel a: terminal alive features per condition (mean +/- SEM)
        rows_a = []
        for ck in cond_keys:
            if ck not in conditions:
                continue
            vals = []
            for s in SEEDS:
                sd = conditions[ck]["per_seed"].get(s, {})
                se = sd.get("selectivity_evolution", {})
                if se:
                    last_key = max(se.keys(), key=lambda k: int(k))
                    v = se[last_key].get("n_alive_mean", None)
                    if v is not None:
                        vals.append(v)
            rows_a.append({
                "condition": ck,
                "terminal_alive_mean": np.mean(vals) if vals else None,
                "terminal_alive_sem": (np.std(vals, ddof=1) / np.sqrt(len(vals))
                                       if len(vals) >= 2 else 0),
                "n_seeds": len(vals),
            })
        df_a = pd.DataFrame(rows_a)

        # Panel b: Di-E fraction trajectories (multi-seed mean +/- SEM)
        traj_conds = ["standard", "between_sc_p03", "random_p03"]
        rows_b = []
        for ck in traj_conds:
            if ck not in conditions:
                continue
            per_seed_fracs = {}
            trans_keys = None
            for s in SEEDS:
                sd = conditions[ck]["per_seed"].get(s, {})
                pe = sd.get("process_events_per_transition", {})
                if not pe:
                    continue
                sorted_trans = sorted(pe.keys(), key=lambda t: int(t.split("->")[0]))
                if trans_keys is None:
                    trans_keys = sorted_trans
                per_seed_fracs[s] = [pe[t].get("di_frac", 0) for t in sorted_trans]

            if trans_keys is None:
                continue
            for ti, t in enumerate(trans_keys):
                vals = [per_seed_fracs[s][ti] for s in per_seed_fracs
                        if ti < len(per_seed_fracs[s])]
                rows_b.append({
                    "condition": ck, "transition": t,
                    "di_frac_mean_pct": np.mean(vals) * 100,
                    "di_frac_sem_pct": (np.std(vals, ddof=1) / np.sqrt(len(vals)) * 100
                                        if len(vals) >= 2 else 0),
                })
        df_b = pd.DataFrame(rows_b)

        with pd.ExcelWriter(OUT / "Source Data ED Fig. 7.xlsx", engine="openpyxl") as w:
            df_a.to_excel(w, sheet_name="a_compensatory_proliferation", index=False)
            df_b.to_excel(w, sheet_name="b_DiE_trajectories", index=False)
        print("  Source Data ED Fig. 7.xlsx")
    except Exception as e:
        print(f"  SKIP ED Fig. 7: {e}")


# ════════════════════════════════════════════════════════════════════════════
# ED Figure 8: Selectivity vs activation magnitude (placeholder)
# ════════════════════════════════════════════════════════════════════════════
def generate_ed_fig8():
    try:
        # This figure requires raw per-feature activation data which may not
        # be in consolidated_findings.json. Extract what we can from
        # feature_landscape (mean_ssi, mean_csi, mean_sai per layer per ckpt).
        lane = data["lanes"]["ResNet18-CIFAR100-seed42"]
        fl = lane["feature_landscape"]
        layers = sorted(lane["metadata"]["layers"])
        checkpoints = sorted(fl.keys(), key=lambda x: int(x) if x.isdigit() else 999)

        rows = []
        for ckpt in checkpoints:
            for layer in layers:
                if layer in fl[ckpt]:
                    entry = fl[ckpt][layer]
                    rows.append({
                        "checkpoint": ckpt, "layer": layer,
                        "n_alive": entry.get("n_alive", 0),
                        "mean_ssi": entry.get("mean_ssi", 0),
                        "mean_csi": entry.get("mean_csi", 0),
                        "mean_sai": entry.get("mean_sai", 0),
                        "n_high_ssi": entry.get("n_high_ssi", 0),
                        "n_high_sai": entry.get("n_high_sai", 0),
                    })

        df = pd.DataFrame(rows)
        note_df = pd.DataFrame([{
            "note": ("Per-feature activation magnitudes are not stored in "
                     "consolidated_findings.json. This sheet provides aggregate "
                     "selectivity metrics per layer per checkpoint as a proxy. "
                     "Raw per-feature data would need to be regenerated from "
                     "SAE activation tensors.")
        }])

        with pd.ExcelWriter(OUT / "Source Data ED Fig. 8.xlsx", engine="openpyxl") as w:
            note_df.to_excel(w, sheet_name="note", index=False)
            df.to_excel(w, sheet_name="selectivity_per_layer", index=False)
        print("  Source Data ED Fig. 8.xlsx (placeholder — see note sheet)")
    except Exception as e:
        print(f"  SKIP ED Fig. 8: {e}")


# ════════════════════════════════════════════════════════════════════════════
# ED Table 1: Overview of all 55 runs
# ════════════════════════════════════════════════════════════════════════════
def generate_ed_table1():
    try:
        rows = []

        # Standard lanes from consolidated_findings.json
        for ln, lane_d in data["lanes"].items():
            meta = lane_d["metadata"]
            se = lane_d["selectivity_evolution"]
            pi = lane_d["process_intensity"]
            total_ab = sum(e["ab_h"] for e in pi)
            total_di = sum(e["di_h"] for e in pi)
            total_tg = sum(e.get("tg_h", 0) for e in pi)
            rows.append({
                "lane": ln,
                "experiment": "standard",
                "architecture": meta["architecture"],
                "dataset": meta["dataset"],
                "seed": meta.get("seed", ""),
                "epochs": meta["epochs"],
                "expansion": meta["expansion"],
                "n_checkpoints": meta["n_checkpoints"],
                "n_transitions": meta["n_transitions"],
                "total_Ab-E": total_ab,
                "total_Di-E": total_di,
                "total_Tg-E": total_tg,
                "terminal_mean_ssi": se[-1]["mean_ssi"] if se else None,
                "terminal_mean_csi": se[-1]["mean_csi"] if se else None,
            })

        # Curriculum switch lanes
        cs_path = ROOT / "data" / "curriculum_switch_summary.json"
        if cs_path.exists():
            with open(cs_path) as f:
                cs = json.load(f)
            for i, cond in enumerate(cs.get("conditions", [])):
                if cond == "standard":
                    continue  # already counted above
                rows.append({
                    "lane": f"curriculum_{cond}",
                    "experiment": "curriculum_switch",
                    "architecture": "ResNet-18",
                    "dataset": "CIFAR-100",
                    "seed": 42,
                    "epochs": 50,
                    "expansion": "4x",
                    "n_checkpoints": None,
                    "n_transitions": None,
                    "total_Ab-E": None,
                    "total_Di-E": None,
                    "total_Tg-E": None,
                    "terminal_mean_ssi": None,
                    "terminal_mean_csi": None,
                    "val_acc_terminal": cs["accuracy"]["val_acc_terminal"][i]
                        if i < len(cs["accuracy"]["val_acc_terminal"]) else None,
                })

        # Label noise lanes
        noise_path = ROOT / "data" / "targeted_label_noise_summary_5seeds.json"
        if noise_path.exists():
            with open(noise_path) as f:
                noise = json.load(f)
            seeds = [str(s) for s in noise["seeds"]]
            for cond_key, cond_data in noise["conditions"].items():
                for s in seeds:
                    sd = cond_data["per_seed"].get(s, {})
                    rows.append({
                        "lane": f"noise_{cond_key}_seed{s}",
                        "experiment": "label_noise",
                        "architecture": "ResNet-18",
                        "dataset": "CIFAR-100",
                        "seed": int(s),
                        "epochs": 30,
                        "expansion": "4x",
                        "n_checkpoints": None,
                        "n_transitions": None,
                        "total_Ab-E": sd.get("process_events_total", {}).get("Ab-H"),
                        "total_Di-E": sd.get("process_events_total", {}).get("Di-H"),
                        "total_Tg-E": sd.get("process_events_total", {}).get("Tg-H"),
                        "terminal_mean_ssi": None,
                        "terminal_mean_csi": None,
                    })

        df = pd.DataFrame(rows)
        with pd.ExcelWriter(OUT / "Source Data ED Table 1.xlsx", engine="openpyxl") as w:
            df.to_excel(w, sheet_name="all_runs", index=False)
        print(f"  Source Data ED Table 1.xlsx  ({len(df)} runs)")
    except Exception as e:
        print(f"  SKIP ED Table 1: {e}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating Nature Source Data Excel files...")
    print(f"Output directory: {OUT}\n")
    print("── Main Figures ──")
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    generate_fig6()
    print("\n── Extended Data Figures ──")
    generate_ed_fig1()
    generate_ed_fig2()
    generate_ed_fig3()
    generate_ed_fig4()
    generate_ed_fig5()
    generate_ed_fig6()
    generate_ed_fig7()
    generate_ed_fig8()
    print("\n── Extended Data Tables ──")
    generate_ed_table1()
    print("\nDone. All source data files written.")
