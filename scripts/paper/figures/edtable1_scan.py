#!/usr/bin/env python3
"""Extended Data Table 1: overview of all training runs, one row per condition.

Scans every run directory (read only) for its checkpoint record (dev_snapshots/metrics/devtrain_snapshots.json)
and SAE metadata (sae_analysis/sae_results.json), then writes
  analysis_outputs/paper/figures/final/source_data/edtable1.csv        one row per condition
  analysis_outputs/paper/figures/final/source_data/edtable1_runs.csv   one row per run (consecutive run numbers)
  OUT_XML                                                                 the table as a single <w:tbl> in the
                                                                          layout of Extended Data Table 1

Usage: python scripts/paper/figures/edtable1_scan.py [--rescan]
The scan reads both experiment directories and is cached in analysis_outputs/run_inventory/edtable1_scan.json.
"""
import argparse
import csv
import json
import os
import re
import statistics
import zipfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from xml.sax.saxutils import escape
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

REPO = Path(__file__).resolve().parents[3]
D = REPO / "data/experiments"                                  # core runs
E = Path(RDT_DATA + "/extended/experiments")  # intervention, control and curriculum runs
FINAL = REPO / "analysis_outputs/paper/figures/final"
OUT_CSV = FINAL / "source_data/edtable1.csv"
OUT_RUNS = FINAL / "source_data/edtable1_runs.csv"
CACHE = REPO / "analysis_outputs/run_inventory/edtable1_scan.json"
OUT_XML = REPO / "analysis_outputs/paper/figures/final/source_data/edtable1_tbl.xml"

PARAMS = {"ResNet-18": "11.2", "ViT-Small": "22.0", "CCT-7": "3.7"}


def params(arch, seed):
    """ViT-Small seeds 42 and 137: width 192, 4 x 4 patches at 32 x 32 (5.4 M); seed 256: 16 x 16 patches at 224 x 224 (21.7 M)."""
    if arch == "ViT-Small":
        return "21.7" if seed == 256 else "5.4"
    return PARAMS[arch]
SEEDS5 = (7, 42, 137, 256, 314)

# scan
CORE = "abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb"
STANDARD = [  # run numbers 1-9
    ("ResNet-18", 42, "7db28a9a-f436-4d8a-a6b4-37e69fbf54eb"),
    ("ResNet-18", 137, "a0bcbcba-94a9-4009-bcb6-d725b47588ee"),
    ("ResNet-18", 256, "935c13ae-a026-4eae-a2b2-0306879b2e8c"),
    ("ViT-Small", 42, "10d31151-68a2-4491-836d-3ead8e00a7ad"),
    ("ViT-Small", 137, "909d7f68-55e1-4cd9-8c7b-68d5793798d3"),
    ("ViT-Small", 256, "abad8975-de55-4e7e-b95d-05067f0eed90"),
    ("CCT-7", 42, "4bebc329-c26a-4eb4-b18c-7586e513f49e"),
    ("CCT-7", 137, "edf8275e-c99f-4436-a13d-4ed50f072a66"),
    ("CCT-7", 256, "445dd4d4-770a-4f33-b82b-c0bc7baed9a7"),
]
NOISE_EXP = "0c28799c-ec9d-4e45-ac30-302dff810a8d"
SWEEP_EXP = "7885dc97-1062-4b2f-9999-61d63b93d097"


def lane_record(lane):
    lane = Path(lane)
    rec = {}
    sp = lane / "dev_snapshots/metrics/devtrain_snapshots.json"
    if sp.exists():
        s = json.load(open(sp))
        rec["ckpt_epochs"] = [x.get("epoch") for x in s]
        rec["ckpt_updates"] = [x.get("weight_updates") for x in s]
        rec["final_epoch"] = s[-1].get("epoch")
        rec["final_val"] = s[-1].get("val_accuracy")
    rp = lane / "sae_analysis/sae_results.json"
    rec["sae_expansion"] = None
    if rp.exists():
        with open(rp) as f:
            head = f.read(400_000)  # "metadata" is the first key; avoid parsing the full file
        m = re.search(r'"expansion_factor":\s*(\d+)', head)
        labels = re.search(r'"checkpoint_labels":\s*\[([^\]]*)\]', head)
        rec["sae_expansion"] = int(m.group(1)) if m else None
        rec["sae_ckpt_labels"] = len(labels.group(1).split(",")) if labels else None
    return rec


def scan():
    titles = json.load(open(REPO / "analysis_outputs/run_inventory/run_titles.json"))
    runs = []

    def add(key, arch, dataset, seed, lane, **kw):
        r = {"key": key, "arch": arch, "dataset": dataset, "seed": seed, "lane": str(lane)}
        r.update(kw)
        r.update(lane_record(lane))
        runs.append(r)

    for arch, seed, lid in STANDARD:
        add(f"standard|{arch}|{seed}", arch, "CIFAR-100", seed, D / CORE / "lanes" / lid)
    add("ablation|200ep", "ResNet-18", "CIFAR-100", 42,
        D / "ce93d3f6-282b-4811-84f8-be6c589c0500/lanes/c645c39e-e21f-4cff-853f-967bf69ee857")
    # The 8x-expansion and independent-initialisation SAE analyses run on copies of the seed-256 standard run's
    # checkpoints (identical checkpoint files, weight-update counts, elapsed times and accuracies).
    add("ablation|8x", "ResNet-18", "CIFAR-100", 256,
        D / "578b5803-1f2f-49c4-99b8-bf7071f43917/lanes/abfc645c-4b05-46b0-b940-0ef3e8c7284f")
    add("ablation|indep", "ResNet-18", "CIFAR-100", 256,
        D / "independent_sae/lanes/independent_sae_seed256")
    add("crossdataset", "ResNet-18", "Tiny ImageNet", 42,
        D / "821eaadb-0160-48f8-9669-ec3d9185c428/lanes/45476b66-90e0-47f0-aec6-41c62aff8c21")
    for lid in sorted(x for x in os.listdir(D / SWEEP_EXP / "lanes") if not x.startswith(".")):
        t = titles[f"{SWEEP_EXP}/{lid}"]["title"]
        m = re.search(r"switch @ epoch (\d+)", t)
        add(f"sweep|{int(m.group(1)) if m else 0}", "ResNet-18", "CIFAR-100", 42, D / SWEEP_EXP / "lanes" / lid)
    for lid in sorted(x for x in os.listdir(D / NOISE_EXP / "lanes") if not x.startswith(".")):
        t = titles[f"{NOISE_EXP}/{lid}"]["title"]
        seed = int(re.search(r"seed (\d+)", t).group(1))
        if "baseline" in t:
            cond = "none"
        else:
            kind = "within" if "within" in t else "between" if "between" in t else "random"
            cond = f"{kind}|{re.search(r'p=([0-9.]+)', t).group(1)}"
        add(f"redrawn|{cond}", "ResNet-18", "CIFAR-100", seed, D / NOISE_EXP / "lanes" / lid)
    for s in SEEDS5:
        for lane, cond in (("standard_resnet18", "none"), ("within_sc_resnet18_throughout", "within|0.3"),
                           ("between_sc_resnet18_throughout", "between|0.3"),
                           ("between_sc_matched_resnet18_throughout", "matched|0.3")):
            add(f"fixed|{cond}", "ResNet-18", "CIFAR-100", s, E / f"static_noise_s{s}/lanes/{lane}")

    def lanes(*parts):
        out = []
        for p in parts:
            d = E / p / "lanes"
            out += sorted(d / x for x in os.listdir(d) if not x.startswith("."))
        return out

    seed_of = lambda p: int(re.search(r"seed(\d+)", p.name).group(1))
    for p in lanes("prevention", "prevention_s42_copy", "critical"):
        b = int(re.search(r"_e(\d\d)_", p.name).group(1))
        add(f"lastblock|{b}|{'sc' if 'scaffold' in p.name else 'rand'}", "ResNet-18", "CIFAR-100", seed_of(p), p,
            branch_epoch=b)
    for p in lanes("randgroup"):  # random-grouping code (RandomState(10000 + seed) grouping), last block, epochs 1-10
        add("lastblock|0|rg", "ResNet-18", "CIFAR-100", seed_of(p), p, branch_epoch=0)
    for p in lanes("critical_sham"):
        b = int(re.search(r"_e(\d\d)_", p.name).group(1))
        add(f"sham|{b}", "ResNet-18", "CIFAR-100", seed_of(p), p, branch_epoch=b)
    for p in lanes("xlayer"):
        add(f"everylayer|{'sc' if 'scaffold' in p.name else 'rand'}", "ResNet-18", "CIFAR-100", seed_of(p), p,
            branch_epoch=0)
    # acquisition blocking (Fig. 5): forward pass unchanged, learning signal projected off a subspace in epochs 1-10;
    # pilots (gradblock_pilot, gradenergy_pilot) are not reported
    # The superclass, random-direction and energy-matched runs were trained to epoch 15 and continued without block
    # from their epoch-15 checkpoint to epoch 30 (gradcont_grad<kind>_r1, same 30-epoch learning-rate schedule); the
    # two lanes form one run (checkpoint updates of the continuation offset by the epoch-15 count; its resume
    # checkpoint duplicates epoch 15 and is dropped as a distinct checkpoint)
    for p in lanes("gradblock", "gradenergy"):
        kind = re.search(r"lesion_grad(sc|rand|energy)_", p.name).group(1)
        add(f"gradblock|{kind}", "ResNet-18", "CIFAR-100", seed_of(p), p, branch_epoch=0)
        r, cont = runs[-1], E / f"gradcont_grad{kind}_r1/lanes/sham_e15_seed{seed_of(p)}"
        c = lane_record(cont)
        base = r["ckpt_updates"][-1]
        r.update(ckpt_epochs=r["ckpt_epochs"] + c["ckpt_epochs"],
                 ckpt_updates=r["ckpt_updates"] + [base + u for u in c["ckpt_updates"]],
                 final_epoch=c["final_epoch"], final_val=c["final_val"], lane_cont=str(cont))
    for p in lanes("gradrg"):  # random-grouping code (RandomState(10000 + seed) grouping), epochs 1-10, to epoch 30
        add("gradblock|rg", "ResNet-18", "CIFAR-100", seed_of(p), p, branch_epoch=0)
    # dose-matched controls trained straight to epoch 30: rank- and share-matched (jointly matched) and random
    # directions redrawn at every re-estimation; and the fully matched pair at layer3 to the pooled layer, to epoch 15
    for p in lanes("gradjoint"):
        add("gradblock|joint", "ResNet-18", "CIFAR-100", seed_of(p), p, branch_epoch=0)
    for p in lanes("gradrandfresh"):
        add("gradblock|randfresh", "ResNet-18", "CIFAR-100", seed_of(p), p, branch_epoch=0)
    for p in lanes("gradl3"):
        kind = re.search(r"lesion_grad(sc|joint)_", p.name).group(1)
        add(f"gradl3|{kind}", "ResNet-18", "CIFAR-100", seed_of(p), p, branch_epoch=0)
    # self-supervised SimCLR (Supplementary Note 1): no labels, no classifier; record = train_log.jsonl (195 weight
    # updates per epoch at batch 256) and the stored checkpoint directories epoch_<EEE> (encoder.pt, pooled_test.npz)
    for d in sorted((E / "simclr_train").glob("seed*")):
        log = [json.loads(x) for x in open(d / "train_log.jsonl")]
        spe = {x["steps"] for x in log}
        assert len(spe) == 1, spe
        eps = sorted(int(c.name.split("_")[1]) for c in d.glob("epoch_*") if (c / "encoder.pt").exists())
        runs.append({"key": "ssl|simclr", "arch": "ResNet-18", "dataset": "CIFAR-100", "seed": int(d.name[4:]),
                     "lane": str(d), "ckpt_epochs": eps, "ckpt_updates": [e * log[0]["steps"]
                                                       for e in eps],
                     "final_epoch": max(x["epoch"] for x in log), "final_val": None, "sae_expansion": None,
                     "train_seconds": float(sum(x["seconds"] for x in log)), "device": log[-1]["device"]})
    for p in lanes("curriculum"):
        add("ctrl|" + re.sub(r"_seed\d+", "", p.name), "ResNet-18", "CIFAR-100", seed_of(p), p)
    for p in lanes("taskirr"):
        add("blind", "ResNet-18", "CIFAR-100", seed_of(p), p)
    # per-epoch CCT-7 (Supplementary Fig. 2): one run with a checkpoint every epoch
    for p in lanes("perepoch_cct"):
        add("perepoch|cct", "CCT-7", "CIFAR-100", int(re.search(r"seed(\d+)", p.name).group(1)), p)
    return runs


# conditions
# (experiment, where the runs are shown, [(key, condition)]); every cell fits one line (checked in build_xml)
BLOCKS = [
    ("Standard replication", "Figs 1–4; ED 1–6; Supp. Figs 4, 5", [
        (f"standard|{a}|{s}", "Standard") for a, s, _ in STANDARD]),
    ("Ablations", "ED 1c–f; Supp. Fig. 4", [
        ("ablation|200ep", "200 epochs"),
        ("ablation|8x", "SAE expansion 8×ᵃ"),
        ("ablation|indep", "Independent SAE seedᵃ")]),
    ("Cross-dataset", "Fig. 2; ED 1, 2", [("crossdataset", "Tiny ImageNet, 200 classes")]),
    ("Curriculum sweep", "Fig. 6b; ED 9e", [
        ("sweep|0", "0 superclass epochs")] + [
        (f"sweep|{k}", "11 superclass epochsᵇ" if k == 15 else f"{k} superclass epochs")
        for k in (5, 10, 15, 20, 25, 30)]),
    ("Label noise, redrawn", "ED 1, 5, 6, 8a,f; Supp. Fig. 3", [
        ("redrawn|none", "No noise"),
        ("redrawn|within|0.1", "Within-superclass, p = 0.1"),
        ("redrawn|within|0.3", "Within-superclass, p = 0.3"),
        ("redrawn|between|0.1", "Between-superclass, p = 0.1"),
        ("redrawn|between|0.3", "Between-superclass, p = 0.3"),
        ("redrawn|random|0.1", "Random, p = 0.1"),
        ("redrawn|random|0.3", "Random, p = 0.3")]),
    ("Label noise, fixed", "ED 1, 5, 6, 7g, 8; Supp. Fig. 3", [
        ("fixed|none", "No noise"),
        ("fixed|within|0.3", "Within-superclass, p = 0.3"),
        ("fixed|between|0.3", "Between-superclass, p = 0.3"),
        ("fixed|matched|0.3", "Similarity-matched, p = 0.3")]),
    ("Suppression, last block", "ED 6c–g, 7g", [
        ("lastblock|0|sc", "Superclass directions"),
        ("lastblock|0|rand", "Random directions"),
        ("lastblock|0|rg", "Random grouping")]),
    ("Sham control", "Fig. 5; ED 6, 7; Supp. Figs 1, 6", [
        ("sham|0", "No projection, epoch 0ᶠ")]),
    ("Acquisition blocking", "Fig. 5; ED 7; Supp. Fig. 6; Supp. Methods", [
        ("gradblock|sc", "Superclass directions"),
        ("gradblock|rg", "Random groupingⁱ"),
        ("gradblock|rand", "Random directions"),
        ("gradblock|randfresh", "Random, redrawnⁱ"),
        ("gradblock|energy", "Energy-matched"),
        ("gradblock|joint", "Jointly matchedⁱ"),
        ("gradl3|sc", "Superclass, layer3+ᵍ"),
        ("gradl3|joint", "Jointly matched, layer3+ᵍ")]),
    ("Controlled curriculum", "Fig. 6; ED 9; Supp. Note 4", [
        ("ctrl|sc_true_K20", "Superclasses 20, fine 30"),
        ("ctrl|sc_rand_K20", "Random grouping 20, fine 30"),
        ("ctrl|scratch_E30", "Fine classes only, 30"),
        ("ctrl|sc_true_K30", "Superclasses 30, fine 20"),
        ("ctrl|sc_rand_K30", "Random grouping 30, fine 20"),
        ("ctrl|scratch_E20", "Fine classes only, 20")]),
    ("Superclass-blind", "Supp. Note 4", [
        ("blind", "5-way within-superclass")]),
    ("Self-supervised", "ED 1h; Supp. Note 1", [
        ("ssl|simclr", "SimCLR, no labelsʰ")]),
    ("Per-epoch run", "Supp. Fig. 2", [
        ("perepoch|cct", "Standard, 50 epochs")]),
]

# Tiny ImageNet: the run's training record (epochs, accuracy, weight updates) was not retained with the run; the
# values below are those reported for the run. Checkpoints are counted from the SAE checkpoint labels.
TIN_CORE = {"final_epoch": 50, "final_val": 53.0}


def distinct_ckpts(updates):
    """Distinct network states: drop the duplicated final checkpoint and resume duplicates (<= 1 update apart)."""
    out = []
    for u in updates:
        if not out or u - out[-1] > 1:
            out.append(u)
    return out


def r1(x):
    """Round half up to one decimal (accuracies are stored with two decimals)."""
    return str(Decimal(str(round(x, 4))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def fmt_int(x):
    return f"{x:,}"


def summarise(rs):
    seeds = sorted(r["seed"] for r in rs)
    vals = [r["final_val"] for r in rs if r.get("final_val") is not None]
    if "ckpt_updates" in rs[0]:
        n_ck = sorted({len(distinct_ckpts(r["ckpt_updates"])) for r in rs})
        gaps = [b - a for r in rs for a, b in zip(distinct_ckpts(r["ckpt_updates"]), distinct_ckpts(r["ckpt_updates"])[1:])]
        interval = (min(gaps), max(gaps))
        epochs = sorted({r["final_epoch"] for r in rs})
        if rs[0]["key"] == "sweep|15":  # resumed run: 17,985 updates ~ 46 epochs of 391 updates (labelled 50)
            epochs = [round(rs[0]["ckpt_updates"][-1] / 391)]
    else:  # Tiny ImageNet
        n_ck = [rs[0]["sae_ckpt_labels"] - 1]  # labels include the duplicated terminal checkpoint
        interval = None
        epochs = [TIN_CORE["final_epoch"]]
        vals = [TIN_CORE["final_val"]]
    sae = sorted({r["sae_expansion"] for r in rs}, key=str)
    return seeds, vals, n_ck, interval, epochs, sae


def build_rows(runs):
    by_key = {}
    for r in runs:
        by_key.setdefault(r["key"], []).append(r)
    rows, run_no = [], 0
    fixed_none_no = {}
    run_rows = []
    for block, figure, conds in BLOCKS:
        for key, cond in conds:
            rs = by_key[key]
            if key.startswith("redrawn|"):  # table order within a condition: 42, 137, 256, 7, 314
                rs = sorted(rs, key=lambda r: [42, 137, 256, 7, 314].index(r["seed"]))
            else:
                rs = sorted(rs, key=lambda r: r["seed"])
            seeds, vals, n_ck, interval, epochs, sae = summarise(rs)
            dup = key == "ctrl|scratch_E30"  # same networks as fixed|none (identical training records)
            first = run_no + 1 if not dup else fixed_none_no[rs[0]["seed"]]
            for r in rs:
                if dup:
                    no = fixed_none_no[r["seed"]]
                else:
                    run_no += 1
                    no = run_no
                    if key == "fixed|none":
                        fixed_none_no[r["seed"]] = no
                d = distinct_ckpts(r["ckpt_updates"]) if "ckpt_updates" in r else None
                gaps = [b - a for a, b in zip(d, d[1:])] if d else None
                run_rows.append({
                    "run": no, "experiment": block, "figure": figure, "condition": cond, "architecture": r["arch"],
                    "params_M": params(r["arch"], r["seed"]), "dataset": r["dataset"], "seed": r["seed"],
                    "epochs": epochs[0],
                    "branch_epoch": r.get("branch_epoch", ""),
                    "checkpoints": len(d) if d else n_ck[0],
                    # completed epochs at each distinct checkpoint (391 updates per CIFAR-100 epoch at batch 128;
                    # branched runs count from their branch epoch)
                    "checkpoint_epochs_completed": " ".join(
                        f"{r.get('branch_epoch', 0) + u / 391:.4g}" for u in d) if d else "",
                    "interval_min_updates": min(gaps) if gaps else "", "interval_max_updates": max(gaps) if gaps else "",
                    "sae_expansion": f"{r['sae_expansion']}x" if r["sae_expansion"] else "none",
                    "val_acc_final": (round(r["final_val"], 2) if r.get("final_val") is not None else
                                      "" if r["key"] == "ssl|simclr" else TIN_CORE["final_val"]),
                })
            if not vals:
                val = "–"
            elif len(vals) == 1:
                val = r1(vals[0])
            else:
                val = f"{r1(statistics.mean(vals))} ± {r1(statistics.stdev(vals))}"
            seed_txt = ", ".join(str(s) for s in seeds)
            rows.append({
                "experiment": block, "figure": figure, "condition": cond,
                "runs": (f"{first}–{first + len(rs) - 1}" if dup else
                         f"{first}" if run_no == first else f"{first}–{run_no}"),
                "architecture": rs[0]["arch"], "params_M": params(rs[0]["arch"], rs[0]["seed"]), "dataset": rs[0]["dataset"],
                "seeds": seed_txt, "seeds_short": ("five" if tuple(seeds) == SEEDS5 else "three" if tuple(seeds) == (7, 42, 137) else seed_txt) + ("ᵉ" if dup else ""), "n": len(rs),
                "epochs": "/".join(str(e) for e in epochs),
                "checkpoints": "/".join(str(c) for c in n_ck),
                "interval_updates": (f"{fmt_int(interval[0])}" if interval[0] == interval[1]
                                     else f"{fmt_int(interval[0])}–{fmt_int(interval[1])}") if interval else "–ᵈ",
                # blocking arms analysed with 4x SAEs for seeds 7, 42, 137 only (blocking_rdt; footnote j)
                "sae": ("/".join(f"{x}×" for x in sae) if sae != [None] else
                        "–ʲ" if key in ("gradblock|sc", "gradblock|rg", "gradblock|rand", "gradblock|energy") else "–"),
                "val_acc": val + ("ᶜ" if key == "blind" else ""),
            })
    return rows, run_rows


# WordprocessingML
# Every row is one text line high (exact row height): column widths are computed from the measured text (Times New
# Roman) so that no cell wraps; the merged Experiment and Figure cells may wrap within the rows of their block.
TABLE_W = 8300            # twips: A4 or US Letter with 3.17-cm margins
CELL_MAR = 60             # twips, left and right cell margin
SUPMAP = {"ᵃ": "a", "ᵇ": "b", "ᶜ": "c", "ᵈ": "d", "ᵉ": "e", "ᶠ": "f", "ᵍ": "g", "ʰ": "h", "ⁱ": "i", "ʲ": "j"}
COLUMNS = [  # (header, csv field)
    ("Experiment", "experiment"), ("Figure", "figure"), ("Condition", "condition"), ("Network", "architecture"),
    ("Seeds", "seeds_short"), ("Epochs", "epochs"), ("Ckpts", "checkpoints"), ("Interval", "interval_updates"),
    ("SAE", "sae"), ("Val Acc", "val_acc")]
MERGED = ("experiment", "figure")
TNR = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
TNR_B = "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"


def _font(pt, bold=False):
    from PIL import ImageFont
    return ImageFont.truetype(TNR_B if bold else TNR, int(round(pt * 10)))


def text_tw(text, pt, bold=False):
    """Width of a cell text in twips (superscript footnote letters at 65%)."""
    f = _font(pt, bold)
    main = "".join(ch for ch in text if ch not in SUPMAP)
    sup = "".join(SUPMAP[ch] for ch in text if ch in SUPMAP)
    return (f.getlength(main) + 0.65 * f.getlength(sup)) / 10 * 20


def lines_needed(text, width_tw, pt):
    """Greedy word wrap: number of lines the text takes in a cell of this content width."""
    sp = text_tw(" ", pt)
    n, cur = 1, None
    for w in text.split(" "):
        lw = text_tw(w, pt)
        if cur is None:
            cur = lw
        elif cur + sp + lw > width_tw:
            n, cur = n + 1, lw
        else:
            cur += sp + lw
    return n


def column_widths(rows, pt):
    """Minimum width per column (content + margins + 30-twip slack) such that every non-merged cell is one line and
    every merged label fits within the rows of its block; spare width goes to Condition and Experiment."""
    blocks = {}
    for r in rows:
        blocks.setdefault(r["experiment"], [r["figure"], 0])[1] += 1
    need = {}
    for h, f in COLUMNS:
        head = text_tw(h, pt, bold=True)
        if f in MERGED:
            w = head
            for exp, (fig, n) in blocks.items():
                label = exp if f == "experiment" else fig
                c = max(text_tw(x, pt) for x in label.split(" "))
                while lines_needed(label, c, pt) > n:
                    c += 10
                w = max(w, c)
        else:
            w = max([head] + [text_tw(str(r[f]), pt) for r in rows])
        need[f] = int(w) + 2 * CELL_MAR + 30
    spare = TABLE_W - sum(need.values())
    if spare >= 0:
        need["condition"] += spare - spare // 3
        need["experiment"] += spare // 3
    return need, spare


def runs_xml(text, sz, bold=False):
    """Text runs; trailing footnote letters become superscripts."""
    rpr = (f'<w:rPr><w:rFonts w:cs="Times New Roman"/>{"<w:b/>" if bold else ""}<w:color w:val="000000"/>'
           f'<w:sz w:val="{sz}"/></w:rPr>')
    sup = (f'<w:rPr><w:rFonts w:cs="Times New Roman"/><w:color w:val="000000"/><w:sz w:val="{sz}"/>'
           f'<w:vertAlign w:val="superscript"/></w:rPr>')
    out = []
    for p in re.split(r"([ᵃᵇᶜᵈᵉᶠᵍʰⁱʲ])", text):
        if not p:
            continue
        if p in SUPMAP:
            out.append(f'<w:r>{sup}<w:t>{SUPMAP[p]}</w:t></w:r>')
        else:
            out.append(f'<w:r>{rpr}<w:t xml:space="preserve">{escape(p)}</w:t></w:r>')
    return "".join(out)


PPR = ('<w:pPr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/><w:rPr><w:rFonts '
       'w:cs="Times New Roman"/></w:rPr></w:pPr>')


def cell(width, text, sz, bold=False, vmerge=None):
    vm = "" if vmerge is None else ('<w:vMerge w:val="restart"/>' if vmerge == "restart" else "<w:vMerge/>")
    tcpr = (f'<w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{vm}<w:shd w:val="clear" w:color="auto" w:fill="FFFFFF"/>'
            f'<w:vAlign w:val="center"/></w:tcPr>' if vmerge is None else
            f'<w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{vm}<w:shd w:val="clear" w:color="auto" w:fill="FFFFFF"/>'
            f'</w:tcPr>')
    body = runs_xml(text, sz, bold) if text else ""
    return f"<w:tc>{tcpr}<w:p>{PPR}{body}</w:p></w:tc>"


def build_xml(rows):
    """One <w:tbl>: fixed layout, exact row height of one text line, widths from column_widths at the largest font
    size (7, 6.5 or 6 pt; Nature's range) at which every cell fits."""
    for pt in (7.0, 6.5, 6.0):
        W, spare = column_widths(rows, pt)
        if spare >= 0:
            break
    else:
        raise ValueError(f"ED Table 1 does not fit {TABLE_W} twips on one line per row even at 6 pt (short {-spare})")
    sz = int(round(pt * 2))
    row_h = int(round(pt * 20 * 1.2)) + 24      # single line spacing of Times New Roman plus a small gap
    for r in rows:                               # one line per cell, merged labels within their block
        for h, f in COLUMNS:
            if f not in MERGED:
                assert lines_needed(str(r[f]), W[f] - 2 * CELL_MAR, pt) == 1, (f, r[f], W[f])
    grid = "".join(f'<w:gridCol w:w="{W[f]}"/>' for _, f in COLUMNS)
    tblpr = (f'<w:tblPr><w:tblStyle w:val="Tabellenraster"/><w:tblW w:w="{sum(W.values())}" w:type="dxa"/>'
             '<w:tblLayout w:type="fixed"/>'
             f'<w:tblCellMar><w:left w:w="{CELL_MAR}" w:type="dxa"/><w:right w:w="{CELL_MAR}" w:type="dxa"/>'
             '</w:tblCellMar>'
             '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" '
             'w:noHBand="0" w:noVBand="1"/></w:tblPr>')
    trpr = f'<w:trPr><w:cantSplit/><w:trHeight w:val="{row_h}" w:hRule="exact"/>'
    trs = [trpr + '<w:tblHeader/></w:trPr>' + "".join(cell(W[f], h, sz, bold=True) for h, f in COLUMNS)]
    trs[0] = "<w:tr>" + trs[0] + "</w:tr>"
    prev_block = None
    for r in rows:
        tcs = []
        for h, f in COLUMNS:
            if f in MERGED:
                if r["experiment"] != prev_block:
                    label = re.sub(r"(Figs?\.?|Note|ED) ", "\\1\u00a0", r[f])
                    tcs.append(cell(W[f], label, sz, vmerge="restart"))
                else:
                    tcs.append(cell(W[f], "", sz, vmerge="continue"))
            else:
                tcs.append(cell(W[f], str(r[f]), sz))
        prev_block = r["experiment"]
        trs.append("<w:tr>" + trpr + "</w:trPr>" + "".join(tcs) + "</w:tr>")
    print(f"ED Table 1: {pt} pt, row height {row_h} twips, widths {W} (total {sum(W.values())})")
    return ('<w:tbl xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            + tblpr + f"<w:tblGrid>{grid}</w:tblGrid>" + "".join(trs) + "</w:tbl>")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescan", action="store_true")
    a = ap.parse_args()
    if a.rescan or not CACHE.exists():
        reported = {k for _, _, conds in BLOCKS for k, _ in conds}
        runs = [r for r in scan() if r["key"] in reported]   # the cache holds the reported runs only
        CACHE.write_text(json.dumps(runs, indent=1))
    runs = json.loads(CACHE.read_text())
    rows, run_rows = build_rows(runs)
    # The scratch-30 rows repeat the fixed-label 'No noise' run numbers. Runs not in BLOCKS (later suppression windows,
    # sham from epoch 10, suppression in every layer, the layer2-to-pooled pair) are not listed.
    n_runs, n_rows = 169, 52
    assert len({r["run"] for r in run_rows}) == n_runs and len(rows) == n_rows, (len({r["run"] for r in run_rows}),
                                                                                  len(rows))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(OUT_RUNS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(run_rows[0]))
        w.writeheader()
        w.writerows(run_rows)
    OUT_XML.write_text(build_xml(rows))
    uniq = {r["run"]: r for r in run_rows}
    print(f"{len(uniq)} runs, {len(rows)} condition rows ({n_rows - 1} distinct conditions), "
          f"{sum(1 for r in run_rows if r['sae_expansion'] != 'none')} SAE analyses in the table")


if __name__ == "__main__":
    main()
