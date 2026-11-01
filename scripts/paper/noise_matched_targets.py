#!/usr/bin/env python3
"""Similarity-matched between-superclass flip targets for the difficulty-matched noise control.

Similarity = cosine similarity of centred class means of the penultimate (avgpool) activations of the five clean
30-epoch ResNet-18 networks (static_noise_s{S}/standard_resnet18, final checkpoint, 100 test images per class),
averaged over seeds. For every class c with siblings s1..s4 (same CIFAR-100 superclass, in descending similarity),
each sibling is matched to the not-yet-chosen class OUTSIDE c's superclass whose similarity to c is closest. The four
matched classes are c's flip targets in the 'between_sc_matched' arm (uniform choice, as 'within_sc' among siblings).
Writes analysis_outputs/paper/NOISE_matched/targets.json and match_quality.json.
Usage: python scripts/paper/noise_matched_targets.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_lib as T  # noqa: E402

OUT = T.REPO / "analysis_outputs" / "paper" / "NOISE_matched"
d = pickle.load(open(T.CIFAR / "test", "rb"), encoding="latin1")
F2S = np.zeros(100, int)
for f, c in zip(d["fine_labels"], d["coarse_labels"]):
    F2S[f] = c


def sims():
    Ss = []
    for s in T.SEEDS:
        lane = T.RUNS_EXTENDED / f"static_noise_s{s}" / "lanes" / "standard_resnet18"
        snaps = json.load(open(lane / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"))
        snaps = snaps.get("snapshots", snaps) if isinstance(snaps, dict) else snaps
        last = [x for x in snaps if x.get("snapshot_type") != "final"][-1]
        a = torch.load(lane / "dev_snapshots" / Path(last["snapshot_dir"]).name / "individual_multilayer_activations.pt",
                       map_location="cpu", weights_only=False)["avgpool"]
        M = np.stack([a[c].numpy().mean(0) for c in range(100)])
        M = M - M.mean(0)
        M = M / np.linalg.norm(M, axis=1, keepdims=True)
        Ss.append(M @ M.T)
    return np.mean(Ss, 0)


def main():
    S = sims()
    targets, q = {}, []
    for c in range(100):
        sib = [k for k in range(100) if F2S[k] == F2S[c] and k != c]
        sib = sorted(sib, key=lambda k: -S[c, k])
        pool = [k for k in range(100) if F2S[k] != F2S[c]]
        chosen = []
        for k in sib:
            j = min((p for p in pool if p not in chosen), key=lambda p: abs(S[c, p] - S[c, k]))
            chosen.append(j)
        targets[c] = chosen
        q.append({"cls": c, "sib_sim": float(np.mean([S[c, k] for k in sib])),
                  "matched_sim": float(np.mean([S[c, k] for k in chosen])),
                  "between_sim": float(np.mean([S[c, k] for k in pool])),
                  "matched_targets_distinct_superclasses": int(len(set(F2S[chosen])))})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "targets.json").write_text(json.dumps({str(k): v for k, v in targets.items()}))
    sib, mat, btw = (np.array([r[k] for r in q]) for k in ("sib_sim", "matched_sim", "between_sim"))
    summ = {"mean_sim_siblings": float(sib.mean()), "mean_sim_matched": float(mat.mean()),
            "mean_sim_between_random": float(btw.mean()), "median_abs_diff_matched_vs_sib": float(np.median(np.abs(mat - sib))),
            "share_classes_matched_within_0.05": float((np.abs(mat - sib) <= 0.05).mean()),
            "mean_distinct_superclasses_among_targets": float(np.mean([r["matched_targets_distinct_superclasses"] for r in q])),
            "per_class": q}
    (OUT / "match_quality.json").write_text(json.dumps(summ, indent=1))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items() if k != "per_class"})


if __name__ == "__main__":
    main()
