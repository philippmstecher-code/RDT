#!/usr/bin/env python3
"""CCT-7 routing specificity (routing, part 2 of routing.py) for the CCT-7 routing panel: seeds 42 and 137, checkpoints
1, 5, 9 (early / middle / final), with epoch and weight-update index per checkpoint, S, control, n_Ab and the 20 x 20
drop matrices (rows = ablated superclass k, columns = evaluated superclass j; canonical CIFAR-100 coarse order).
Same computation as routing_transformers.py. Writes analysis_outputs/paper/ROUTING/routing_cct7.json."""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import routing_transformers as R  # noqa: E402
T = R.T


def main():
    out = {"superclass_names": pickle.load(open(T.CIFAR / "meta", "rb"), encoding="latin1")["coarse_label_names"],
           "definition": "S = mean diagonal - mean off-diagonal drop in superclass accuracy (base - ablated); control = "
                         "per superclass the same number of alive non-Ab non-Tg features matched in activation mass",
           "runs": {}}
    for key in ("core__CCT-7__s42", "core__CCT-7__s137"):
        lane = [l for l in T.registry() if l["key"] == key][0]
        snaps = json.load(open(Path(lane["dir"]) / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"))
        snaps = snaps.get("snapshots", snaps) if isinstance(snaps, dict) else snaps
        meta = {Path(s["snapshot_dir"]).name.split("_")[-1]: {"epoch": s.get("epoch"), "weight_updates": s.get("weight_updates")}
                for s in snaps if "snapshot_dir" in s}
        for label in ("1", "5", "9"):
            r = R.analyse({**lane, "label": label}, label)
            out["runs"][f"{key}|{label}"] = {"seed": lane.get("seed"), "checkpoint": label, **meta.get(label, {}),
                                             **{k: r[k] for k in ("S", "S_ctrl", "n_ab", "n_di", "n_ab_per_group",
                                                                  "base_sc_acc", "matrix", "matrix_ctrl")}}
            print(key, label, meta.get(label), round(r["S"], 3), round(r["S_ctrl"], 3), flush=True)
    json.dump(out, open(R.A.OUT / "routing_cct7.json", "w"), indent=1)


if __name__ == "__main__":
    main()
