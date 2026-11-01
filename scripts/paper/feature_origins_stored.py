#!/usr/bin/env python3
"""Origins of class-selective features with stored thresholds: feature_origins.py with each
run's stored pipeline thresholds for SSI, CSI and SAI (SAE_outcomes EN null cache, key 'stored'), instead of the per-layer
alive-only null p95 and SAI > 0.9. Everything else (lanes, checkpoints, matching, permutations, same-data control,
analysis) is unchanged. Writes analysis_outputs/paper/ORIGINS_stored/.
Usage: feature_origins_stored.py run ; feature_origins_stored.py analyze
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scaffold_profile_lib as E  # noqa: E402
import feature_origins as S  # noqa: E402

_orig = E.thresholds


def thresholds(nl, layer, kind):
    if kind != "stored":
        return _orig(nl, layer, kind)
    st, ati = nl["stored"]["thresholds"], nl["stored"]["ati"]
    return {"ssi": float(st["ssi_abstraction_thresh"]), "csi": float(st["csi_differentiation_thresh"]),
            "sai": float(st["sai_task_general_thresh"]),
            "ssi_mean": float(ati.get("ssi_null_mean", 0.0)), "ssi_std": float(ati.get("ssi_null_std", 1.0)),
            "csi_mean": float(ati.get("csi_null_mean", 0.0)), "csi_std": float(ati.get("csi_null_std", 1.0))}


E.thresholds = thresholds
S.TKS = ("stored",)
S.OUT = E.REPO / "analysis_outputs" / "paper" / "ORIGINS_stored"
S.CD = S.OUT / "cache"

if __name__ == "__main__":
    cmd = sys.argv[1]
    S.OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "run":
        for ln in E.lanes():
            if ln["cond"] != "clean":
                continue
            if (S.CD / f"{ln['key']}_tables.npz").exists():
                print(ln["key"], "cached")
                continue
            print(S.lane_run(ln), flush=True)
    else:
        try:
            S.analyze()
        except KeyError as e:  # feature_origins.analyze ends by printing summ['layer']; its outputs are written before that
            print("analysis written (final print skipped:", e, ")")
