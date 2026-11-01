#!/usr/bin/env python3
"""Event counts and census with every feature counted, including weakly firing ones (9 core CIFAR-100 runs + Tiny ImageNet).

Stored per-run thresholds throughout. The only change is the 'alive' rule (pipeline: active on >= 1% of the 10,000
images) used by Phase-5 matching (saeanalysis._phase5_feature_matching, patched by source substitution of
`n_samples * 0.01`) and therefore by Phase-6 counting (noise_null_lib.classify_counts uses the matching's alive masks):
  p01   alive = >= 1% of images   (check: must reproduce sae_results.json['process_intensity'] exactly)
  all   alive = >= 1 image        (every feature the SAE uses)
Also a census per checkpoint (census layers: all layers except ResNet layer4, which duplicates avgpool; exclusive rule
Tg = SAI > t, Ab = SSI > t & CSI <= t & not Tg, Di = CSI > t & not Tg) for both alive rules.
Writes analysis_outputs/paper/THRESHOLD_impact/core_events_allfeat.csv and core_census_allfeat.csv.
Usage: core_events_allfeat.py
"""
import glob
import inspect
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noise_null_lib  # noqa: E402
from noise_null_lib import REPO, Lane, classify_counts  # noqa: E402
from core_events_alive import EXP, RUNS  # noqa: E402

sa = noise_null_lib.sa
OUT = REPO / "analysis_outputs" / "paper" / "THRESHOLD_impact"
P = ("tg_h", "ab_h", "di_h", "as_h", "de_h", "stable")
_src = inspect.getsource(sa._phase5_feature_matching)
assert _src.count("n_samples * 0.01") == 1, "patch target not unique"
exec(compile(_src.replace("n_samples * 0.01", "n_samples * _ALIVE_FRAC_PATCH"), sa.__file__, "exec"), sa.__dict__)
_PATCHED = sa._phase5_feature_matching


def match_frac(Ha, Hb, frac):
    sa._ALIVE_FRAC_PATCH = frac
    sa._phase5_feature_matching = _PATCHED
    return noise_null_lib.match(Ha, Hb)


def census(H, sel, th, amin):
    cnt = (np.abs(H) > 1e-6).sum(0)
    alive = cnt >= amin
    sai, ssi = np.asarray(sel["feature_sai"], float), np.asarray(sel["feature_ssi"], float)
    csi = np.nan_to_num(np.asarray(sel["feature_csi"], float), nan=-1)
    tg = alive & (sai > th["sai_task_general_thresh"])
    ab = alive & (ssi > th["ssi_abstraction_thresh"]) & (csi <= th["csi_differentiation_thresh"]) & ~tg
    di = alive & (csi > th["csi_differentiation_thresh"]) & ~tg
    return {"n_alive": int(alive.sum()), "n_tg": int(tg.sum()), "n_ab": int(ab.sum()), "n_di": int(di.sum())}


def job(run):
    arch, seed, exp, prefix, name = run
    d = Path(glob.glob(str(EXP / exp / "lanes" / f"{prefix}*"))[0])
    lane = Lane(d, dataset_id="tiny_imagenet" if arch == "tiny" else "cifar100")
    pi = lane.results["process_intensity"]
    trans = [t["transition"] for t in pi[:-1]]
    sel = lane.results["selectivity"]
    th = lane.thresholds
    rows, crow = [], []
    for L in lane.layers:
        codes = {}
        for k, t in enumerate(trans):
            a, b = t.split("->")
            for x in (a, b):
                if x not in codes:
                    codes[x] = lane.load_codes(x, L)
                    n = codes[x].shape[0]
                    for nm, amin in (("p01", max(1, n * 0.01)), ("all", 1)):
                        crow.append({"arch": arch, "seed": seed, "run": name, "variant": nm, "layer": L, "label": x,
                                     **census(codes[x], sel[x][L], th, amin)})
            for nm, frac in (("p01", 0.01), ("all", 0.0)):
                md = match_frac(codes[a], codes[b], frac)
                c = classify_counts(codes[a], codes[b], md, sel[a][L], sel[b][L], thresholds=th, ati=lane.ati)
                rows.append({"arch": arch, "seed": seed, "run": name, "variant": nm, "layer": L, "transition": k,
                             "label": t, "n_alive_a": int(np.sum(md["alive_a"])), **{p: int(c[p]) for p in P}})
            codes.pop(a, None)
    df = pd.DataFrame(rows)
    st = df[df.variant == "p01"].groupby("transition")[["tg_h", "ab_h", "di_h", "as_h", "de_h"]].sum()
    ok = all(int(st.loc[k, p]) == int(pi[k][p]) for k in range(len(trans)) for p in ("tg_h", "ab_h", "di_h", "as_h", "de_h"))
    df["p01_reproduces_stored"] = ok
    print(name, "p01 reproduces stored" if ok else "P01 MISMATCH", flush=True)
    return df, pd.DataFrame(crow)


def main():
    with ProcessPoolExecutor(3) as ex:
        out = list(ex.map(job, RUNS))
    pd.concat([o[0] for o in out]).to_csv(OUT / "core_events_allfeat.csv", index=False)
    pd.concat([o[1] for o in out]).drop_duplicates().to_csv(OUT / "core_census_allfeat.csv", index=False)
    print("written")


if __name__ == "__main__":
    main()
