#!/usr/bin/env python3
"""Control for the reconstruction of the fixed noisy labels (noise_fit_features.py): at the final checkpoint (epoch 30)
of each between- and within-superclass run, the share of flipped training images predicted as their reconstructed
wrong label (own seed's draw) against the share predicted as the wrong label of another seed's draw (seed + 1), on the
images that the other draw flips and the own draw keeps (chance about 1%).
Writes analysis_outputs/paper/RDT_noise_fit/verify_control.json. Read-only; CPU.
Usage: python scripts/paper/noise_fit_verify_control.py
"""
import json
import sys
from pathlib import Path
import importlib.util
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("nf", HERE / "noise_fit_features.py")
nf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nf)
X, y = nf.train_set()
res = {}
for s in nf.N.SEEDS:
    for kind in ("between", "within"):
        yn, fl = nf.noisy_labels(y, kind, s)
        yo, flo = nf.noisy_labels(y, kind, s + 1)
        _, Z = nf.pooled(nf.load_net(nf.N.lane_dir(kind, s), 6), X)
        pred = Z.argmax(1).numpy()
        other = flo & ~fl
        res[f"{kind}|{s}"] = {"own_draw_pred_eq_noisy_on_flipped": float((pred[fl] == yn[fl]).mean()),
                              "other_draw_pred_eq_noisy": float((pred[other] == yo[other]).mean()),
                              "n_other": int(other.sum())}
        print(kind, s, res[f"{kind}|{s}"], flush=True)
json.dump(res, open(nf.OUT / "verify_control.json", "w"), indent=1)
