"""Developmental timetable, step 1: per-image logits + pooled activations in canonical order, for every open lane/checkpoint.

ResNet-18 / ViT-Small: logits = head(stored pooled activation)  (avgpool / CLS of encoder.ln).
CCT-7: stored 'norm' is mean-pooled (model uses attention pooling) -> CPU forward pass (timetable_cct_forward.py).
Every logit set is verified against stored sample_predictions argmax.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timetable_lib import CACHE, lanes, to_canonical_perm  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)

HEAD = {"ResNet-18": ("avgpool", "fc"), "ViT-Small": ("encoder.ln", "heads.head")}


def canon_rows(perm):
    """canonical row order -> lane row index. lane rows are lane_class*100 + j."""
    inv = np.empty(100, int)
    inv[perm] = np.arange(100)  # canonical class -> lane class
    return (inv[:, None] * 100 + np.arange(100)[None, :]).ravel(), inv


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    log = []
    for key, lane in lanes().items():
        if lane["arch"] == "CCT-7":
            continue
        perm = to_canonical_perm(lane)
        rows, inv = canon_rows(perm)
        layer, head = HEAD[lane["arch"]]
        for lab in lane["labels"]:
            f_log = CACHE / f"{key}_m{lab}_logits.npy"
            f_act = CACHE / f"{key}_m{lab}_pooled.npy"
            f_cor = CACHE / f"{key}_m{lab}_storedcorrect.npy"
            if f_log.exists() and f_act.exists() and f_cor.exists():
                continue
            d = Path(lane["dir"]) / "dev_snapshots" / f"milestone_{lab}"
            A = torch.load(d / "individual_multilayer_activations.pt", map_location="cpu", weights_only=False)[layer]
            X = torch.cat([A[i].float() for i in range(100)])  # lane rows
            del A
            sd = torch.load(d / "checkpoint.pt", map_location="cpu", weights_only=False)
            sd = sd.get("model_state_dict", sd)
            W, b = sd[f"{head}.weight"].float(), sd[f"{head}.bias"].float()
            Z = (X @ W.T + b).numpy()  # lane rows x lane classes
            Zc = Z[rows][:, inv]  # canonical rows x canonical classes
            Xc = X.numpy()[rows]
            sp = torch.load(d / "sample_predictions.pt", map_location="cpu", weights_only=False)
            pred_lane = np.concatenate([sp[i]["predictions"].numpy() for i in range(100)])
            cor_lane = np.concatenate([sp[i]["correct"].numpy() for i in range(100)]).astype(bool)
            agree = float((Z.argmax(1) == pred_lane).mean())
            np.save(f_log, Zc.astype(np.float32))
            np.save(f_act, Xc.astype(np.float16))
            np.save(f_cor, cor_lane[rows])
            log.append(f"{key} m{lab} argmax-agree={agree:.4f} acc={cor_lane.mean():.4f}")
            print(log[-1], flush=True)
    with open(CACHE / "extract_log.txt", "a") as f:
        f.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
