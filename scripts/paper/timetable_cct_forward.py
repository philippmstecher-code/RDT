"""CPU forward passes of the core CCT-7 s42 checkpoints on the CIFAR-100 test set -> canonical logits and pooled
representation (the attention-pooled token the classifier sees), verified against the stored predictions.
Usage: timetable_cct_forward.py [norm_mean_std_tag] [n_check] [checkpoint_label]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timetable_lib import CACHE, lanes, test_canonical, to_canonical_perm  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)
from models import create_network  # noqa: E402

NORMS = {"half": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         "cifar": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))}


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "half"
    n_check = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    key = "core_CCT-7_s42"
    lane = lanes()[key]
    perm = to_canonical_perm(lane)  # lane class -> canonical
    inv = np.empty(100, int)
    inv[perm] = np.arange(100)
    X, y = test_canonical()
    mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in NORMS[tag])
    Xt = (torch.from_numpy(X).permute(0, 3, 1, 2).float() / 255.0 - mean) / std
    net = create_network("cct_7", num_classes=100)
    pooled = {}

    def hook(m, i, o):
        pooled["x"] = o.detach()

    labs = lane["labels"] if len(sys.argv) <= 3 else [sys.argv[3]]
    for lab in labs:
        f_log = CACHE / f"{key}_m{lab}_logits.npy"
        if f_log.exists() and not n_check:
            continue
        d = Path(lane["dir"]) / "dev_snapshots" / f"milestone_{lab}"
        sd = torch.load(d / "checkpoint.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(sd.get("model_state_dict", sd))
        net.eval()
        h = net.fc.register_forward_pre_hook(lambda m, inp: pooled.__setitem__("x", inp[0].detach()))
        idx = np.arange(10000) if not n_check else np.concatenate([np.arange(c * 100, c * 100 + n_check) for c in range(100)])
        Z, P = [], []
        with torch.no_grad():
            for s in range(0, len(idx), 250):
                z = net(Xt[idx[s:s + 250]])
                Z.append(z.numpy())
                P.append(pooled["x"].numpy())
        h.remove()
        Z = np.concatenate(Z)
        # lane class j -> canonical perm[j]; reorder columns to canonical
        Zc = np.empty_like(Z)
        Zc[:, perm] = Z
        sp = torch.load(d / "sample_predictions.pt", map_location="cpu", weights_only=False)
        pred_lane = np.concatenate([sp[i]["predictions"].numpy() for i in range(100)])
        cor_lane = np.concatenate([sp[i]["correct"].numpy() for i in range(100)]).astype(bool)
        rows = (inv[:, None] * 100 + np.arange(100)[None, :]).ravel()
        pred_c = perm[pred_lane[rows]]  # canonical predicted class
        agree = float((Zc.argmax(1) == pred_c[idx]).mean())
        msg = f"{key} m{lab} norm={tag} n={len(idx)} argmax-agree={agree:.4f} acc={(Zc.argmax(1) == y[idx]).mean():.4f}"
        print(msg, flush=True)
        if n_check:
            return
        np.save(f_log, Zc.astype(np.float32))
        np.save(CACHE / f"{key}_m{lab}_pooled.npy", np.concatenate(P).astype(np.float16))
        np.save(CACHE / f"{key}_m{lab}_storedcorrect.npy", cor_lane[rows])
        with open(CACHE / "extract_log.txt", "a") as f:
            f.write(msg + "\n")


if __name__ == "__main__":
    main()
