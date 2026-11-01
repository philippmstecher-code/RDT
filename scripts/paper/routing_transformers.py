#!/usr/bin/env python3
"""Routing specificity in transformers (core runs, 2 seeds each): targeted ablation of superclass-selective (Ab) SAE
features at the classifier-input site, with the matched-feature control (routing, part 2 of routing.py).

CCT-7: SAE input = mean over tokens of the final norm output (codes correlate 0.9997 with the stored codes); the
classifier reads the attention-pooled tokens, so a feature set is removed from every token by a mean-preserving
projection onto the complement of the span of its decoder directions (lesion-hook semantics).
ViT-Small: SAE input = CLS token after encoder.ln (= the classifier input); the same projection is applied to it.
Categories, best superclass, thresholds and control as in routing.py (codes of all 10,000 test images, the
lane's adaptive thresholds, canonical class order). Writes analysis_outputs/paper/ROUTING/routing_tf.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import transfer_lib as T  # noqa: E402
import routing as A  # noqa: E402
sys.path.insert(0, str(T.REPO / "packages" / "ml" / "src"))
from sae import SparseAutoencoder  # noqa: E402

torch.set_num_threads(4)


@torch.no_grad()
def site_tokens(net, arch, X):
    """Per batch: tokens at the SAE site (CCT: norm output (B,T,C); ViT: CLS after encoder.ln as (B,1,C))."""
    out = []
    for s in range(0, len(X), 250):
        x = T.cifar_batch(X[s:s + 250])
        if arch == "CCT-7":
            out.append(net.norm(net.transformer(net.tokenizer(x))))
        elif arch == "ResNet-18":
            h = net.maxpool(net.relu(net.bn1(net.conv1(x))))
            h = net.layer4(net.layer3(net.layer2(net.layer1(h))))
            out.append(torch.flatten(net.avgpool(h), 1)[:, None, :])
        else:
            h = net._process_input(x)
            h = torch.cat([net.class_token.expand(h.shape[0], -1, -1), h], dim=1)
            out.append(net.encoder(h)[:, :1])
    return torch.cat(out)


@torch.no_grad()
def head(net, arch, tok):
    if arch == "CCT-7":
        w = torch.softmax(net.attention_pool(tok), dim=1)
        return net.fc((tok * w).sum(1))
    if arch == "ResNet-18":
        return net.fc(tok[:, 0])
    return net.heads(tok[:, 0])


def analyse(lane, label):
    arch = lane["arch"]
    net = T.load_net(lane, label)
    cf = T.cifar_test()
    c2c = T.lane_class_to_canonical(lane)
    y = cf["y"]
    tok = site_tokens(net, arch, cf["X"])
    xin = tok.mean(1)
    d = Path(lane["dir"])
    site = {"CCT-7": "norm", "ViT-Small": "encoder.ln", "ResNet-18": "avgpool"}[arch]
    f = d / "sae_analysis" / "saes" / label / f"{site.replace('.', '_')}.pt"
    sd = torch.load(f if f.exists() else d / "sae_analysis" / "saes" / label / f"{site}.pt", map_location="cpu",
                    weights_only=False)
    din = sd["decoder.weight"].shape[0]
    sae = SparseAutoencoder(din, sd["encoder.weight"].shape[0] // din, 32)
    sae.load_state_dict(sd)
    sae.eval()
    with torch.no_grad():
        H = sae.encode(xin).numpy().astype(np.float64)
    D = sae.decoder.weight.detach().double()
    md = json.load(open(d / "sae_analysis" / "sae_results.json"))["metadata"]["adaptive_thresholds"]
    th = {"sai": md["sai_adaptive_thresh"], "ssi": md["ssi_adaptive_thresh"], "csi": md["csi_adaptive_thresh"]}
    grp = A.TRUE
    cat = A.categories(H, y, grp, th)
    mu = tok.reshape(-1, tok.shape[-1]).mean(0).double()

    def sc_acc(logits):
        pred = c2c[logits.argmax(1).numpy()]
        return np.array([(grp[pred[grp[y] == j]] == j).mean() for j in range(20)])

    base = sc_acc(head(net, arch, tok))

    @torch.no_grad()
    def ablate(idx):
        Q, _ = torch.linalg.qr(D[:, idx])
        t = tok.double()
        tc = t - mu
        t2 = t - (tc @ Q) @ Q.T
        return sc_acc(head(net, arch, t2.float()))

    used = set()
    pool = np.flatnonzero(cat["other"] | cat["di"])
    lm = np.log(cat["mass"] + 1e-12)
    Mt, Mc, nk = np.full((20, 20), np.nan), np.full((20, 20), np.nan), []
    for k in range(20):
        S = np.flatnonzero(cat["ab"] & (cat["best"] == k))
        nk.append(int(len(S)))
        if len(S) == 0:
            continue
        ctrl = []
        for fidx in S:
            cand = [q for q in pool[np.argsort(np.abs(lm[pool] - lm[fidx]))] if q not in used]
            if cand:
                ctrl.append(int(cand[0]))
                used.add(cand[0])
        Mt[k] = base - ablate(S)
        if ctrl:
            Mc[k] = base - ablate(np.array(ctrl))

    def spec(M):
        rows = np.flatnonzero(~np.isnan(M).all(1))
        if not len(rows):
            return float("nan")
        diag = np.array([M[k, k] for k in rows])
        off = (np.nansum(M[rows], 1) - diag) / 19
        return float(diag.mean() - off.mean())
    return {"S": spec(Mt), "S_ctrl": spec(Mc), "n_ab_per_group": nk, "n_ab": int(cat["ab"].sum()),
            "n_di": int(cat["di"].sum()), "base_sc_acc": float(base.mean()), "matrix": Mt.tolist(), "matrix_ctrl": Mc.tolist()}


def main():
    res = {}
    for key in ("core__CCT-7__s42", "core__CCT-7__s137", "core__ViT-Small__s42", "core__ViT-Small__s137",
                "core__ResNet-18__s42", "core__ResNet-18__s137"):
        reg = [l for l in T.registry() if l["key"] == key]
        if not reg:
            print("missing", key)
            continue
        lane = reg[0]
        labels = json.load(open(Path(lane["dir"]) / "sae_analysis" / "sae_results.json"))["metadata"]["checkpoint_labels"]
        real = [l for l in labels if l != "terminal"]
        for label in (real[1], real[len(real) // 2], real[-1]):
            if lane["arch"] == "ResNet-18":
                continue  # ResNet: routing.py (exact subtraction); here only the transformers
            r = analyse({**lane, "label": label}, label)
            res[f"{key}|{label}"] = r
            print(f"{key} ckpt {label}: S {r['S']:.3f}  S_ctrl {r['S_ctrl']:.3f}  nAb {r['n_ab']} nDi {r['n_di']}  "
                  f"base sc acc {r['base_sc_acc']:.3f}", flush=True)
    json.dump(res, open(A.OUT / "routing_tf.json", "w"), indent=1)


if __name__ == "__main__":
    main()
