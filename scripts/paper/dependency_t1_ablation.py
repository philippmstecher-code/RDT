#!/usr/bin/env python3
"""T1 (and inputs for T2): cross-layer input dependency of class-specific (CSI)
SAE features on layer3 scaffold (SAI / SSI) SAE features, by forward-pass ablation.

For every standard lane (5 seeds) and milestone m = 0..6:
  * forward the 10k CIFAR-100 test images to the spatial layer3 output (N,256,2,2);
    layer4 on a 2x2 map is computed exactly as a BN-folded MLP (verified to
    reproduce the stored avgpool activations to <4e-3 abs);
  * ablation = mean-preserving projection of the span of a set of layer3 SAE decoder
    directions out of EVERY spatial position:  x' = x - P_S (x - mu_pos);
  * effect = relative drop of each target avgpool-SAE feature's encoder
    pre-activation, over the samples on which that feature fires in the clean run:
        drop_f = sum_{s active}(pre_clean - pre_abl) / sum_{s active} pre_clean
  * scaffold sets (layer3 SAE, checkpoint m):  SAI, SSI (ssi>t, csi<=t, not SAI),
    SCAF = SAI u SSI, SCAFpipe = SAI u SSI_pipe (pipeline Ab definition);
    controls: 1,000 random sets of non-scaffold alive layer3 features matched in
    count and activation mass (each member replaced by one of its 5 nearest
    neighbours in log activation mass);
  * SAE-free variant (T1b): project out the 19-d between-superclass subspace of
    layer3 (SCsub) vs the 19-d within-superclass class subspace (WSCsub).
Target groups: own avgpool SAE, own layer4 SAE (identical data, independent SAE),
and the FINAL (m=6) avgpool / layer4 SAEs applied to checkpoint m (fixed-dictionary
tracking, used by T2).
Outputs: per-feature rows (CSV.gz), set-level rows in analysis_outputs/paper/DEPENDENCY/raw/.
Usage: python scripts/paper/dependency_t1_ablation.py [--seeds S ...] [--milestones M ...] [--draws 1000]
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from dependency_lib import (COARSE_TO_FINE, MILESTONES, OUT, Layer4, LaneData, encode,
                      layer3_spatial, selectivity, standard_lanes)

N_DRAWS = 1000
K_NN = 5
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def matched_draws(set_idx, pool_idx, mass, n_draws, rng):
    """Random sets matched in count and activation mass (nearest-K in log mass, no repeats)."""
    lm = np.log(mass + 1e-12)
    cand = {i: pool_idx[np.argsort(np.abs(lm[pool_idx] - lm[i]))] for i in set_idx}
    draws = []
    for _ in range(n_draws):
        chosen = []
        for i in rng.permutation(set_idx):
            c = [j for j in cand[i] if j not in chosen]
            if not c:
                break
            chosen.append(rng.choice(c[:K_NN]))
        draws.append(np.array(chosen))
    return draws


def orth(V):
    """Orthonormal basis (columns) of the span of V (d,k)."""
    q, r = torch.linalg.qr(V)
    keep = torch.abs(torch.diagonal(r)) > 1e-6 * torch.abs(torch.diagonal(r)).max()
    return q[:, keep]


def subspace_bases(l3_pooled):
    """19-d between-superclass and 19-d within-superclass class subspaces of pooled layer3."""
    X = l3_pooled - l3_pooled.mean(0)
    cm = X.reshape(100, 100, -1).mean(1)  # class means
    scm = np.stack([cm[COARSE_TO_FINE[c]].mean(0) for c in range(20)])
    U, _, _ = np.linalg.svd(scm.T, full_matrices=False)
    sc = U[:, :19]
    resid = cm - scm[np.array([next(c for c in range(20) if k in COARSE_TO_FINE[c]) for k in range(100)])]
    U, _, _ = np.linalg.svd(resid.T, full_matrices=False)
    wsc = U[:, :19]
    return sc, wsc


class Ablator:
    def __init__(self, l4: Layer4, X: torch.Tensor, targets):
        self.l4 = l4
        self.mu = X.mean(0, keepdim=True)
        self.Xc = X - self.mu  # (N,4,256)
        flat = X.reshape(X.shape[0], -1)
        self.z1, self.zd = l4.pre(flat)
        self.a = l4.from_pre(self.z1, self.zd)
        self.W1p = [l4.W1[:, p * 256:(p + 1) * 256] for p in range(4)]
        self.targets = targets  # list of dicts with W (n,512) b (n) A (N,n) float
        for t in targets:
            t["pre"] = self.a @ t["W"].T + t["b"]
            t["den"] = (t["pre"] * t["A"]).sum(0)

    def run(self, Q):
        """Q (256,r) orthonormal. Returns list of per-target relative drop arrays, removed variance."""
        C = torch.einsum("npc,cr->npr", self.Xc, Q)  # (N,4,r)
        dz1 = sum(C[:, p] @ (self.W1p[p] @ Q).T for p in range(4))
        dzd = C[:, 0] @ (self.l4.Wd @ Q).T
        a2 = self.l4.from_pre(self.z1 - dz1, self.zd - dzd)
        var = float((C ** 2).sum(-1).mean())
        out = []
        for t in self.targets:
            pre2 = a2 @ t["W"].T + t["b"]
            out.append(((t["pre"] - pre2) * t["A"]).sum(0) / t["den"])
        return out, var


def target_group(lane, sae, codes, m_sel, csi_t, ssi_t, sai_t, which, dev):
    """Select target features (alive at this checkpoint) and their category."""
    fire = (codes != 0).mean(0)
    alive = fire >= 0.01
    s = m_sel
    sai = (s["sai"] > sai_t) & alive
    csi = (s["csi"] > csi_t) & alive & ~sai
    ssi = (s["ssi"] > ssi_t) & alive & ~sai & ~csi
    if which == "own":
        idx = np.where(csi | ssi | sai)[0]
    else:  # final-dictionary tracking: features that are CSI at the final checkpoint
        idx = np.where(s["final_csi"] & (fire * 10000 >= 10))[0]
    cat = np.where(csi[idx], "CSI", np.where(ssi[idx], "SSI", np.where(sai[idx], "SAI", "other")))
    return dict(idx=idx, cat=cat, fire=fire[idx],
                W=sae["encoder.weight"][idx].to(dev), b=sae["encoder.bias"][idx].to(dev),
                A=torch.from_numpy((codes[:, idx] != 0).astype(np.float32)).to(dev))


def run_lane(r, milestones, n_draws):
    lane = LaneData(r["dir"])
    seed = r["seed"]
    rng = np.random.default_rng(1000 + seed)
    fin = {ly: lane.sae(6, ly) for ly in ("avgpool", "layer4")}
    fin_csi = {}
    for ly in ("avgpool", "layer4"):
        s6 = lane.sel["6"][ly]
        fire6 = (lane.codes(6, ly) != 0).mean(0)
        fin_csi[ly] = (s6["feature_csi"] > lane.csi_t) & (s6["feature_sai"] <= lane.sai_t) & (fire6 >= 0.01)
    feat_rows, draw_rows, set_rows = [], [], []
    for m in milestones:
        t0 = time.time()
        raw = lane.raw(m)
        sp = layer3_spatial(lane, m, device=DEV)
        l4 = Layer4(lane.checkpoint(m), device=DEV)
        X = torch.from_numpy(sp).to(DEV)
        # targets
        groups, targets = [], []
        for which in ("own", "final"):
            for ly in ("avgpool", "layer4"):
                if which == "own":
                    sae = lane.sae(m, ly)
                    codes = lane.codes(m, ly)
                    st = lane.sel[str(m)][ly]
                    s = {"sai": st["feature_sai"], "csi": st["feature_csi"], "ssi": st["feature_ssi"]}
                else:
                    sae = fin[ly]
                    codes = encode(sae, raw["avgpool"])
                    sl = selectivity(codes)
                    s = {"sai": sl["sai"], "csi": sl["csi"], "ssi": sl["ssi"], "final_csi": fin_csi[ly]}
                g = target_group(lane, sae, codes, s, lane.csi_t, lane.ssi_t, lane.sai_t, which, DEV)
                g.update(which=which, layer=ly, ssi=s["ssi"][g["idx"]], csi=s["csi"][g["idx"]], sai=s["sai"][g["idx"]])
                groups.append(g)
                targets.append(g)
        ab = Ablator(l4, X, targets)
        err = float(torch.abs(ab.a.cpu() - torch.from_numpy(raw["avgpool"])).max())
        # layer3 SAE sets
        sae3 = lane.sae(m, "layer3")
        D3 = torch.nn.functional.normalize(sae3["decoder.weight"].float(), dim=0).to(DEV)  # (256,1024)
        c3 = lane.codes(m, "layer3")
        fire3 = (c3 != 0).mean(0)
        alive3 = fire3 >= 0.01
        mass3 = np.abs(c3).sum(0)
        # Threshold-defined layer3 SAI/SSI sets are (near-)empty after epoch 5 (census_features.csv),
        # so scaffold sets are rank-defined among alive layer3 features (top quartile):
        #   SSIq: top-25% SSI among alive features with CSI <= csi_t
        #   SAIq: top-25% SAI among alive features, excluding SSIq
        #   SCAFq = SAIq u SSIq; CSIq (contrast): top-25% CSI among alive, excluding SCAFq
        st3 = lane.sel[str(m)]["layer3"]
        ai = np.where(alive3)[0]
        q = max(1, int(round(0.25 * len(ai))))
        def top(score, elig):
            c = ai[elig[ai]]
            out = np.zeros_like(alive3)
            out[c[np.argsort(-score[c])][:q]] = True
            return out
        ssiq = top(st3["feature_ssi"], st3["feature_csi"] <= lane.csi_t)
        saiq = top(st3["feature_sai"], ~ssiq)
        csiq = top(st3["feature_csi"], ~(ssiq | saiq))
        sets = {"SAIq": saiq, "SSIq": ssiq, "SCAFq": saiq | ssiq, "CSIq": csiq}
        pool_scaf = np.where(alive3 & ~(saiq | ssiq))[0]
        pool_csi = np.where(alive3 & ~csiq)[0]
        # SAE-free subspaces
        scb, wscb = subspace_bases(raw["layer3"])
        res = {}
        for name, Q in (("SCsub", scb), ("WSCsub", wscb)):
            d, v = ab.run(torch.from_numpy(Q.astype(np.float32)).to(DEV))
            res[name] = ([x.cpu().numpy() for x in d], v)
        for sname, smask in sets.items():
            sidx = np.where(smask)[0]
            pool = pool_csi if sname == "CSIq" else pool_scaf
            row = dict(seed=seed, milestone=m, set=sname, n_set=len(sidx), n_pool=len(pool), n_alive3=int(alive3.sum()),
                       mean_ssi=float(st3["feature_ssi"][sidx].mean()) if len(sidx) else np.nan,
                       mean_sai=float(st3["feature_sai"][sidx].mean()) if len(sidx) else np.nan,
                       mean_csi=float(st3["feature_csi"][sidx].mean()) if len(sidx) else np.nan)
            if len(sidx) == 0 or len(pool) < len(sidx):
                set_rows.append(row)
                continue
            d_s, v_s = ab.run(orth(D3[:, sidx]))
            d_s = [x.cpu().numpy() for x in d_s]
            draws = matched_draws(sidx, pool, mass3, n_draws if len(pool) > len(sidx) + 2 else 1, rng)  # forced complement -> 1 draw
            n = len(groups)
            S1 = [np.zeros(len(g["idx"])) for g in groups]
            S2 = [np.zeros(len(g["idx"])) for g in groups]
            GE = [np.zeros(len(g["idx"])) for g in groups]
            SN = [np.zeros(len(g["idx"])) for g in groups]
            vr = []
            med_draw = {gi: [] for gi in range(n)}
            for dr in draws:
                d_r, v_r = ab.run(orth(D3[:, dr]))
                vr.append(v_r)
                for gi in range(n):
                    x = d_r[gi].cpu().numpy()
                    S1[gi] += x; S2[gi] += x * x; GE[gi] += x >= d_s[gi]; SN[gi] += x / max(v_r, 1e-12)
                    csi_m = groups[gi]["cat"] == "CSI" if groups[gi]["which"] == "own" else np.ones(len(x), bool)
                    med_draw[gi].append(np.median(x[csi_m]) if csi_m.any() else np.nan)
            nd = len(draws)
            row.update(var_scaf=v_s, var_rand_mean=float(np.mean(vr)), var_rand_sd=float(np.std(vr)),
                       mass_scaf=float(mass3[sidx].sum()), mass_rand_mean=float(np.mean([mass3[d].sum() for d in draws])))
            set_rows.append(row)
            for gi, g in enumerate(groups):
                mu_r = S1[gi] / nd
                sd_r = np.sqrt(np.maximum(S2[gi] / nd - mu_r ** 2, 0))
                csi_m = g["cat"] == "CSI" if g["which"] == "own" else np.ones(len(g["idx"]), bool)
                md = np.array(med_draw[gi])
                draw_rows.append(dict(seed=seed, milestone=m, set=sname, which=g["which"], layer=g["layer"],
                                      n_target=int(csi_m.sum()),
                                      med_scaf=float(np.median(d_s[gi][csi_m])) if csi_m.any() else np.nan,
                                      med_rand_mean=float(np.nanmean(md)), med_rand_q95=float(np.nanpercentile(md, 95)) if np.isfinite(md).any() else np.nan,
                                      p_set=float((np.sum(md >= np.median(d_s[gi][csi_m])) + 1) / (nd + 1)) if csi_m.any() else np.nan))
                for k in range(len(g["idx"])):
                    feat_rows.append(dict(seed=seed, milestone=m, set=sname, which=g["which"], layer=g["layer"],
                                          fid=int(g["idx"][k]), cat=g["cat"][k], fire=float(g["fire"][k]),
                                          ssi=float(g["ssi"][k]), csi=float(g["csi"][k]), sai=float(g["sai"][k]),
                                          drop_scaf=float(d_s[gi][k]), drop_rand_mean=float(mu_r[k]), drop_rand_sd=float(sd_r[k]),
                                          p_feat=float((GE[gi][k] + 1) / (nd + 1)),
                                          dropn_scaf=float(d_s[gi][k] / max(v_s, 1e-12)), dropn_rand_mean=float(SN[gi][k] / nd),
                                          drop_SCsub=float(res["SCsub"][0][gi][k]), drop_WSCsub=float(res["WSCsub"][0][gi][k]),
                                          var_SCsub=res["SCsub"][1], var_WSCsub=res["WSCsub"][1]))
        # subspace rows even when a set is empty (attach to set='none')
        if not any(fr["milestone"] == m and fr["seed"] == seed for fr in feat_rows):
            for gi, g in enumerate(groups):
                for k in range(len(g["idx"])):
                    feat_rows.append(dict(seed=seed, milestone=m, set="none", which=g["which"], layer=g["layer"],
                                          fid=int(g["idx"][k]), cat=g["cat"][k], fire=float(g["fire"][k]),
                                          ssi=float(g["ssi"][k]), csi=float(g["csi"][k]), sai=float(g["sai"][k]),
                                          drop_SCsub=float(res["SCsub"][0][gi][k]), drop_WSCsub=float(res["WSCsub"][0][gi][k]),
                                          var_SCsub=res["SCsub"][1], var_WSCsub=res["WSCsub"][1]))
        print(f"seed {seed} m{m}: fwd err {err:.2e}, sets " + ", ".join(f"{k}={int(v.sum())}" for k, v in sets.items())
              + f", pool={len(pool_scaf)}, targets " + ", ".join(f"{g['which']}/{g['layer']}={len(g['idx'])}" for g in groups)
              + f" [{time.time() - t0:.0f}s]", flush=True)
        del X, ab
    raw_dir = OUT / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(feat_rows).to_csv(raw_dir / f"t1_features_seed{seed}.csv.gz", index=False, float_format="%.6g")
    pd.DataFrame(draw_rows).to_csv(raw_dir / f"t1_setlevel_seed{seed}.csv", index=False, float_format="%.6g")
    pd.DataFrame(set_rows).to_csv(raw_dir / f"t1_sets_seed{seed}.csv", index=False, float_format="%.6g")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--milestones", type=int, nargs="*", default=MILESTONES)
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    a = ap.parse_args()
    for r in standard_lanes():
        if a.seeds and r["seed"] not in a.seeds:
            continue
        run_lane(r, a.milestones, a.draws)


if __name__ == "__main__":
    main()
