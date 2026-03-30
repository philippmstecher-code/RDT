#!/usr/bin/env python3
"""Feature survival analysis stratified by task-general (Tg-H) status.

Extends the base survival analysis to answer:
- Do task-general features (SAI > 0.9) survive longer than non-Tg features?
- How does Tg-H survival compare to high-SSI and high-CSI cohorts?
- Log-rank test: Tg-H vs non-Tg-H
- Spearman: SAI vs lifespan

Usage:
    python scripts/compute_feature_survival_tg.py
"""
import json
import sys
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CONSOLIDATED = DATA_DIR / "consolidated_findings.json"

LANES = [
    ("abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb", "7db28a9a-f436-4d8a-a6b4-37e69fbf54eb", "ResNet18-CIFAR100-seed42"),
    ("abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb", "a0bcbcba-94a9-4009-bcb6-d725b47588ee", "ResNet18-CIFAR100-seed137"),
    ("abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb", "935c13ae-a026-4eae-a2b2-0306879b2e8c", "ResNet18-CIFAR100-seed256"),
    ("abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb", "10d31151-68a2-4491-836d-3ead8e00a7ad", "ViTSmall-CIFAR100-seed42"),
    ("abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb", "909d7f68-55e1-4cd9-8c7b-68d5793798d3", "ViTSmall-CIFAR100-seed137"),
    ("abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb", "abad8975-de55-4e7e-b95d-05067f0eed90", "ViTSmall-CIFAR100-seed256"),
    ("abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb", "4bebc329-c26a-4eb4-b18c-7586e513f49e", "CCT7-CIFAR100-seed42"),
    ("abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb", "edf8275e-c99f-4436-a13d-4ed50f072a66", "CCT7-CIFAR100-seed137"),
    ("abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb", "445dd4d4-770a-4f33-b82b-c0bc7baed9a7", "CCT7-CIFAR100-seed256"),
    ("ce93d3f6-282b-4811-84f8-be6c589c0500", "c645c39e-e21f-4cff-853f-967bf69ee857", "ResNet18-CIFAR100-200ep-seed42"),
    ("578b5803-1f2f-49c4-99b8-bf7071f43917", "abfc645c-4b05-46b0-b940-0ef3e8c7284f", "ResNet18-CIFAR100-8x-seed42"),
    ("821eaadb-0160-48f8-9669-ec3d9185c428", "45476b66-90e0-47f0-aec6-41c62aff8c21", "ResNet18-TinyImageNet-seed42"),
]

SAI_THRESHOLD = 0.9


def _load_adaptive_thresholds() -> dict[str, dict]:
    """Load per-lane adaptive thresholds from consolidated_findings.json."""
    if not CONSOLIDATED.exists():
        print(f"WARNING: {CONSOLIDATED} not found, falling back to median splits")
        return {}
    with open(CONSOLIDATED) as f:
        cf = json.load(f)
    thresholds = {}
    for label, lane_data in cf.get('lanes', {}).items():
        at = lane_data.get('metadata', {}).get('adaptive_thresholds', {})
        if at:
            thresholds[label] = at
    return thresholds


def _safe_layer_filename(layer: str) -> str:
    return layer.replace("/", "_").replace(".", "_")


def compute_survival_with_tg(sae_dir: Path, layers: list[str], checkpoint_labels: list[str],
                              sae_data: dict, adaptive_thresh: Optional[dict] = None) -> Optional[dict]:
    """Feature survival stratified by Tg-H (SAI > 0.9), SSI, and CSI."""

    if len(checkpoint_labels) < 3:
        return None

    transitions_dir = sae_dir / "transitions"
    selectivity = sae_data.get('selectivity', {})
    n_transitions = len(checkpoint_labels) - 1

    feature_records: list[dict] = []

    for layer in layers:
        # Build forward maps
        forward_maps: list[dict[int, int]] = [dict() for _ in range(n_transitions)]

        for t_idx in range(n_transitions):
            ckpt_a = checkpoint_labels[t_idx]
            ckpt_b = checkpoint_labels[t_idx + 1]
            pair_key = f"{ckpt_a}_to_{ckpt_b}"
            trans_file = transitions_dir / pair_key / f"{_safe_layer_filename(layer)}.json"
            if not trans_file.exists():
                continue
            try:
                with open(trans_file, 'r') as f:
                    samples = json.load(f)
            except Exception:
                continue

            link_counts: dict[tuple[int, int], int] = {}
            for sample in samples:
                for entry in sample.get('stable', []):
                    fid_a = entry.get('fid_a')
                    fid_b = entry.get('fid_b')
                    if fid_a is not None and fid_b is not None:
                        key = (fid_a, fid_b)
                        link_counts[key] = link_counts.get(key, 0) + 1

            src_to_best: dict[int, int] = {}
            for (fid_a, fid_b), count in link_counts.items():
                if fid_a not in src_to_best or count > link_counts.get((fid_a, src_to_best[fid_a]), 0):
                    src_to_best[fid_a] = fid_b

            forward_maps[t_idx] = src_to_best

        # Trace features
        for birth_idx in range(len(checkpoint_labels)):
            ckpt_label = checkpoint_labels[birth_idx]
            sel_ckpt = selectivity.get(ckpt_label, {}).get(layer, {})
            ssi_list = sel_ckpt.get('feature_ssi', [])
            csi_list = sel_ckpt.get('feature_csi', [])
            sai_list = sel_ckpt.get('feature_sai', [])
            if not ssi_list:
                continue

            alive_fids: set[int] = set()
            if birth_idx < n_transitions:
                alive_fids = set(forward_maps[birth_idx].keys())
            for fid in range(len(ssi_list)):
                if ssi_list[fid] > 0 or (fid < len(csi_list) and csi_list[fid] > 0):
                    alive_fids.add(fid)

            for fid in alive_fids:
                current_fid = fid
                lifespan = 0
                for t_idx in range(birth_idx, n_transitions):
                    if current_fid in forward_maps[t_idx]:
                        lifespan += 1
                        current_fid = forward_maps[t_idx][current_fid]
                    else:
                        break

                survived_to_terminal = (birth_idx + lifespan) >= n_transitions
                initial_ssi = ssi_list[fid] if fid < len(ssi_list) else 0.0
                initial_csi = csi_list[fid] if fid < len(csi_list) else 0.0
                initial_sai = sai_list[fid] if fid < len(sai_list) else 0.0

                feature_records.append({
                    'layer': layer,
                    'birth_idx': birth_idx,
                    'lifespan': lifespan,
                    'max_possible': n_transitions - birth_idx,
                    'censored': survived_to_terminal,
                    'initial_ssi': initial_ssi,
                    'initial_csi': initial_csi,
                    'initial_sai': initial_sai,
                    'is_tg': initial_sai > SAI_THRESHOLD,
                })

    if not feature_records:
        return None

    # ── Cohort definitions (adaptive thresholds from null permutation test) ──
    tg_features = [r for r in feature_records if r['is_tg']]
    non_tg_features = [r for r in feature_records if not r['is_tg']]

    if adaptive_thresh:
        ssi_threshold = adaptive_thresh['ssi_adaptive_thresh']
        csi_threshold = adaptive_thresh['csi_adaptive_thresh']
        threshold_source = 'adaptive'
    else:
        ssi_threshold = float(np.median([r['initial_ssi'] for r in feature_records]))
        csi_threshold = float(np.median([r['initial_csi'] for r in feature_records]))
        threshold_source = 'median'

    high_ssi = [r for r in feature_records if r['initial_ssi'] >= ssi_threshold]
    low_ssi = [r for r in feature_records if r['initial_ssi'] < ssi_threshold]

    high_csi = [r for r in feature_records if r['initial_csi'] >= csi_threshold]
    low_csi = [r for r in feature_records if r['initial_csi'] < csi_threshold]

    # Tg-H features that also have high SSI (task-general + abstract)
    tg_high_ssi = [r for r in feature_records if r['is_tg'] and r['initial_ssi'] >= ssi_threshold]
    # Non-Tg with high SSI (category-specific abstraction)
    non_tg_high_ssi = [r for r in feature_records if not r['is_tg'] and r['initial_ssi'] >= ssi_threshold]

    def _cohort_stats(records, name):
        if not records:
            return {'name': name, 'n': 0}
        lifespans = [r['lifespan'] for r in records]
        n_surviving = sum(1 for r in records if r['censored'])
        return {
            'name': name,
            'n': len(records),
            'survival_rate': round(n_surviving / len(records), 4),
            'mean_lifespan': round(float(np.mean(lifespans)), 4),
            'median_lifespan': round(float(np.median(lifespans)), 2),
            'max_lifespan': max(lifespans),
            'pct_surviving_to_terminal': round(100 * n_surviving / len(records), 2),
        }

    def _logrank(group1, group2):
        """Log-rank test between two cohorts."""
        if not group1 or not group2:
            return None, None
        all_recs = group1 + group2
        max_time = max(r['max_possible'] for r in all_recs)
        chi2_sum = 0.0
        var_sum = 0.0
        for t in range(max_time + 1):
            n1 = sum(1 for r in group1 if r['lifespan'] >= t)
            n2 = sum(1 for r in group2 if r['lifespan'] >= t)
            d1 = sum(1 for r in group1 if r['lifespan'] == t and not r['censored'])
            d2 = sum(1 for r in group2 if r['lifespan'] == t and not r['censored'])
            n = n1 + n2
            d = d1 + d2
            if n > 1 and d > 0:
                e1 = d * n1 / n
                var = d * (n - d) * n1 * n2 / (n * n * max(1, n - 1))
                chi2_sum += (d1 - e1)
                var_sum += var
        if var_sum > 0:
            chi2 = round(chi2_sum ** 2 / var_sum, 4)
            p = round(float(1.0 - stats.chi2.cdf(chi2, df=1)), 8)
            return chi2, p
        return None, None

    # ── Log-rank tests ──
    lr_tg_chi2, lr_tg_p = _logrank(tg_features, non_tg_features)
    lr_ssi_chi2, lr_ssi_p = _logrank(high_ssi, low_ssi)
    lr_csi_chi2, lr_csi_p = _logrank(high_csi, low_csi)
    lr_tg_vs_nontg_highssi_chi2, lr_tg_vs_nontg_highssi_p = _logrank(tg_high_ssi, non_tg_high_ssi)

    # ── Spearman correlations ──
    lifespans = np.array([r['lifespan'] for r in feature_records])
    sais = np.array([r['initial_sai'] for r in feature_records])
    ssis = np.array([r['initial_ssi'] for r in feature_records])
    csis = np.array([r['initial_csi'] for r in feature_records])

    def _spearman(x, y):
        if len(x) > 10:
            try:
                rho, p = stats.spearmanr(x, y)
                return round(float(rho), 4), round(float(p), 8)
            except Exception:
                pass
        return None, None

    sai_rho, sai_p = _spearman(sais, lifespans)
    ssi_rho, ssi_p = _spearman(ssis, lifespans)
    csi_rho, csi_p = _spearman(csis, lifespans)

    # ── Per-layer Tg-H breakdown ──
    per_layer_tg = {}
    for layer in layers:
        layer_recs = [r for r in feature_records if r['layer'] == layer]
        layer_tg = [r for r in layer_recs if r['is_tg']]
        layer_non_tg = [r for r in layer_recs if not r['is_tg']]
        tg_surv = sum(1 for r in layer_tg if r['censored']) / len(layer_tg) if layer_tg else 0
        non_tg_surv = sum(1 for r in layer_non_tg if r['censored']) / len(layer_non_tg) if layer_non_tg else 0
        tg_mean = float(np.mean([r['lifespan'] for r in layer_tg])) if layer_tg else 0
        non_tg_mean = float(np.mean([r['lifespan'] for r in layer_non_tg])) if layer_non_tg else 0
        per_layer_tg[layer] = {
            'n_tg': len(layer_tg),
            'n_non_tg': len(layer_non_tg),
            'tg_survival_rate': round(tg_surv, 4),
            'non_tg_survival_rate': round(non_tg_surv, 4),
            'tg_mean_lifespan': round(tg_mean, 4),
            'non_tg_mean_lifespan': round(non_tg_mean, 4),
        }

    # ── Birth checkpoint distribution for Tg-H ──
    tg_birth_dist = {}
    for r in tg_features:
        b = r['birth_idx']
        tg_birth_dist[b] = tg_birth_dist.get(b, 0) + 1

    return {
        'n_total': len(feature_records),
        'threshold_source': threshold_source,
        'ssi_threshold': ssi_threshold,
        'csi_threshold': csi_threshold,
        'cohorts': {
            'tg': _cohort_stats(tg_features, 'Task-General (SAI>0.9)'),
            'non_tg': _cohort_stats(non_tg_features, 'Non-Tg (SAI≤0.9)'),
            'high_ssi': _cohort_stats(high_ssi, f'High SSI (≥{ssi_threshold:.4f}, {threshold_source})'),
            'low_ssi': _cohort_stats(low_ssi, f'Low SSI (<{ssi_threshold:.4f}, {threshold_source})'),
            'high_csi': _cohort_stats(high_csi, f'High CSI (≥{csi_threshold:.4f}, {threshold_source})'),
            'low_csi': _cohort_stats(low_csi, f'Low CSI (<{csi_threshold:.4f}, {threshold_source})'),
            'tg_high_ssi': _cohort_stats(tg_high_ssi, 'Tg-H ∩ High-SSI'),
            'non_tg_high_ssi': _cohort_stats(non_tg_high_ssi, 'Non-Tg ∩ High-SSI'),
        },
        'logrank': {
            'tg_vs_non_tg': {'chi2': lr_tg_chi2, 'p': lr_tg_p},
            'high_ssi_vs_low_ssi': {'chi2': lr_ssi_chi2, 'p': lr_ssi_p},
            'high_csi_vs_low_csi': {'chi2': lr_csi_chi2, 'p': lr_csi_p},
            'tg_highssi_vs_nontg_highssi': {'chi2': lr_tg_vs_nontg_highssi_chi2, 'p': lr_tg_vs_nontg_highssi_p},
        },
        'spearman': {
            'sai_vs_lifespan': {'rho': sai_rho, 'p': sai_p},
            'ssi_vs_lifespan': {'rho': ssi_rho, 'p': ssi_p},
            'csi_vs_lifespan': {'rho': csi_rho, 'p': csi_p},
        },
        'per_layer_tg': per_layer_tg,
        'tg_birth_checkpoint_distribution': tg_birth_dist,
    }


def main():
    adaptive_thresholds = _load_adaptive_thresholds()
    if adaptive_thresholds:
        print(f"  Loaded adaptive thresholds for {len(adaptive_thresholds)} lanes")
    else:
        print("  WARNING: No adaptive thresholds found, using median fallback")

    results = {}
    for exp_id, lane_id, label in LANES:
        sae_dir = DATA_DIR / "experiments" / exp_id / "lanes" / lane_id / "sae_analysis"
        results_file = sae_dir / "sae_results.json"

        if not results_file.exists():
            print(f"  SKIP {label}")
            continue

        print(f"  {label}...", end=" ", flush=True)

        with open(results_file, 'r') as f:
            sae_data = json.load(f)

        layers = sae_data['metadata']['layers']
        checkpoint_labels = sae_data['metadata']['checkpoint_labels']

        # Look up adaptive thresholds for this lane
        # Try exact label match, then strip "-seed*" suffix variants
        lane_thresh = adaptive_thresholds.get(label)
        if lane_thresh:
            print(f"[adaptive: ssi={lane_thresh['ssi_adaptive_thresh']:.4f}, csi={lane_thresh['csi_adaptive_thresh']:.4f}] ", end="")
        else:
            # Try alternate label mapping (e.g. "ResNet18-CIFAR100-200ep-seed42" -> "ResNet18-CIFAR100-200ep")
            for alt_label in adaptive_thresholds:
                if label.startswith(alt_label) or alt_label.startswith(label.rsplit('-seed', 1)[0]):
                    lane_thresh = adaptive_thresholds[alt_label]
                    print(f"[adaptive via {alt_label}] ", end="")
                    break
            if not lane_thresh:
                print("[median fallback] ", end="")

        result = compute_survival_with_tg(sae_dir, layers, checkpoint_labels, sae_data,
                                           adaptive_thresh=lane_thresh)

        if result is None:
            print("NO DATA")
            continue

        results[label] = result
        tg = result['cohorts']['tg']
        non_tg = result['cohorts']['non_tg']
        lr = result['logrank']['tg_vs_non_tg']
        sp = result['spearman']['sai_vs_lifespan']
        print(f"Tg-H: n={tg['n']}, surv={tg.get('survival_rate','?')}, mean_life={tg.get('mean_lifespan','?')} | "
              f"Non-Tg: surv={non_tg.get('survival_rate','?')}, mean_life={non_tg.get('mean_lifespan','?')} | "
              f"LR p={lr['p']} | SAI ρ={sp['rho']}")

    out_path = DATA_DIR / "feature_survival_tg_expanded.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # ── Summary table ──
    print("\n" + "=" * 140)
    print(f"{'Lane':<35} {'Tg n':>6} {'Tg surv%':>9} {'Tg mean':>8} {'nonTg surv%':>11} {'nonTg mean':>10} "
          f"{'LR χ²':>8} {'LR p':>12} {'SAI ρ':>8} {'SAI p':>12} {'TgHiSSI vs nonTgHiSSI p':>24}")
    print("-" * 140)
    for label, r in results.items():
        tg = r['cohorts']['tg']
        ntg = r['cohorts']['non_tg']
        lr = r['logrank']['tg_vs_non_tg']
        lr2 = r['logrank']['tg_highssi_vs_nontg_highssi']
        sp = r['spearman']['sai_vs_lifespan']
        print(f"{label:<35} {tg['n']:>6} {tg.get('survival_rate',0):>9.4f} {tg.get('mean_lifespan',0):>8.3f} "
              f"{ntg.get('survival_rate',0):>11.4f} {ntg.get('mean_lifespan',0):>10.3f} "
              f"{lr['chi2'] or 'N/A':>8} {lr['p'] or 'N/A':>12} "
              f"{sp['rho'] or 'N/A':>8} {sp['p'] or 'N/A':>12} "
              f"{lr2['p'] or 'N/A':>24}")
    print("=" * 140)

    # ── Per-layer Tg-H breakdown for primary lane ──
    print("\nPer-layer Tg-H breakdown (ResNet18-CIFAR100-seed42):")
    if 'ResNet18-CIFAR100-seed42' in results:
        for layer, info in results['ResNet18-CIFAR100-seed42']['per_layer_tg'].items():
            print(f"  {layer:<15} Tg: n={info['n_tg']:>4}, surv={info['tg_survival_rate']:.4f}, "
                  f"mean_life={info['tg_mean_lifespan']:.3f} | "
                  f"Non-Tg: n={info['n_non_tg']:>5}, surv={info['non_tg_survival_rate']:.4f}, "
                  f"mean_life={info['non_tg_mean_lifespan']:.3f}")

    # ── Tg-H birth checkpoint distribution ──
    print("\nTg-H birth checkpoint distribution (ResNet18-CIFAR100-seed42):")
    if 'ResNet18-CIFAR100-seed42' in results:
        dist = results['ResNet18-CIFAR100-seed42']['tg_birth_checkpoint_distribution']
        total = sum(dist.values())
        for ckpt in sorted(dist.keys(), key=int):
            print(f"  Checkpoint {ckpt}: {dist[ckpt]} Tg-H features ({100*dist[ckpt]/total:.1f}%)")


if __name__ == "__main__":
    main()
