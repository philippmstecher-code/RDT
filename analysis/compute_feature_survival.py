#!/usr/bin/env python3
"""Compute feature survival (Kaplan-Meier) for all 9 completed SAEANALYSIS lanes.

Outputs a JSON summary with per-lane survival statistics:
- Log-rank test (high-SSI vs low-SSI)
- Spearman correlations (SSI/CSI vs lifespan)
- Per-cohort median survival
- Per-layer survival rates

Usage:
    python scripts/compute_feature_survival.py
"""
import json
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

# ── Paths ──
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# All 9 lanes: (experiment_id, lane_id, label)
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


def _safe_layer_filename(layer: str) -> str:
    return layer.replace("/", "_").replace(".", "_")


def compute_feature_survival(sae_dir: Path, layers: list[str], checkpoint_labels: list[str],
                              sae_data: dict, matching_method: str = "activation") -> Optional[dict]:
    """Adapted from saeanalysis.py feature survival computation — returns plain dict."""
    import numpy as np
    from scipy import stats

    if len(checkpoint_labels) < 3:
        return None

    transitions_dir = sae_dir / ("transitions_weight" if matching_method == "weight" else "transitions")
    selectivity = sae_data.get('selectivity', {})
    n_transitions = len(checkpoint_labels) - 1

    feature_records: list[dict] = []

    for layer in layers:
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
                if fid_a not in src_to_best:
                    src_to_best[fid_a] = fid_b
                else:
                    old_b = src_to_best[fid_a]
                    old_count = link_counts.get((fid_a, old_b), 0)
                    if count > old_count:
                        src_to_best[fid_a] = fid_b

            forward_maps[t_idx] = src_to_best

        for birth_idx in range(len(checkpoint_labels)):
            ckpt_label = checkpoint_labels[birth_idx]
            sel_ckpt = selectivity.get(ckpt_label, {}).get(layer, {})
            ssi_list = sel_ckpt.get('feature_ssi', [])
            csi_list = sel_ckpt.get('feature_csi', [])
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
                    fwd = forward_maps[t_idx]
                    if current_fid in fwd:
                        lifespan += 1
                        current_fid = fwd[current_fid]
                    else:
                        break

                survived_to_terminal = (birth_idx + lifespan) >= n_transitions
                initial_ssi = ssi_list[fid] if fid < len(ssi_list) else 0.0
                initial_csi = csi_list[fid] if fid < len(csi_list) else 0.0
                max_possible = n_transitions - birth_idx

                feature_records.append({
                    'layer': layer,
                    'birth_idx': birth_idx,
                    'lifespan': lifespan,
                    'max_possible': max_possible,
                    'censored': survived_to_terminal,
                    'initial_ssi': initial_ssi,
                    'initial_csi': initial_csi,
                })

    if not feature_records:
        return None

    # ── Kaplan-Meier ──
    transition_labels = [f"{checkpoint_labels[i]}->{checkpoint_labels[i+1]}" for i in range(n_transitions)]

    def _build_km_curve(records):
        if not records:
            return []
        max_time = max(r['max_possible'] for r in records)
        n_at_risk = len(records)
        survival = 1.0
        curve = []
        for t in range(max_time + 1):
            n_events = sum(1 for r in records if r['lifespan'] == t and not r['censored'])
            n_censored = sum(1 for r in records if r['lifespan'] == t and r['censored'])
            if n_at_risk > 0 and n_events > 0:
                survival *= (1.0 - n_events / n_at_risk)
            trans_label = transition_labels[t] if t < len(transition_labels) else f"t={t}"
            curve.append({
                'transition_idx': t, 'transition': trans_label,
                'n_at_risk': n_at_risk, 'n_events': n_events,
                'n_censored': n_censored, 'survival_prob': round(survival, 4),
            })
            n_at_risk -= (n_events + n_censored)
            if n_at_risk <= 0:
                break
        return curve

    def _median_survival(curve):
        for pt in curve:
            if pt['survival_prob'] <= 0.5:
                return float(pt['transition_idx'])
        return None

    overall_curve = _build_km_curve(feature_records)
    overall_median = _median_survival(overall_curve)

    # ── Stratify by SSI ──
    ssi_values = [r['initial_ssi'] for r in feature_records]
    ssi_median = float(np.median(ssi_values)) if ssi_values else 0.3

    high_ssi = [r for r in feature_records if r['initial_ssi'] >= ssi_median]
    low_ssi = [r for r in feature_records if r['initial_ssi'] < ssi_median]

    high_ssi_curve = _build_km_curve(high_ssi)
    low_ssi_curve = _build_km_curve(low_ssi)

    # ── Log-rank test ──
    logrank_chi2 = None
    logrank_p = None
    if high_ssi and low_ssi:
        try:
            max_time = max(r['max_possible'] for r in feature_records)
            chi2_sum = 0.0
            var_sum = 0.0
            for t in range(max_time + 1):
                n1 = sum(1 for r in high_ssi if r['lifespan'] >= t)
                n2 = sum(1 for r in low_ssi if r['lifespan'] >= t)
                d1 = sum(1 for r in high_ssi if r['lifespan'] == t and not r['censored'])
                d2 = sum(1 for r in low_ssi if r['lifespan'] == t and not r['censored'])
                n = n1 + n2
                d = d1 + d2
                if n > 1 and d > 0:
                    e1 = d * n1 / n
                    var = d * (n - d) * n1 * n2 / (n * n * max(1, n - 1))
                    chi2_sum += (d1 - e1)
                    var_sum += var
            if var_sum > 0:
                logrank_chi2 = round(chi2_sum ** 2 / var_sum, 4)
                logrank_p = round(float(1.0 - stats.chi2.cdf(logrank_chi2, df=1)), 6)
        except Exception:
            pass

    # ── Spearman correlations ──
    lifespans = np.array([r['lifespan'] for r in feature_records])
    initial_ssis = np.array([r['initial_ssi'] for r in feature_records])
    initial_csis = np.array([r['initial_csi'] for r in feature_records])

    ssi_corr, ssi_p_val = None, None
    csi_corr, csi_p_val = None, None
    if len(lifespans) > 10:
        try:
            rho, p = stats.spearmanr(initial_ssis, lifespans)
            ssi_corr = round(float(rho), 4)
            ssi_p_val = round(float(p), 6)
        except Exception:
            pass
        try:
            rho, p = stats.spearmanr(initial_csis, lifespans)
            csi_corr = round(float(rho), 4)
            csi_p_val = round(float(p), 6)
        except Exception:
            pass

    # ── Per-layer summary ──
    per_layer = {}
    for layer in layers:
        layer_recs = [r for r in feature_records if r['layer'] == layer]
        if not layer_recs:
            per_layer[layer] = {'n_tracked': 0}
            continue
        ls = [r['lifespan'] for r in layer_recs]
        n_surviving = sum(1 for r in layer_recs if r['censored'])
        per_layer[layer] = {
            'n_tracked': len(layer_recs),
            'survival_rate': round(n_surviving / len(layer_recs), 4),
            'median_lifespan': round(float(np.median(ls)), 2),
            'mean_lifespan': round(float(np.mean(ls)), 2),
        }

    n_surviving_total = sum(1 for r in feature_records if r['censored'])

    return {
        'n_total_tracked': len(feature_records),
        'overall_survival_rate': round(n_surviving_total / len(feature_records), 4),
        'overall_median_lifespan': overall_median,
        'ssi_median_threshold': round(ssi_median, 4),
        'high_ssi_n': len(high_ssi),
        'low_ssi_n': len(low_ssi),
        'high_ssi_median_survival': _median_survival(high_ssi_curve),
        'low_ssi_median_survival': _median_survival(low_ssi_curve),
        'logrank_chi2': logrank_chi2,
        'logrank_p': logrank_p,
        'ssi_survival_corr': ssi_corr,
        'ssi_survival_p': ssi_p_val,
        'csi_survival_corr': csi_corr,
        'csi_survival_p': csi_p_val,
        'per_layer': per_layer,
    }


def main():
    results = {}
    for exp_id, lane_id, label in LANES:
        sae_dir = DATA_DIR / "experiments" / exp_id / "lanes" / lane_id / "sae_analysis"
        results_file = sae_dir / "sae_results.json"

        if not results_file.exists():
            print(f"  SKIP {label}: no sae_results.json")
            continue

        print(f"  Computing {label}...", end=" ", flush=True)

        with open(results_file, 'r') as f:
            sae_data = json.load(f)

        layers = sae_data['metadata']['layers']
        checkpoint_labels = sae_data['metadata']['checkpoint_labels']

        result = compute_feature_survival(sae_dir, layers, checkpoint_labels, sae_data)

        if result is None:
            print("NO DATA")
            continue

        results[label] = result
        print(f"OK — {result['n_total_tracked']} features tracked, "
              f"logrank p={result['logrank_p']}, "
              f"SSI-lifespan ρ={result['ssi_survival_corr']} (p={result['ssi_survival_p']})")

    # Save aggregated results
    out_path = DATA_DIR / "feature_survival_all_lanes.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Lane':<40} {'N feat':>8} {'Surv%':>8} {'Med life':>10} "
          f"{'LR χ²':>8} {'LR p':>10} {'SSI ρ':>8} {'SSI p':>10} {'CSI ρ':>8} {'CSI p':>10}")
    print("-" * 100)
    for label, r in results.items():
        print(f"{label:<40} {r['n_total_tracked']:>8} {r['overall_survival_rate']:>8.4f} "
              f"{r['overall_median_lifespan'] or 'N/A':>10} "
              f"{r['logrank_chi2'] or 'N/A':>8} {r['logrank_p'] or 'N/A':>10} "
              f"{r['ssi_survival_corr'] or 'N/A':>8} {r['ssi_survival_p'] or 'N/A':>10} "
              f"{r['csi_survival_corr'] or 'N/A':>8} {r['csi_survival_p'] or 'N/A':>10}")
    print("=" * 100)


if __name__ == "__main__":
    main()
