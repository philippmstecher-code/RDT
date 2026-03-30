"""
SAEANALYSIS — Per-checkpoint SAE feature decomposition pipeline.

Trains a SEPARATE SAE per layer at EACH checkpoint, giving each milestone its
own feature dictionary.  Features across checkpoints are matched by correlating
their sample-activation columns (functional equivalence — which samples activate
this feature?).

Methodological foundation:
  - SAE architecture: top-k sparsity (Gao et al. 2024), unit-norm decoder
    columns (Bricken et al. 2023).
  - Feature matching: Hungarian algorithm on Pearson r of activation columns,
    with decoder cosine similarity as secondary signal (Bau et al. 2025).
  - Selectivity indices: SSI/CSI/SAI formulas from Morcos et al. 2018, with
    thresholds justified by permutation null baseline.
  - Task-general abstraction: SAI (Super Abstraction Index) measures
    uniformity of activation across all classes via normalized entropy.
    SAI = H(p_f) / log(N_classes), bounded [0,1]. Superclass-level
    selectivity is handled separately by SSI.
  - Shared initialisation across checkpoints ensures comparable decoder
    directions (motivated by Bau et al. 2025's finding that different seeds
    yield only ~30% shared features; related to Bai et al. 2024 SAE-Track).

Key references:
  [1] Bricken et al. 2023, "Towards Monosemanticity"
  [2] Gao et al. 2024, "Scaling and Evaluating Sparse Autoencoders"
  [3] Bau et al. 2025, "SAEs Trained on Same Data Learn Different Features"
  [4] Bai et al. 2024, "SAE-Track: Tracking Feature Dynamics in LLM Training"
  [5] Morcos et al. 2018, "On the Importance of Single Directions"
  [6] Gorton et al. 2025, "Sparse Autoencoders for Vision Models"
  [7] Cunningham et al. 2023, "SAEs Find Highly Interpretable Features"

Eleven phases:

  1. Data loading                            (0-10 %)
  2. Per-checkpoint SAE training             (10-35 %)
  3. Per-sample encoding                     (35-50 %)
  4. SSI / CSI / SAI computation               (50-58 %)
  5. Feature matching (activation + weight)   (58-70 %)
 5c. Within-checkpoint control (SAE stab.)   (70-72 %)
  6. Process classification (per-sample)     (72-78 %)
  7. Aggregation (class/superclass/global)   (78-90 %)
  8. Hypothesis testing & null baseline      (90-96 %)
  9. Output                                  (96-100 %)

Output:
  {lane_dir}/sae_analysis/sae_results.json        — aggregated results
  {lane_dir}/sae_analysis/saes/{ckpt}/{layer}.pt   — SAE weights
  {lane_dir}/sae_analysis/activations/{ckpt}/{layer}.pt  — activation matrices
  {lane_dir}/sae_analysis/transitions/{pair}/{ci}_{si}.json — per-sample events
"""
import gc
import json
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sp_stats
from scipy.optimize import linear_sum_assignment

from cifar100_hierarchy import (
    build_superclass_map as _cifar100_build_superclass_map,
    get_superclass_groups,
    has_hierarchical_structure,
)
from inat_hierarchy import (
    build_superclass_map as _inat_build_superclass_map,
)
from tiny_imagenet_hierarchy import (
    build_superclass_map as _tiny_imagenet_build_superclass_map,
)


def _resolve_build_superclass_map(dataset_id: str):
    """Return the correct build_superclass_map for the given dataset."""
    if dataset_id == "inat_family_genus":
        return _inat_build_superclass_map
    if dataset_id == "tiny_imagenet":
        return _tiny_imagenet_build_superclass_map
    return _cifar100_build_superclass_map


# Default for backward compatibility
build_superclass_map = _cifar100_build_superclass_map
from sae import SparseAutoencoder, train_sae, create_sae_init


# ───────────────────── helpers ──────────────────────────────────────────────


_progress_file: Optional[Path] = None


def _progress(cb: Optional[Callable], pct: float, msg: str) -> None:
    print(f"[SAEANALYSIS {pct:5.1f}%] {msg}", flush=True)
    if cb is not None:
        cb(pct, msg)
    if _progress_file is not None:
        try:
            tmp = _progress_file.with_suffix(".tmp")
            tmp.write_text(json.dumps({"progress_pct": pct, "message": msg}))
            tmp.rename(_progress_file)
        except Exception:
            pass


def _to_list(obj: Any) -> Any:
    """Recursively convert numpy/torch objects to JSON-safe Python types."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _to_list(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_list(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        return obj
    return obj


def _clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _safe_layer_filename(layer_name: str) -> str:
    """Replace dots with underscores for filesystem-safe layer names."""
    return layer_name.replace(".", "_")


def _sort_checkpoint_labels(labels):
    """Sort checkpoint labels numerically."""
    def _key(k):
        try:
            return int(k)
        except (ValueError, TypeError):
            return 999999
    return sorted(labels, key=_key)


def _safe_float(x: Any) -> float:
    """Convert any numeric to a plain Python float, handling nan/inf."""
    v = float(x)
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v


# ─────────── data discovery ────────────────────────────────────────────────


def _discover_snapshots(lane_dir: Path) -> List[Tuple[int, Path]]:
    """Return sorted (milestone_idx, path) pairs."""
    snap_base = lane_dir / "dev_snapshots"
    if not snap_base.exists():
        return []
    snapshots = []
    for entry in sorted(snap_base.iterdir()):
        if entry.is_dir() and entry.name.startswith("milestone_"):
            try:
                idx = int(entry.name.split("_")[1])
                snapshots.append((idx, entry))
            except (ValueError, IndexError):
                continue
    snapshots.sort(key=lambda x: x[0])
    return snapshots


def _select_key_checkpoints(
    snapshots: List[Tuple[int, Path]],
) -> List[Tuple[str, Path]]:
    """Select ALL snapshots — train SAE at every captured milestone.

    The last DEVTRAIN milestone is the terminal checkpoint, so no separate
    terminal_analysis directory is needed.
    """
    selected: List[Tuple[str, Path]] = []

    for idx, path in snapshots:
        selected.append((str(idx), path))

    return selected


def _load_multilayer_activations(
    path: Path,
) -> Optional[Dict[str, Dict[int, torch.Tensor]]]:
    """Load individual_multilayer_activations.pt."""
    act_file = path / "individual_multilayer_activations.pt"
    if not act_file.exists():
        return None
    data = torch.load(act_file, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        return None
    return data


def _load_sample_predictions(
    path: Path,
) -> Optional[Dict[int, Dict[str, torch.Tensor]]]:
    """Load sample_predictions.pt from a snapshot dir."""
    pred_file = path / "sample_predictions.pt"
    if not pred_file.exists():
        return None
    data = torch.load(pred_file, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        return None
    return data


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Data Loading (0-10%)
# ═══════════════════════════════════════════════════════════════════════════


def _phase1_load(
    lane_dir: Path,
    cb: Optional[Callable],
) -> Tuple[
    Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    Dict[str, Dict[int, Dict[str, torch.Tensor]]],
    List[str],
    List[str],
]:
    """
    Load multilayer activations and sample predictions at key checkpoints.

    Returns:
        all_data:       {checkpoint_label: {layer_name: {class_idx: Tensor[N, d]}}}
        predictions:    {checkpoint_label: {class_idx: {predictions, correct, confidences}}}
        layers:         Sorted list of layer names
        labels:         Sorted checkpoint labels
    """
    _progress(cb, 0, "Discovering snapshots")
    snapshots = _discover_snapshots(lane_dir)
    checkpoints = _select_key_checkpoints(snapshots)

    if not checkpoints:
        _progress(cb, 10, "No checkpoints found")
        return {}, {}, [], []

    _progress(cb, 2, f"Loading data from {len(checkpoints)} checkpoints")

    all_data: Dict[str, Dict[str, Dict[int, torch.Tensor]]] = {}
    predictions: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}
    total = len(checkpoints)
    for step, (label, path) in enumerate(checkpoints):
        pct = 2 + 8 * step / max(total, 1)
        _progress(cb, pct, f"Loading checkpoint {label}")
        ml_data = _load_multilayer_activations(path)
        if ml_data is not None:
            all_data[label] = ml_data
        pred_data = _load_sample_predictions(path)
        if pred_data is not None:
            predictions[label] = pred_data

    layers: List[str] = []
    if all_data:
        # Use the last checkpoint (terminal) to discover layers
        last_key = _sort_checkpoint_labels(list(all_data.keys()))[-1]
        layers = sorted(all_data[last_key].keys())

    labels = _sort_checkpoint_labels(list(all_data.keys()))
    _progress(cb, 10, f"Loaded {len(all_data)} checkpoints, {len(layers)} layers")
    return all_data, predictions, layers, labels


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — Per-Checkpoint SAE Training (10-35%)
#
# One SAE per (checkpoint, layer) pair.  All SAEs for a given layer share
# initial weights (shared_init_seed) so decoder directions are comparable
# across checkpoints — motivated by Bau et al. 2025's finding that SAEs
# with different seeds learn only ~30% shared features.  Related to Bai et
# al. 2024 (SAE-Track) recurrent init, but we share *initial* weights
# rather than *trained* weights from the prior checkpoint.
# ═══════════════════════════════════════════════════════════════════════════


def _phase2_train_per_checkpoint_saes(
    all_data: Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    output_dir: Path,
    layers: List[str],
    labels: List[str],
    expansion_factor: int,
    k_sparse: int,
    n_steps: int,
    cb: Optional[Callable],
    shared_init_seed: Optional[int] = 42,
) -> Dict[str, Dict[str, SparseAutoencoder]]:
    """
    Train one SAE per layer per checkpoint.
    Uses shared init so decoder directions are comparable (secondary signal).

    Returns:
        {checkpoint_label: {layer_name: SparseAutoencoder}}
    """
    _progress(cb, 10, f"Training SAEs at {len(labels)} checkpoints × {len(layers)} layers")

    # Create shared init per layer
    layer_inits: Dict[str, Optional[dict]] = {}
    if shared_init_seed is not None:
        first_label = labels[0]
        first_data = all_data.get(first_label, {})
        for layer_name in layers:
            class_acts = first_data.get(layer_name, {})
            d_input = None
            for ci in sorted(class_acts.keys()):
                t = class_acts[ci]
                if t.numel() > 0:
                    d_input = t.shape[-1]
                    break
            if d_input is not None:
                use_topk = k_sparse is not None and k_sparse > 0
                layer_inits[layer_name] = create_sae_init(
                    d_input, expansion_factor,
                    k_sparse if use_topk else None,
                    seed=shared_init_seed + hash(layer_name) % 2**31,
                )

    all_saes: Dict[str, Dict[str, SparseAutoencoder]] = {}
    total_work = len(labels) * len(layers)
    work_done = 0

    for label in labels:
        all_saes[label] = {}
        layer_data = all_data.get(label, {})
        if not layer_data:
            continue

        # Save SAEs to disk
        sae_dir = output_dir / "saes" / label
        sae_dir.mkdir(parents=True, exist_ok=True)

        for layer_name in layers:
            pct = 10 + 25 * work_done / max(total_work, 1)
            _progress(cb, pct, f"Training SAE {label}/{layer_name}")

            class_acts = layer_data.get(layer_name, {})
            all_acts = []
            for ci in sorted(class_acts.keys()):
                t = class_acts[ci]
                if t.numel() > 0:
                    all_acts.append(t.float())
            if not all_acts:
                work_done += 1
                continue

            X = torch.cat(all_acts, dim=0)
            sae = train_sae(
                activations=X,
                expansion_factor=expansion_factor,
                k_sparse=k_sparse,
                n_steps=n_steps,
                batch_size=min(256, X.shape[0]),
                init_state=layer_inits.get(layer_name),
            )
            all_saes[label][layer_name] = sae

            safe_name = _safe_layer_filename(layer_name)
            torch.save(sae.state_dict(), sae_dir / f"{safe_name}.pt")

            del X, all_acts
            _clear_memory()
            work_done += 1

    _progress(cb, 35, f"Trained SAEs at {len(all_saes)} checkpoints")
    return all_saes


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — Per-Sample Encoding (35-50%)
# ═══════════════════════════════════════════════════════════════════════════


def _phase3_encode_per_checkpoint(
    all_saes: Dict[str, Dict[str, SparseAutoencoder]],
    all_data: Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    labels: List[str],
    layers: List[str],
    output_dir: Path,
    k_sparse: int,
    cb: Optional[Callable],
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, Dict[str, float]]],
]:
    """
    Encode every sample through each checkpoint's OWN SAE.

    For each (checkpoint, layer): produces an activation matrix A of shape
    (n_total_samples, d_hidden) where each row is one sample's sparse feature
    vector.  This is the atomic data unit.

    Returns:
        activation_matrices:   {checkpoint: {layer: ndarray(n_samples, d_hidden)}}
        reconstruction_quality: {checkpoint: {layer: {mse, cosine_sim}}}

    Also saves activation matrices to disk for later sample-level queries.
    """
    _progress(cb, 35, "Encoding all samples through per-checkpoint SAEs")

    act_matrices: Dict[str, Dict[str, np.ndarray]] = {}
    recon_quality: Dict[str, Dict[str, Dict[str, float]]] = {}

    total_work = len(labels) * len(layers)
    work_done = 0

    for label in labels:
        act_matrices[label] = {}
        recon_quality[label] = {}
        checkpoint_data = all_data.get(label, {})
        checkpoint_saes = all_saes.get(label, {})

        act_dir = output_dir / "activations" / label
        act_dir.mkdir(parents=True, exist_ok=True)

        for layer_name in layers:
            layer_data = checkpoint_data.get(layer_name, {})
            sae = checkpoint_saes.get(layer_name)
            if sae is None or not layer_data:
                work_done += 1
                continue

            # Collect all class activations in order (class_idx sorted)
            all_originals = []
            for ci in sorted(layer_data.keys()):
                t = layer_data[ci]
                if t.numel() > 0:
                    all_originals.append(t.float())

            if not all_originals:
                work_done += 1
                continue

            X = torch.cat(all_originals, dim=0)  # (N_total, d_input)

            with torch.no_grad():
                H = sae.encode(X)          # (N_total, d_hidden) — sparse
                X_hat = sae.decode(H)      # (N_total, d_input)

            # Store activation matrix as numpy
            H_np = H.cpu().numpy()
            act_matrices[label][layer_name] = H_np

            # Save to disk
            safe_layer = _safe_layer_filename(layer_name)
            np.save(act_dir / f"{safe_layer}.npy", H_np)

            # Reconstruction quality
            mse = float(F.mse_loss(X_hat, X).item())
            cos_sim = float(F.cosine_similarity(X, X_hat, dim=1).mean().item())
            recon_quality[label][layer_name] = {
                "mse": _safe_float(mse),
                "cosine_sim": _safe_float(cos_sim),
            }

            del X, H, X_hat
            _clear_memory()
            work_done += 1
            pct = 35 + 15 * work_done / max(total_work, 1)
            if work_done % max(1, total_work // 5) == 0:
                _progress(cb, pct, f"Encoded {label}/{layer_name}")

    _progress(cb, 50, "Per-sample encoding complete")
    return act_matrices, recon_quality


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4 — SSI / CSI Computation (50-58%)
# ═══════════════════════════════════════════════════════════════════════════


def _phase4_ssi_csi(
    act_matrices: Dict[str, Dict[str, np.ndarray]],
    all_data: Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    superclass_map: Dict[int, str],
    superclass_groups: Dict[str, List[int]],
    labels: List[str],
    layers: List[str],
    cb: Optional[Callable],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Compute SSI, CSI, SAI, and entropy per (checkpoint, layer).

    Uses the activation matrices from phase 3 to compute feature × class
    mean activation magnitudes, then derives selectivity indices.

    CSI is computed as within-superclass class selectivity (CSI_local) for
    hierarchical datasets, or normalized max-class fraction for flat datasets.
    SAI is normalized entropy: H(p_f) / log(N_classes).

    Returns:
        Dict keyed by (checkpoint, layer) with:
            feature_ssi, feature_csi, feature_sai, feature_entropy,
            n_high_ssi, n_high_sai, mean_ssi, mean_csi, mean_sai, mean_entropy,
            n_active_features, feature_magnitudes (in-memory ndarray),
            best_superclass, best_class
    """
    _progress(cb, 50, "Computing SSI & CSI per layer")
    results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    total_work = len(labels) * len(layers)
    work_done = 0

    has_hierarchy = has_hierarchical_structure(superclass_map)

    for label in labels:
        for layer_name in layers:
            H = act_matrices.get(label, {}).get(layer_name)
            if H is None:
                work_done += 1
                continue

            # Reconstruct class boundaries from all_data
            layer_data = all_data.get(label, {}).get(layer_name, {})
            class_indices = sorted(layer_data.keys())
            n_classes = len(class_indices)
            if n_classes == 0:
                work_done += 1
                continue

            n_features = H.shape[1]

            # Build feature × class mean magnitude matrix + activation mass
            # H rows are ordered by class (sorted class_idx), each class has N_samples rows
            M = np.zeros((n_features, n_classes), dtype=np.float32)
            activation_mass: Dict[int, List[int]] = {}  # class_idx → per-feature active sample counts
            offset = 0
            for col_idx, ci in enumerate(class_indices):
                t = layer_data[ci]
                n_samples_ci = t.shape[0] if t.numel() > 0 else 0
                if n_samples_ci > 0:
                    class_activations = H[offset:offset + n_samples_ci, :]  # (N, d_hidden)
                    M[:, col_idx] = np.abs(class_activations).mean(axis=0)
                    activation_mass[int(ci)] = (np.abs(class_activations) > 1e-6).sum(axis=0).tolist()
                    offset += n_samples_ci

            eps = 1e-10
            feat_total = M.sum(axis=1)
            active_mask = feat_total > eps
            n_active = int(active_mask.sum())

            # Best class per feature (for reporting)
            best_class_col = M.argmax(axis=1)
            best_class_indices = [class_indices[c] for c in best_class_col]

            # SSI: superclass selectivity — normalized max-superclass fraction.
            # SSI = (best_sc_frac - 1/N_sc) / (1 - 1/N_sc)
            # Maps uniform across superclasses → 0, all activation in one SC → 1.
            # Within-SC distribution is CSI's job, not SSI's.
            feature_ssi = np.zeros(n_features, dtype=np.float32)
            best_superclass = [""] * n_features

            if has_hierarchy:
                sc_col_masks: Dict[str, np.ndarray] = {}
                for sc_name, sc_class_list in superclass_groups.items():
                    mask = np.zeros(n_classes, dtype=bool)
                    for col_idx, ci in enumerate(class_indices):
                        if ci in sc_class_list:
                            mask[col_idx] = True
                    if mask.any():
                        sc_col_masks[sc_name] = mask

                n_sc = len(sc_col_masks)

                for fi in range(n_features):
                    if feat_total[fi] < eps:
                        continue
                    row = M[fi, :]
                    row_sum = feat_total[fi]
                    best_sc_frac = 0.0
                    best_sc = ""
                    for sc_name, sc_mask in sc_col_masks.items():
                        sc_frac = row[sc_mask].sum() / (row_sum + eps)
                        if sc_frac > best_sc_frac:
                            best_sc_frac = sc_frac
                            best_sc = sc_name

                    # Normalize: uniform → 0, single-superclass → 1
                    feature_ssi[fi] = max(
                        (best_sc_frac - 1.0 / n_sc) / (1.0 - 1.0 / n_sc + eps),
                        0.0,
                    ) if n_sc > 1 else 0.0
                    best_superclass[fi] = best_sc

            # Feature entropy
            feature_entropy = np.zeros(n_features, dtype=np.float32)
            for fi in range(n_features):
                if feat_total[fi] < eps:
                    continue
                probs = M[fi, :] / (feat_total[fi] + eps)
                probs_nz = probs[probs > 0]
                if len(probs_nz) > 0:
                    feature_entropy[fi] = float(-np.sum(probs_nz * np.log(probs_nz + eps)))

            # SAI: normalized entropy of class activation distribution.
            # SAI = H(p_f) / log(N) where p_f[c] = M[f,c] / sum_c(M[f,c]).
            # Bounded [0,1], continuous, handles zeros gracefully.
            log_n_classes = math.log(n_classes) if n_classes > 1 else 1.0
            feature_sai = np.where(active_mask, feature_entropy / log_n_classes, 0.0).astype(np.float32)

            # CSI: within-superclass class selectivity (CSI_local).
            # Measures class selectivity WITHIN the best superclass, not globally.
            # This decouples Di-H from Ab-H: SSI measures "which superclass?"
            # while CSI measures "which class within that superclass?"
            # For non-hierarchical datasets, falls back to normalized max-class fraction.
            feat_max = M.max(axis=1)
            raw_global_csi = np.where(feat_total > eps, feat_max / (feat_total + eps), 0.0)
            feature_csi = np.where(
                active_mask,
                np.clip((raw_global_csi - 1.0 / n_classes) / (1.0 - 1.0 / n_classes + eps), 0.0, 1.0),
                0.0,
            ).astype(np.float32)
            if has_hierarchy:
                for fi in range(n_features):
                    if feat_total[fi] < eps or not best_superclass[fi]:
                        continue
                    sc_mask = sc_col_masks.get(best_superclass[fi])
                    if sc_mask is None or int(sc_mask.sum()) < 2:
                        continue
                    within_row = M[fi, sc_mask]
                    within_total = within_row.sum()
                    if within_total > eps:
                        n_in_sc = int(sc_mask.sum())
                        raw_local = within_row.max() / (within_total + eps)
                        feature_csi[fi] = max(
                            (raw_local - 1.0 / n_in_sc) / (1.0 - 1.0 / n_in_sc + eps),
                            0.0,
                        )

            n_high_ssi = int(np.sum(feature_ssi > 0.3))
            n_high_sai = int(np.sum(feature_sai > 0.5))
            mean_ssi = float(np.mean(feature_ssi[active_mask])) if n_active > 0 else 0.0
            mean_csi = float(np.mean(feature_csi[active_mask])) if n_active > 0 else 0.0
            mean_entropy = float(np.mean(feature_entropy[active_mask])) if n_active > 0 else 0.0
            mean_sai = float(np.mean(feature_sai[active_mask])) if n_active > 0 else 0.0

            results[(label, layer_name)] = {
                "feature_ssi": feature_ssi.tolist(),
                "feature_csi": feature_csi.tolist(),
                "feature_sai": feature_sai.tolist(),
                "feature_entropy": feature_entropy.tolist(),
                "feature_magnitudes": M,
                "activation_mass": activation_mass,
                "class_indices": class_indices,
                "best_superclass": best_superclass,
                "best_class": best_class_indices,
                "n_high_ssi": n_high_ssi,
                "n_high_sai": n_high_sai,
                "mean_ssi": _safe_float(mean_ssi),
                "mean_csi": _safe_float(mean_csi),
                "mean_sai": _safe_float(mean_sai),
                "mean_entropy": _safe_float(mean_entropy),
                "n_active_features": n_active,
            }

            work_done += 1
            pct = 50 + 8 * work_done / max(total_work, 1)
            if work_done % max(1, total_work // 5) == 0:
                _progress(cb, pct, f"SSI/CSI {label}/{layer_name}")

    _progress(cb, 58, "SSI, CSI & SAI complete")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4b — Adaptive SSI/CSI Thresholds from Permutation Null Distribution
# ═══════════════════════════════════════════════════════════════════════════


def _phase4b_adaptive_thresholds(
    ssi_csi_data: Dict[Tuple[str, str], Dict[str, Any]],
    act_matrices: Dict[str, Dict[str, np.ndarray]],
    all_data: Dict[str, Dict[str, Dict[int, Any]]],
    superclass_map: Dict[int, str],
    superclass_groups: Dict[str, List[int]],
    labels: List[str],
    layers: List[str],
    n_permutations: int = 100,
    percentile: float = 95.0,
    ssi_floor: float = 0.1,
    csi_floor: float = 0.15,
    sai_floor: float = 0.5,
    cb: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Compute adaptive SSI, CSI, and SAI thresholds from permutation null distributions.

    Instead of fixed thresholds (SSI > 0.3, CSI > 0.4, SAI > 0.9), we compute
    the 95th percentile of each metric under random label assignments. This makes
    thresholds adaptive to the number of classes, superclass structure, and
    dataset size.

    SSI null: permute class→superclass assignments (same as _compute_null_baseline
    but collecting per-feature values instead of means).

    SAI null: same permutation as SSI — SAI measures uniformity across superclasses,
    so permuting class→superclass assignments tests whether apparent uniformity
    exceeds what random groupings would produce.

    CSI null: permute sample→class assignments by shuffling rows of the activation
    matrix, then recomputing per-class mean magnitudes and CSI.

    Returns dict with adaptive thresholds and diagnostics.
    """
    _progress(cb, 58, "Computing adaptive SSI/CSI/SAI thresholds")

    all_class_indices = sorted(superclass_map.keys())
    sc_names = sorted(superclass_groups.keys())
    sc_sizes = [len(superclass_groups[sc]) for sc in sc_names]
    rng = np.random.RandomState(42)

    # ── SSI + SAI null distribution (same permutation: class→superclass) ──
    # Vectorized: process all features at once per (permutation, checkpoint, layer)
    null_ssi_all: List[float] = []
    null_sai_all: List[float] = []

    for perm_i in range(n_permutations):
        # Randomly assign classes to superclass groups (preserving group sizes)
        perm = rng.permutation(all_class_indices)
        perm_groups: Dict[str, List[int]] = {}
        offset = 0
        for sc, size in zip(sc_names, sc_sizes):
            perm_groups[sc] = list(perm[offset:offset + size])
            offset += size

        for label in labels:
            for layer_name in layers:
                entry = ssi_csi_data.get((label, layer_name))
                if not entry or "feature_magnitudes" not in entry:
                    continue
                M = entry["feature_magnitudes"]  # (n_features, n_classes)
                layer_class_indices = entry.get("class_indices", list(range(M.shape[1])))
                n_features, n_cols = M.shape
                eps = 1e-10
                feat_total = M.sum(axis=1)  # (n_features,)
                active_mask = feat_total > eps

                # Map class_idx → col_idx
                class_to_col = {ci: col for col, ci in enumerate(layer_class_indices)}

                # Build column masks as a (n_superclasses, n_cols) bool array
                sc_mask_list = []
                for sc_name, sc_class_list in perm_groups.items():
                    mask = np.zeros(n_cols, dtype=bool)
                    for ci in sc_class_list:
                        col = class_to_col.get(ci)
                        if col is not None:
                            mask[col] = True
                    if mask.any():
                        sc_mask_list.append(mask)

                if not sc_mask_list:
                    continue

                # Stack masks: (n_sc, n_cols)
                sc_masks = np.stack(sc_mask_list)  # (n_sc, n_cols)
                n_sc = len(sc_mask_list)

                # Vectorized SSI: normalized max-superclass fraction
                # sc_fracs[fi, sc] = sum of activation in sc / total activation
                sc_fracs = (M @ sc_masks.T) / (feat_total[:, None] + eps)  # (n_features, n_sc)
                best_sc_frac = sc_fracs.max(axis=1)  # (n_features,)

                # Normalize: uniform → 0, single-superclass → 1
                best_ssi = np.clip(
                    (best_sc_frac - 1.0 / n_sc) / (1.0 - 1.0 / n_sc + eps),
                    0.0, 1.0,
                )
                best_ssi = np.where(active_mask, best_ssi, 0.0)

                # Collect positive SSI values
                pos_mask = best_ssi > 0
                if pos_mask.any():
                    null_ssi_all.extend(best_ssi[pos_mask].tolist())

                # SAI null: normalized entropy under permuted superclass assignments
                # SAI = H(p_f) / log(n_cols) — same formula as Phase 4
                if n_cols > 1:
                    probs = M / (feat_total[:, None] + eps)  # (n_features, n_cols)
                    log_p = np.where(probs > 0, np.log(probs + eps), 0.0)
                    entropy = -np.sum(probs * log_p, axis=1)  # (n_features,)
                    log_n = math.log(n_cols)
                    fbi_vals = np.where(active_mask, entropy / log_n, 0.0)
                    fbi_pos = fbi_vals > 0
                    if fbi_pos.any():
                        null_sai_all.extend(fbi_vals[fbi_pos].tolist())

    ssi_p95 = float(np.percentile(null_ssi_all, percentile)) if null_ssi_all else 0.0
    ssi_adaptive = max(ssi_p95, ssi_floor)
    sai_p95 = float(np.percentile(null_sai_all, percentile)) if null_sai_all else 0.0
    sai_adaptive = max(sai_p95, sai_floor)

    _progress(cb, 59, f"SSI null p{percentile:.0f}={ssi_p95:.4f} → {ssi_adaptive:.4f}, "
              f"SAI null p{percentile:.0f}={sai_p95:.4f} → {sai_adaptive:.4f}")

    # ── CSI null distribution (local, within-superclass) ──
    # Permute sample→class assignments, then compute CSI_local: for each
    # feature, find its best superclass and compute max/total within that
    # superclass, normalized for within-SC class count. This matches the
    # actual classification metric (CSI_local) rather than global CSI.
    null_csi_all: List[float] = []
    eps = 1e-10

    # Build real superclass column masks (same structure as Phase 4)
    has_hierarchy = has_hierarchical_structure(superclass_map)

    for label in labels:
        for layer_name in layers:
            H = act_matrices.get(label, {}).get(layer_name)
            if H is None or H.shape[0] == 0:
                continue

            entry = ssi_csi_data.get((label, layer_name))
            if not entry or "feature_magnitudes" not in entry:
                continue

            layer_class_indices = entry.get("class_indices", list(range(entry["feature_magnitudes"].shape[1])))
            n_samples_total = H.shape[0]
            n_features_h = H.shape[1]
            n_classes = len(layer_class_indices)

            # Determine original class sizes from all_data
            layer_data = all_data.get(label, {}).get(layer_name, {})
            class_sizes = []
            for ci in layer_class_indices:
                t = layer_data.get(ci)
                if t is not None and hasattr(t, 'shape'):
                    class_sizes.append(t.shape[0] if t.numel() > 0 else 0)
                else:
                    class_sizes.append(0)

            if sum(class_sizes) != n_samples_total or n_classes < 2:
                continue

            # Build superclass column masks for local CSI computation
            sc_col_masks_real: Dict[str, np.ndarray] = {}
            if has_hierarchy:
                for sc_name, sc_class_list in superclass_groups.items():
                    mask = np.zeros(n_classes, dtype=bool)
                    for col_idx, ci in enumerate(layer_class_indices):
                        if ci in sc_class_list:
                            mask[col_idx] = True
                    if mask.any() and mask.sum() >= 2:
                        sc_col_masks_real[sc_name] = mask

            # Precompute abs(H) once
            H_abs = np.abs(H)  # (n_samples, n_features)

            # Build cumulative class boundaries
            boundaries = np.cumsum([0] + class_sizes)

            for perm_i in range(n_permutations):
                perm_indices = rng.permutation(n_samples_total)

                # Recompute M with permuted sample→class assignments
                M_perm = np.zeros((n_features_h, n_classes), dtype=np.float32)
                for col_idx, size in enumerate(class_sizes):
                    if size > 0:
                        start = boundaries[col_idx]
                        end = boundaries[col_idx + 1]
                        M_perm[:, col_idx] = H_abs[perm_indices[start:end], :].mean(axis=0)

                feat_total_perm = M_perm.sum(axis=1)
                active_perm = feat_total_perm > eps

                if has_hierarchy and sc_col_masks_real:
                    # Compute CSI_local: for each feature, find best superclass
                    # by SSI, then compute normalized max/total within that SC
                    sc_masks_arr = np.stack(list(sc_col_masks_real.values()))  # (n_sc, n_cols)
                    sc_sizes_arr = sc_masks_arr.sum(axis=1)  # (n_sc,)

                    # Best SC per feature via SSI-like criterion
                    within_fracs = (M_perm @ sc_masks_arr.T) / (feat_total_perm[:, None] + eps)  # (n_feat, n_sc)
                    best_sc_idx = within_fracs.argmax(axis=1)  # (n_feat,)
                    best_sc_masks = sc_masks_arr[best_sc_idx]  # (n_feat, n_cols)
                    best_sc_sizes = sc_sizes_arr[best_sc_idx]  # (n_feat,)

                    # Within-SC max and total
                    M_within = np.where(best_sc_masks, M_perm, 0.0)
                    within_max = M_within.max(axis=1)
                    within_total = M_within.sum(axis=1)
                    raw_local = np.where(within_total > eps, within_max / (within_total + eps), 0.0)

                    # Normalize: (raw - 1/n_in_sc) / (1 - 1/n_in_sc)
                    baseline = 1.0 / best_sc_sizes
                    csi_local_norm = np.where(
                        active_perm & (best_sc_sizes >= 2),
                        np.maximum((raw_local - baseline) / (1.0 - baseline + eps), 0.0),
                        0.0,
                    )
                    # Only include null CSI from features in superclasses with
                    # enough classes (≥5) to prevent small groups from inflating
                    # the null distribution. With 3 classes, random concentration
                    # trivially produces CSI~1.0, making the threshold unusable.
                    min_sc_for_null = 5
                    size_ok = best_sc_sizes >= min_sc_for_null
                    csi_pos = (csi_local_norm > eps) & size_ok
                    if csi_pos.any():
                        null_csi_all.extend(csi_local_norm[csi_pos].tolist())
                else:
                    # No hierarchy: fall back to global CSI normalized
                    feat_max = M_perm.max(axis=1)
                    feature_csi = np.where(feat_total_perm > eps, feat_max / (feat_total_perm + eps), 0.0)
                    baseline_global = 1.0 / n_classes
                    csi_norm = np.where(
                        active_perm,
                        np.clip((feature_csi - baseline_global) / (1.0 - baseline_global + eps), 0.0, 1.0),
                        0.0,
                    )
                    csi_pos = csi_norm > eps
                    if csi_pos.any():
                        null_csi_all.extend(csi_norm[csi_pos].tolist())

    csi_p95 = float(np.percentile(null_csi_all, percentile)) if null_csi_all else 0.0
    # Cap CSI threshold: values above 0.8 indicate the null is dominated by
    # small superclass groups where random concentration saturates the metric.
    # In such cases, use a conservative cap that still exceeds the fixed default.
    csi_cap = 0.7
    csi_adaptive = max(min(csi_p95, csi_cap), csi_floor)

    _progress(cb, 60, f"CSI null p{percentile:.0f}={csi_p95:.4f} → {csi_adaptive:.4f}")

    return {
        "ssi_adaptive_thresh": ssi_adaptive,
        "csi_adaptive_thresh": csi_adaptive,
        "sai_adaptive_thresh": sai_adaptive,
        "ssi_null_percentile": ssi_p95,
        "csi_null_percentile": csi_p95,
        "sai_null_percentile": sai_p95,
        # Null distribution mean/std for z-score computation
        "ssi_null_mean": float(np.mean(null_ssi_all)) if null_ssi_all else 0.0,
        "ssi_null_std": float(np.std(null_ssi_all)) if null_ssi_all else 1.0,
        "csi_null_mean": float(np.mean(null_csi_all)) if null_csi_all else 0.0,
        "csi_null_std": float(np.std(null_csi_all)) if null_csi_all else 1.0,
        "sai_null_mean": float(np.mean(null_sai_all)) if null_sai_all else 0.0,
        "sai_null_std": float(np.std(null_sai_all)) if null_sai_all else 1.0,
        "ssi_fixed_default": 0.3,
        "csi_fixed_default": 0.4,
        "sai_fixed_default": 0.9,
        "n_permutations": n_permutations,
        "percentile": percentile,
        "ssi_floor": ssi_floor,
        "csi_floor": csi_floor,
        "sai_floor": sai_floor,
        "n_null_ssi_samples": len(null_ssi_all),
        "n_null_csi_samples": len(null_csi_all),
        "n_null_sai_samples": len(null_sai_all),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Analysis Thresholds — Configurable via analyze_sae_features()
#
# Literature context for each threshold:
#
#   Feature matching uses two complementary signals:
#     (a) Activation-column Pearson correlation — "functional equivalence":
#         do the same samples activate this feature?  Comparable performance
#         to cosine similarity for matching (feature-flow literature).
#     (b) Decoder-weight cosine similarity — "geometric equivalence":
#         do the decoder directions point the same way?
#         Hungarian algorithm for optimal 1-to-1 matching: directly supported
#         by Bau et al. 2025 ("SAEs on Same Data Learn Different Features").
#
#   weight_stable_thresh=0.7 matches Bau et al. 2025 exactly (their cutoff
#   for "shared" features across seeds).
#
#   match_stable_thresh=0.5 (Pearson r) and match_death_thresh=0.2 are
#   custom thresholds with no direct SAE literature precedent.  They follow
#   standard statistical conventions: r≥0.5 = moderate-to-strong correlation,
#   r<0.2 = negligible correlation.
#
#   weight_death_thresh=0.3 is consistent with the bimodal cosine-similarity
#   distributions observed by Bau et al. 2025 (high peak ≈ shared, low peak
#   ≈ orphan), though they did not specify a numeric cutoff.
#
#   alive_sample_frac=0.01 (≥1% of samples) is more conservative than the
#   literature — Gao et al. 2024 define dead features as "not activated in
#   10M tokens" (effectively a zero threshold).  Our higher bar is adapted
#   for small vision datasets, where a feature firing on only a handful of
#   images is likely noise.
#
#   SSI = normalized max-superclass fraction:
#     SSI = (best_sc_frac - 1/N_sc) / (1 - 1/N_sc)
#   Maps uniform → 0, all activation in one superclass → 1.
#   CSI_local measures within-SC class selectivity (orthogonal to SSI).
#   SAI = normalized entropy across all classes (task-generality).
#   Thresholds (SSI>0.3, CSI>0.4) are justified empirically via the
#   permutation null baseline rather than by literature convention.
#
# Key references:
#   - Bau et al. 2025, "SAEs Trained on Same Data Learn Different Features"
#     — Hungarian matching, cos≥0.7 for shared features, seed dependence
#   - Morcos et al. 2018, "On the Importance of Single Directions for
#     Generalization" — class selectivity index formula
#   - Gao et al. 2024, "Scaling and Evaluating Sparse Autoencoders"
#     — dead feature definition, L1 ranges, top-k SAEs
#   - Bai et al. 2024, "SAE-Track: Tracking Feature Dynamics in LLM
#     Training" — feature tracking across checkpoints, recurrent init
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_THRESHOLDS: Dict[str, float] = {
    # Feature matching — activation-column correlation (Pearson r)
    # No direct SAE precedent; follows standard statistical conventions.
    "match_stable_thresh": 0.5,       # r ≥ 0.5 = stable match (moderate-to-strong)
    "match_death_thresh": 0.2,        # r < 0.2 = genuinely died (negligible)
    # Feature matching — decoder weight cosine similarity
    # weight_stable_thresh matches Bau et al. 2025 exactly.
    "weight_stable_thresh": 0.7,      # cos ≥ 0.7 = stable (Bau et al. 2025)
    "weight_death_thresh": 0.3,       # cos < 0.3 = died (consistent w/ Bau bimodal dist.)
    # Feature lifecycle
    # More conservative than Gao et al. 2024 (near-zero); adapted for small datasets.
    "alive_sample_frac": 0.01,        # feature must fire on ≥1% of samples to be "alive"
    "alive_activation_eps": 1e-6,     # |activation| > this to count as firing
    # Process classification — selectivity thresholds
    # Formula: Morcos et al. 2018.  Thresholds: empirical (justified by null baseline).
    "ssi_abstraction_thresh": 0.3,    # SSI above this → Ab-H (superclass abstraction)
    "sai_task_general_thresh": 0.9,   # SAI above this → Tg-H (task-general abstraction)
    "csi_differentiation_thresh": 0.4,  # CSI above this → Di-H (differentiation)
    # Hypothesis testing
    "baseline_exceedance_ratio": 2.0,     # observed rate must exceed false rate by this factor
    "death_fraction_thresh": 0.15,        # De-H: deaths must be >15% of total events
    "assembly_fraction_thresh": 0.15,     # As-H: births must be >15% of total events
    "ssi_increase_factor": 1.1,           # LEGACY: kept for backward compat; z-score delta (0.5 SD) used instead
    "sai_increase_factor": 1.1,           # LEGACY: kept for backward compat; z-score delta (0.5 SD) used instead
    "csi_increase_factor": 1.1,           # LEGACY: kept for backward compat; z-score delta (0.5 SD) used instead
    # Adaptive thresholds — override fixed SSI/CSI thresholds using permutation null
    "adaptive_thresholds_enabled": 1.0,   # 1.0 = enabled, 0.0 = disabled (use fixed thresholds)
    "adaptive_threshold_percentile": 95.0, # percentile of null distribution to use as threshold
    "ssi_adaptive_floor": 0.1,            # minimum SSI threshold even if null is lower
    "csi_adaptive_floor": 0.15,           # minimum CSI threshold even if null is lower
    "sai_adaptive_floor": 0.5,            # minimum SAI threshold even if null is lower
}


def _apply_thresholds(overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Merge user overrides into defaults and set module-level variables."""
    global MATCH_STABLE_THRESH, MATCH_DEATH_THRESH
    global WEIGHT_STABLE_THRESH, WEIGHT_DEATH_THRESH

    active = {**DEFAULT_THRESHOLDS}
    if overrides:
        for k, v in overrides.items():
            if k in active:
                active[k] = v

    # Set module-level constants used by phase functions
    MATCH_STABLE_THRESH = active["match_stable_thresh"]
    MATCH_DEATH_THRESH = active["match_death_thresh"]
    WEIGHT_STABLE_THRESH = active["weight_stable_thresh"]
    WEIGHT_DEATH_THRESH = active["weight_death_thresh"]

    return active


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5 — Feature Matching by Activation-Column Correlation (58-70%)
#
# Matches features across consecutive checkpoints via Hungarian algorithm
# (Bau et al. 2025) on Pearson r of activation columns.  Activation-column
# correlation captures functional equivalence (same samples → same feature),
# which is arguably more meaningful for developmental tracking than the
# decoder cosine similarity used as primary signal by Bau et al.
# ═══════════════════════════════════════════════════════════════════════════

# Module-level defaults (overwritten by _apply_thresholds at analysis start)
# See DEFAULT_THRESHOLDS block above for literature context.
MATCH_STABLE_THRESH = 0.5      # Pearson r above this = stable match
MATCH_DEATH_THRESH = 0.2       # Pearson r below this = genuinely died


def _phase5_feature_matching(
    act_matrices: Dict[str, Dict[str, np.ndarray]],
    all_data: Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    ssi_csi_data: Dict[Tuple[str, str], Dict[str, Any]],
    labels: List[str],
    layers: List[str],
    cb: Optional[Callable],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Match features across consecutive checkpoints by correlating their
    activation columns (which samples activate this feature?).

    For checkpoints k and k+1:
      A_k: (n_samples, d_hidden_k)  — activation matrix at checkpoint k
      A_{k+1}: (n_samples, d_hidden_{k+1}) — activation matrix at k+1
      Correlate columns of A_k with columns of A_{k+1}.
      Same samples → same feature (functional equivalence).

    Returns:
        {f"{label_a}->{label_b}": {layer: {
            stable: [(idx_a, idx_b, corr)],
            born: [idx_b, ...],
            died: [idx_a, ...],
            n_stable, n_born, n_died,
            correlation_matrix: ndarray  (kept in memory for process classification)
        }}}
    """
    _progress(cb, 58, "Feature matching by activation-column correlation")

    pairs = list(zip(labels[:-1], labels[1:]))
    if not pairs:
        return {}

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    total_work = sum(len(layers) for _ in pairs)
    work_done = 0

    for label_a, label_b in pairs:
        pair_key = f"{label_a}->{label_b}"
        results[pair_key] = {}

        for layer_name in layers:
            H_a = act_matrices.get(label_a, {}).get(layer_name)
            H_b = act_matrices.get(label_b, {}).get(layer_name)

            if H_a is None or H_b is None:
                work_done += 1
                continue

            n_samples = H_a.shape[0]
            n_feat_a = H_a.shape[1]
            n_feat_b = H_b.shape[1]

            # Determine which features are "alive" (fire on >1% of samples)
            alive_thresh = max(1, n_samples * 0.01)
            alive_a = (np.abs(H_a) > 1e-6).sum(axis=0) >= alive_thresh  # (n_feat_a,)
            alive_b = (np.abs(H_b) > 1e-6).sum(axis=0) >= alive_thresh  # (n_feat_b,)

            # Compute correlation matrix between alive columns
            # C[i, j] = Pearson correlation between column i of A_k and column j of A_{k+1}
            C = np.zeros((n_feat_a, n_feat_b), dtype=np.float32)

            # Vectorised: normalise columns, compute dot product
            eps = 1e-10
            A_a = H_a.copy()
            A_b = H_b.copy()
            # Center columns
            A_a -= A_a.mean(axis=0, keepdims=True)
            A_b -= A_b.mean(axis=0, keepdims=True)
            # Normalise
            norm_a = np.sqrt((A_a ** 2).sum(axis=0, keepdims=True) + eps)
            norm_b = np.sqrt((A_b ** 2).sum(axis=0, keepdims=True) + eps)
            A_a /= norm_a
            A_b /= norm_b
            # Correlation matrix
            C = (A_a.T @ A_b)  # (n_feat_a, n_feat_b)

            # Optimal one-to-one matching via Hungarian algorithm.
            # Only match alive features; use negative correlation as cost.
            alive_idx_a = np.where(alive_a)[0]
            alive_idx_b = np.where(alive_b)[0]

            stable = []
            died = []
            transformed = []
            matched_b = set()       # B features claimed by stable matches
            transformed_b = set()   # B features claimed by transformed matches

            if len(alive_idx_a) > 0 and len(alive_idx_b) > 0:
                # Sub-matrix of alive×alive correlations
                C_sub = C[np.ix_(alive_idx_a, alive_idx_b)]  # (n_alive_a, n_alive_b)
                # Hungarian minimises cost → negate correlation to maximise
                cost = -C_sub
                row_ind, col_ind = linear_sum_assignment(cost)

                matched_a_set = set()
                for r, c in zip(row_ind, col_ind):
                    i = int(alive_idx_a[r])
                    j = int(alive_idx_b[c])
                    corr = float(C_sub[r, c])
                    matched_a_set.add(i)
                    if corr >= MATCH_STABLE_THRESH:
                        stable.append((i, j, round(corr, 4)))
                        matched_b.add(j)
                    elif corr >= MATCH_DEATH_THRESH:
                        transformed.append((i, j, round(corr, 4)))
                        transformed_b.add(j)
                    else:
                        died.append(i)

                # Alive A features not assigned by Hungarian (when n_alive_a > n_alive_b)
                for i in alive_idx_a:
                    if int(i) not in matched_a_set:
                        died.append(int(i))
            else:
                # Edge case: all features dead on one side
                for i in alive_idx_a:
                    died.append(int(i))

            # Born: alive features in B not claimed by any stable or transformed A
            claimed_b = matched_b | transformed_b
            born = [int(j) for j in range(n_feat_b) if alive_b[j] and j not in claimed_b]

            results[pair_key][layer_name] = {
                "stable": stable,
                "born": born,
                "died": died,
                "transformed": transformed,
                "n_stable": len(stable),
                "n_born": len(born),
                "n_died": len(died),
                "n_transformed": len(transformed),
                "correlation_matrix": C,
                "alive_a": alive_a,
                "alive_b": alive_b,
            }

            del A_a, A_b
            work_done += 1
            pct = 58 + 12 * work_done / max(total_work, 1)
            if work_done % max(1, total_work // 5) == 0:
                _progress(cb, pct, f"Matching {pair_key}/{layer_name}")

    _progress(cb, 70, f"Feature matching complete: {len(results)} pairs")
    return results


# ── Weight-space matching thresholds ──
# WEIGHT_STABLE_THRESH=0.7 matches Bau et al. 2025 exactly (their cutoff for
# "shared" features across different random seeds).
# WEIGHT_DEATH_THRESH=0.3 is consistent with the bimodal cosine-similarity
# distribution they observed (shared peak vs. orphan peak).
WEIGHT_STABLE_THRESH = 0.7     # cosine ≥ 0.7 = stable (Bau et al. 2025)
WEIGHT_DEATH_THRESH = 0.3      # cosine < 0.3 = died
# Features with cosine 0.3–0.7 are "transformed" — excluded from event counts


def _phase5b_weight_matching(
    all_saes: Dict[str, Dict[str, "SparseAutoencoder"]],
    act_matrices: Dict[str, Dict[str, np.ndarray]],
    labels: List[str],
    layers: List[str],
    cb: Optional[Callable],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Match features across consecutive checkpoints by decoder weight cosine
    similarity.  Each SAE feature j has a decoder column W_dec[:, j] that
    represents its direction in representation space.

    For checkpoints k and k+1:
      S[i, j] = cos_sim(W_dec_k[:, i], W_dec_{k+1}[:, j])

    Same thresholds logic as activation matching but on cosine similarity.
    """
    _progress(cb, 58, "Weight-space feature matching (decoder cosine similarity)")

    pairs = list(zip(labels[:-1], labels[1:]))
    if not pairs:
        return {}

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    total_work = sum(len(layers) for _ in pairs)
    work_done = 0

    for label_a, label_b in pairs:
        pair_key = f"{label_a}->{label_b}"
        results[pair_key] = {}

        for layer_name in layers:
            sae_a = all_saes.get(label_a, {}).get(layer_name)
            sae_b = all_saes.get(label_b, {}).get(layer_name)
            H_a = act_matrices.get(label_a, {}).get(layer_name)
            H_b = act_matrices.get(label_b, {}).get(layer_name)

            if sae_a is None or sae_b is None or H_a is None or H_b is None:
                work_done += 1
                continue

            n_samples = H_a.shape[0]
            n_feat_a = H_a.shape[1]
            n_feat_b = H_b.shape[1]

            # Determine alive features (same criterion as activation matching)
            alive_thresh = max(1, n_samples * 0.01)
            alive_a = (np.abs(H_a) > 1e-6).sum(axis=0) >= alive_thresh
            alive_b = (np.abs(H_b) > 1e-6).sum(axis=0) >= alive_thresh

            # Extract decoder weights: decoder.weight has shape (d_input, d_hidden)
            # Column j = decoder direction for feature j
            with torch.no_grad():
                W_a = sae_a.decoder.weight.detach().float()  # (d_input, d_hidden_a)
                W_b = sae_b.decoder.weight.detach().float()  # (d_input, d_hidden_b)

            # L2-normalise columns
            W_a = F.normalize(W_a, dim=0)  # (d_input, n_feat_a)
            W_b = F.normalize(W_b, dim=0)  # (d_input, n_feat_b)

            # Cosine similarity matrix: S[i, j] = cos_sim(W_a[:, i], W_b[:, j])
            S = (W_a.T @ W_b).cpu().numpy()  # (n_feat_a, n_feat_b)

            # Optimal one-to-one matching via Hungarian algorithm
            alive_idx_a = np.where(alive_a)[0]
            alive_idx_b = np.where(alive_b)[0]

            stable = []
            died = []
            transformed = []
            matched_b = set()
            transformed_b = set()

            if len(alive_idx_a) > 0 and len(alive_idx_b) > 0:
                S_sub = S[np.ix_(alive_idx_a, alive_idx_b)]
                cost = -S_sub
                row_ind, col_ind = linear_sum_assignment(cost)

                matched_a_set = set()
                for r, c in zip(row_ind, col_ind):
                    i = int(alive_idx_a[r])
                    j = int(alive_idx_b[c])
                    sim = float(S_sub[r, c])
                    matched_a_set.add(i)
                    if sim >= WEIGHT_STABLE_THRESH:
                        stable.append((i, j, round(sim, 4)))
                        matched_b.add(j)
                    elif sim >= WEIGHT_DEATH_THRESH:
                        transformed.append((i, j, round(sim, 4)))
                        transformed_b.add(j)
                    else:
                        died.append(i)

                for i in alive_idx_a:
                    if int(i) not in matched_a_set:
                        died.append(int(i))
            else:
                for i in alive_idx_a:
                    died.append(int(i))

            claimed_b = matched_b | transformed_b
            born = [int(j) for j in range(n_feat_b) if alive_b[j] and j not in claimed_b]

            results[pair_key][layer_name] = {
                "stable": stable,
                "born": born,
                "died": died,
                "transformed": transformed,
                "n_stable": len(stable),
                "n_born": len(born),
                "n_died": len(died),
                "n_transformed": len(transformed),
                "correlation_matrix": S,  # cosine similarity matrix
                "alive_a": alive_a,
                "alive_b": alive_b,
            }

            del W_a, W_b
            work_done += 1
            pct = 58 + 12 * work_done / max(total_work, 1)
            if work_done % max(1, total_work // 5) == 0:
                _progress(cb, pct, f"Weight matching {pair_key}/{layer_name}")

    _progress(cb, 70, f"Weight-space matching complete: {len(results)} pairs")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5c — Within-Checkpoint Control (SAE stability baseline)
# ═══════════════════════════════════════════════════════════════════════════

CONTROL_SAE_SEED_OFFSET = 999_999  # Different from shared_init_seed


def _phase5c_within_checkpoint_control(
    all_data: Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    act_matrices: Dict[str, Dict[str, np.ndarray]],
    labels: List[str],
    layers: List[str],
    expansion_factor: int,
    k_sparse: int,
    n_steps: int,
    cb: Optional[Callable],
) -> Dict[str, Any]:
    """
    Within-checkpoint SAE stability control.

    For a subset of checkpoints, train a SECOND SAE with a different random
    seed on the SAME activations, encode the same samples, and run Hungarian
    matching between the two SAEs' feature spaces.

    Any features classified as 'born' or 'died' between two SAEs trained on
    identical data represent the FALSE EVENT RATE — pure SAE stochasticity.
    Cross-checkpoint event rates must significantly exceed this baseline to
    claim genuine developmental processes.

    Tests on first, middle, and last checkpoints (≤3 total) for efficiency.

    Returns:
        {
            "per_checkpoint": [{
                "checkpoint": label,
                "per_layer": [{
                    "layer": name,
                    "n_alive_primary": int,
                    "n_alive_control": int,
                    "n_stable": int,
                    "n_transformed": int,
                    "n_false_born": int,   # born in control but not primary → artifact
                    "n_false_died": int,   # died in primary but exists in control → artifact
                    "stable_rate": float,  # n_stable / n_alive_primary
                    "false_death_rate": float,  # n_false_died / n_alive_primary
                    "false_birth_rate": float,  # n_false_born / n_alive_control
                    "mean_match_corr": float,   # mean correlation of stable matches
                }]
            }],
            "summary": {
                "mean_stable_rate": float,
                "mean_false_death_rate": float,
                "mean_false_birth_rate": float,
                "mean_match_correlation": float,
            }
        }
    """
    # Select checkpoints to test: first, middle, last
    if len(labels) <= 3:
        test_labels = labels[:]
    else:
        mid = len(labels) // 2
        test_labels = [labels[0], labels[mid], labels[-1]]

    _progress(cb, 70, f"Within-checkpoint control: testing {len(test_labels)} checkpoints")

    per_checkpoint_results = []
    all_stable_rates = []
    all_death_rates = []
    all_birth_rates = []
    all_corrs = []

    for ckpt_i, label in enumerate(test_labels):
        pct = 70 + 2 * (ckpt_i + 1) / len(test_labels)
        _progress(cb, pct, f"Control SAE for checkpoint {label}")

        layer_data = all_data.get(label, {})
        if not layer_data:
            continue

        per_layer_results = []

        for layer_name in layers:
            class_acts = layer_data.get(layer_name, {})
            all_acts = []
            for ci in sorted(class_acts.keys()):
                t = class_acts[ci]
                if t.numel() > 0:
                    all_acts.append(t.float())
            if not all_acts:
                continue

            X = torch.cat(all_acts, dim=0)
            d_input = X.shape[-1]

            # Train control SAE with different seed (no shared init)
            use_topk = k_sparse is not None and k_sparse > 0
            control_init = create_sae_init(
                d_input, expansion_factor,
                k_sparse if use_topk else None,
                seed=CONTROL_SAE_SEED_OFFSET + hash(layer_name) % 2**31,
            )
            control_sae = train_sae(
                activations=X,
                expansion_factor=expansion_factor,
                k_sparse=k_sparse,
                n_steps=n_steps,
                batch_size=min(256, X.shape[0]),
                init_state=control_init,
            )

            # Encode samples through control SAE
            control_sae.eval()
            with torch.no_grad():
                _, H_control_t = control_sae(X.float())
            H_control = H_control_t.cpu().numpy()

            # Get primary SAE activations
            H_primary = act_matrices.get(label, {}).get(layer_name)
            if H_primary is None:
                del X, control_sae, H_control
                _clear_memory()
                continue

            n_samples = H_primary.shape[0]
            n_feat_p = H_primary.shape[1]
            n_feat_c = H_control.shape[1]

            # Alive features
            alive_thresh = max(1, n_samples * 0.01)
            alive_p = (np.abs(H_primary) > 1e-6).sum(axis=0) >= alive_thresh
            alive_c = (np.abs(H_control) > 1e-6).sum(axis=0) >= alive_thresh

            alive_idx_p = np.where(alive_p)[0]
            alive_idx_c = np.where(alive_c)[0]

            n_alive_p = len(alive_idx_p)
            n_alive_c = len(alive_idx_c)

            if n_alive_p == 0 or n_alive_c == 0:
                del X, control_sae, H_control
                _clear_memory()
                continue

            # Correlation matrix between primary and control
            eps = 1e-10
            A_p = H_primary.copy()
            A_c = H_control.copy()
            A_p -= A_p.mean(axis=0, keepdims=True)
            A_c -= A_c.mean(axis=0, keepdims=True)
            norm_p = np.sqrt((A_p ** 2).sum(axis=0, keepdims=True) + eps)
            norm_c = np.sqrt((A_c ** 2).sum(axis=0, keepdims=True) + eps)
            A_p /= norm_p
            A_c /= norm_c
            C = A_p.T @ A_c  # (n_feat_p, n_feat_c)

            # Hungarian matching on alive features
            C_sub = C[np.ix_(alive_idx_p, alive_idx_c)]
            cost = -C_sub
            row_ind, col_ind = linear_sum_assignment(cost)

            n_stable = 0
            n_transformed = 0
            n_false_died = 0
            match_corrs = []
            matched_p = set()
            matched_c = set()

            for r, c_idx in zip(row_ind, col_ind):
                corr = float(C_sub[r, c_idx])
                i_p = int(alive_idx_p[r])
                j_c = int(alive_idx_c[c_idx])
                matched_p.add(i_p)
                matched_c.add(j_c)

                if corr >= MATCH_STABLE_THRESH:
                    n_stable += 1
                    match_corrs.append(corr)
                elif corr >= MATCH_DEATH_THRESH:
                    n_transformed += 1
                else:
                    n_false_died += 1

            # Unmatched primary features = also false deaths
            for i in alive_idx_p:
                if int(i) not in matched_p:
                    n_false_died += 1

            # False births: alive control features not matched
            n_false_born = sum(1 for j in alive_idx_c if int(j) not in matched_c)

            stable_rate = n_stable / max(n_alive_p, 1)
            false_death_rate = n_false_died / max(n_alive_p, 1)
            false_birth_rate = n_false_born / max(n_alive_c, 1)
            mean_corr = float(np.mean(match_corrs)) if match_corrs else 0.0

            per_layer_results.append({
                "layer": layer_name,
                "n_alive_primary": n_alive_p,
                "n_alive_control": n_alive_c,
                "n_stable": n_stable,
                "n_transformed": n_transformed,
                "n_false_born": n_false_born,
                "n_false_died": n_false_died,
                "stable_rate": round(stable_rate, 4),
                "false_death_rate": round(false_death_rate, 4),
                "false_birth_rate": round(false_birth_rate, 4),
                "mean_match_corr": round(mean_corr, 4),
            })

            all_stable_rates.append(stable_rate)
            all_death_rates.append(false_death_rate)
            all_birth_rates.append(false_birth_rate)
            if match_corrs:
                all_corrs.extend(match_corrs)

            del X, control_sae, H_control, A_p, A_c, C, C_sub
            _clear_memory()

        per_checkpoint_results.append({
            "checkpoint": label,
            "per_layer": per_layer_results,
        })

    summary = {
        "mean_stable_rate": round(float(np.mean(all_stable_rates)), 4) if all_stable_rates else 0.0,
        "mean_false_death_rate": round(float(np.mean(all_death_rates)), 4) if all_death_rates else 0.0,
        "mean_false_birth_rate": round(float(np.mean(all_birth_rates)), 4) if all_birth_rates else 0.0,
        "mean_match_correlation": round(float(np.mean(all_corrs)), 4) if all_corrs else 0.0,
    }

    _progress(cb, 72, f"Control complete: {summary['mean_stable_rate']:.0%} stable, "
              f"{summary['mean_false_death_rate']:.0%} false death rate")

    return {
        "per_checkpoint": per_checkpoint_results,
        "summary": summary,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6 — Per-Sample Process Classification (70-78%)
# ═══════════════════════════════════════════════════════════════════════════


def _phase6_process_classification(
    act_matrices: Dict[str, Dict[str, np.ndarray]],
    all_data: Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    feature_matching: Dict[str, Dict[str, Dict[str, Any]]],
    ssi_csi_data: Dict[Tuple[str, str], Dict[str, Any]],
    labels: List[str],
    layers: List[str],
    output_dir: Path,
    cb: Optional[Callable],
    transitions_subdir: str = "transitions",
    thresholds: Optional[Dict[str, float]] = None,
    adaptive_threshold_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Classify each feature event PER SAMPLE into developmental processes.

    For each sample at each transition, determine which of its active features
    are born/died/stable, then label each event:
      - born → As-H (every new feature is assembled into the representation)
      - SSI > thresh → Ab-H (superclass abstraction)
      - SAI > thresh → Tg-H + Ab-H (task-general abstraction)
      - CSI_local > thresh → Di-H (within-superclass differentiation)
      - died → De-H (feature lost)

    Stable features get Ab-H/Di-H/Tg-H only if their z-score increased by
    ≥0.5 SD across the transition (replaces old 1.1× factor).
    Uses CSI_local (within-superclass) for Di-H instead of global CSI.
    A feature can have MULTIPLE process labels.

    Returns:
        {pair_key: {layer: [per_sample_events]}}
    """
    _progress(cb, 70, "Per-sample process classification")

    t = thresholds or DEFAULT_THRESHOLDS
    ssi_thresh = t["ssi_abstraction_thresh"]
    sai_thresh = t["sai_task_general_thresh"]
    csi_thresh = t["csi_differentiation_thresh"]
    # Z-score delta for stable features (replaces 1.1× increase factor)
    z_delta_min = 0.5  # Cohen's d small effect

    # Extract null distribution stats for z-score computation
    eps_z = 1e-10
    if adaptive_threshold_info:
        ssi_null_mean = adaptive_threshold_info.get("ssi_null_mean", 0.0)
        ssi_null_std = max(adaptive_threshold_info.get("ssi_null_std", 1.0), eps_z)
        csi_null_mean = adaptive_threshold_info.get("csi_null_mean", 0.0)
        csi_null_std = max(adaptive_threshold_info.get("csi_null_std", 1.0), eps_z)
        sai_null_mean = adaptive_threshold_info.get("sai_null_mean", 0.0)
        sai_null_std = max(adaptive_threshold_info.get("sai_null_std", 1.0), eps_z)
    else:
        # Fallback: no z-score normalization (raw values, effectively z=val)
        ssi_null_mean = 0.0; ssi_null_std = 1.0
        csi_null_mean = 0.0; csi_null_std = 1.0
        sai_null_mean = 0.0; sai_null_std = 1.0

    def _z(val: float, mean: float, std: float) -> float:
        return (val - mean) / std

    results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    pairs = list(zip(labels[:-1], labels[1:]))
    total_work = sum(len(layers) for _ in pairs)
    work_done = 0

    for label_a, label_b in pairs:
        pair_key = f"{label_a}->{label_b}"
        results[pair_key] = {}

        for layer_name in layers:
            H_a = act_matrices.get(label_a, {}).get(layer_name)
            H_b = act_matrices.get(label_b, {}).get(layer_name)
            match_data = feature_matching.get(pair_key, {}).get(layer_name)

            if H_a is None or H_b is None or match_data is None:
                work_done += 1
                continue

            # Get SSI/CSI/SAI for features at checkpoints A and B
            ssi_entry_b = ssi_csi_data.get((label_b, layer_name), {})
            feature_ssi_b = np.array(ssi_entry_b.get("feature_ssi", []))
            feature_csi_b = np.array(ssi_entry_b.get("feature_csi", []))
            feature_sai_b = np.array(ssi_entry_b.get("feature_sai", []))

            ssi_entry_a = ssi_csi_data.get((label_a, layer_name), {})
            feature_ssi_a = np.array(ssi_entry_a.get("feature_ssi", []))
            feature_csi_a = np.array(ssi_entry_a.get("feature_csi", []))
            feature_sai_a = np.array(ssi_entry_a.get("feature_sai", []))

            # Build lookup sets from global matching
            stable_set_a = {s[0] for s in match_data["stable"]}       # A features that are stable
            stable_map = {s[0]: s[1] for s in match_data["stable"]}   # A→B mapping for stable
            died_set = set(match_data["died"])                         # genuinely lost (corr < 0.2)
            born_set = set(match_data["born"])                         # genuinely new (no match from A)
            transformed_set_a = {t[0] for t in match_data.get("transformed", [])}  # ambiguous (0.2–0.5)

            alive_a = match_data["alive_a"]
            alive_b = match_data["alive_b"]

            n_samples = H_a.shape[0]
            alive_thresh_val = 1e-6

            per_sample_events: List[Dict[str, Any]] = []

            # Determine class boundaries
            layer_data_a = all_data.get(label_a, {}).get(layer_name, {})
            class_indices = sorted(layer_data_a.keys())
            class_boundaries = []
            offset = 0
            for ci in class_indices:
                t = layer_data_a[ci]
                n_ci = t.shape[0] if t.numel() > 0 else 0
                class_boundaries.append((ci, offset, offset + n_ci))
                offset += n_ci

            for ci, start, end in class_boundaries:
                for si in range(end - start):
                    sample_idx = start + si
                    h_a = H_a[sample_idx, :]  # (d_hidden_a,)
                    h_b = H_b[sample_idx, :]  # (d_hidden_b,)

                    # Which features are active for THIS sample
                    active_a_sample = np.abs(h_a) > alive_thresh_val
                    active_b_sample = np.abs(h_b) > alive_thresh_val

                    sample_born = []
                    sample_died = []
                    sample_stable = []
                    counts = {"ab_h": 0, "tg_h": 0, "di_h": 0, "as_h": 0, "de_h": 0,
                              "stable": 0, "unclassified": 0}

                    # Stable features active in this sample
                    for idx_a in range(len(h_a)):
                        if not active_a_sample[idx_a]:
                            continue
                        if not alive_a[idx_a]:
                            continue

                        if idx_a in stable_set_a:
                            idx_b = stable_map[idx_a]
                            if active_b_sample[idx_b]:
                                # Stable features get labels if z-score increased by ≥0.5 SD
                                ssi_a = float(feature_ssi_a[idx_a]) if idx_a < len(feature_ssi_a) else 0.0
                                ssi_b = float(feature_ssi_b[idx_b]) if idx_b < len(feature_ssi_b) else 0.0
                                csi_a = float(feature_csi_a[idx_a]) if idx_a < len(feature_csi_a) else 0.0
                                csi_b = float(feature_csi_b[idx_b]) if idx_b < len(feature_csi_b) else 0.0
                                sai_a_val = float(feature_sai_a[idx_a]) if idx_a < len(feature_sai_a) else 0.0
                                sai_b_val = float(feature_sai_b[idx_b]) if idx_b < len(feature_sai_b) else 0.0

                                z_ssi_a = _z(ssi_a, ssi_null_mean, ssi_null_std)
                                z_ssi_b = _z(ssi_b, ssi_null_mean, ssi_null_std)
                                z_csi_a = _z(csi_a, csi_null_mean, csi_null_std)
                                z_csi_b = _z(csi_b, csi_null_mean, csi_null_std)
                                z_sai_a = _z(sai_a_val, sai_null_mean, sai_null_std)
                                z_sai_b = _z(sai_b_val, sai_null_mean, sai_null_std)

                                stable_procs: List[str] = []
                                # Ab-H: SSI exceeds threshold AND z-score increased by ≥0.5 SD
                                has_ab = (ssi_b > ssi_thresh and (z_ssi_b - z_ssi_a) > z_delta_min)
                                # Tg-H: SAI exceeds threshold AND z-score increased
                                has_tg = (sai_b_val > sai_thresh and (z_sai_b - z_sai_a) > z_delta_min)
                                if has_tg:
                                    stable_procs.append("tg_h")
                                elif has_ab:
                                    stable_procs.append("ab_h")
                                # Di-H: CSI exceeds threshold AND z-score increased
                                if csi_b > csi_thresh and (z_csi_b - z_csi_a) > z_delta_min:
                                    stable_procs.append("di_h")
                                sample_stable.append({
                                    "fid_a": int(idx_a), "fid_b": int(idx_b),
                                    "processes": stable_procs,
                                    "ssi": round(ssi_b, 4), "csi": round(csi_b, 4),
                                    "sai": round(sai_b_val, 4),
                                    "z_ssi": round(z_ssi_b, 2), "z_csi": round(z_csi_b, 2),
                                    "z_sai": round(z_sai_b, 2),
                                })
                                counts["stable"] += 1
                                for p in stable_procs:
                                    counts[p] = counts.get(p, 0) + 1
                        elif idx_a in transformed_set_a:
                            pass  # ambiguous — excluded from process breakdown
                        elif idx_a in died_set:
                            processes = ["de_h"]
                            sample_died.append({"fid": int(idx_a), "processes": processes})
                            counts["de_h"] += 1

                    # Born features active in this sample
                    for idx_b in born_set:
                        if not active_b_sample[idx_b]:
                            continue
                        if not alive_b[idx_b]:
                            continue
                        # Every born feature is assembly (As-H) by definition.
                        # Additionally label by selectivity (absolute threshold).
                        processes = ["as_h"]
                        ssi_val = float(feature_ssi_b[idx_b]) if idx_b < len(feature_ssi_b) else 0.0
                        csi_val = float(feature_csi_b[idx_b]) if idx_b < len(feature_csi_b) else 0.0
                        sai_val = float(feature_sai_b[idx_b]) if idx_b < len(feature_sai_b) else 0.0
                        z_ssi_val = _z(ssi_val, ssi_null_mean, ssi_null_std)
                        z_csi_val = _z(csi_val, csi_null_mean, csi_null_std)
                        z_sai_val = _z(sai_val, sai_null_mean, sai_null_std)
                        # Ab-H: superclass abstraction (SSI only, not task-general)
                        has_ab = ssi_val > ssi_thresh
                        # Tg-H: task-general (SAI)
                        has_tg = sai_val > sai_thresh
                        if has_tg:
                            processes.append("tg_h")
                        elif has_ab:
                            processes.append("ab_h")
                        if csi_val > csi_thresh:
                            processes.append("di_h")
                        sample_born.append({
                            "fid": int(idx_b),
                            "processes": processes,
                            "ssi": round(ssi_val, 4),
                            "csi": round(csi_val, 4),
                            "sai": round(sai_val, 4),
                            "z_ssi": round(z_ssi_val, 2),
                            "z_csi": round(z_csi_val, 2),
                            "z_sai": round(z_sai_val, 2),
                        })
                        for p in processes:
                            counts[p] = counts.get(p, 0) + 1

                    per_sample_events.append({
                        "class_idx": int(ci),
                        "sample_idx_in_class": int(si),
                        "born": sample_born,
                        "died": sample_died,
                        "stable": sample_stable,
                        "process_counts": counts,
                    })

            results[pair_key][layer_name] = per_sample_events

            work_done += 1
            pct = 70 + 8 * work_done / max(total_work, 1)
            if work_done % max(1, total_work // 5) == 0:
                _progress(cb, pct, f"Classifying {pair_key}/{layer_name}")

    # Save per-sample transitions to disk
    trans_dir = output_dir / transitions_subdir
    for pair_key, layer_dict in results.items():
        for layer_name, events in layer_dict.items():
            safe_layer = _safe_layer_filename(layer_name)
            pair_dir = trans_dir / pair_key.replace("->", "_to_")
            pair_dir.mkdir(parents=True, exist_ok=True)
            with open(pair_dir / f"{safe_layer}.json", "w") as f:
                json.dump(_to_list(events), f)

    _progress(cb, 78, "Process classification complete")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7 — Aggregation (78-90%)
# ═══════════════════════════════════════════════════════════════════════════


def _phase8_aggregation(
    process_events: Dict[str, Dict[str, List[Dict[str, Any]]]],
    act_matrices: Dict[str, Dict[str, np.ndarray]],
    all_data: Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    ssi_csi_data: Dict[Tuple[str, str], Dict[str, Any]],
    superclass_map: Dict[int, str],
    superclass_groups: Dict[str, List[int]],
    selected_classes: List[str],
    labels: List[str],
    layers: List[str],
    cb: Optional[Callable],
    control_results: Optional[Dict[str, Any]] = None,
    feature_matching: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """
    Aggregate per-sample process events into class, superclass, and global summaries.

    Returns dict with:
        class_process_summary, superclass_summary, sample_consistency,
        process_intensity, feature_landscape
    """
    _progress(cb, 82, "Aggregating results")

    class_to_idx = {name: i for i, name in enumerate(selected_classes)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    n_classes = len(selected_classes)

    # ── Per-class process counts ──
    class_process: Dict[int, Dict[str, int]] = {}
    class_per_transition: Dict[int, List[Dict[str, Any]]] = {}

    # ── Per-sample dominant process (for sample consistency) ──
    sample_totals: Dict[int, Dict[str, int]] = {}  # class_idx → {ab_h_dom, di_h_dom, ...}

    for pair_key, layer_dict in process_events.items():
        for layer_name, events in layer_dict.items():
            for ev in events:
                ci = ev["class_idx"]
                counts = ev["process_counts"]

                # Accumulate class totals
                if ci not in class_process:
                    class_process[ci] = {"ab_h": 0, "tg_h": 0, "di_h": 0, "as_h": 0, "de_h": 0, "unclassified": 0, "stable": 0}
                for proc, count in counts.items():
                    class_process[ci][proc] = class_process[ci].get(proc, 0) + count

    # Per-class per-transition breakdown
    for pair_key, layer_dict in process_events.items():
        pair_class_counts: Dict[int, Dict[str, int]] = {}
        for layer_name, events in layer_dict.items():
            for ev in events:
                ci = ev["class_idx"]
                counts = ev["process_counts"]
                if ci not in pair_class_counts:
                    pair_class_counts[ci] = {"ab_h": 0, "tg_h": 0, "di_h": 0, "as_h": 0, "de_h": 0, "unclassified": 0}
                for proc, count in counts.items():
                    if proc != "stable":
                        pair_class_counts[ci][proc] = pair_class_counts[ci].get(proc, 0) + count

        for ci, counts in pair_class_counts.items():
            class_per_transition.setdefault(ci, []).append({
                "transition": pair_key,
                **counts,
            })

    # ── Sample consistency: dominant process per sample ──
    # For each sample, sum events across all transitions and layers,
    # determine dominant process.
    sample_events_total: Dict[Tuple[int, int], Dict[str, int]] = {}  # (ci, si) → counts
    for pair_key, layer_dict in process_events.items():
        for layer_name, events in layer_dict.items():
            for ev in events:
                ci = ev["class_idx"]
                si = ev["sample_idx_in_class"]
                key = (ci, si)
                if key not in sample_events_total:
                    sample_events_total[key] = {"ab_h": 0, "tg_h": 0, "di_h": 0, "as_h": 0, "de_h": 0}
                for proc in ["ab_h", "tg_h", "di_h", "as_h", "de_h"]:
                    sample_events_total[key][proc] += ev["process_counts"].get(proc, 0)

    sample_consistency: Dict[int, Dict[str, int]] = {}
    for (ci, si), counts in sample_events_total.items():
        if ci not in sample_consistency:
            sample_consistency[ci] = {"ab_h_dominant": 0, "tg_h_dominant": 0, "di_h_dominant": 0,
                                      "as_h_dominant": 0, "de_h_dominant": 0, "mixed": 0}
        total = sum(counts.values())
        if total == 0:
            sample_consistency[ci]["mixed"] += 1
            continue
        dominant = max(counts, key=counts.get)
        dominant_pct = counts[dominant] / total
        if dominant_pct > 0.4:
            sample_consistency[ci][f"{dominant}_dominant"] += 1
        else:
            sample_consistency[ci]["mixed"] += 1

    # ── Class process summary ──
    class_process_summary: Dict[str, Any] = {}
    for ci in sorted(class_process.keys()):
        counts = class_process[ci]
        # tg_h is a subcategory of ab_h — don't double-count
        total = counts["ab_h"] + counts["di_h"] + counts["as_h"] + counts["de_h"] + counts["unclassified"]
        dominant = "none"
        if total > 0:
            # tg_h excluded from dominant — it's a subcategory of ab_h
            proc_counts = {k: counts[k] for k in ["ab_h", "di_h", "as_h", "de_h"]}
            dominant = max(proc_counts, key=proc_counts.get)

        class_name = idx_to_class.get(ci, str(ci))
        class_process_summary[str(ci)] = {
            "class_name": class_name,
            "ab_h": counts["ab_h"],
            "tg_h": counts["tg_h"],
            "di_h": counts["di_h"],
            "as_h": counts["as_h"],
            "de_h": counts["de_h"],
            "unclassified": counts["unclassified"],
            "stable": counts["stable"],
            "total": total,
            "dominant": dominant,
            "per_transition": class_per_transition.get(ci, []),
        }

    # ── Superclass summary ──
    superclass_summary: Dict[str, Any] = {}
    for sc_name, sc_class_list in superclass_groups.items():
        sc_counts = {"ab_h": 0, "tg_h": 0, "di_h": 0, "as_h": 0, "de_h": 0, "unclassified": 0}
        per_class_process: Dict[str, Dict[str, int]] = {}

        for ci in sc_class_list:
            class_name = idx_to_class.get(ci, str(ci))
            c_counts = class_process.get(ci, {})
            for proc in sc_counts:
                sc_counts[proc] += c_counts.get(proc, 0)
            per_class_process[class_name] = {
                "ab_h": c_counts.get("ab_h", 0),
                "tg_h": c_counts.get("tg_h", 0),
                "di_h": c_counts.get("di_h", 0),
                "as_h": c_counts.get("as_h", 0),
                "de_h": c_counts.get("de_h", 0),
            }

        # tg_h is a subcategory of ab_h — don't double-count
        sc_total = sum(sc_counts.values()) - sc_counts["tg_h"]

        # SSI evolution for this superclass (only features selective for this SC)
        ssi_evolution = []
        for label in labels:
            sc_ssi_vals = []
            for layer_name in layers:
                entry = ssi_csi_data.get((label, layer_name))
                if entry:
                    f_ssi = np.array(entry["feature_ssi"])
                    f_best_sc = entry["best_superclass"]
                    sc_mask = np.array([bs == sc_name for bs in f_best_sc])
                    if sc_mask.any():
                        sc_ssi_vals.append(float(np.mean(f_ssi[sc_mask])))
            if sc_ssi_vals:
                ssi_evolution.append({
                    "checkpoint": label,
                    "mean_ssi": _safe_float(float(np.mean(sc_ssi_vals))),
                })

        # Per-class CSI evolution (features selective for each fine class within this SC)
        per_class_csi_evolution: Dict[str, list] = {}
        for ci in sc_class_list:
            class_name = idx_to_class.get(ci, str(ci))
            csi_evo = []
            for label in labels:
                class_csi_vals = []
                for layer_name in layers:
                    entry = ssi_csi_data.get((label, layer_name))
                    if entry:
                        f_csi = np.array(entry["feature_csi"])
                        f_best_class = entry["best_class"]
                        cls_mask = np.array([bc == ci for bc in f_best_class])
                        if cls_mask.any():
                            class_csi_vals.append(float(np.mean(f_csi[cls_mask])))
                if class_csi_vals:
                    csi_evo.append({
                        "checkpoint": label,
                        "mean_csi": _safe_float(float(np.mean(class_csi_vals))),
                    })
            per_class_csi_evolution[class_name] = csi_evo

        # Pairwise feature overlap at terminal checkpoint (last milestone)
        pairwise_overlap = _compute_pairwise_overlap(
            act_matrices, all_data, sc_class_list, labels[-1], layers,
        )

        # Feature sharing breadth at terminal
        feature_sharing = _compute_feature_sharing_breadth(
            act_matrices, all_data, sc_class_list, labels[-1], layers,
        )

        # Shared feature growth over training
        shared_growth = []
        for label in labels:
            n_shared = _count_shared_features(
                act_matrices, all_data, sc_class_list, label, layers,
            )
            shared_growth.append({"checkpoint": label, "n_shared_all": n_shared})

        superclass_summary[sc_name] = {
            "process_breakdown": {
                "ab_h": sc_counts["ab_h"],
                "tg_h": sc_counts["tg_h"],
                "di_h": sc_counts["di_h"],
                "as_h": sc_counts["as_h"],
                "de_h": sc_counts["de_h"],
                "unclassified": sc_counts["unclassified"],
                "total": sc_total,
            },
            "ssi_evolution": ssi_evolution,
            "per_class_csi_evolution": per_class_csi_evolution,
            "per_class_process": per_class_process,
            "pairwise_overlap": pairwise_overlap,
            "feature_sharing_breadth": feature_sharing,
            "shared_feature_growth": shared_growth,
        }

    # ── Feature landscape ──
    feature_landscape: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for label in labels:
        feature_landscape[label] = {}
        for layer_name in layers:
            entry = ssi_csi_data.get((label, layer_name))
            if entry:
                feature_landscape[label][layer_name] = {
                    "n_alive": entry["n_active_features"],
                    "mean_ssi": entry["mean_ssi"],
                    "mean_csi": entry["mean_csi"],
                    "mean_entropy": entry["mean_entropy"],
                    "mean_sai": entry["mean_sai"],
                    "n_high_ssi": entry["n_high_ssi"],
                    "n_high_sai": entry["n_high_sai"],
                }

    # ── Global process intensity per transition ──
    process_intensity: List[Dict[str, Any]] = []
    for pair_key, layer_dict in process_events.items():
        totals = {"ab_h": 0, "tg_h": 0, "di_h": 0, "as_h": 0, "de_h": 0, "unclassified": 0, "stable": 0}
        for layer_name, events in layer_dict.items():
            for ev in events:
                for proc, count in ev["process_counts"].items():
                    totals[proc] = totals.get(proc, 0) + count

        # tg_h is a subcategory of ab_h — don't double-count
        event_total = totals["ab_h"] + totals["di_h"] + totals["as_h"] + totals["de_h"] + totals["unclassified"]
        churn = 0.0
        if event_total + totals["stable"] > 0:
            churn = event_total / (event_total + totals["stable"])

        dominant = "none"
        if event_total > 0:
            # tg_h excluded from dominant — it's a subcategory of ab_h
            proc_counts = {k: totals[k] for k in ["ab_h", "di_h", "as_h", "de_h"]}
            dominant = max(proc_counts, key=proc_counts.get)

        pi_entry: Dict[str, Any] = {
            "transition": pair_key,
            "ab_h": totals["ab_h"],
            "tg_h": totals["tg_h"],
            "di_h": totals["di_h"],
            "as_h": totals["as_h"],
            "de_h": totals["de_h"],
            "unclassified": totals["unclassified"],
            "total": event_total,
            "churn": _safe_float(churn),
            "dominant": dominant,
        }

        # Adjusted counts: subtract expected false events from control baseline
        if control_results and feature_matching:
            ctrl_summary = control_results.get("summary", {})
            false_birth_rate = ctrl_summary.get("mean_false_birth_rate", 0.0)
            false_death_rate = ctrl_summary.get("mean_false_death_rate", 0.0)

            # Count alive features for this transition to estimate expected false events
            match_data = feature_matching.get(pair_key, {})
            total_alive_a, total_alive_b = 0, 0
            total_born, total_died = 0, 0
            for layer_name, mi in match_data.items():
                alive_a = mi.get("alive_a")
                alive_b = mi.get("alive_b")
                if alive_a is not None:
                    total_alive_a += int(alive_a.sum())
                if alive_b is not None:
                    total_alive_b += int(alive_b.sum())
                total_born += mi.get("n_born", 0)
                total_died += mi.get("n_died", 0)

            expected_false_births = int(round(total_alive_b * false_birth_rate))
            expected_false_deaths = int(round(total_alive_a * false_death_rate))

            # Adjusted born/died counts (floor at 0)
            adj_born = max(0, total_born - expected_false_births)
            adj_died = max(0, total_died - expected_false_deaths)

            # Scale process event counts proportionally
            birth_scale = adj_born / max(total_born, 1)
            death_scale = adj_died / max(total_died, 1)

            # As-H and born-dependent counts scale with birth_scale;
            # De-H scales with death_scale; Ab-H and Di-H on born features scale too
            pi_entry["adjusted_as_h"] = max(0, int(round(totals["as_h"] * birth_scale)))
            pi_entry["adjusted_de_h"] = max(0, int(round(totals["de_h"] * death_scale)))
            # Ab-H, Tg-H, and Di-H come from both stable features and born features;
            # conservative: scale only the born-feature contribution (not stable)
            # Since we can't separate them here, use birth_scale as upper bound
            pi_entry["adjusted_ab_h"] = totals["ab_h"]  # stable-driven, keep raw
            pi_entry["adjusted_tg_h"] = totals["tg_h"]  # stable-driven, keep raw
            pi_entry["adjusted_di_h"] = totals["di_h"]  # stable-driven, keep raw
            # tg_h is a subcategory of ab_h — don't double-count
            adj_total = (pi_entry["adjusted_ab_h"]
                         + pi_entry["adjusted_di_h"]
                         + pi_entry["adjusted_as_h"] + pi_entry["adjusted_de_h"]
                         + totals["unclassified"])
            pi_entry["adjusted_total"] = adj_total
            pi_entry["expected_false_births"] = expected_false_births
            pi_entry["expected_false_deaths"] = expected_false_deaths

        process_intensity.append(pi_entry)

    _progress(cb, 90, "Aggregation complete")
    return {
        "class_process_summary": class_process_summary,
        "superclass_summary": superclass_summary,
        "sample_consistency": {str(k): v for k, v in sample_consistency.items()},
        "process_intensity": process_intensity,
        "feature_landscape": feature_landscape,
    }


def _compute_pairwise_overlap(
    act_matrices: Dict[str, Dict[str, np.ndarray]],
    all_data: Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    class_list: List[int],
    checkpoint: str,
    layers: List[str],
) -> Dict[str, float]:
    """Compute Jaccard similarity of active feature sets between class pairs."""
    overlap: Dict[str, float] = {}

    for layer_name in layers:
        H = act_matrices.get(checkpoint, {}).get(layer_name)
        if H is None:
            continue
        layer_data = all_data.get(checkpoint, {}).get(layer_name, {})
        class_indices = sorted(layer_data.keys())

        # Get active features per class
        class_active: Dict[int, set] = {}
        offset = 0
        for ci in class_indices:
            t = layer_data[ci]
            n_ci = t.shape[0] if t.numel() > 0 else 0
            if n_ci > 0 and ci in class_list:
                class_h = H[offset:offset + n_ci, :]
                # Feature is active for this class if mean |activation| > threshold
                mean_act = np.abs(class_h).mean(axis=0)
                active_features = set(np.where(mean_act > 0.01)[0])
                class_active[ci] = active_features
            offset += n_ci

    # Pairwise Jaccard across layers (average)
    for ci_a, ci_b in combinations(class_list, 2):
        if ci_a not in class_active or ci_b not in class_active:
            continue
        set_a = class_active.get(ci_a, set())
        set_b = class_active.get(ci_b, set())
        if len(set_a | set_b) == 0:
            jaccard = 0.0
        else:
            jaccard = len(set_a & set_b) / len(set_a | set_b)
        key = f"{ci_a}-{ci_b}"
        overlap[key] = _safe_float(jaccard)

    return overlap


def _compute_feature_sharing_breadth(
    act_matrices: Dict[str, Dict[str, np.ndarray]],
    all_data: Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    class_list: List[int],
    checkpoint: str,
    layers: List[str],
) -> Dict[str, int]:
    """Count features shared by exactly 1, 2, ..., N classes in the superclass."""
    n_sc = len(class_list)
    breadth: Dict[str, int] = {f"n_shared_{i}": 0 for i in range(1, n_sc + 1)}

    for layer_name in layers:
        H = act_matrices.get(checkpoint, {}).get(layer_name)
        if H is None:
            continue
        layer_data = all_data.get(checkpoint, {}).get(layer_name, {})
        class_indices = sorted(layer_data.keys())

        n_features = H.shape[1]
        # For each feature, count how many classes in this superclass activate it
        feature_class_count = np.zeros(n_features, dtype=int)
        offset = 0
        for ci in class_indices:
            t = layer_data[ci]
            n_ci = t.shape[0] if t.numel() > 0 else 0
            if n_ci > 0 and ci in class_list:
                class_h = H[offset:offset + n_ci, :]
                mean_act = np.abs(class_h).mean(axis=0)
                feature_class_count += (mean_act > 0.01).astype(int)
            offset += n_ci

        # Count breadth
        for count in range(1, n_sc + 1):
            n_feat = int(np.sum(feature_class_count == count))
            breadth[f"n_shared_{count}"] += n_feat

    return breadth


def _count_shared_features(
    act_matrices: Dict[str, Dict[str, np.ndarray]],
    all_data: Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    class_list: List[int],
    checkpoint: str,
    layers: List[str],
) -> int:
    """Count features active for ALL classes in the superclass."""
    n_sc = len(class_list)
    total_shared = 0

    for layer_name in layers:
        H = act_matrices.get(checkpoint, {}).get(layer_name)
        if H is None:
            continue
        layer_data = all_data.get(checkpoint, {}).get(layer_name, {})
        class_indices = sorted(layer_data.keys())

        n_features = H.shape[1]
        feature_class_count = np.zeros(n_features, dtype=int)
        offset = 0
        for ci in class_indices:
            t = layer_data[ci]
            n_ci = t.shape[0] if t.numel() > 0 else 0
            if n_ci > 0 and ci in class_list:
                class_h = H[offset:offset + n_ci, :]
                mean_act = np.abs(class_h).mean(axis=0)
                feature_class_count += (mean_act > 0.01).astype(int)
            offset += n_ci

        total_shared += int(np.sum(feature_class_count >= n_sc))

    return total_shared


# ═══════════════════════════════════════════════════════════════════════════
# Phase 8 — Hypothesis Testing & Null Baseline (90-96%)
# ═══════════════════════════════════════════════════════════════════════════


def _phase9_hypothesis_testing(
    aggregation: Dict[str, Any],
    ssi_csi_data: Dict[Tuple[str, str], Dict[str, Any]],
    superclass_map: Dict[int, str],
    superclass_groups: Dict[str, List[int]],
    labels: List[str],
    layers: List[str],
    null_permutations: int,
    cb: Optional[Callable],
    control_results: Optional[Dict[str, Any]] = None,
    feature_matching: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Statistical hypothesis testing and null baseline.

    Returns dict with:
        hypotheses: {Ab-H, Tg-H, Di-H, As-H, De-H} with verdict + evidence
        null_baseline: per-checkpoint SSI vs random permutation
        selectivity_evolution: per-checkpoint mean metrics
        discrimination_gradients: layer × checkpoint discrimination scores
    """
    _progress(cb, 90, "Hypothesis testing")

    t = thresholds or DEFAULT_THRESHOLDS
    process_intensity = aggregation["process_intensity"]
    feature_landscape = aggregation["feature_landscape"]

    # ── Selectivity evolution table ──
    selectivity_evolution = []
    for label in labels:
        layer_entries = []
        for layer_name in layers:
            entry = ssi_csi_data.get((label, layer_name))
            if entry:
                layer_entries.append(entry)

        if layer_entries:
            mean_ssi = float(np.mean([e["mean_ssi"] for e in layer_entries]))
            mean_csi = float(np.mean([e["mean_csi"] for e in layer_entries]))
            mean_sai = float(np.mean([e["mean_sai"] for e in layer_entries]))
            n_alive = float(np.mean([e["n_active_features"] for e in layer_entries]))
            selectivity_evolution.append({
                "checkpoint": label,
                "mean_ssi": _safe_float(mean_ssi),
                "mean_csi": _safe_float(mean_csi),
                "mean_sai": _safe_float(mean_sai),
                "n_alive_mean": _safe_float(n_alive),
            })

    # ── Hypothesis verdicts ──
    total_events = sum(pi["total"] for pi in process_intensity)
    hypotheses: Dict[str, Any] = {}

    # Ab-H: SSI increases over time + Ab-H events dominate early
    ab_h_events = sum(pi["ab_h"] for pi in process_intensity)
    ab_h_evidence = []
    if len(selectivity_evolution) >= 2:
        ssi_values = [s["mean_ssi"] for s in selectivity_evolution]
        if ssi_values[-1] > ssi_values[0] * t["ssi_increase_factor"]:
            ab_h_evidence.append({
                "metric": "ssi_increase",
                "value": _safe_float(ssi_values[-1] - ssi_values[0]),
                "significant": True,
            })
        # Spearman rank correlation of SSI with checkpoint index
        if len(ssi_values) >= 3:
            rho, p_val = sp_stats.spearmanr(range(len(ssi_values)), ssi_values)
            ab_h_evidence.append({
                "metric": "ssi_trend_spearman",
                "rho": _safe_float(rho),
                "p_value": _safe_float(p_val),
                "significant": p_val < 0.05,
            })

    n_met = sum(1 for e in ab_h_evidence if e.get("significant"))
    hypotheses["Ab-H"] = {
        "verdict": "confirmed" if n_met >= 1 and ab_h_events > 0 else (
            "partially_supported" if ab_h_events > 0 else "insufficient_data"),
        "events": ab_h_events,
        "events_pct": _safe_float(ab_h_events / max(total_events, 1) * 100),
        "evidence_met": n_met,
        "evidence_total": len(ab_h_evidence),
        "evidence": ab_h_evidence,
    }

    # Tg-H: SAI stable or increases over time — task-general features persist
    tg_h_events = sum(pi["tg_h"] for pi in process_intensity)
    tg_h_evidence = []
    if len(selectivity_evolution) >= 2:
        sai_values = [s["mean_sai"] for s in selectivity_evolution]
        if sai_values[-1] > sai_values[0] * t["sai_increase_factor"]:
            tg_h_evidence.append({
                "metric": "sai_increase",
                "value": _safe_float(sai_values[-1] - sai_values[0]),
                "significant": True,
            })
        if len(sai_values) >= 3:
            rho, p_val = sp_stats.spearmanr(range(len(sai_values)), sai_values)
            tg_h_evidence.append({
                "metric": "sai_trend_spearman",
                "rho": _safe_float(rho),
                "p_value": _safe_float(p_val),
                "significant": p_val < 0.05,
            })

    n_met = sum(1 for e in tg_h_evidence if e.get("significant"))
    hypotheses["Tg-H"] = {
        "verdict": "confirmed" if n_met >= 1 and tg_h_events > 0 else (
            "partially_supported" if tg_h_events > 0 else "insufficient_data"),
        "events": tg_h_events,
        "events_pct": _safe_float(tg_h_events / max(total_events, 1) * 100),
        "evidence_met": n_met,
        "evidence_total": len(tg_h_evidence),
        "evidence": tg_h_evidence,
    }

    # Di-H: CSI increases over time + Di-H events dominate mid-to-late
    di_h_events = sum(pi["di_h"] for pi in process_intensity)
    di_h_evidence = []
    if len(selectivity_evolution) >= 2:
        csi_values = [s["mean_csi"] for s in selectivity_evolution]
        if csi_values[-1] > csi_values[0] * t["csi_increase_factor"]:
            di_h_evidence.append({
                "metric": "csi_increase",
                "value": _safe_float(csi_values[-1] - csi_values[0]),
                "significant": True,
            })
        if len(csi_values) >= 3:
            rho, p_val = sp_stats.spearmanr(range(len(csi_values)), csi_values)
            di_h_evidence.append({
                "metric": "csi_trend_spearman",
                "rho": _safe_float(rho),
                "p_value": _safe_float(p_val),
                "significant": p_val < 0.05,
            })

    n_met = sum(1 for e in di_h_evidence if e.get("significant"))
    hypotheses["Di-H"] = {
        "verdict": "confirmed" if n_met >= 1 and di_h_events > 0 else (
            "partially_supported" if di_h_events > 0 else "insufficient_data"),
        "events": di_h_events,
        "events_pct": _safe_float(di_h_events / max(total_events, 1) * 100),
        "evidence_met": n_met,
        "evidence_total": len(di_h_evidence),
        "evidence": di_h_evidence,
    }

    # As-H: novel features born that co-fire with existing features
    # Assembly = born features joining the active representation. Evidence: birth
    # events increase or form a substantial fraction of total events.
    as_h_events = sum(pi["as_h"] for pi in process_intensity)
    as_h_evidence = []
    # Evidence 1: As-H fraction of total events
    as_h_frac = as_h_events / max(total_events, 1)
    as_h_evidence.append({
        "metric": "assembly_fraction",
        "value": _safe_float(as_h_frac),
        "significant": as_h_frac > t["assembly_fraction_thresh"],
    })
    # Evidence 2: birth count trend over training (Spearman)
    if len(process_intensity) >= 3:
        birth_values = [pi["as_h"] for pi in process_intensity]
        rho, p_val = sp_stats.spearmanr(range(len(birth_values)), birth_values)
        as_h_evidence.append({
            "metric": "birth_trend_spearman",
            "rho": _safe_float(rho),
            "p_value": _safe_float(p_val),
            "significant": p_val < 0.05,
        })

    n_met = sum(1 for e in as_h_evidence if e.get("significant"))
    hypotheses["As-H"] = {
        "verdict": "confirmed" if n_met >= 2 and as_h_events > 0 else (
            "partially_supported" if n_met >= 1 and as_h_events > 0 else (
                "not_met" if as_h_events > 0 else "insufficient_data")),
        "events": as_h_events,
        "events_pct": _safe_float(as_h_events / max(total_events, 1) * 100),
        "evidence_met": n_met,
        "evidence_total": len(as_h_evidence),
        "evidence": as_h_evidence,
    }

    # De-H: feature deaths — need actual statistical evidence, not just event count
    de_h_events = sum(pi["de_h"] for pi in process_intensity)
    de_h_evidence = []
    # Evidence 1: death fraction of total events (substantial = >15%)
    death_frac = de_h_events / max(total_events, 1)
    death_frac_sig = death_frac > t["death_fraction_thresh"]
    de_h_evidence.append({
        "metric": "death_fraction",
        "value": _safe_float(death_frac),
        "significant": death_frac_sig,
    })
    # Evidence 2: deaths increase over training (later transitions have more deaths)
    if len(process_intensity) >= 3:
        death_values = [pi["de_h"] for pi in process_intensity]
        rho, p_val = sp_stats.spearmanr(range(len(death_values)), death_values)
        de_h_evidence.append({
            "metric": "death_trend_spearman",
            "rho": _safe_float(rho),
            "p_value": _safe_float(p_val),
            "significant": p_val < 0.05,
        })

    n_met = sum(1 for e in de_h_evidence if e.get("significant"))
    hypotheses["De-H"] = {
        "verdict": "confirmed" if n_met >= 2 and de_h_events > 0 else (
            "partially_supported" if n_met >= 1 and de_h_events > 0 else (
                "not_met" if de_h_events > 0 else "insufficient_data")),
        "events": de_h_events,
        "events_pct": _safe_float(de_h_events / max(total_events, 1) * 100),
        "evidence_met": n_met,
        "evidence_total": len(de_h_evidence),
        "evidence": de_h_evidence,
    }

    # ── Baseline exceedance: compare observed event rates vs control false-event rates ──
    if control_results and feature_matching:
        ctrl_summary = control_results.get("summary", {})
        false_birth_rate = ctrl_summary.get("mean_false_birth_rate", 0.0)
        false_death_rate = ctrl_summary.get("mean_false_death_rate", 0.0)

        # Compute observed birth/death rates from feature matching
        obs_births, obs_deaths, obs_alive = 0, 0, 0
        for pair_key, layer_dict in feature_matching.items():
            for layer_name, match_info in layer_dict.items():
                n_born = match_info.get("n_born", 0)
                n_died = match_info.get("n_died", 0)
                alive_a = match_info.get("alive_a")
                alive_b = match_info.get("alive_b")
                n_alive_a = int(alive_a.sum()) if alive_a is not None else 0
                n_alive_b = int(alive_b.sum()) if alive_b is not None else 0
                obs_births += n_born
                obs_deaths += n_died
                obs_alive += max(n_alive_a, n_alive_b)

        obs_birth_rate = obs_births / max(obs_alive, 1)
        obs_death_rate = obs_deaths / max(obs_alive, 1)

        # Birth-rate exceedance ratio: how many times above baseline
        birth_exceedance = obs_birth_rate / max(false_birth_rate, 1e-6)
        death_exceedance = obs_death_rate / max(false_death_rate, 1e-6)
        exceedance_ratio = t["baseline_exceedance_ratio"]
        birth_exceeds = birth_exceedance > exceedance_ratio
        death_exceeds = death_exceedance > exceedance_ratio

        baseline_evidence = {
            "metric": "baseline_exceedance",
            "obs_birth_rate": _safe_float(obs_birth_rate),
            "false_birth_rate": _safe_float(false_birth_rate),
            "birth_exceedance_ratio": _safe_float(birth_exceedance),
            "obs_death_rate": _safe_float(obs_death_rate),
            "false_death_rate": _safe_float(false_death_rate),
            "death_exceedance_ratio": _safe_float(death_exceedance),
            "birth_exceeds_baseline": birth_exceeds,
            "death_exceeds_baseline": death_exceeds,
        }

        # Gate As-H (driven by births): downgrade if birth rate ≤ 2× baseline
        as_ev = {"metric": "birth_vs_baseline", "significant": birth_exceeds,
                 "obs_rate": _safe_float(obs_birth_rate),
                 "baseline_rate": _safe_float(false_birth_rate),
                 "exceedance": _safe_float(birth_exceedance)}
        hypotheses["As-H"]["evidence"].append(as_ev)
        hypotheses["As-H"]["evidence_total"] += 1
        if birth_exceeds:
            hypotheses["As-H"]["evidence_met"] += 1
        elif hypotheses["As-H"]["verdict"] == "confirmed":
            hypotheses["As-H"]["verdict"] = "partially_supported"

        # Gate De-H (driven by deaths): downgrade if death rate ≤ 2× baseline
        de_ev = {"metric": "death_vs_baseline", "significant": death_exceeds,
                 "obs_rate": _safe_float(obs_death_rate),
                 "baseline_rate": _safe_float(false_death_rate),
                 "exceedance": _safe_float(death_exceedance)}
        hypotheses["De-H"]["evidence"].append(de_ev)
        hypotheses["De-H"]["evidence_total"] += 1
        if death_exceeds:
            hypotheses["De-H"]["evidence_met"] += 1
        # De-H: baseline gate — promote if deaths exceed baseline, downgrade only from confirmed
        if death_exceeds and hypotheses["De-H"]["verdict"] == "not_met":
            hypotheses["De-H"]["verdict"] = "partially_supported"
        elif not death_exceeds and hypotheses["De-H"]["verdict"] == "confirmed":
            hypotheses["De-H"]["verdict"] = "partially_supported"

        # Ab-H, Tg-H, and Di-H: add baseline context (these depend on births too)
        for hyp_id in ["Ab-H", "Tg-H", "Di-H"]:
            bev = {"metric": "birth_vs_baseline", "significant": birth_exceeds,
                   "obs_rate": _safe_float(obs_birth_rate),
                   "baseline_rate": _safe_float(false_birth_rate),
                   "exceedance": _safe_float(birth_exceedance),
                   "note": "Born features drive Ab-H/Tg-H/Di-H labels; baseline context only"}
            hypotheses[hyp_id]["evidence"].append(bev)
            hypotheses[hyp_id]["evidence_total"] += 1
            if birth_exceeds:
                hypotheses[hyp_id]["evidence_met"] += 1

    # ── Null baseline: SSI vs random superclass permutations ──
    null_baseline: Dict[str, Any] = {}
    if null_permutations > 0 and has_hierarchical_structure(superclass_map):
        _progress(cb, 93, f"Null baseline ({null_permutations} permutations)")
        null_baseline = _compute_null_baseline(
            ssi_csi_data, superclass_map, superclass_groups,
            labels, layers, null_permutations,
        )

    # ── Discrimination gradients (global + per-superclass) ──
    discrimination = {"superclass": {}, "fineclass": {}, "per_superclass": {}}
    for label in labels:
        for layer_name in layers:
            entry = ssi_csi_data.get((label, layer_name))
            if entry:
                sc_disc = entry["mean_ssi"]
                fc_disc = entry["mean_csi"]
                discrimination["superclass"].setdefault(layer_name, {})[label] = _safe_float(sc_disc)
                discrimination["fineclass"].setdefault(layer_name, {})[label] = _safe_float(fc_disc)

                # Per-superclass: SSI/CSI of features selective for each SC
                f_ssi = np.array(entry["feature_ssi"])
                f_csi = np.array(entry["feature_csi"])
                f_best_sc = entry["best_superclass"]
                for sc_name in superclass_groups:
                    sc_mask = np.array([bs == sc_name for bs in f_best_sc])
                    if sc_mask.any():
                        sc_entry = discrimination["per_superclass"].setdefault(sc_name, {})
                        sc_layer = sc_entry.setdefault(layer_name, {})
                        sc_layer[label] = {
                            "mean_ssi": _safe_float(float(np.mean(f_ssi[sc_mask]))),
                            "mean_csi": _safe_float(float(np.mean(f_csi[sc_mask]))),
                            "n_features": int(sc_mask.sum()),
                        }

    _progress(cb, 96, "Hypothesis testing complete")
    return {
        "hypotheses": hypotheses,
        "null_baseline": null_baseline,
        "selectivity_evolution": selectivity_evolution,
        "discrimination_gradients": discrimination,
    }


def _compute_null_baseline(
    ssi_csi_data: Dict[Tuple[str, str], Dict[str, Any]],
    superclass_map: Dict[int, str],
    superclass_groups: Dict[str, List[int]],
    labels: List[str],
    layers: List[str],
    n_permutations: int,
) -> Dict[str, Any]:
    """Permutation test: SSI under random class→superclass assignments (vectorized)."""
    all_class_indices = sorted(superclass_map.keys())
    sc_names = sorted(superclass_groups.keys())
    sc_sizes = [len(superclass_groups[sc]) for sc in sc_names]

    per_checkpoint = []

    for label in labels:
        # Observed mean SSI
        observed_ssi_vals = []
        for layer_name in layers:
            entry = ssi_csi_data.get((label, layer_name))
            if entry:
                observed_ssi_vals.append(entry["mean_ssi"])
        observed_mean = float(np.mean(observed_ssi_vals)) if observed_ssi_vals else 0.0

        # Permutation null distribution (vectorized — matches Phase 4b)
        null_ssi_vals = []
        rng = np.random.RandomState(42)
        for _ in range(n_permutations):
            perm = rng.permutation(all_class_indices)
            perm_groups: Dict[str, List[int]] = {}
            offset = 0
            for sc, size in zip(sc_names, sc_sizes):
                perm_groups[sc] = list(perm[offset:offset + size])
                offset += size

            perm_ssi = []
            for layer_name in layers:
                entry = ssi_csi_data.get((label, layer_name))
                if not entry or "feature_magnitudes" not in entry:
                    continue
                M = entry["feature_magnitudes"]
                layer_class_indices = entry.get("class_indices", list(range(M.shape[1])))
                n_features, n_cols = M.shape
                eps = 1e-10
                feat_total = M.sum(axis=1)
                active_mask = feat_total > eps

                class_to_col = {ci: col for col, ci in enumerate(layer_class_indices)}
                sc_mask_list = []
                for sc_class_list in perm_groups.values():
                    mask = np.zeros(n_cols, dtype=bool)
                    for ci in sc_class_list:
                        col = class_to_col.get(ci)
                        if col is not None:
                            mask[col] = True
                    if mask.any():
                        sc_mask_list.append(mask)

                if not sc_mask_list:
                    continue

                sc_masks = np.stack(sc_mask_list)
                n_sc = len(sc_mask_list)

                # Vectorized SSI: normalized max-superclass fraction
                sc_fracs = (M @ sc_masks.T) / (feat_total[:, None] + eps)
                best_sc_frac = sc_fracs.max(axis=1)
                best_ssi = np.clip(
                    (best_sc_frac - 1.0 / n_sc) / (1.0 - 1.0 / n_sc + eps),
                    0.0, 1.0,
                )
                best_ssi = np.where(active_mask, best_ssi, 0.0)

                n_active = int(active_mask.sum())
                perm_mean_ssi = float(best_ssi.sum()) / max(n_active, 1)
                perm_ssi.append(perm_mean_ssi)

            null_mean = float(np.mean(perm_ssi)) if perm_ssi else 0.0
            null_ssi_vals.append(null_mean)

        null_mean_overall = float(np.mean(null_ssi_vals)) if null_ssi_vals else 0.0
        p_value = float(np.mean(np.array(null_ssi_vals) >= observed_mean)) if null_ssi_vals else 1.0

        per_checkpoint.append({
            "checkpoint": label,
            "observed_ssi": _safe_float(observed_mean),
            "null_ssi": _safe_float(null_mean_overall),
            "p_value": _safe_float(p_value),
        })

    return {"per_checkpoint": per_checkpoint, "n_permutations": n_permutations}


# ═══════════════════════════════════════════════════════════════════════════
# Reclassify Entry Point — Reuse saved SAEs & activations, re-run Phases 4b-8
# ═══════════════════════════════════════════════════════════════════════════


def reclassify_from_saved(
    lane_dir: str,
    selected_classes: List[str],
    experiment_config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None,
    expansion_factor: int = 4,
    k_sparse: int = 32,
    progress_file: Optional[str] = None,
    analysis_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Re-run process classification with new thresholds using saved artifacts.

    Skips SAE training (Phase 2) and encoding (Phase 3) by loading activation
    matrices from disk. Re-runs Phase 4 (SSI/CSI), Phase 4b (adaptive thresholds),
    Phase 5 (feature matching), and Phases 6-8 (classification, aggregation,
    hypothesis testing).

    Phase 5c (within-checkpoint control) is loaded from the existing results
    since it is unaffected by threshold changes.

    Args:
        lane_dir: Path to the lane directory (must have sae_analysis/ from a prior run).
        selected_classes: Fine-class names used in this experiment.
        experiment_config: Full experiment/NETINIT configuration dict.
        progress_callback: Optional (pct, msg) callback.
        expansion_factor: SAE hidden layer multiplier (must match original run).
        k_sparse: Top-k sparsity (must match original run).
        progress_file: Optional path to progress file.
        analysis_thresholds: Optional overrides for classification thresholds.

    Returns:
        Dict with all results (also written to sae_results.json).
    """
    global _progress_file
    lane_path = Path(lane_dir)
    cb = progress_callback
    output_dir = lane_path / "sae_analysis"

    if not output_dir.exists():
        return {"error": "No sae_analysis directory found — run full analysis first"}

    if progress_file is not None:
        _progress_file = Path(progress_file)
    else:
        _progress_file = output_dir / "saeanalysis_progress.json"

    # Apply analysis thresholds
    active_thresholds = _apply_thresholds(analysis_thresholds)

    # Extract SAE policy from config
    sae_policy = experiment_config.get("sae_policy", {}) or {}
    if sae_policy.get("expansion_factor"):
        expansion_factor = sae_policy["expansion_factor"]
    if sae_policy.get("k_sparse"):
        k_sparse = sae_policy["k_sparse"]
    null_permutations: int = sae_policy.get("null_permutations", 100)

    # Build hierarchy (resolve correct module based on dataset)
    _build_fn = _resolve_build_superclass_map(experiment_config.get("dataset_id", "cifar100"))
    class_to_idx = {name: i for i, name in enumerate(selected_classes)}
    superclass_map = _build_fn(selected_classes, class_to_idx)
    superclass_groups = get_superclass_groups(superclass_map)
    num_classes = len(selected_classes)

    warnings: List[str] = ["reclassify_from_saved: reusing saved SAEs and activations"]
    if not has_hierarchical_structure(superclass_map):
        warnings.append("No classes share a superclass -- superclass analyses limited.")

    # ── Phase 1: Load raw data (for class boundaries) ──
    _progress(cb, 0, "Loading checkpoint data")
    all_data, predictions, layers, labels = _phase1_load(lane_path, cb)
    if not all_data:
        return {"error": "No multilayer activation data found", "warnings": warnings}
    if not layers:
        return {"error": "No layers discovered", "warnings": warnings}

    _clear_memory()

    # ── Load activation matrices from disk (skip Phase 2+3) ──
    _progress(cb, 15, "Loading saved activation matrices")
    act_matrices: Dict[str, Dict[str, np.ndarray]] = {}
    for label in labels:
        act_matrices[label] = {}
        act_dir = output_dir / "activations" / label
        for layer_name in layers:
            safe_name = _safe_layer_filename(layer_name)
            npy_path = act_dir / f"{safe_name}.npy"
            if npy_path.exists():
                act_matrices[label][layer_name] = np.load(npy_path)
            else:
                warnings.append(f"Missing activation file: {npy_path}")

    if not any(act_matrices[label] for label in labels):
        return {"error": "No saved activation matrices found", "warnings": warnings}

    # ── Load SAE weights from disk (for weight matching) ──
    _progress(cb, 20, "Loading saved SAE weights")
    all_saes: Dict[str, Dict[str, SparseAutoencoder]] = {}
    for label in labels:
        all_saes[label] = {}
        sae_dir = output_dir / "saes" / label
        for layer_name in layers:
            safe_name = _safe_layer_filename(layer_name)
            pt_path = sae_dir / f"{safe_name}.pt"
            if not pt_path.exists():
                continue
            # Determine d_input from activation matrix
            H = act_matrices.get(label, {}).get(layer_name)
            if H is None:
                continue
            state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
            d_input = state_dict["encoder.weight"].shape[1]
            sae = SparseAutoencoder(d_input, expansion_factor, k_sparse)
            sae.load_state_dict(state_dict)
            sae.eval()
            all_saes[label][layer_name] = sae

    _progress(cb, 25, f"Loaded SAEs for {sum(len(v) for v in all_saes.values())} (checkpoint, layer) pairs")

    # ── Load previous control results (Phase 5c — unaffected by thresholds) ──
    prev_results_path = output_dir / "sae_results.json"
    control_results: Optional[Dict[str, Any]] = None
    prev_recon_quality: Dict[str, Any] = {}
    if prev_results_path.exists():
        with open(prev_results_path) as f:
            prev_results = json.load(f)
        control_results = prev_results.get("within_checkpoint_control")
        prev_recon_quality = prev_results.get("reconstruction_quality", {})

    # ── Phase 4: SSI & CSI computation (fast — just means) ──
    ssi_csi_data = _phase4_ssi_csi(
        act_matrices, all_data, superclass_map, superclass_groups,
        labels, layers, cb,
    )

    # ── Phase 4b: Adaptive thresholds from null distribution ──
    adaptive_threshold_info: Optional[Dict[str, Any]] = None
    if (active_thresholds.get("adaptive_thresholds_enabled", 1.0) > 0
            and has_hierarchical_structure(superclass_map)):
        adaptive_threshold_info = _phase4b_adaptive_thresholds(
            ssi_csi_data, act_matrices, all_data,
            superclass_map, superclass_groups,
            labels, layers,
            n_permutations=null_permutations,
            percentile=active_thresholds.get("adaptive_threshold_percentile", 95.0),
            ssi_floor=active_thresholds.get("ssi_adaptive_floor", 0.1),
            csi_floor=active_thresholds.get("csi_adaptive_floor", 0.15),
            sai_floor=active_thresholds.get("sai_adaptive_floor", 0.5),
            cb=cb,
        )
        active_thresholds["ssi_abstraction_thresh"] = adaptive_threshold_info["ssi_adaptive_thresh"]
        active_thresholds["csi_differentiation_thresh"] = adaptive_threshold_info["csi_adaptive_thresh"]
        active_thresholds["sai_task_general_thresh"] = adaptive_threshold_info["sai_adaptive_thresh"]

    _clear_memory()

    # ── Phase 5: Feature matching (fast — correlation + Hungarian) ──
    _progress(cb, 62, "Running activation-column feature matching")
    feature_matching_activation = _phase5_feature_matching(
        act_matrices, all_data, ssi_csi_data, labels, layers, cb,
    )

    _progress(cb, 65, "Running weight-space feature matching")
    feature_matching_weight = _phase5b_weight_matching(
        all_saes, act_matrices, labels, layers, cb,
    )

    _clear_memory()

    # ── Run phases 6-8 for EACH matching method ──
    matching_methods = {
        "activation": {
            "matching": feature_matching_activation,
            "transitions_subdir": "transitions",
        },
        "weight": {
            "matching": feature_matching_weight,
            "transitions_subdir": "transitions_weight",
        },
    }

    method_results: Dict[str, Dict[str, Any]] = {}

    for method_name, method_cfg in matching_methods.items():
        fm = method_cfg["matching"]
        tr_subdir = method_cfg["transitions_subdir"]
        _progress(cb, 70, f"Processing {method_name} matching → classification")

        # Phase 6: Per-sample process classification
        process_events = _phase6_process_classification(
            act_matrices, all_data, fm, ssi_csi_data,
            labels, layers, output_dir, cb,
            transitions_subdir=tr_subdir,
            thresholds=active_thresholds,
            adaptive_threshold_info=adaptive_threshold_info,
        )

        _clear_memory()

        # Phase 7: Aggregation
        aggregation = _phase8_aggregation(
            process_events, act_matrices, all_data, ssi_csi_data,
            superclass_map, superclass_groups, selected_classes,
            labels, layers, cb,
            control_results=control_results,
            feature_matching=fm,
        )

        _clear_memory()

        # Phase 8: Hypothesis testing & null baseline
        hypothesis_results = _phase9_hypothesis_testing(
            aggregation, ssi_csi_data,
            superclass_map, superclass_groups,
            labels, layers, null_permutations, cb,
            control_results=control_results,
            feature_matching=fm,
            thresholds=active_thresholds,
        )

        # Serialise feature matching counts
        fm_serialised: Dict[str, Any] = {}
        for pair_key, layer_dict in fm.items():
            fm_serialised[pair_key] = {}
            for layer_name, match_info in layer_dict.items():
                fm_serialised[pair_key][layer_name] = {
                    "n_stable": match_info["n_stable"],
                    "n_born": match_info["n_born"],
                    "n_died": match_info["n_died"],
                    "n_transformed": match_info.get("n_transformed", 0),
                }

        method_results[method_name] = {
            "feature_matching": fm_serialised,
            "feature_landscape": aggregation["feature_landscape"],
            "class_process_summary": aggregation["class_process_summary"],
            "superclass_summary": aggregation["superclass_summary"],
            "sample_consistency": aggregation["sample_consistency"],
            "process_intensity": aggregation["process_intensity"],
            "hypotheses": hypothesis_results["hypotheses"],
            "null_baseline": hypothesis_results["null_baseline"],
            "selectivity_evolution": hypothesis_results["selectivity_evolution"],
            "discrimination_gradients": hypothesis_results["discrimination_gradients"],
        }

    # ── Serialise selectivity data ──
    selectivity_serialised: Dict[str, Any] = {}
    for (ckpt, layer), val in ssi_csi_data.items():
        serialised_val = {
            k: v for k, v in val.items()
            if k not in ("feature_magnitudes", "class_indices", "best_superclass", "best_class")
        }
        selectivity_serialised.setdefault(ckpt, {})[layer] = serialised_val

    # ── Derive n_samples_per_class ──
    _n_samples_per_class = 0
    if act_matrices and num_classes > 0:
        for _lbl_acts in act_matrices.values():
            for _H_mat in _lbl_acts.values():
                if _H_mat is not None and _H_mat.shape[0] > 0:
                    _n_samples_per_class = _H_mat.shape[0] // num_classes
                    break
            if _n_samples_per_class > 0:
                break

    # ── Assemble results ──
    results = {
        "metadata": {
            "approach": "per_checkpoint_sae",
            "reclassified": True,
            "layers": layers,
            "checkpoint_labels": labels,
            "n_classes": num_classes,
            "n_layers": len(layers),
            "expansion_factor": expansion_factor,
            "k_sparse": k_sparse,
            "n_samples_per_class": _n_samples_per_class,
            "selected_classes": selected_classes,
            "matching_methods": ["activation", "weight"],
            "analysis_thresholds": active_thresholds,
            "adaptive_thresholds": adaptive_threshold_info,
        },
        "warnings": warnings,
        "reconstruction_quality": prev_recon_quality,
        "within_checkpoint_control": control_results,
        "selectivity": selectivity_serialised,
        # Default (activation) results at top level for backward compat
        "feature_matching": method_results["activation"]["feature_matching"],
        "feature_landscape": method_results["activation"]["feature_landscape"],
        "class_process_summary": method_results["activation"]["class_process_summary"],
        "superclass_summary": method_results["activation"]["superclass_summary"],
        "sample_consistency": method_results["activation"]["sample_consistency"],
        "process_intensity": method_results["activation"]["process_intensity"],
        "hypotheses": method_results["activation"]["hypotheses"],
        "null_baseline": method_results["activation"]["null_baseline"],
        "selectivity_evolution": method_results["activation"]["selectivity_evolution"],
        "discrimination_gradients": method_results["activation"]["discrimination_gradients"],
        # Weight matching results under separate key
        "weight_matching": method_results["weight"],
    }

    # Write to disk
    output_file = output_dir / "sae_results.json"
    with open(output_file, "w") as f:
        json.dump(_to_list(results), f, indent=2)

    _progress(cb, 100, f"Reclassification complete — results written to {output_file}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════


def analyze_sae_features(
    lane_dir: str,
    selected_classes: List[str],
    experiment_config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None,
    expansion_factor: int = 4,
    k_sparse: int = 32,
    n_steps: int = 10_000,
    progress_file: Optional[str] = None,
    analysis_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Run the per-checkpoint SAE analysis pipeline.

    Trains a separate SAE per layer per checkpoint, encodes all samples,
    matches features by activation-column correlation, classifies process
    events per sample, and aggregates into class/superclass/global summaries.

    Args:
        lane_dir: Path to the lane directory.
        selected_classes: Fine-class names used in this experiment.
        experiment_config: Full experiment/NETINIT configuration dict.
        progress_callback: Optional (pct, msg) callback.
        expansion_factor: SAE hidden layer multiplier (default 4).
            Conservative vs. literature (8x–256x); see sae.py docstring.
        k_sparse: Top-k sparsity (default 32).
            Directly from Gao et al. 2024 (Figs 15–18).
        n_steps: SAE training steps per layer (default 10,000).
            No direct precedent; adapted for small per-checkpoint datasets.
        progress_file: Optional path to progress file.
        analysis_thresholds: Optional overrides for classification thresholds.
            See DEFAULT_THRESHOLDS for available keys, defaults, and
            literature context for each parameter.

    Returns:
        Dict with all results (also written to sae_results.json).
    """
    global _progress_file
    lane_path = Path(lane_dir)
    cb = progress_callback
    output_dir = lane_path / "sae_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    if progress_file is not None:
        _progress_file = Path(progress_file)
    else:
        _progress_file = output_dir / "saeanalysis_progress.json"

    # Apply analysis thresholds (merge user overrides into defaults)
    active_thresholds = _apply_thresholds(analysis_thresholds)

    # Extract SAE policy from config
    sae_policy = experiment_config.get("sae_policy", {}) or {}
    if sae_policy.get("expansion_factor"):
        expansion_factor = sae_policy["expansion_factor"]
    if sae_policy.get("k_sparse"):
        k_sparse = sae_policy["k_sparse"]
    if sae_policy.get("n_steps"):
        n_steps = sae_policy["n_steps"]
    shared_init_seed: Optional[int] = sae_policy.get("shared_init_seed", 42)
    null_permutations: int = sae_policy.get("null_permutations", 100)

    # Build hierarchy (resolve correct module based on dataset)
    _build_fn = _resolve_build_superclass_map(experiment_config.get("dataset_id", "cifar100"))
    class_to_idx = {name: i for i, name in enumerate(selected_classes)}
    superclass_map = _build_fn(selected_classes, class_to_idx)
    superclass_groups = get_superclass_groups(superclass_map)
    num_classes = len(selected_classes)

    warnings: List[str] = []
    if not has_hierarchical_structure(superclass_map):
        warnings.append("No classes share a superclass -- superclass analyses limited.")
    if k_sparse and k_sparse > 0:
        warnings.append(f"k-sparse SAE with k={k_sparse}: L0 is constant.")

    # ── Phase 1: Load data ──
    all_data, predictions, layers, labels = _phase1_load(lane_path, cb)
    if not all_data:
        _progress(cb, 100, "No multilayer activation data found")
        return {"error": "No multilayer activation data found", "warnings": warnings}
    if not layers:
        _progress(cb, 100, "No layers discovered")
        return {"error": "No layers discovered", "warnings": warnings}

    _clear_memory()

    # ── Phase 2: Train SAEs per checkpoint ──
    all_saes = _phase2_train_per_checkpoint_saes(
        all_data, output_dir, layers, labels,
        expansion_factor, k_sparse, n_steps, cb,
        shared_init_seed=shared_init_seed,
    )
    if not all_saes:
        _progress(cb, 100, "No SAEs trained")
        return {"error": "SAE training failed", "warnings": warnings}

    _clear_memory()

    # ── Phase 3: Encode all samples through per-checkpoint SAEs ──
    act_matrices, recon_quality = _phase3_encode_per_checkpoint(
        all_saes, all_data, labels, layers, output_dir, k_sparse, cb,
    )

    _clear_memory()

    # ── Phase 4: SSI & CSI computation ──
    ssi_csi_data = _phase4_ssi_csi(
        act_matrices, all_data, superclass_map, superclass_groups,
        labels, layers, cb,
    )

    # ── Phase 4b: Adaptive thresholds from null distribution ──
    adaptive_threshold_info: Optional[Dict[str, Any]] = None
    if (active_thresholds.get("adaptive_thresholds_enabled", 1.0) > 0
            and has_hierarchical_structure(superclass_map)):
        adaptive_threshold_info = _phase4b_adaptive_thresholds(
            ssi_csi_data, act_matrices, all_data,
            superclass_map, superclass_groups,
            labels, layers,
            n_permutations=null_permutations,
            percentile=active_thresholds.get("adaptive_threshold_percentile", 95.0),
            ssi_floor=active_thresholds.get("ssi_adaptive_floor", 0.1),
            csi_floor=active_thresholds.get("csi_adaptive_floor", 0.15),
            sai_floor=active_thresholds.get("sai_adaptive_floor", 0.5),
            cb=cb,
        )
        # Override fixed thresholds with adaptive ones
        active_thresholds["ssi_abstraction_thresh"] = adaptive_threshold_info["ssi_adaptive_thresh"]
        active_thresholds["csi_differentiation_thresh"] = adaptive_threshold_info["csi_adaptive_thresh"]
        active_thresholds["sai_task_general_thresh"] = adaptive_threshold_info["sai_adaptive_thresh"]

    _clear_memory()

    # ── Phase 5: Feature matching — BOTH methods ──
    _progress(cb, 58, "Running activation-column feature matching")
    feature_matching_activation = _phase5_feature_matching(
        act_matrices, all_data, ssi_csi_data, labels, layers, cb,
    )

    _progress(cb, 58, "Running weight-space feature matching")
    feature_matching_weight = _phase5b_weight_matching(
        all_saes, act_matrices, labels, layers, cb,
    )

    _clear_memory()

    # ── Phase 5c: Within-checkpoint SAE stability control ──
    control_results = _phase5c_within_checkpoint_control(
        all_data, act_matrices, labels, layers,
        expansion_factor, k_sparse, n_steps, cb,
    )

    _clear_memory()

    # ── Run phases 6-9 for EACH matching method ──
    matching_methods = {
        "activation": {
            "matching": feature_matching_activation,
            "transitions_subdir": "transitions",
        },
        "weight": {
            "matching": feature_matching_weight,
            "transitions_subdir": "transitions_weight",
        },
    }

    method_results: Dict[str, Dict[str, Any]] = {}

    for method_name, method_cfg in matching_methods.items():
        fm = method_cfg["matching"]
        tr_subdir = method_cfg["transitions_subdir"]
        _progress(cb, 70, f"Processing {method_name} matching → classification")

        # Phase 6: Per-sample process classification
        process_events = _phase6_process_classification(
            act_matrices, all_data, fm, ssi_csi_data,
            labels, layers, output_dir, cb,
            transitions_subdir=tr_subdir,
            thresholds=active_thresholds,
            adaptive_threshold_info=adaptive_threshold_info,
        )

        _clear_memory()

        # Phase 7: Aggregation
        aggregation = _phase8_aggregation(
            process_events, act_matrices, all_data, ssi_csi_data,
            superclass_map, superclass_groups, selected_classes,
            labels, layers, cb,
            control_results=control_results,
            feature_matching=fm,
        )

        _clear_memory()

        # Phase 8: Hypothesis testing & null baseline
        hypothesis_results = _phase9_hypothesis_testing(
            aggregation, ssi_csi_data,
            superclass_map, superclass_groups,
            labels, layers, null_permutations, cb,
            control_results=control_results,
            feature_matching=fm,
            thresholds=active_thresholds,
        )

        # Serialise feature matching counts
        fm_serialised: Dict[str, Any] = {}
        for pair_key, layer_dict in fm.items():
            fm_serialised[pair_key] = {}
            for layer_name, match_info in layer_dict.items():
                fm_serialised[pair_key][layer_name] = {
                    "n_stable": match_info["n_stable"],
                    "n_born": match_info["n_born"],
                    "n_died": match_info["n_died"],
                    "n_transformed": match_info.get("n_transformed", 0),
                }

        method_results[method_name] = {
            "feature_matching": fm_serialised,
            "feature_landscape": aggregation["feature_landscape"],
            "class_process_summary": aggregation["class_process_summary"],
            "superclass_summary": aggregation["superclass_summary"],
            "sample_consistency": aggregation["sample_consistency"],
            "process_intensity": aggregation["process_intensity"],
            "hypotheses": hypothesis_results["hypotheses"],
            "null_baseline": hypothesis_results["null_baseline"],
            "selectivity_evolution": hypothesis_results["selectivity_evolution"],
            "discrimination_gradients": hypothesis_results["discrimination_gradients"],
        }

    # ── Serialise selectivity data (shared) ──
    selectivity_serialised: Dict[str, Any] = {}
    for (ckpt, layer), val in ssi_csi_data.items():
        serialised_val = {
            k: v for k, v in val.items()
            if k not in ("feature_magnitudes", "class_indices", "best_superclass", "best_class")
        }
        selectivity_serialised.setdefault(ckpt, {})[layer] = serialised_val

    # ── Derive n_samples_per_class from activation matrices ──
    _n_samples_per_class = 0
    if act_matrices and num_classes > 0:
        for _lbl_acts in act_matrices.values():
            for _H_mat in _lbl_acts.values():
                if _H_mat is not None and _H_mat.shape[0] > 0:
                    _n_samples_per_class = _H_mat.shape[0] // num_classes
                    break
            if _n_samples_per_class > 0:
                break

    # ── Assemble results — activation as default, weight as secondary ──
    results = {
        "metadata": {
            "approach": "per_checkpoint_sae",
            "layers": layers,
            "checkpoint_labels": labels,
            "n_classes": num_classes,
            "n_layers": len(layers),
            "expansion_factor": expansion_factor,
            "k_sparse": k_sparse,
            "n_steps": n_steps,
            "n_samples_per_class": _n_samples_per_class,
            "selected_classes": selected_classes,
            "matching_methods": ["activation", "weight"],
            "analysis_thresholds": active_thresholds,
            "adaptive_thresholds": adaptive_threshold_info,
        },
        "warnings": warnings,
        "reconstruction_quality": recon_quality,
        "within_checkpoint_control": _to_list(control_results),
        "selectivity": selectivity_serialised,
        # Default (activation) results at top level for backward compat
        "feature_matching": method_results["activation"]["feature_matching"],
        "feature_landscape": method_results["activation"]["feature_landscape"],
        "class_process_summary": method_results["activation"]["class_process_summary"],
        "superclass_summary": method_results["activation"]["superclass_summary"],
        "sample_consistency": method_results["activation"]["sample_consistency"],
        "process_intensity": method_results["activation"]["process_intensity"],
        "hypotheses": method_results["activation"]["hypotheses"],
        "null_baseline": method_results["activation"]["null_baseline"],
        "selectivity_evolution": method_results["activation"]["selectivity_evolution"],
        "discrimination_gradients": method_results["activation"]["discrimination_gradients"],
        # Weight matching results under separate key
        "weight_matching": method_results["weight"],
    }

    # Write to disk
    output_file = output_dir / "sae_results.json"
    with open(output_file, "w") as f:
        json.dump(_to_list(results), f, indent=2)

    _progress(cb, 100, f"Results written to {output_file}")
    return results
