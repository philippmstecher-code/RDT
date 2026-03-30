"""
PATHWAYS -- Causal pathway analysis over SAE features.

Implements edge attribution patching (EAP) to establish that SAE features
identified by SAEANALYSIS play genuine causal roles in the network's
computation.  This provides "Level 2: Mechanistic" evidence complementing
the "Level 1: Representational" evidence from SAE feature tracking.

Two analyses (run per checkpoint):

  1. Feature Attribution   -- per-SAE-feature causal importance scores
     (how much each feature contributes to correct classification).
  2. Class-Conditional Attribution -- which features are causally important
     for each class (connects to SSI/CSI from SAEANALYSIS).

Method: Attribution patching (Nanda 2023; Marks et al., ICLR 2025)
  Score(f) = (h_clean[f] - h_corrupt[f]) . dL/dh[f] |_{corrupt}
  Requires 2 forward passes + 1 backward pass per batch.

References:
  Marks, Rager, Michaud et al.  "Sparse Feature Circuits" (ICLR 2025)
  Nanda  "Attribution Patching" (2023)
  Syed, Rager, Conmy  "Attribution Patching Outperforms ACDC" (EMNLP 2024)

Output -> {lane_dir}/sae_analysis/pathway_results.json
"""

import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sae import SparseAutoencoder


# ---- helpers ---------------------------------------------------------------


def _progress(cb: Optional[Callable], pct: float, msg: str) -> None:
    if cb is not None:
        cb(pct, msg)


def _clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _safe_float(x: Any) -> float:
    v = float(x)
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v


def _safe_layer_filename(layer_name: str) -> str:
    return layer_name.replace(".", "_")


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---- layer hook discovery --------------------------------------------------


def _get_layer_hooks(model: nn.Module) -> List[str]:
    """
    Identify hookable layers, matching the pattern in training.py.
    """
    hooks: List[str] = []

    if hasattr(model, "tokenizer") and hasattr(model, "transformer"):  # CCT
        for i, _ in enumerate(model.transformer.layers):
            hooks.append(f"transformer.layers.{i}")
        if hasattr(model, "norm"):
            hooks.append("norm")

    elif hasattr(model, "fc") and not hasattr(model, "tokenizer"):  # ResNet-like
        for name, _ in model.named_modules():
            if "layer" in name and name.count(".") == 0:
                hooks.append(name)
        hooks.append("avgpool")

    elif hasattr(model, "classifier") and isinstance(
        model.classifier, nn.Sequential
    ):  # VGG
        for i, module in enumerate(model.features):
            if isinstance(module, nn.MaxPool2d):
                hooks.append(f"features.{i}")

    elif hasattr(model, "heads"):  # ViT
        for i, _ in enumerate(model.encoder.layers):
            hooks.append(f"encoder.layers.{i}")
        hooks.append("encoder.ln")

    else:
        for name, module in model.named_modules():
            if isinstance(
                module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d)
            ):
                hooks.append(name)

    return hooks


def _resolve_module(model: nn.Module, layer_path: str) -> nn.Module:
    """Walk dotted path to resolve a submodule."""
    module = model
    for part in layer_path.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


# ---- SAE intervention hook -------------------------------------------------


class _SAEHook:
    """
    Forward hook that extracts SAE feature activations for attribution.

    Matches the extraction method used in SAEANALYSIS:
      - CNN (4-D: B×C×H×W): global average pool -> (B, C)
      - ViT (3-D: B×seq×D): CLS token [:, 0, :] -> (B, D)
      - Already 2-D: use as-is

    When record_grad=False (clean pass): records features without modifying
    the computation graph.

    When record_grad=True (corrupt pass): records features WITH gradients.
    The pooled representation is encoded, grad is retained, then decoded
    back and added to the original output as a zero-residual branch so
    that gradients flow through h while the forward output is unchanged
    (decode(encode(x)) - x ≈ 0 for well-trained SAEs, and even if not,
    the attribution formula uses the gradient direction, not magnitude).
    """

    def __init__(self, sae: SparseAutoencoder, *, record_grad: bool = False):
        self.sae = sae
        self.record_grad = record_grad
        self.features: Optional[torch.Tensor] = None

    def _pool(self, output: torch.Tensor) -> torch.Tensor:
        """Pool/extract to match SAEANALYSIS activation extraction."""
        if output.dim() == 4:
            # CNN: B×C×H×W -> global average pool -> B×C
            return output.mean(dim=(2, 3))
        elif output.dim() == 3:
            # ViT/Transformer: B×seq×D -> CLS token -> B×D
            return output[:, 0, :].contiguous()
        return output

    def __call__(
        self,
        module: nn.Module,
        _input: Any,
        output: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self._pool(output)

        # Skip if dimension doesn't match SAE
        if pooled.shape[-1] != self.sae.d_input:
            self.features = None
            return output

        h = self.sae.encode(pooled)
        if self.record_grad:
            h.retain_grad()
        self.features = h

        if self.record_grad:
            # Create a gradient path: add a zero-residual branch through
            # the SAE so that loss gradients flow back to h.
            # residual = decode(h) - pooled.detach()  (≈ 0 for good SAE)
            reconstructed = self.sae.decode(h)
            residual = reconstructed - pooled.detach()
            # Broadcast residual back into original shape
            if output.dim() == 4:
                # B×C -> B×C×1×1, broadcast to B×C×H×W
                return output + residual.unsqueeze(-1).unsqueeze(-1)
            elif output.dim() == 3:
                # B×D -> add only to CLS token position
                delta = torch.zeros_like(output)
                delta[:, 0, :] = residual
                return output + delta
            else:
                return output + residual

        return output

    def reset(self) -> None:
        self.features = None


# ---- core attribution patching ---------------------------------------------


def compute_feature_attributions(
    model: nn.Module,
    saes: Dict[str, SparseAutoencoder],
    data_loader: DataLoader,
    device: str,
    num_classes: int,
    max_batches: int = 50,
    cb: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Per-SAE-feature causal attribution scores via attribution patching.

    Uses zero ablation as baseline (corrupt input = zero tensor).  For
    normalised datasets this corresponds to the dataset-mean image.

    Returns:
        feature_scores:   {layer: list[float]}         mean |attribution|
        class_scores:     {layer: {class_str: [float]}} per-class attribution
        top_features:     {layer: [(idx, score, best_class), ...]}
        causal_coverage:  {layer: float}  top-10 % share of total attribution
    """
    model.eval()
    model.to(device)
    for sae in saes.values():
        sae.to(device)
        sae.eval()

    layer_names = sorted(saes.keys())
    if not layer_names:
        return {}

    # -- accumulators --------------------------------------------------------
    feat_attr_sum: Dict[str, np.ndarray] = {}
    class_attr_sum: Dict[str, Dict[int, np.ndarray]] = {}
    n_samples = 0
    n_class_samples: Dict[int, int] = {c: 0 for c in range(num_classes)}

    for layer in layer_names:
        nf = saes[layer].d_hidden
        feat_attr_sum[layer] = np.zeros(nf, dtype=np.float64)
        class_attr_sum[layer] = {
            c: np.zeros(nf, dtype=np.float64) for c in range(num_classes)
        }

    total_batches = min(max_batches, len(data_loader))

    for batch_idx, (images, labels) in enumerate(data_loader):
        if batch_idx >= max_batches:
            break
        if cb and batch_idx % max(1, total_batches // 10) == 0:
            _progress(
                cb,
                batch_idx / total_batches * 100,
                f"Attribution batch {batch_idx + 1}/{total_batches}",
            )

        images = images.to(device)
        labels = labels.to(device)
        bsz = images.size(0)

        # -- 1. Clean forward (SAE inserted, no grad) -----------------------
        clean_hooks: Dict[str, _SAEHook] = {}
        handles: List[Any] = []
        for ln in layer_names:
            hk = _SAEHook(saes[ln], record_grad=False)
            clean_hooks[ln] = hk
            handles.append(
                _resolve_module(model, ln).register_forward_hook(hk)
            )

        with torch.no_grad():
            model(images)

        h_clean: Dict[str, torch.Tensor] = {}
        for ln, hk in clean_hooks.items():
            if hk.features is not None:
                h_clean[ln] = hk.features.detach().clone()
        for h in handles:
            h.remove()

        # -- 2. Corrupt forward (SAE inserted, WITH grad) -------------------
        corrupt_input = torch.zeros_like(images)

        corrupt_hooks: Dict[str, _SAEHook] = {}
        handles = []
        for ln in layer_names:
            hk = _SAEHook(saes[ln], record_grad=True)
            corrupt_hooks[ln] = hk
            handles.append(
                _resolve_module(model, ln).register_forward_hook(hk)
            )

        logits_corrupt = model(corrupt_input)
        loss = F.cross_entropy(logits_corrupt, labels)
        loss.backward()

        # -- 3. Score = (h_clean - h_corrupt) * grad -------------------------
        for ln in layer_names:
            if ln not in h_clean:
                continue
            hk = corrupt_hooks[ln]
            if hk.features is None or hk.features.grad is None:
                continue

            h_c = h_clean[ln]
            h_corr = hk.features.detach()
            grad = hk.features.grad.detach()

            attr = ((h_c - h_corr) * grad).cpu().numpy()  # (B, nf)

            feat_attr_sum[ln] += np.abs(attr).sum(axis=0)

            labs_np = labels.cpu().numpy()
            for i in range(bsz):
                ci = int(labs_np[i])
                class_attr_sum[ln][ci] += np.abs(attr[i])
                if ln == layer_names[0]:
                    n_class_samples[ci] += 1

        for h in handles:
            h.remove()
        n_samples += bsz
        model.zero_grad()

    # -- 4. Normalise and assemble -------------------------------------------
    feature_scores: Dict[str, List[float]] = {}
    class_scores: Dict[str, Dict[str, List[float]]] = {}
    top_features: Dict[str, List[Any]] = {}
    causal_coverage: Dict[str, float] = {}

    for ln in layer_names:
        nf = saes[ln].d_hidden
        if n_samples == 0:
            feature_scores[ln] = [0.0] * nf
            class_scores[ln] = {}
            top_features[ln] = []
            causal_coverage[ln] = 0.0
            continue

        mean_attr = feat_attr_sum[ln] / max(n_samples, 1)
        feature_scores[ln] = [_safe_float(v) for v in mean_attr]

        cs: Dict[str, List[float]] = {}
        for ci in range(num_classes):
            nc = n_class_samples.get(ci, 0)
            if nc > 0:
                cs[str(ci)] = [
                    _safe_float(v)
                    for v in class_attr_sum[ln][ci] / nc
                ]
        class_scores[ln] = cs

        ranked = sorted(enumerate(mean_attr), key=lambda x: x[1], reverse=True)
        top_list = []
        for fi, sc in ranked[:50]:
            best_c, best_v = 0, 0.0
            for ci, arr in class_attr_sum[ln].items():
                nc = n_class_samples.get(ci, 1)
                v = arr[fi] / max(nc, 1)
                if v > best_v:
                    best_v = v
                    best_c = ci
            top_list.append((fi, _safe_float(sc), best_c))
        top_features[ln] = top_list

        total_attr = mean_attr.sum()
        if total_attr > 0:
            n_top = max(1, nf // 10)
            top_attr = sum(s for _, s in ranked[:n_top])
            causal_coverage[ln] = _safe_float(top_attr / total_attr)
        else:
            causal_coverage[ln] = 0.0

    for sae in saes.values():
        sae.cpu()
    _clear_memory()

    return {
        "feature_scores": feature_scores,
        "class_scores": class_scores,
        "top_features": top_features,
        "causal_coverage": causal_coverage,
        "n_samples": n_samples,
    }


# ---- per-sample cross-layer edge attribution ------------------------------


def compute_sample_causal_edges(
    model: nn.Module,
    saes: Dict[str, SparseAutoencoder],
    sample_image: torch.Tensor,  # (1, C, H, W) single image
    sample_label: int,
    device: str,
) -> Dict[str, Any]:
    """
    Per-sample cross-layer edge attribution for active SAE features.

    For each pair of adjacent hookable layers (L_n, L_{n+1}):
      1. Run clean forward pass with SAE hooks to get h_clean per layer.
      2. Run corrupt forward pass (zero input) with grad to get per-feature
         attribution scores via attribution patching.
      3. Approximate cross-layer edge scores as the product of individual
         per-feature attributions:
           edge_score[f_j -> f_k] = attr_n[f_j] * attr_{n+1}[f_k]
         This first-order approximation avoids expensive partial forward
         passes while capturing pairwise causal importance.
      4. Only retain edges where both features are active (non-zero in
         h_clean) and both have positive attribution.  Return top ~50
         edges per layer pair by score magnitude.

    Args:
        model:        The classifier (eval mode assumed).
        saes:         Dict mapping layer name -> trained SparseAutoencoder.
        sample_image: Single image tensor of shape (1, C, H, W).
        sample_label: Integer class label for the sample.
        device:       Device string ("cpu", "cuda", "mps").

    Returns:
        {
            "edges": [
                {
                    "from_layer": str,
                    "to_layer": str,
                    "edge_scores": [[from_feat_idx, to_feat_idx, score], ...],
                    "from_active_features": [idx, ...],
                    "to_active_features": [idx, ...],
                },
                ...
            ],
            "feature_attributions": {layer: [[idx, score], ...]},
            "computation_time_ms": float,
        }
    """
    t_start = time.time()

    model.eval()
    model.to(device)
    for sae in saes.values():
        sae.to(device)
        sae.eval()

    # Determine ordered layers that have matching SAEs
    all_hook_layers = _get_layer_hooks(model)
    layer_names = [ln for ln in all_hook_layers if ln in saes]

    if not layer_names:
        return {
            "edges": [],
            "feature_attributions": {},
            "computation_time_ms": _safe_float((time.time() - t_start) * 1000),
        }

    image = sample_image.to(device)
    label = torch.tensor([sample_label], dtype=torch.long, device=device)

    # ---- 1. Clean forward pass (SAE inserted, no grad) ---------------------
    clean_hooks: Dict[str, _SAEHook] = {}
    handles: List[Any] = []
    for ln in layer_names:
        hk = _SAEHook(saes[ln], record_grad=False)
        clean_hooks[ln] = hk
        handles.append(
            _resolve_module(model, ln).register_forward_hook(hk)
        )

    with torch.no_grad():
        model(image)

    h_clean: Dict[str, torch.Tensor] = {}
    for ln, hk in clean_hooks.items():
        if hk.features is not None:
            h_clean[ln] = hk.features.detach().clone()
    for h in handles:
        h.remove()

    # ---- 2. Corrupt forward pass (zero input, with grad) -------------------
    corrupt_input = torch.zeros_like(image)

    corrupt_hooks: Dict[str, _SAEHook] = {}
    handles = []
    for ln in layer_names:
        hk = _SAEHook(saes[ln], record_grad=True)
        corrupt_hooks[ln] = hk
        handles.append(
            _resolve_module(model, ln).register_forward_hook(hk)
        )

    logits_corrupt = model(corrupt_input)
    loss = F.cross_entropy(logits_corrupt, label)
    loss.backward()

    # ---- 3. Per-feature attribution: (h_clean - h_corrupt) * grad ----------
    per_layer_attr: Dict[str, torch.Tensor] = {}  # layer -> (nf,) abs attr
    for ln in layer_names:
        if ln not in h_clean:
            continue
        hk = corrupt_hooks[ln]
        if hk.features is None or hk.features.grad is None:
            continue

        h_c = h_clean[ln]           # (1, nf)
        h_corr = hk.features.detach()  # (1, nf)
        grad = hk.features.grad.detach()  # (1, nf)

        attr = ((h_c - h_corr) * grad).abs().squeeze(0)  # (nf,)
        per_layer_attr[ln] = attr

    for h in handles:
        h.remove()
    model.zero_grad()

    # ---- 4. Build feature_attributions output (top features per layer) ------
    feature_attributions: Dict[str, List[List[Any]]] = {}
    for ln in layer_names:
        if ln not in per_layer_attr:
            feature_attributions[ln] = []
            continue

        attr = per_layer_attr[ln]  # (nf,)
        # Get non-zero attribution indices, sorted by score descending
        nonzero_mask = attr > 0
        if not nonzero_mask.any():
            feature_attributions[ln] = []
            continue

        indices = nonzero_mask.nonzero(as_tuple=False).squeeze(-1)
        scores = attr[indices]
        sorted_order = scores.argsort(descending=True)
        indices = indices[sorted_order]
        scores = scores[sorted_order]

        feature_attributions[ln] = [
            [int(idx.item()), _safe_float(sc.item())]
            for idx, sc in zip(indices, scores)
        ]

    # ---- 5. Cross-layer edge scores ----------------------------------------
    edges: List[Dict[str, Any]] = []

    # Only consider layers that have both h_clean and attributions
    valid_layers = [ln for ln in layer_names if ln in h_clean and ln in per_layer_attr]

    for i in range(len(valid_layers) - 1):
        layer_n = valid_layers[i]
        layer_np1 = valid_layers[i + 1]

        h_n = h_clean[layer_n].squeeze(0)       # (nf_n,)
        h_np1 = h_clean[layer_np1].squeeze(0)   # (nf_np1,)
        attr_n = per_layer_attr[layer_n]         # (nf_n,)
        attr_np1 = per_layer_attr[layer_np1]     # (nf_np1,)

        # Active features: non-zero in h_clean (SAE uses k-sparse, so
        # exactly k features are non-zero after encoding)
        active_n = h_n.nonzero(as_tuple=False).squeeze(-1)     # (k_n,)
        active_np1 = h_np1.nonzero(as_tuple=False).squeeze(-1) # (k_np1,)

        if active_n.numel() == 0 or active_np1.numel() == 0:
            edges.append({
                "from_layer": layer_n,
                "to_layer": layer_np1,
                "edge_scores": [],
                "from_active_features": active_n.tolist(),
                "to_active_features": active_np1.tolist(),
            })
            continue

        # Filter to active features that also have positive attribution
        attr_n_active = attr_n[active_n]           # (k_n,)
        attr_np1_active = attr_np1[active_np1]     # (k_np1,)

        pos_mask_n = attr_n_active > 0
        pos_mask_np1 = attr_np1_active > 0

        pos_indices_n = active_n[pos_mask_n]         # feature indices in layer_n
        pos_scores_n = attr_n_active[pos_mask_n]     # their attribution scores
        pos_indices_np1 = active_np1[pos_mask_np1]   # feature indices in layer_{n+1}
        pos_scores_np1 = attr_np1_active[pos_mask_np1]

        if pos_indices_n.numel() == 0 or pos_indices_np1.numel() == 0:
            edges.append({
                "from_layer": layer_n,
                "to_layer": layer_np1,
                "edge_scores": [],
                "from_active_features": active_n.tolist(),
                "to_active_features": active_np1.tolist(),
            })
            continue

        # Compute pairwise edge scores via outer product of attributions
        # edge_score[j, k] = attr_n[j] * attr_{n+1}[k]
        edge_matrix = pos_scores_n.unsqueeze(1) * pos_scores_np1.unsqueeze(0)
        # shape: (|pos_n|, |pos_np1|)

        # Flatten, sort, and take top 50
        n_from = pos_indices_n.numel()
        n_to = pos_indices_np1.numel()
        flat_scores = edge_matrix.reshape(-1)
        n_edges = min(50, flat_scores.numel())
        top_vals, top_flat_idx = flat_scores.topk(n_edges, sorted=True)

        edge_scores_list: List[List[Any]] = []
        for rank in range(n_edges):
            fi = int(top_flat_idx[rank].item() // n_to)
            ti = int(top_flat_idx[rank].item() % n_to)
            from_feat = int(pos_indices_n[fi].item())
            to_feat = int(pos_indices_np1[ti].item())
            score = _safe_float(top_vals[rank].item())
            if score > 0:
                edge_scores_list.append([from_feat, to_feat, score])

        edges.append({
            "from_layer": layer_n,
            "to_layer": layer_np1,
            "edge_scores": edge_scores_list,
            "from_active_features": active_n.tolist(),
            "to_active_features": active_np1.tolist(),
        })

    # ---- cleanup -----------------------------------------------------------
    for sae in saes.values():
        sae.cpu()
    _clear_memory()

    computation_time_ms = _safe_float((time.time() - t_start) * 1000)

    return {
        "edges": edges,
        "feature_attributions": feature_attributions,
        "computation_time_ms": computation_time_ms,
    }


# ---- SSI/CSI vs causal agreement ------------------------------------------


def compute_sae_causal_agreement(
    feature_scores: Dict[str, List[float]],
    class_scores: Dict[str, Dict[str, List[float]]],
    ssi_csi_data: Dict,
    checkpoint_label: str,
    layers: List[str],
) -> Dict[str, Any]:
    """
    Rank-correlation between SAE selectivity metrics (SSI, CSI) and causal
    attribution scores.

    Returns ssi/csi correlations and top-feature overlap fractions.
    """
    from scipy.stats import spearmanr

    ssi_corrs: Dict[str, float] = {}
    csi_corrs: Dict[str, float] = {}
    ssi_causal_overlap: Dict[str, float] = {}
    causal_ssi_overlap: Dict[str, float] = {}

    for layer in layers:
        if layer not in feature_scores:
            continue

        sel_key = (checkpoint_label, layer)
        sel = ssi_csi_data.get(sel_key) if isinstance(ssi_csi_data, dict) else None
        if sel is None:
            continue

        ssi = np.array(sel.get("feature_ssi", []))
        csi = np.array(sel.get("feature_csi", []))
        causal = np.array(feature_scores[layer])

        n = min(len(ssi), len(causal))
        if n < 5:
            continue
        ssi, csi, causal = ssi[:n], csi[:n], causal[:n]

        try:
            r, _ = spearmanr(ssi, causal)
            ssi_corrs[layer] = _safe_float(r)
        except Exception:
            ssi_corrs[layer] = 0.0
        try:
            r, _ = spearmanr(csi, causal)
            csi_corrs[layer] = _safe_float(r)
        except Exception:
            csi_corrs[layer] = 0.0

        n_top = max(1, n // 10)
        top_ssi = set(np.argsort(ssi)[-n_top:])
        top_cau = set(np.argsort(causal)[-n_top:])
        inter = top_ssi & top_cau
        ssi_causal_overlap[layer] = _safe_float(len(inter) / max(len(top_ssi), 1))
        causal_ssi_overlap[layer] = _safe_float(len(inter) / max(len(top_cau), 1))

    return {
        "ssi_causal_correlation": ssi_corrs,
        "csi_causal_correlation": csi_corrs,
        "high_ssi_causal_overlap": ssi_causal_overlap,
        "high_causal_ssi_overlap": causal_ssi_overlap,
    }


# ---- hypothesis evidence generation ---------------------------------------


def generate_pathway_evidence(
    attributions: Dict[str, Any],
    agreement: Dict[str, Any],
    layers: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Produce hypothesis evidence items from pathway analysis, compatible
    with the SAEANALYSIS Phase-8 evidence format.
    """
    evidence: Dict[str, Dict[str, Any]] = {}

    # Ab-H: superclass-selective features are causally important
    ssi_corrs = agreement.get("ssi_causal_correlation", {})
    overlaps = agreement.get("high_ssi_causal_overlap", {})
    if ssi_corrs:
        mc = float(np.mean(list(ssi_corrs.values())))
        mo = float(np.mean(list(overlaps.values()))) if overlaps else 0.0
        evidence["Ab-H"] = {
            "name": "ssi_causal_agreement",
            "met": mc > 0.2 and mo > 0.15,
            "p_value": None,
            "effect_size": _safe_float(mc),
            "significant_bonferroni": False,
            "details": {
                "mean_ssi_causal_correlation": _safe_float(mc),
                "mean_overlap": _safe_float(mo),
                "per_layer": {k: _safe_float(v) for k, v in ssi_corrs.items()},
                "interpretation": (
                    "Positive correlation means SSI-identified superclass-"
                    "selective features genuinely drive classification."
                ),
            },
        }

    # Di-H: fine-class selective features are causally class-specific
    csi_corrs = agreement.get("csi_causal_correlation", {})
    if csi_corrs:
        mc = float(np.mean(list(csi_corrs.values())))
        evidence["Di-H"] = {
            "name": "csi_causal_agreement",
            "met": mc > 0.2,
            "p_value": None,
            "effect_size": _safe_float(mc),
            "significant_bonferroni": False,
            "details": {
                "mean_csi_causal_correlation": _safe_float(mc),
                "per_layer": {k: _safe_float(v) for k, v in csi_corrs.items()},
                "interpretation": (
                    "Positive CSI-causal correlation means fine-class "
                    "selective features genuinely drive class-specific "
                    "decisions."
                ),
            },
        }

    # As-H: causal influence is concentrated (assembled representations)
    cov = attributions.get("causal_coverage", {})
    if cov:
        mc = float(np.mean(list(cov.values())))
        evidence["As-H"] = {
            "name": "causal_concentration",
            "met": mc > 0.5,
            "p_value": None,
            "effect_size": _safe_float(mc),
            "significant_bonferroni": False,
            "details": {
                "mean_top10pct_coverage": _safe_float(mc),
                "per_layer": {k: _safe_float(v) for k, v in cov.items()},
                "interpretation": (
                    "High concentration means a small feature set "
                    "carries most causal influence, consistent with "
                    "assembly into integrated representations."
                ),
            },
        }

    # De-H: sparse causal influence spread (potential decay signature)
    # If causal coverage is LOW, it means features are not well-assembled
    # which could indicate degraded representations. This is weak evidence
    # but contributes to the overall picture.
    if cov:
        mc = float(np.mean(list(cov.values())))
        if mc < 0.3:
            evidence["De-H"] = {
                "name": "causal_diffusion",
                "met": True,
                "p_value": None,
                "effect_size": _safe_float(1.0 - mc),
                "significant_bonferroni": False,
                "details": {
                    "mean_top10pct_coverage": _safe_float(mc),
                    "per_layer": {k: _safe_float(v) for k, v in cov.items()},
                    "interpretation": (
                        "Low causal concentration suggests features are "
                        "not well-integrated, possibly reflecting representational "
                        "degradation."
                    ),
                },
            }

    return evidence


# ===========================================================================
# Top-level orchestrator
# ===========================================================================


def run_pathway_analysis(
    lane_dir: str,
    selected_classes: List[str],
    experiment_config: Dict[str, Any],
    ssi_csi_data: Optional[Dict] = None,
    checkpoint_labels: Optional[List[str]] = None,
    max_batches: int = 50,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    Run the full pathway analysis pipeline.

    Loads model checkpoints and trained SAEs from SAEANALYSIS output,
    creates a data loader, and runs attribution patching at selected
    checkpoints.

    Args:
        lane_dir:           Path to the lane directory.
        selected_classes:   Fine-class names used in this experiment.
        experiment_config:  Dict; must include 'netinit' with 'network_type'.
        ssi_csi_data:       SSI/CSI data from SAEANALYSIS phase 4 keyed by
                            (checkpoint_label, layer_name).  If None the
                            agreement analysis is skipped.
        checkpoint_labels:  Which checkpoints to analyse.  Default: terminal.
        max_batches:        Max batches per checkpoint (controls runtime).
        progress_callback:  Optional (pct, msg) callback.

    Returns:
        Dict with all pathway results (also written to pathway_results.json).
    """
    from models import create_network
    from training import get_dataset, RemappedDataset

    cb = progress_callback
    lane_path = Path(lane_dir)
    sae_dir = lane_path / "sae_analysis"
    device = _get_device()
    num_classes = len(selected_classes)

    # -- resolve config ------------------------------------------------------
    netinit = experiment_config.get("netinit", {})
    if hasattr(netinit, "network_type"):
        # Pydantic object
        network_type = netinit.network_type
        batch_size = getattr(netinit, "batch_size", 64)
        experiment_id = getattr(netinit, "experiment_id", None)
        transform_config = None
        if hasattr(netinit, "transform"):
            tc = netinit.transform
            transform_config = {
                "resize": getattr(tc, "resize", None),
                "center_crop": getattr(tc, "center_crop", None),
                "normalize_mean": getattr(tc, "normalize_mean", [0.5, 0.5, 0.5]),
                "normalize_std": getattr(tc, "normalize_std", [0.5, 0.5, 0.5]),
            }
    else:
        # Plain dict
        network_type = netinit.get("network_type", "resnet18")
        batch_size = netinit.get("batch_size", 64)
        experiment_id = netinit.get("experiment_id")
        transform_config = netinit.get("transform")

    if experiment_id is None:
        experiment_id = experiment_config.get("experiment_id")
    dataset_id = experiment_config.get("dataset_id", "cifar100")

    _progress(cb, 0, "Loading model and dataset")

    # -- data loader ---------------------------------------------------------
    test_dataset, _ = get_dataset(
        dataset_id=dataset_id,
        selected_classes=selected_classes,
        train=False,
        experiment_id=experiment_id,
        transform_config=transform_config,
    )
    data_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    # -- read SAE results metadata ------------------------------------------
    k_sparse_val: Optional[int] = None
    sae_results_data: Dict[str, Any] = {}
    results_file = sae_dir / "sae_results.json"
    if results_file.exists():
        try:
            with open(results_file) as f:
                sae_results_data = json.load(f)
            meta = sae_results_data.get("metadata", {})
            k_val = meta.get("k_sparse", 32)
            if k_val and k_val > 0:
                k_sparse_val = k_val
        except Exception:
            pass

    if checkpoint_labels is None:
        # Use all available checkpoints from metadata
        meta = sae_results_data.get("metadata", {})
        checkpoint_labels = meta.get("checkpoint_labels", ["terminal"])

    # -- build SSI/CSI lookup from sae_results if not provided ---------------
    if ssi_csi_data is None and "selectivity" in sae_results_data:
        ssi_csi_data = {}
        sel = sae_results_data["selectivity"]
        for ckpt_lbl, layers_dict in sel.items():
            if isinstance(layers_dict, dict):
                for layer_name, layer_data in layers_dict.items():
                    ssi_csi_data[(str(ckpt_lbl), layer_name)] = layer_data

    # -- attribution per checkpoint ------------------------------------------
    _progress(cb, 5, f"Running attribution on {len(checkpoint_labels)} checkpoints")
    all_results: Dict[str, Any] = {}
    all_saes_layers: List[str] = []
    total_ckpts = len(checkpoint_labels)

    for ci, ckpt_label in enumerate(checkpoint_labels):
        base_pct = 5 + 90 * ci / max(total_ckpts, 1)
        _progress(cb, base_pct, f"Checkpoint {ckpt_label} ({ci + 1}/{total_ckpts})")

        # -- locate model checkpoint ----------------------------------------
        if ckpt_label == "terminal":
            ckpt_dir = lane_path / "terminal_analysis"
        else:
            ckpt_dir = lane_path / "dev_snapshots" / f"milestone_{ckpt_label}"

        ckpt_path = ckpt_dir / "checkpoint.pt"
        if not ckpt_path.exists():
            all_results[ckpt_label] = {"error": f"Not found: {ckpt_path}"}
            continue

        # -- load SAEs for this checkpoint -----------------------------------
        sae_ckpt_dir = sae_dir / "saes" / str(ckpt_label)
        if not sae_ckpt_dir.exists():
            all_results[ckpt_label] = {
                "error": f"No SAE dir: {sae_ckpt_dir}",
            }
            continue

        ckpt_saes: Dict[str, SparseAutoencoder] = {}
        for pt_file in sorted(sae_ckpt_dir.glob("*.pt")):
            layer_safe = pt_file.stem  # e.g. "encoder_layers_0" or "layer1"
            # Reconstruct layer name: match against metadata layers
            meta_layers = sae_results_data.get("metadata", {}).get("layers", [])
            layer_name = None
            for ml in meta_layers:
                if _safe_layer_filename(ml) == layer_safe:
                    layer_name = ml
                    break
            if layer_name is None:
                # Fallback: simple dot replacement
                layer_name = layer_safe.replace("_", ".")

            state = torch.load(pt_file, map_location="cpu", weights_only=True)
            d_input = state["encoder.weight"].shape[1]
            d_hidden = state["encoder.weight"].shape[0]
            expansion = d_hidden // d_input if d_input > 0 else 4

            sae = SparseAutoencoder(
                d_input=d_input,
                expansion_factor=expansion,
                k_sparse=k_sparse_val,
            )
            sae.load_state_dict(state)
            sae.eval()
            ckpt_saes[layer_name] = sae

        if not ckpt_saes:
            all_results[ckpt_label] = {
                "error": f"No SAE .pt files in {sae_ckpt_dir}",
            }
            continue

        if not all_saes_layers:
            all_saes_layers = list(ckpt_saes.keys())

        # -- load model at this checkpoint -----------------------------------
        model = create_network(
            network_type=network_type,
            num_classes=num_classes,
            pretrained=False,
        )
        model.load_state_dict(
            torch.load(ckpt_path, map_location="cpu", weights_only=True)
        )
        model.to(device)
        model.eval()

        # Filter SAEs to matching layers
        model_layers = set(_get_layer_hooks(model))
        matched_saes = {k: v for k, v in ckpt_saes.items() if k in model_layers}

        if not matched_saes:
            all_results[ckpt_label] = {
                "error": "No SAE layers match model hooks",
                "model_layers": list(model_layers),
                "sae_layers": list(ckpt_saes.keys()),
            }
            del model
            _clear_memory()
            continue

        def _sub_cb(pct: float, msg: str) -> None:
            overall = base_pct + pct * 0.9 / max(total_ckpts, 1)
            _progress(cb, overall, f"[{ckpt_label}] {msg}")

        attributions = compute_feature_attributions(
            model=model,
            saes=matched_saes,
            data_loader=data_loader,
            device=device,
            num_classes=num_classes,
            max_batches=max_batches,
            cb=_sub_cb,
        )

        agreement = {}
        if ssi_csi_data is not None:
            agreement = compute_sae_causal_agreement(
                feature_scores=attributions.get("feature_scores", {}),
                class_scores=attributions.get("class_scores", {}),
                ssi_csi_data=ssi_csi_data,
                checkpoint_label=ckpt_label,
                layers=list(matched_saes.keys()),
            )

        pathway_evidence = generate_pathway_evidence(
            attributions=attributions,
            agreement=agreement,
            layers=list(matched_saes.keys()),
        )

        all_results[ckpt_label] = {
            "attributions": attributions,
            "agreement": agreement,
            "hypothesis_evidence": pathway_evidence,
        }

        del model
        for sae in ckpt_saes.values():
            sae.cpu()
        del ckpt_saes
        _clear_memory()

    # -- serialise -----------------------------------------------------------
    def _ser(obj: Any) -> Any:
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, dict):
            return {str(k): _ser(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_ser(v) for v in obj]
        return obj

    output = {
        "metadata": {
            "method": "edge_attribution_patching",
            "corruption": "zero_ablation",
            "max_batches": max_batches,
            "checkpoints_analysed": checkpoint_labels,
            "sae_layers": all_saes_layers,
        },
        "checkpoint_results": all_results,
    }

    output_file = sae_dir / "pathway_results.json"
    with open(output_file, "w") as f:
        json.dump(_ser(output), f, indent=2)

    _progress(cb, 100, f"Pathway results -> {output_file}")
    return output
