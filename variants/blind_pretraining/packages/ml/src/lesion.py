"""
Mid-training lesions of SAE-identified feature subspaces via orthogonal-projection
forward hooks.

At a branch checkpoint of a finished clean run, a set of SAE features is selected
per tracked layer (high-SSI scaffold, high-CSI refinement control, or random alive
control). During a lesion window the activations of each hooked layer are projected
orthogonally to the frozen decoder directions of the selected features:

    x  <-  (I - Q Q^T) x        per spatial position, in channel space,

where Q is an orthonormal basis of the selected (unit-norm but non-orthogonal)
decoder columns at the branch checkpoint. The subspace is fixed at lesion onset,
so the measured quantity is the network's reorganisation under a fixed constraint.

The projection acts on the raw (not mean-centred) activation, so the mean component
along Q is removed as well, identically in all conditions.

Because decoder columns are not orthogonal, summed code energy sum_f E[h_f^2] only
approximates the ablated variance. Features are ranked by code energy; variance
matching across conditions uses the exact ablated energy fraction
v(S) = E[||Q_S^T x||^2] / E[||x||^2] on the branch checkpoint's raw pooled
activations, i.e. the quantity the hook removes.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# Alive definition shared with the feature-fate analysis:
# a feature is alive iff it fires (|a| > EPS_FIRE) on >= ALIVE_FRAC of samples.
ALIVE_FRAC = 0.01
EPS_FIRE = 1e-6


# Projection
def _resolve_module(model: nn.Module, layer_path: str) -> nn.Module:
    """Walk a dotted module path (same convention as pathways._resolve_module)."""
    module = model
    for part in layer_path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _safe_layer_filename(layer_name: str) -> str:
    return layer_name.replace(".", "_")


def orthonormal_basis(
    decoder_weight: torch.Tensor,
    feature_indices: Sequence[int],
    rank_tol: float = 1e-6,
) -> torch.Tensor:
    """Orthonormal basis Q (C x r) of the selected decoder columns, on CPU float32.

    Uses SVD rather than QR: decoder columns of high-SSI features can be
    near-collinear, and SVD gives a numerically clean rank cut
    (drop singular values < rank_tol * s_max).
    """
    idx = torch.as_tensor(list(feature_indices), dtype=torch.long)
    A = decoder_weight.detach().cpu().float()[:, idx]  # (C, k)
    # SVD on CPU: MPS linalg support is unreliable, and this runs once per lane.
    U, S, _ = torch.linalg.svd(A, full_matrices=False)
    keep = S > (rank_tol * S[0].clamp(min=1e-12))
    return U[:, keep].contiguous()


def build_projector(
    decoder_weight: torch.Tensor,
    feature_indices: Sequence[int],
    rank_tol: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Build P = I - Q Q^T for the selected decoder columns.

    Returns:
        (P, Q, effective_rank), all CPU float32. P is (C, C), Q is (C, r).
    """
    Q = orthonormal_basis(decoder_weight, feature_indices, rank_tol)
    C = decoder_weight.shape[0]
    P = torch.eye(C, dtype=torch.float32) - Q @ Q.T
    return P, Q, Q.shape[1]


class ProjectionLesionHook:
    """Forward hook applying x <- P x in channel space, per spatial position.

    The hook returns the modified tensor: PyTorch replaces the module output with a
    hook's non-None return value, so downstream layers and later-registered
    (capture/probe) hooks see the lesioned activation.
    """

    def __init__(self, projector: torch.Tensor, layer_name: str):
        self.P = projector  # moved to the model's device/dtype at install time
        self.layer_name = layer_name
        self.enabled = True

    def to(self, device, dtype=torch.float32) -> "ProjectionLesionHook":
        self.P = self.P.to(device=device, dtype=dtype)
        return self

    def __call__(self, module, inputs, output):
        if not self.enabled:
            return output
        P = self.P if self.P.dtype == output.dtype else self.P.to(output.dtype)
        if output.dim() == 4:
            # (B,C,H,W) conv stages; avgpool's (B,512,1,1) is the H=W=1 case.
            return torch.einsum("dc,bchw->bdhw", P, output)
        if output.dim() == 2:
            return output @ P.T
        return output  # other shapes are left unchanged


class LesionController:
    """Owns the per-layer projection hooks for one lesion spec.

    install()/remove() are idempotent and keyed to epoch boundaries by the
    training loop. set_enabled() toggles the projection without re-registering
    (used for the lesion_during_capture=False capture path).
    """

    def __init__(
        self,
        model: nn.Module,
        spec: dict,
        layers: Optional[List[str]] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.spec = spec
        sae_dir = Path(spec["sae_dir"])
        self.hooks: Dict[str, ProjectionLesionHook] = {}
        self.bases: Dict[str, torch.Tensor] = {}  # Q per layer, for diagnostics
        for layer_name, ld in spec["layers"].items():
            if layers is not None and layer_name not in layers:
                continue
            sae_path = sae_dir / f"{_safe_layer_filename(layer_name)}.pt"
            state = torch.load(sae_path, map_location="cpu", weights_only=True)
            P, Q, rank = build_projector(state["decoder.weight"], ld["feature_indices"])
            expected = ld.get("subspace_rank")
            if expected is not None and rank != expected:
                print(f"[LESION] WARNING {layer_name}: projector rank {rank} != "
                      f"spec subspace_rank {expected}")
            self.hooks[layer_name] = ProjectionLesionHook(P, layer_name).to(device)
            self.bases[layer_name] = Q
        if not self.hooks:
            raise ValueError("LesionController: no layers selected from spec")
        # Frozen-base rank per layer: sliding-window re-projection never evicts
        # the initial spec subspace, only dynamically added directions.
        self.base_rank = {l: q.shape[1] for l, q in self.bases.items()}
        self._handles = []

    @classmethod
    def from_spec_path(
        cls,
        model: nn.Module,
        spec_path: str,
        layers: Optional[List[str]] = None,
        device: str = "cpu",
    ) -> "LesionController":
        with open(spec_path) as f:
            spec = json.load(f)
        return cls(model, spec, layers=layers, device=device)

    @property
    def active(self) -> bool:
        return bool(self._handles)

    @property
    def layer_names(self) -> List[str]:
        return list(self.hooks.keys())

    def install(self) -> None:
        if self._handles:
            return
        for layer_name, hook in self.hooks.items():
            module = _resolve_module(self.model, layer_name)
            self._handles.append(module.register_forward_hook(hook))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def set_enabled(self, flag: bool) -> None:
        for hook in self.hooks.values():
            hook.enabled = flag

    def _extend_basis(self, layer_name: str, new_dirs: torch.Tensor,
                      max_rank: int, rank_policy: str = "accumulate") -> int:
        """Add directions (C x m, CPU float32) to a layer's lesioned subspace.

        New directions are orthogonalised against the existing basis; near-zero
        residuals (already inside the subspace) are dropped. The total rank is
        capped at max_rank. Returns the number of directions actually added.
        The hook's projector is rebuilt in place (device preserved).

        rank_policy='accumulate': stop adding once max_rank is reached, so the
        suppression weakens over long windows.
        rank_policy='sliding': at the rank budget, evict the oldest dynamically
        added directions (never the frozen spec base) to admit the new ones, so
        the current code stays suppressed at constant rank. Evicted directions
        can be re-expressed about budget/q refits later.
        """
        Q_old = self.bases[layer_name]
        if new_dirs.numel() == 0:
            return 0
        evicted = False
        if rank_policy == "sliding":
            n_new = min(new_dirs.shape[1], max(max_rank - self.base_rank[layer_name], 0))
            overflow = Q_old.shape[1] + n_new - max_rank
            if overflow > 0:
                base = self.base_rank[layer_name]
                Q_old = torch.cat([Q_old[:, :base],
                                   Q_old[:, base + overflow:]], dim=1)
                self.bases[layer_name] = Q_old
                evicted = True
        room = max_rank - Q_old.shape[1]
        if room <= 0:
            if evicted:
                self._rebuild_projector(layer_name)
            return 0
        # unit-normalise candidates so the residual threshold below is absolute
        norms = new_dirs.norm(dim=0)
        new_dirs = new_dirs[:, norms > 1e-8] / norms[norms > 1e-8]
        if new_dirs.numel() == 0:
            return 0
        # residualise against the current basis, then orthonormalise the residual.
        # The threshold is absolute (inputs are unit norm): a candidate already inside
        # the subspace leaves only numerical noise, which must not be added.
        R = new_dirs - Q_old @ (Q_old.T @ new_dirs)
        U, S, _ = torch.linalg.svd(R, full_matrices=False)
        keep = S > 1e-3
        U = U[:, keep][:, :room]
        if U.shape[1] == 0:
            if evicted:
                self._rebuild_projector(layer_name)
            return 0
        self.bases[layer_name] = torch.cat([Q_old, U], dim=1)
        self._rebuild_projector(layer_name)
        return int(U.shape[1])

    def _rebuild_projector(self, layer_name: str) -> None:
        Q = self.bases[layer_name]
        P = torch.eye(Q.shape[0], dtype=torch.float32) - Q @ Q.T
        hook = self.hooks[layer_name]
        hook.P = P.to(device=hook.P.device, dtype=hook.P.dtype)

    @torch.no_grad()
    def reproject(
        self,
        model: nn.Module,
        val_loader,
        device: str,
        selected_classes: List[str],
        mode: str = "superclass",     # 'superclass' | 'taskgeneral' | 'random' (dose-matched control)
        samples_per_class: int = 20,
        q_max: int = 19,              # between-SC scatter has rank <= n_sc - 1
        evr_floor: float = 0.02,      # min fraction of between-SC variance per dir
        max_rank_frac: float = 0.5,   # cap: lesioned rank <= frac * channels
        rank_policy: str = "accumulate",  # 'accumulate' | 'sliding' (evict oldest)
        rng_seed: int = 0,
    ) -> Dict[str, dict]:
        """Re-estimate the linear superclass code from the current (lesioned)
        activations and add it to the projected-out set.

        With hooks active the extracted activations are post-projection, so any
        between-superclass structure found here was rebuilt by the network in
        directions orthogonal to the current lesion. Each call measures this
        re-encoding (between-SC explained variance, returned and logged) and
        removes it, so the network cannot maintain a linear superclass code in
        these layers.

        mode='random' is the dose-matched control: it determines how many
        directions the superclass refit would add and adds that many random
        directions (orthogonal to the existing basis) instead.

        mode='taskgeneral' removes the task-general code instead: the same number
        of directions the superclass refit would add (rank-matched to the
        superclass arm and the random control), taken as the top directions of
        class-agnostic activation energy (taskgeneral_code_directions).
        """
        from training import extract_individual_multilayer_activations
        from cifar100_hierarchy import build_superclass_map

        class_to_idx = {n: i for i, n in enumerate(selected_classes)}
        sc_map = build_superclass_map(selected_classes, class_to_idx)
        sc_names = sorted(set(sc_map.values()))
        sc_idx = {s: i for i, s in enumerate(sc_names)}

        acts = extract_individual_multilayer_activations(
            model, val_loader, device, num_classes=len(selected_classes),
            samples_per_class=samples_per_class, selected_layers=self.layer_names)

        report: Dict[str, dict] = {}
        for layer in self.layer_names:
            per_class = acts.get(layer, {})
            if not per_class:
                continue
            X = torch.cat([per_class[c] for c in sorted(per_class)], 0).float().cpu()
            y = torch.cat([
                torch.full((per_class[c].shape[0],), sc_idx[sc_map[c]], dtype=torch.long)
                for c in sorted(per_class)], 0)
            dirs, evr = superclass_code_directions(X, y, q_max=q_max,
                                                   evr_floor=evr_floor)
            if mode == "taskgeneral":
                y_cls = torch.cat([torch.full((per_class[c].shape[0],), int(c), dtype=torch.long)
                                   for c in sorted(per_class)], 0)
                dirs, evr = taskgeneral_code_directions(X, y_cls, n_dirs=int(dirs.shape[1]))
            C = X.shape[1]
            max_rank = int(max_rank_frac * C)
            if mode == "random":
                g = torch.Generator().manual_seed(
                    rng_seed + (hash(layer) % (2 ** 31)))
                dirs = torch.randn(C, dirs.shape[1], generator=g)
            n_added = self._extend_basis(layer, dirs, max_rank=max_rank,
                                         rank_policy=rank_policy)
            report[layer] = {
                "n_candidate_dirs": int(dirs.shape[1]),
                "n_added": n_added,
                "total_rank": int(self.bases[layer].shape[1]),
                "max_rank": max_rank,
                "between_sc_evr": [round(float(v), 4) for v in evr[:q_max]],  # TG mode: energy ratios
            }
        return report


def superclass_code_directions(
    X: torch.Tensor,
    y_sc: torch.Tensor,
    q_max: int = 19,
    evr_floor: float = 0.02,
) -> Tuple[torch.Tensor, list]:
    """Top directions of the between-superclass scatter of activations X.

    Returns (dirs (C x q), explained-variance ratios of ALL eigenvalues, desc).
    Directions are eigenvectors of the between-SC covariance whose eigenvalue
    holds >= evr_floor of total between-SC variance, capped at q_max. These are
    the directions a linear superclass readout uses; projecting them out removes
    the current linear superclass code.
    """
    X = X.float()
    grand = X.mean(0)
    B = torch.zeros(X.shape[1], X.shape[1])
    n_total = X.shape[0]
    for s in y_sc.unique():
        mask = y_sc == s
        mu = X[mask].mean(0) - grand
        B += (mask.sum().item() / n_total) * torch.outer(mu, mu)
    evals, evecs = torch.linalg.eigh(B)
    evals, evecs = evals.flip(0), evecs.flip(1)  # descending
    total = float(evals.clamp(min=0).sum().clamp(min=1e-12))
    evr = [float(v) / total for v in evals.clamp(min=0)]
    q = min(q_max, int(sum(1 for r in evr if r >= evr_floor)))
    return evecs[:, :q].contiguous(), evr


def taskgeneral_code_directions(
    X: torch.Tensor,
    y_cls: torch.Tensor,
    n_dirs: int,
    between_frac: float = 0.99,
) -> Tuple[torch.Tensor, list]:
    """Top n_dirs directions of the class-agnostic activation energy of X.

    Linear analogue of task-general (high-SAI) features, which respond alike to all classes:
    (1) remove the subspace spanned by the between-class scatter (eigenvectors holding
    `between_frac` of the between-class variance), so no remaining direction separates class
    means; (2) return the top eigenvectors of the uncentered second moment of the residual
    (class-agnostic energy, including the shared mean response).
    Returns (dirs (C x n_dirs), energy ratios of all eigenvalues, desc).
    """
    X = X.float()
    n, C = X.shape
    grand = X.mean(0)
    B = torch.zeros(C, C)
    for c in y_cls.unique():
        mask = y_cls == c
        mu = X[mask].mean(0) - grand
        B += (mask.sum().item() / n) * torch.outer(mu, mu)
    eb, Ub = torch.linalg.eigh(B)
    eb, Ub = eb.flip(0).clamp(min=0), Ub.flip(1)
    cum = torch.cumsum(eb, 0) / eb.sum().clamp(min=1e-12)
    k = int((cum < between_frac).sum().item()) + 1
    U = Ub[:, :k]
    R = X - (X @ U) @ U.T
    M = R.T @ R / n
    em, V = torch.linalg.eigh(M)
    em, V = em.flip(0).clamp(min=0), V.flip(1)
    total = float(em.sum().clamp(min=1e-12))
    return V[:, :n_dirs].contiguous(), [float(v) / total for v in em]


# Lesion spec construction (feature selection and variance matching)
def _load_branch_checkpoint_label(clean_lane_dir: Path, branch_epoch: int) -> str:
    """Resolve a branch epoch to its milestone label via devtrain_snapshots.json."""
    snaps_file = clean_lane_dir / "dev_snapshots" / "metrics" / "devtrain_snapshots.json"
    with open(snaps_file) as f:
        snaps = json.load(f)
    if isinstance(snaps, dict):
        snaps = snaps.get("snapshots", [])
    matches = [s for s in snaps
               if s.get("epoch") == branch_epoch
               and s.get("snapshot_type") in ("intermediate", "t0_baseline")]
    if not matches:
        raise ValueError(
            f"no intermediate snapshot at epoch {branch_epoch} in {snaps_file} "
            f"(available: {[(s.get('milestone_index'), s.get('epoch')) for s in snaps]})")
    return str(matches[0]["milestone_index"])


def _exact_subspace_variance_fraction(
    X: torch.Tensor, decoder_weight: torch.Tensor, feature_indices: Sequence[int]
) -> float:
    """v(S) = E[||Q_S^T x||^2] / E[||x||^2] on raw (uncentred) activations."""
    if len(feature_indices) == 0:
        return 0.0
    Q = orthonormal_basis(decoder_weight, feature_indices)
    total = float((X ** 2).sum())
    if total <= 0:
        return 0.0
    return float(((X @ Q) ** 2).sum()) / total


def _greedy_variance_match(
    X: torch.Tensor,
    decoder_weight: torch.Tensor,
    ranked_candidates: List[int],
    target_v: float,
    tol: float,
) -> Tuple[List[int], float, bool]:
    """Add candidates in rank order until the exact ablated fraction reaches
    target_v * (1 - tol). Returns (indices, achieved_v, feasible)."""
    chosen: List[int] = []
    v = 0.0
    for f in ranked_candidates:
        chosen.append(f)
        v = _exact_subspace_variance_fraction(X, decoder_weight, chosen)
        if v >= target_v * (1.0 - tol):
            return chosen, v, True
    return chosen, v, False


def load_branch_artifacts(
    clean_lane_dir, branch_epoch: int, layers: Sequence[str]
) -> dict:
    """Load everything build_lesion_spec needs from a finished clean lane.

    Returns dict with: ckpt_label, sae_dir, per-layer {decoder, H (codes),
    X (raw pooled activations, class-sorted rows), ssi, csi}, thresholds.
    """
    clean_lane_dir = Path(clean_lane_dir)
    ckpt = _load_branch_checkpoint_label(clean_lane_dir, branch_epoch)
    sae_dir = clean_lane_dir / "sae_analysis" / "saes" / ckpt
    act_dir = clean_lane_dir / "sae_analysis" / "activations" / ckpt

    with open(clean_lane_dir / "sae_analysis" / "sae_results.json") as f:
        results = json.load(f)
    sel = results["selectivity"][ckpt]
    thresholds = results.get("metadata", {}).get("analysis_thresholds", {})

    raw_path = clean_lane_dir / "dev_snapshots" / f"milestone_{ckpt}" / \
        "individual_multilayer_activations.pt"
    raw = torch.load(raw_path, map_location="cpu", weights_only=True)

    per_layer = {}
    for layer in layers:
        safe = _safe_layer_filename(layer)
        state = torch.load(sae_dir / f"{safe}.pt", map_location="cpu", weights_only=True)
        H = np.load(act_dir / f"{safe}.npy")
        # Raw pooled activations, concatenated in sorted class order (same
        # class-blocked row convention as the saved SAE code matrices).
        cls_acts = raw[layer]
        X = torch.cat([cls_acts[c] for c in sorted(cls_acts.keys())], dim=0).float()
        per_layer[layer] = {
            "decoder": state["decoder.weight"],
            "H": H,
            "X": X,
            "ssi": np.asarray(sel[layer]["feature_ssi"], dtype=np.float64),
            "csi": np.asarray(sel[layer]["feature_csi"], dtype=np.float64),
            "sai": np.asarray(sel[layer]["feature_sai"], dtype=np.float64),
        }
    return {"ckpt_label": ckpt, "sae_dir": str(sae_dir),
            "per_layer": per_layer, "thresholds": thresholds}


def build_lesion_spec(
    clean_lane_dir,
    branch_epoch: int,
    layers: Sequence[str],
    lesion_type: str,            # 'scaffold' | 'taskgeneral' | 'refinement' | 'random'
    top_n: int = 32,
    matching: str = "variance",  # 'count' | 'variance' (selects active indices)
    seed: int = 42,
    alive_frac: float = ALIVE_FRAC,
    eps_fire: float = EPS_FIRE,
    exclude_overlap: bool = True,
    variance_tol: float = 0.10,
    ssi_gate: bool = True,
    target_code: str = "scaffold",  # 'scaffold' (SSI) | 'taskgeneral' (SAI)
    _artifacts: Optional[dict] = None,  # pass preloaded load_branch_artifacts()
) -> dict:
    """Build the per-layer feature selection for one lesion condition.

    Both the count-matched and variance-matched selections are recorded for
    every layer; `feature_indices` is the active one per `matching`. For the
    primary-code condition (scaffold/taskgeneral) the two coincide (that set
    defines the variance target), so such lanes are shared across matchings.

    `target_code` picks the primary code that defines the variance target v* and
    (under the gate) which layers carry a lesion at all:
      - 'scaffold': alive features ranked by SSI, gated at ssi_abstraction_thresh
        (superclass-selective features, layer4/avgpool).
      - 'taskgeneral': alive features ranked by SAI, gated at sai_task_general_thresh
        (task-general features, layer1-2).
    lesion_type 'scaffold'/'taskgeneral' lesion the primary set directly (and set
    target_code accordingly); 'refinement'/'random' are variance-matched controls
    that inherit the target via target_code (so a taskgeneral run's random control
    matches the SAI v* and inherits the SAI layer exclusions).

    ssi_gate=True (default): the primary set is restricted to alive features above
    the lane's adaptive threshold (SSI or SAI per target_code, capped at top_n),
    and layers with no above-threshold feature are excluded from the lesion in all
    conditions. In ResNet-18/CIFAR-100, above-SSI features occur in layer4/avgpool
    and above-SAI features in layer1-2 only; a flat top-N elsewhere would lesion
    near-zero-selectivity features, i.e. a nonspecific capacity lesion.
    ssi_gate=False uses the flat top-N-by-primary-code selection at every layer.
    """
    if lesion_type not in ("scaffold", "taskgeneral", "refinement", "random"):
        raise ValueError(f"unknown lesion_type {lesion_type!r}")
    if matching not in ("count", "variance"):
        raise ValueError(f"unknown matching {matching!r}")
    # The primary lesion types pin the target code they ablate; controls keep the
    # caller-supplied target_code so they match the right v*.
    if lesion_type in ("scaffold", "taskgeneral"):
        target_code = lesion_type
    if target_code not in ("scaffold", "taskgeneral"):
        raise ValueError(f"unknown target_code {target_code!r}")

    art = _artifacts or load_branch_artifacts(clean_lane_dir, branch_epoch, layers)
    ssi_thresh = art["thresholds"].get("ssi_abstraction_thresh")
    sai_thresh = art["thresholds"].get("sai_task_general_thresh")
    csi_thresh = art["thresholds"].get("csi_differentiation_thresh")
    if target_code == "scaffold":
        primary_thresh, primary_name, thresh_key = ssi_thresh, "SSI", "ssi_abstraction_thresh"
    else:
        primary_thresh, primary_name, thresh_key = sai_thresh, "SAI", "sai_task_general_thresh"
    if ssi_gate and primary_thresh is None:
        raise ValueError(f"ssi_gate=True but no {thresh_key} in the clean lane's "
                         f"analysis_thresholds")
    spec_layers = {}
    excluded_layers = {}
    for layer in layers:
        d = art["per_layer"][layer]
        H, X, W_dec = d["H"], d["X"], d["decoder"]
        ssi, csi, sai = d["ssi"], d["csi"], d["sai"]
        primary_sel = sai if target_code == "taskgeneral" else ssi
        n_feat = H.shape[1]
        assert len(ssi) == n_feat, f"{layer}: SSI length {len(ssi)} != d_hidden {n_feat}"

        alive = (np.abs(H) > eps_fire).mean(axis=0) >= alive_frac
        alive_idx = np.flatnonzero(alive)
        energy = (H ** 2).mean(axis=0)

        # Primary set always computed: it defines the variance target v* and
        # (under the gate) which layers carry a lesion at all. Ranked by the
        # primary selectivity (SSI for scaffold, SAI for taskgeneral).
        scaffold_order = alive_idx[np.argsort(-primary_sel[alive_idx], kind="stable")]
        if ssi_gate:
            scaffold_order = scaffold_order[primary_sel[scaffold_order] > primary_thresh]
        n_scaffold = int(min(top_n, len(scaffold_order)))
        scaffold_set = [int(i) for i in scaffold_order[:n_scaffold]]
        if not scaffold_set:
            excluded_layers[layer] = {
                "reason": f"no alive feature above the adaptive {primary_name} threshold",
                "target_code": target_code, "primary_threshold": primary_thresh,
                "n_alive": int(alive.sum()),
            }
            continue
        v_star = _exact_subspace_variance_fraction(X, W_dec, scaffold_set)

        if lesion_type in ("scaffold", "taskgeneral"):
            count_sel, var_sel = list(scaffold_set), list(scaffold_set)
            achieved_v, feasible = v_star, True
            pool_overlap = 0
            n_gated_in_pool = len(scaffold_set)
        else:
            if lesion_type == "refinement":
                # High-CSI control: above-CSI-threshold candidates first
                # (refinement features), then the remaining alive features by CSI, so variance
                # matching can extend past the gated pool if needed (recorded).
                csi_order = alive_idx[np.argsort(-csi[alive_idx], kind="stable")]
                if csi_thresh is not None:
                    gated = csi_order[csi[csi_order] > csi_thresh]
                    rest = csi_order[csi[csi_order] <= csi_thresh]
                    pool = np.concatenate([gated, rest])
                    n_gated_in_pool = int(len(gated))
                else:
                    pool = csi_order
                    n_gated_in_pool = len(csi_order)
            else:  # random
                rng = np.random.default_rng(seed + (hash(layer) % (2 ** 31)))
                pool = rng.permutation(alive_idx)
                n_gated_in_pool = len(pool)
            pool = [int(i) for i in pool]
            pool_overlap = len(set(pool[:n_scaffold]) & set(scaffold_set))
            if exclude_overlap:
                pool = [f for f in pool if f not in set(scaffold_set)]
            count_sel = pool[:n_scaffold]
            var_sel, achieved_v, feasible = _greedy_variance_match(
                X, W_dec, pool, v_star, variance_tol)

        active = var_sel if matching == "variance" else count_sel
        _, _, rank = build_projector(W_dec, active)

        spec_layers[layer] = {
            "feature_indices": active,
            "count_matched_indices": count_sel,
            "variance_matched_indices": var_sel,
            "n_alive": int(alive.sum()),
            "requested_n": int(top_n),
            "n_scaffold": int(n_scaffold),
            "actual_n": len(active),
            "subspace_rank": int(rank),
            "code_energy_fraction": float(energy[active].sum() / max(energy.sum(), 1e-12)),
            "exact_subspace_variance_fraction":
                float(_exact_subspace_variance_fraction(X, W_dec, active)),
            "target_variance_fraction": float(v_star),
            "variance_match_feasible": bool(feasible),
            "target_code": target_code,
            "ssi_threshold": ssi_thresh,
            "sai_threshold": sai_thresh,
            "csi_threshold": csi_thresh,
            "n_above_ssi_threshold":
                int((ssi[active] > ssi_thresh).sum()) if ssi_thresh is not None else None,
            "n_above_sai_threshold":
                int((sai[active] > sai_thresh).sum()) if sai_thresh is not None else None,
            "n_gated_in_pool": int(n_gated_in_pool),
            "overlap_with_scaffold_topN": int(pool_overlap),
            "mean_ssi_selected": float(ssi[active].mean()) if active else 0.0,
            "mean_sai_selected": float(sai[active].mean()) if active else 0.0,
            "mean_csi_selected": float(csi[active].mean()) if active else 0.0,
        }
        if not feasible:
            print(f"[LESION-SPEC] WARNING {layer}: variance matching infeasible "
                  f"(target {v_star:.4f}, best {achieved_v:.4f} with all "
                  f"{len(var_sel)} candidates): recorded; consider scaffold-down matching")
    if not spec_layers:
        raise ValueError(
            f"no layer has an alive feature above the adaptive {primary_name} threshold "
            f"({primary_thresh}) at branch epoch {branch_epoch}: nothing to lesion")

    # Binding (manipulation) check for the target code. The superclass
    # probe detects SSI removal but not SAI removal (task-general features
    # are not superclass-selective), so taskgeneral lanes use the task probe;
    # the choice is recorded in the lane metadata.
    if target_code == "taskgeneral":
        binding_check = ("overall val-acc / fine-acc drop at install (whole-network, "
                         "behavioral) + per-layer 100-way TASK linear probe on lesioned "
                         "activations; NOT the superclass probe (SAI features are not "
                         "superclass-selective)")
    else:
        binding_check = "superclass linear probe on lesioned activations + hierarchical accuracy"

    return {
        "clean_lane": str(clean_lane_dir),
        "branch_epoch": int(branch_epoch),
        "branch_ckpt_label": art["ckpt_label"],
        "lesion_type": lesion_type,
        "target_code": target_code,
        "binding_check": binding_check,
        "matching": matching,
        "seed": int(seed),
        "top_n": int(top_n),
        "variance_tol": float(variance_tol),
        "exclude_overlap": bool(exclude_overlap),
        "alive_frac": float(alive_frac),
        "ssi_gate": bool(ssi_gate),
        "sae_dir": art["sae_dir"],
        "layers": spec_layers,
        "excluded_layers": excluded_layers,
    }


# Probe-based manipulation check
def run_lesion_probe_check(
    model: nn.Module,
    val_loader,
    device: str,
    selected_classes: List[str],
    layers: List[str],
    samples_per_class: int = 50,
    probe_steps: int = 300,
    probe_seed: int = 0,
    label_level: str = "superclass",  # 'superclass' (SSI binding) | 'class' (SAI binding)
) -> Dict[str, dict]:
    """Train a small logistic probe per layer on the current activations.

    With lesion hooks installed the extracted activations are lesioned (the
    extraction hooks register after the persistent lesion hooks, which return
    their modified output), so this measures whether the lesion binds.

    label_level='superclass' (default): superclass probe, the binding check for
    scaffold (SSI) lesions.
    label_level='class': fine-class (task) probe, the binding check for
    task-general (SAI) lesions, which a superclass probe cannot detect because
    task-general features are not superclass-selective.
    Result key is 'superclass_probe_acc' or 'task_probe_acc'.

    Kept lightweight (about 1-2 min) for use every epoch;
    training.train_linear_probes covers every layer and is too slow for this.
    """
    from training import extract_individual_multilayer_activations
    from cifar100_hierarchy import build_superclass_map

    if label_level not in ("superclass", "class"):
        raise ValueError(f"unknown label_level {label_level!r}")
    class_to_idx = {name: i for i, name in enumerate(selected_classes)}
    # extract_individual_multilayer_activations keys per_class by class index
    # (int), as does build_superclass_map, so the probe label is derived from the
    # integer key.
    if label_level == "superclass":
        sc_map = build_superclass_map(selected_classes, class_to_idx)
        names = sorted(set(sc_map.values()))
        name_to_idx = {s: i for i, s in enumerate(names)}
        label_of = lambda c: name_to_idx[sc_map[c]]
        n_labels, acc_key = len(names), "superclass_probe_acc"
    else:
        label_of = lambda c: int(c)
        n_labels, acc_key = len(selected_classes), "task_probe_acc"

    acts = extract_individual_multilayer_activations(
        model, val_loader, device, num_classes=len(selected_classes),
        samples_per_class=samples_per_class, selected_layers=layers)

    results: Dict[str, dict] = {}
    g = torch.Generator().manual_seed(probe_seed)
    for layer in layers:
        per_class = acts.get(layer, {})
        if not per_class:
            continue
        X = torch.cat([per_class[c] for c in sorted(per_class.keys())], dim=0).float()
        y = torch.cat([
            torch.full((per_class[c].shape[0],), label_of(c), dtype=torch.long)
            for c in sorted(per_class.keys())], dim=0)
        # Standardise; seeded 80/20 split
        mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp(min=1e-6)
        X = (X - mu) / sd
        perm = torch.randperm(X.shape[0], generator=g)
        n_train = int(0.8 * X.shape[0])
        tr, va = perm[:n_train], perm[n_train:]

        probe = nn.Linear(X.shape[1], n_labels)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss()
        Xtr, ytr = X[tr], y[tr]
        for _ in range(probe_steps):
            opt.zero_grad()
            loss = lossf(probe(Xtr), ytr)
            loss.backward()
            opt.step()
        with torch.no_grad():
            acc = (probe(X[va]).argmax(1) == y[va]).float().mean().item()
        results[layer] = {
            acc_key: round(acc, 4),
            "n_train": int(n_train),
            "n_val": int(X.shape[0] - n_train),
        }
    return results
