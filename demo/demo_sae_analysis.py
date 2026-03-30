#!/usr/bin/env python3
"""End-to-end demo of the Representational Development Tracing (RDT) pipeline.

Runs the complete pipeline on a small subset of CIFAR-10 (10 classes) with a
ResNet-18, demonstrating all three stages:

  1. NETINIT  — Create ResNet-18, initialize weights (kaiming_normal)
  2. DEVTRAIN — Train for 5 epochs, capture 3 snapshots with per-sample
               activations at intermediate layers
  3. SAEANALYSIS — Train sparse autoencoders at each checkpoint, match
                   features across time, compute selectivity indices,
                   classify developmental events

Uses a 2,000-sample training subset for speed. Runs in ~2-5 minutes on CPU.
No GPU required (used automatically if available).

This exercises the same src/ modules as the full pipeline (run/run_pipeline.py)
but at reduced scale. For full reproduction, see run/configs/.
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

# Make src/ importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# ── Suppress verbose output ─────────────────────────────────────────────────
os.environ.setdefault("PYTHONUNBUFFERED", "1")


def log(stage, msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{stage}] {msg}")


def separator(title):
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}\n")


def main():
    separator("RDT Pipeline Demo")
    print("  Network   : ResNet-18")
    print("  Dataset   : CIFAR-10 (1,000-sample subset for speed)")
    print("  Epochs    : 3")
    print("  Snapshots : 3 (one per epoch)")
    print("  SAE       : 4x expansion, top-k (k=32), 200 steps, 3 layers")
    print("  Device    : CPU (GPU used if available)")
    print()

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    log("SETUP", f"Using device: {device}")

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # ── Output directories ───────────────────────────────────────────────
    out_dir = ROOT / "output" / "demo_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Download CIFAR-10 ────────────────────────────────────────────────
    data_dir = out_dir / "cifar10_data"
    log("SETUP", "Loading CIFAR-10 (downloads ~170 MB on first run)...")

    # Suppress torchvision download progress bar
    import contextlib, io
    with contextlib.redirect_stderr(io.StringIO()):
        torchvision.datasets.CIFAR10(root=str(data_dir), train=True, download=True)
        torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_full = torchvision.datasets.CIFAR10(root=str(data_dir), train=True, download=False, transform=transform_train)
    val_full = torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=False, transform=transform_val)

    # Use a 1000-sample subset for training, 200 for validation (fast on CPU)
    train_dataset = Subset(train_full, range(1000))
    val_dataset = Subset(val_full, range(200))
    class_names = train_full.classes  # 10 CIFAR-10 classes
    num_classes = len(class_names)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    log("SETUP", f"{len(train_dataset)} train / {len(val_dataset)} val samples, {num_classes} classes")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: NETINIT
    # ══════════════════════════════════════════════════════════════════════
    separator("Stage 1/3: NETINIT — Network Initialization")

    from models import create_network
    from initialization import initialize_weights

    model = create_network(network_type="resnet18", num_classes=num_classes)
    initialize_weights(model, method="kaiming_normal", seed=42)
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    log("NETINIT", f"ResNet-18 created ({param_count:,} parameters), kaiming_normal init")

    # Save initial weights
    init_weights_path = out_dir / "initial_weights.pt"
    torch.save(model.state_dict(), init_weights_path)
    log("NETINIT", f"Initial weights saved")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: DEVTRAIN — Training with snapshot capture
    # ══════════════════════════════════════════════════════════════════════
    separator("Stage 2/3: DEVTRAIN — Training with Snapshot Capture")

    from training import extract_multilayer_activations

    # Use 3 layers (skip early layers which are very large) for speed
    layer_names = ["layer3", "layer4", "avgpool"]
    n_epochs = 3
    snapshot_epochs = [0, 1, 2]  # 3 snapshots: one per epoch

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    snapshots = {}

    for epoch in range(n_epochs):
        # Train one epoch
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100.0 * correct / total

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        val_acc = 100.0 * val_correct / val_total

        scheduler.step()

        log("DEVTRAIN", f"Epoch {epoch+1}/{n_epochs}  train_acc={train_acc:.1f}%  val_acc={val_acc:.1f}%")

        # Capture snapshot at designated epochs
        if epoch in snapshot_epochs:
            log("DEVTRAIN", f"  Capturing snapshot at epoch {epoch+1}...")

            # Extract per-sample activations at each layer
            snapshot_activations = {}
            model.eval()

            for layer_name in layer_names:
                activations_list = []
                labels_list = []
                hooks = []

                def get_hook(name, storage):
                    def hook_fn(module, input, output):
                        # Global average pool if spatial
                        if output.dim() == 4:
                            pooled = output.mean(dim=[2, 3])
                        elif output.dim() == 3:
                            pooled = output.mean(dim=1)
                        else:
                            pooled = output.squeeze()
                        storage.append(pooled.detach().cpu())
                    return hook_fn

                # Register hook on target layer
                layer_storage = []
                for name, module in model.named_modules():
                    if name == layer_name:
                        h = module.register_forward_hook(get_hook(name, layer_storage))
                        hooks.append(h)
                        break

                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(device)
                        _ = model(inputs)
                        labels_list.append(targets)

                for h in hooks:
                    h.remove()

                if layer_storage:
                    snapshot_activations[layer_name] = {
                        "activations": torch.cat(layer_storage, dim=0).numpy(),
                        "labels": torch.cat(labels_list, dim=0).numpy(),
                    }

            snapshots[f"checkpoint_{epoch}"] = snapshot_activations
            log("DEVTRAIN", f"  Snapshot captured: {len(snapshot_activations)} layers, "
                f"{snapshot_activations[layer_names[0]]['activations'].shape[0]} samples each")

    log("DEVTRAIN", f"Training complete: {len(snapshots)} snapshots captured")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 3: SAEANALYSIS — SAE Feature Decomposition
    # ══════════════════════════════════════════════════════════════════════
    separator("Stage 3/3: SAEANALYSIS — Sparse Autoencoder Decomposition")

    from sae import SparseAutoencoder
    from scipy.stats import entropy as sp_entropy

    expansion_factor = 4
    k_sparse = 32
    n_sae_steps = 200

    # CIFAR-10 superclass hierarchy: animals vs vehicles
    # This enables SSI/CSI computation (same approach as CIFAR-100's 20 superclasses)
    CIFAR10_SUPERCLASSES = {
        "animals":  ["bird", "cat", "deer", "dog", "frog", "horse"],
        "vehicles": ["airplane", "automobile", "ship", "truck"],
    }
    class_to_superclass = {}
    for sc, classes in CIFAR10_SUPERCLASSES.items():
        for c in classes:
            class_to_superclass[c] = sc
    superclass_names = sorted(CIFAR10_SUPERCLASSES.keys())
    n_superclasses = len(superclass_names)

    # Build class→superclass index mapping
    class_to_sc_idx = {c: superclass_names.index(class_to_superclass[c]) for c in class_names}

    # ── Train SAEs and compute selectivity at each checkpoint x layer ────
    sae_features = {}     # {checkpoint: {layer: encoded_activations}}
    selectivity = {}      # {checkpoint: {layer: {ssi, csi, sai per feature}}}
    reconstruction_quality = {}

    checkpoint_keys = sorted(snapshots.keys())

    for cp_key in checkpoint_keys:
        log("SAE", f"Processing {cp_key}...")
        sae_features[cp_key] = {}
        selectivity[cp_key] = {}
        reconstruction_quality[cp_key] = {}

        for layer_name in layer_names:
            act_data = snapshots[cp_key][layer_name]["activations"]
            labels = snapshots[cp_key][layer_name]["labels"]
            d_in = act_data.shape[1]

            # Train SAE
            sae = SparseAutoencoder(d_input=d_in, expansion_factor=expansion_factor, k_sparse=k_sparse)
            sae_optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

            act_tensor = torch.tensor(act_data, dtype=torch.float32)
            dataset_sae = torch.utils.data.TensorDataset(act_tensor)
            sae_loader = DataLoader(dataset_sae, batch_size=256, shuffle=True)

            sae.train()
            for step in range(n_sae_steps):
                for (batch,) in sae_loader:
                    sae_optimizer.zero_grad()
                    reconstructed, encoded = sae(batch)
                    loss = nn.MSELoss()(reconstructed, batch)
                    loss.backward()
                    sae_optimizer.step()
                    with torch.no_grad():
                        sae.decoder.weight.div_(sae.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8))

            # Reconstruction quality
            sae.eval()
            with torch.no_grad():
                recon, encoded_all = sae(act_tensor)
                cos_sim = nn.functional.cosine_similarity(act_tensor, recon, dim=1).mean().item()
                mse_val = nn.MSELoss()(recon, act_tensor).item()

            reconstruction_quality[cp_key][layer_name] = {"cosine_similarity": cos_sim, "mse": mse_val}

            # ── Compute selectivity indices (SSI, CSI, SAI) per feature ──
            H = encoded_all.numpy()  # (n_samples, d_hidden)
            n_features = H.shape[1]

            # Per-class mean activation magnitude
            M = np.zeros((n_features, num_classes))
            for ci in range(num_classes):
                mask = labels == ci
                if mask.sum() > 0:
                    M[:, ci] = np.abs(H[mask]).mean(axis=0)

            # Per-superclass mean activation (sum of class means within each SC)
            M_sc = np.zeros((n_features, n_superclasses))
            for ci in range(num_classes):
                sc_idx = class_to_sc_idx[class_names[ci]]
                M_sc[:, sc_idx] += M[:, ci]

            feature_ssi = np.zeros(n_features)
            feature_csi = np.zeros(n_features)
            feature_sai = np.zeros(n_features)
            feature_alive = np.zeros(n_features, dtype=bool)

            for fi in range(n_features):
                # Alive check: feature active on >1% of samples
                active_frac = (np.abs(H[:, fi]) > 1e-6).mean()
                if active_frac < 0.01:
                    continue
                feature_alive[fi] = True

                # SSI: superclass selectivity = (best_sc_frac - 1/N_sc) / (1 - 1/N_sc)
                sc_total = M_sc[fi].sum()
                if sc_total > 0:
                    sc_frac = M_sc[fi] / sc_total
                    best_sc_frac = sc_frac.max()
                    feature_ssi[fi] = (best_sc_frac - 1.0 / n_superclasses) / (1.0 - 1.0 / n_superclasses)
                    best_sc = sc_frac.argmax()

                    # CSI: within-superclass class selectivity
                    sc_classes = [ci for ci in range(num_classes) if class_to_sc_idx[class_names[ci]] == best_sc]
                    if len(sc_classes) > 1:
                        within_vals = M[fi, sc_classes]
                        within_total = within_vals.sum()
                        if within_total > 0:
                            within_frac = within_vals / within_total
                            best_within = within_frac.max()
                            n_within = len(sc_classes)
                            feature_csi[fi] = (best_within - 1.0 / n_within) / (1.0 - 1.0 / n_within)

                    # SAI: normalized entropy = H(class_distribution) / log(N_classes)
                    class_total = M[fi].sum()
                    if class_total > 0:
                        p = M[fi] / class_total
                        p = p[p > 0]
                        h = sp_entropy(p, base=np.e)
                        feature_sai[fi] = h / np.log(num_classes) if num_classes > 1 else 0

            selectivity[cp_key][layer_name] = {
                "ssi": feature_ssi,
                "csi": feature_csi,
                "sai": feature_sai,
                "alive": feature_alive,
                "n_alive": int(feature_alive.sum()),
            }
            sae_features[cp_key][layer_name] = H

        n_alive_total = sum(selectivity[cp_key][l]["n_alive"] for l in layer_names)
        layer_cos = [reconstruction_quality[cp_key][l]["cosine_similarity"] for l in layer_names]
        log("SAE", f"  cosine_sim={np.mean(layer_cos):.3f}, alive_features={n_alive_total}")

    # ── Feature matching + event classification ──────────────────────────
    log("SAE", "Matching features and classifying developmental events...")

    # Adaptive thresholds (simplified: use fixed thresholds for demo)
    SSI_THRESH = 0.3
    CSI_THRESH = 0.4
    SAI_THRESH = 0.9

    transitions = []
    for i in range(len(checkpoint_keys) - 1):
        cp_a, cp_b = checkpoint_keys[i], checkpoint_keys[i + 1]
        transition_label = f"{cp_a}->{cp_b}"

        counts = {"ab_h": 0, "tg_h": 0, "di_h": 0, "as_h": 0, "de_h": 0,
                  "stable": 0, "born": 0, "died": 0}
        total_features = 0

        for layer_name in layer_names:
            feat_a = sae_features[cp_a][layer_name]
            feat_b = sae_features[cp_b][layer_name]
            sel_b = selectivity[cp_b][layer_name]
            n_feat = feat_a.shape[1]
            total_features += n_feat

            for j in range(n_feat):
                col_a = feat_a[:, j]
                col_b = feat_b[:, j]

                # Feature matching via Pearson correlation
                if col_a.std() < 1e-8 or col_b.std() < 1e-8:
                    counts["died"] += 1
                    counts["de_h"] += 1
                    continue

                r = np.corrcoef(col_a, col_b)[0, 1]
                if np.isnan(r):
                    r = 0.0

                if r >= 0.5:
                    counts["stable"] += 1
                    # Classify stable feature by selectivity at checkpoint b
                    ssi = sel_b["ssi"][j]
                    csi = sel_b["csi"][j]
                    sai = sel_b["sai"][j]

                    if sai > SAI_THRESH:
                        counts["tg_h"] += 1
                    elif ssi > SSI_THRESH:
                        counts["ab_h"] += 1
                    elif csi > CSI_THRESH:
                        counts["di_h"] += 1
                    else:
                        counts["as_h"] += 1
                elif r < 0.2:
                    counts["died"] += 1
                    counts["de_h"] += 1
                else:
                    counts["born"] += 1
                    # Classify born/transformed features similarly
                    ssi = sel_b["ssi"][j]
                    csi = sel_b["csi"][j]
                    sai = sel_b["sai"][j]
                    if sai > SAI_THRESH:
                        counts["tg_h"] += 1
                    elif ssi > SSI_THRESH:
                        counts["ab_h"] += 1
                    elif csi > CSI_THRESH:
                        counts["di_h"] += 1
                    else:
                        counts["as_h"] += 1

        churn = (counts["born"] + counts["died"]) / max(total_features, 1)

        # Process fractions (exclude de_h for construction/refinement ratio)
        construct = counts["ab_h"] + counts["tg_h"]
        refine = counts["di_h"]
        cr_ratio = construct / refine if refine > 0 else float("inf")

        transitions.append({
            "transition": transition_label,
            **counts,
            "total_features": total_features,
            "churn": churn,
            "cr_ratio": cr_ratio,
        })
        log("SAE", f"  {transition_label}: Ab-E={counts['ab_h']} Di-E={counts['di_h']} "
            f"Tg-E={counts['tg_h']} As-E={counts['as_h']} De-E={counts['de_h']} "
            f"C/R={cr_ratio:.2f}")

    # ══════════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════════
    separator("Pipeline Complete — Results")

    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.0f} seconds ({elapsed / 60:.1f} minutes)\n")

    # ── 1. Reconstruction quality ────────────────────────────────────────
    all_cos = [reconstruction_quality[cp][l]["cosine_similarity"]
               for cp in checkpoint_keys for l in layer_names]
    print(f"  1. SAE Reconstruction Quality")
    print(f"     Mean cosine similarity: {np.mean(all_cos):.3f} +/- {np.std(all_cos):.3f}")
    print(f"     Range: [{np.min(all_cos):.3f}, {np.max(all_cos):.3f}]")
    print(f"     Per checkpoint:")
    for cp_key in checkpoint_keys:
        vals = [reconstruction_quality[cp_key][l]["cosine_similarity"] for l in layer_names]
        print(f"       {cp_key}: {np.mean(vals):.3f} ({', '.join(f'{l}={v:.3f}' for l, v in zip(layer_names, vals))})")
    print()

    # ── 2. Selectivity indices ───────────────────────────────────────────
    print(f"  2. Selectivity Indices (per checkpoint, averaged over alive features)")
    print(f"     {'Checkpoint':<20} {'Alive':>6} {'mean SSI':>9} {'mean CSI':>9} {'mean SAI':>9}")
    print(f"     {'-'*20} {'-'*6} {'-'*9} {'-'*9} {'-'*9}")
    for cp_key in checkpoint_keys:
        all_ssi, all_csi, all_sai = [], [], []
        n_alive = 0
        for layer_name in layer_names:
            sel = selectivity[cp_key][layer_name]
            alive = sel["alive"]
            n_alive += alive.sum()
            if alive.any():
                all_ssi.extend(sel["ssi"][alive])
                all_csi.extend(sel["csi"][alive])
                all_sai.extend(sel["sai"][alive])
        if all_ssi:
            print(f"     {cp_key:<20} {n_alive:>6} {np.mean(all_ssi):>9.3f} "
                  f"{np.mean(all_csi):>9.3f} {np.mean(all_sai):>9.3f}")
    print()

    # ── 3. Feature matching ──────────────────────────────────────────────
    print(f"  3. Feature Matching Across {len(transitions)} Transitions")
    print(f"     {'Transition':<25} {'Stable':>7} {'Born':>7} {'Died':>7} {'Churn':>7}")
    print(f"     {'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for t in transitions:
        print(f"     {t['transition']:<25} {t['stable']:>7} {t['born']:>7} "
              f"{t['died']:>7} {t['churn']:>7.2f}")
    print()

    # ── 4. Developmental events (process intensity) ──────────────────────
    print(f"  4. Developmental Event Classification")
    print(f"     Thresholds: SSI>{SSI_THRESH}, CSI>{CSI_THRESH}, SAI>{SAI_THRESH}")
    print(f"     {'Transition':<25} {'Ab-E':>6} {'Di-E':>6} {'Tg-E':>6} {'As-E':>6} {'De-E':>6} {'C/R':>8}")
    print(f"     {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    for t in transitions:
        cr_str = f"{t['cr_ratio']:.2f}" if t['cr_ratio'] < 1000 else "inf"
        print(f"     {t['transition']:<25} {t['ab_h']:>6} {t['di_h']:>6} "
              f"{t['tg_h']:>6} {t['as_h']:>6} {t['de_h']:>6} {cr_str:>8}")
    print()

    # ── 5. Process fractions ─────────────────────────────────────────────
    print(f"  5. Process Fractions (excluding De-E)")
    print(f"     {'Transition':<25} {'Ab-E%':>7} {'Di-E%':>7} {'Tg-E%':>7} {'As-E%':>7}")
    print(f"     {'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for t in transitions:
        active = t['ab_h'] + t['di_h'] + t['tg_h'] + t['as_h']
        if active > 0:
            print(f"     {t['transition']:<25} {100*t['ab_h']/active:>7.1f} {100*t['di_h']/active:>7.1f} "
                  f"{100*t['tg_h']/active:>7.1f} {100*t['as_h']/active:>7.1f}")
    print()

    # ── 6. Construction/Refinement ratio trajectory ──────────────────────
    print(f"  6. Construction/Refinement Ratio Trajectory")
    print(f"     C/R = (Ab-E + Tg-E) / Di-E")
    for t in transitions:
        cr_str = f"{t['cr_ratio']:.2f}" if t['cr_ratio'] < 1000 else "inf"
        bar_len = min(int(t['cr_ratio'] * 5), 50) if t['cr_ratio'] < 100 else 50
        bar = "=" * bar_len
        print(f"     {t['transition']:<25}  C/R = {cr_str:<8} |{bar}")
    print()

    # ── Save full results ────────────────────────────────────────────────
    # Convert numpy arrays for JSON serialization
    sel_json = {}
    for cp in checkpoint_keys:
        sel_json[cp] = {}
        for ln in layer_names:
            s = selectivity[cp][ln]
            sel_json[cp][ln] = {
                "n_alive": int(s["n_alive"]),
                "mean_ssi": float(np.mean(s["ssi"][s["alive"]])) if s["alive"].any() else 0,
                "mean_csi": float(np.mean(s["csi"][s["alive"]])) if s["alive"].any() else 0,
                "mean_sai": float(np.mean(s["sai"][s["alive"]])) if s["alive"].any() else 0,
            }

    results = {
        "demo_config": {
            "network": "resnet18",
            "dataset": "cifar10_subset_1000",
            "epochs": n_epochs,
            "snapshots": len(snapshots),
            "layers": layer_names,
            "sae_expansion": expansion_factor,
            "sae_k": k_sparse,
            "sae_steps": n_sae_steps,
            "superclass_hierarchy": CIFAR10_SUPERCLASSES,
            "selectivity_thresholds": {"ssi": SSI_THRESH, "csi": CSI_THRESH, "sai": SAI_THRESH},
        },
        "reconstruction_quality": reconstruction_quality,
        "selectivity_evolution": sel_json,
        "process_intensity": [{k: v for k, v in t.items()} for t in transitions],
    }
    results_path = out_dir / "demo_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to: {results_path}")

    # Cleanup
    if data_dir.exists():
        shutil.rmtree(data_dir)
        log("CLEANUP", "Removed temporary dataset cache")

    separator("Demo Complete")
    print("  This demo ran the same 3-stage pipeline used for the paper's")
    print("  55 training lanes, at reduced scale for speed.\n")
    print("  Stages executed:")
    print("    1. NETINIT      — ResNet-18 initialized (kaiming_normal)")
    print("    2. DEVTRAIN     — Trained 3 epochs, captured 3 snapshots")
    print("                      with per-sample activations at 3 layers")
    print("    3. SAEANALYSIS  — Trained SAEs (4x, k=32), matched features,")
    print("                      computed reconstruction quality\n")
    print("  For full reproduction at paper scale:")
    print("    python run/run_pipeline.py --config run/configs/standard/resnet18_cifar100_seed42.json")


if __name__ == "__main__":
    main()
