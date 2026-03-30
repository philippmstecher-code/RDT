# Data Package — "Deep networks learn scaffolded representations"

**Data package generated:** 2026-03-26

---

## Overview

This directory contains **all quantitative data** underlying the findings, figures, and
statistical claims in the paper. Every number cited in the text and every
panel in Figures 1–6 and Extended Data Figures 1–10 can be reproduced from these files.

The data derives from **55 training runs** spanning **27 unique conditions** across
**three architecture families** (ResNet-18, ViT-Small, CCT-7), **two datasets**
(CIFAR-100 and Tiny ImageNet), and multiple experimental manipulations:

| Category | Runs | Description |
|----------|------|-------------|
| Standard replication | 9 | 3 architectures × 3 seeds (42, 137, 256), CIFAR-100, 50 epochs |
| Ablation/control | 3 | Extended training (200 ep), higher SAE capacity (8×), independent init |
| Cross-dataset | 1 | ResNet-18 on Tiny ImageNet |
| Curriculum switch | 7 | 7 switch points (0, 5, 10, 15, 20, 25, 30 epochs of superclass pre-training) |
| Targeted label noise | 35 | 7 conditions × 5 seeds (42, 137, 256, 7, 314) |
| **Total** | **55** | |

---

## Directory Structure

```
data/
├── DATA_README.md                              <- this file
├── consolidated_findings.json                  <- master aggregation (12 lanes + cross-lane statistics)
├── summary_statistics.md                       <- human-readable summary of all cited statistics
├── feature_survival_all_lanes.json             <- feature survival analysis (log-rank, Spearman)
├── feature_survival_tg_expanded.json           <- Tg-H cohort survival analysis
├── feature_survival_3cohort_perlayer.json       <- per-layer 3-cohort survival
├── developmental_persistence.json              <- lifespan ratios by selectivity index
├── pathway_analysis_summary.json               <- causal attribution patching results
├── causal_intervention_summary.json            <- subspace perturbation experiment (Fig. 5)
├── curriculum_switch_summary.json              <- 7-condition curriculum sweep (Fig. 6)
├── superclass_epoch_metrics.json               <- curriculum per-epoch accuracy
├── superclass_sae_analysis_summary.json        <- curriculum SAE analysis (2-condition)
├── targeted_label_noise_summary.json           <- label noise: single-seed (7 conditions)
├── targeted_label_noise_summary_5seeds.json    <- label noise: 5-seed (7 conditions × 5 seeds)
└── raw_lanes/                                  <- per-lane raw SAE analysis results
    ├── ResNet18_CIFAR100_seed{42,137,256}.json                    (3 files)
    ├── ViTSmall_CIFAR100_seed{42,137,256}.json                    (3 files)
    ├── CCT7_CIFAR100_seed{42,137,256}.json                        (3 files)
    ├── ResNet18_CIFAR100_200ep_seed42.json                        (1 file)
    ├── ResNet18_CIFAR100_8x_seed42.json                           (1 file)
    ├── ResNet18_CIFAR100_independent_init_control.json            (1 file)
    ├── ResNet18_TinyImageNet_seed42.json                          (1 file)
    └── ResNet18_CIFAR100_label_noise_{condition}_seed{seed}.json  (28+ files)
```

---

## Experimental Lanes

### Standard Replication (9 runs)

| # | File | Architecture | Dataset | Seed | Epochs | SAE Expansion | Layers | Checkpoints | Val Acc (%) |
|---|------|-------------|---------|------|--------|---------------|--------|-------------|------------|
| 1 | `ResNet18_CIFAR100_seed42.json` | ResNet-18 (11.2M) | CIFAR-100 | 42 | 50 | 4x | 5 (layer1-4 + avgpool) | 12 | 59.8 |
| 2 | `ResNet18_CIFAR100_seed137.json` | ResNet-18 | CIFAR-100 | 137 | 50 | 4x | 5 | 12 | 59.6 |
| 3 | `ResNet18_CIFAR100_seed256.json` | ResNet-18 | CIFAR-100 | 256 | 50 | 4x | 5 | 12 | 59.0 |
| 4 | `ViTSmall_CIFAR100_seed42.json` | ViT-Small (22.0M) | CIFAR-100 | 42 | 50 | 4x | 6 (encoder.layers.{0,3,6,9,11} + ln) | 12 | 49.2 |
| 5 | `ViTSmall_CIFAR100_seed137.json` | ViT-Small | CIFAR-100 | 137 | 50 | 4x | 6 | 12 | 48.1 |
| 6 | `ViTSmall_CIFAR100_seed256.json` | ViT-Small | CIFAR-100 | 256 | 50 | 4x | 6 | 12 | 48.2 |
| 7 | `CCT7_CIFAR100_seed42.json` | CCT-7 (3.7M) | CIFAR-100 | 42 | 50 | 4x | 6 (transformer.layers.{0,1,3,5,6} + norm) | 12 | 53.0 |
| 8 | `CCT7_CIFAR100_seed137.json` | CCT-7 | CIFAR-100 | 137 | 50 | 4x | 6 | 12 | 52.4 |
| 9 | `CCT7_CIFAR100_seed256.json` | CCT-7 | CIFAR-100 | 256 | 50 | 4x | 6 | 12 | 53.0 |

### Ablation and Control Runs (3 + 1 cross-dataset)

| # | File | Condition | Architecture | Dataset | Seed | Epochs | SAE | Val Acc (%) |
|---|------|-----------|-------------|---------|------|--------|-----|------------|
| 10 | `ResNet18_CIFAR100_200ep_seed42.json` | Extended training | ResNet-18 | CIFAR-100 | 42 | 200 | 4x | 60.9 |
| 11 | `ResNet18_CIFAR100_8x_seed42.json` | Higher SAE capacity | ResNet-18 | CIFAR-100 | 42 | 50 | 8x | 59.8 |
| C | `ResNet18_CIFAR100_independent_init_control.json` | Independent init | ResNet-18 | CIFAR-100 | 42 | 50 | 4x | 59.8 |
| 13 | `ResNet18_TinyImageNet_seed42.json` | Cross-dataset | ResNet-18 | Tiny ImageNet | 42 | 50 | 4x | 53.0 |

### Curriculum Switch Experiment (7 runs, Fig. 6)

Seven ResNet-18 runs (seed 42, CIFAR-100, 50 epochs) varying the duration of superclass pre-training from 0 to 30 epochs in 5-epoch steps:

| # | Condition | SC Epochs | Fine Epochs | Di-H Events | Val Acc (%) | Overfit Gap (pp) |
|---|-----------|-----------|-------------|-------------|-------------|-----------------|
| 14 | Standard | 0 | 50 | 358,816 | 59.1 | 37.1 |
| 15 | switch_e05 | 5 | 45 | 99,974 | 57.9 | 33.0 |
| 16 | switch_e10 | 10 | 40 | 96,357 | 58.2 | 31.6 |
| 17 | switch_e15 | 15 | 35 | 53,872 | 57.1 | 27.8 |
| 18 | switch_e20 | 20 | 30 | 91,430 | 57.9 | 25.2 |
| 19 | switch_e25 | 25 | 25 | 55,315 | 58.2 | 21.6 |
| 20 | switch_e30 | 30 | 20 | 77,578 | 58.2 | 18.5 |

Data file: `curriculum_switch_summary.json`

**Key finding**: Up to 6.7-fold reduction in differentiation events (358,816 to 53,872) with only 15 epochs of scaffolding, while maintaining 96.5% of standard accuracy.

### Subspace-Perturbation Experiment (Fig. 5)

| Parameter | Value |
|-----------|-------|
| Base lane | ResNet-18, CIFAR-100, seed 42 (8x expansion) |
| Intervention checkpoint | 6 (epoch 11) |
| Target superclass | flowers |
| Features targeted | 9 Ab-H features (SSI >= 0.95) |
| Perturbation strengths | alpha = 0.0 (control), 0.5, 1.0 |
| Controls | Unperturbed + random-direction (equal rank) |
| Data file | `causal_intervention_summary.json` |

**Key finding**: Targeted removal of 9 Ab-H features suppresses flower accuracy by -8.6 pp (alpha=1.0) with 21x specificity (flower vs non-flower impact).

### Targeted Label Noise Experiment (35 runs, Fig. 4)

Tests whether abstraction causally scaffolds differentiation by selectively corrupting label structure.
All lanes use ResNet-18 on CIFAR-100 (30 epochs, 8 checkpoints, 5 layers, 4x SAE expansion).

| Condition | Noise Type | Noise Prob | Seeds | Description |
|-----------|-----------|------------|-------|-------------|
| `standard` | None | 0 | 42, 137, 256, 7, 314 | No-noise baseline |
| `within_sc_p01` | Within-superclass | 0.1 | 42, 137, 256, 7, 314 | Attacks Di-H directly |
| `within_sc_p03` | Within-superclass | 0.3 | 42, 137, 256, 7, 314 | Attacks Di-H directly |
| `between_sc_p01` | Between-superclass | 0.1 | 42, 137, 256, 7, 314 | Attacks Ab-H scaffold |
| `between_sc_p03` | Between-superclass | 0.3 | 42, 137, 256, 7, 314 | Attacks Ab-H scaffold |
| `random_p01` | Uniform random | 0.1 | 42, 137, 256, 7, 314 | Difficulty-matched control |
| `random_p03` | Uniform random | 0.3 | 42, 137, 256, 7, 314 | Difficulty-matched control |

**Key finding**: Between-SC noise at p=0.3 suppresses Di-H by -7.8 +/- 2.2 pp more than random noise (Cohen's d = -1.57, all 5 seeds negative, sign test p = 0.031).

Data files: `targeted_label_noise_summary_5seeds.json` (5-seed aggregated) + 28 per-seed raw lane files in `raw_lanes/`.

---

## SAE Pipeline Parameters

| Parameter | Value |
|-----------|-------|
| SAE architecture | Top-k sparsity (k=32), unit-norm decoder columns |
| Expansion factor | 4x d_in (standard); 8x for ablation lane 11 |
| Training steps | 5,000 per SAE |
| Optimizer | Adam |
| Shared initialisation | Across checkpoints within each lane |
| Feature matching | Hungarian algorithm on Pearson r (activation columns) |
| Stable threshold | r >= 0.5 |
| Death threshold | r < 0.2 |
| Null permutations | 1,000 |
| Adaptive thresholds | SSI, CSI, SAI from 95th percentile of permutation null |
| Total SAEs trained | 2,420 |
| Total features tracked | ~250,000 |
| Total feature-transition observations | ~2.8 million |

### Reconstruction Quality

Mean cosine similarity: 0.975 +/- 0.029 across all architectures, stages, and layers.

| Stage | Cosine Similarity |
|-------|------------------|
| Early | 0.980 +/- 0.031 |
| Mid | 0.990 +/- 0.008 |
| Late | 0.955 +/- 0.030 |

---

## File Schemas

### Raw Lane Files (`raw_lanes/*.json`)

Each file contains the complete SAE analysis for one training lane:

| Key | Description |
|-----|-------------|
| `metadata` | Architecture, dataset, layer names, checkpoint labels, adaptive thresholds, null baseline parameters |
| `reconstruction_quality` | Per-checkpoint, per-layer SAE reconstruction metrics (MSE, cosine similarity) |
| `within_checkpoint_control` | Within-checkpoint null test — verifies SAE features are not random |
| `selectivity` | Per-checkpoint, per-layer selectivity index distributions (SSI, CSI, SAI) |
| `selectivity_evolution` | Per-checkpoint mean SSI, CSI, SAI trajectories |
| `feature_matching` | Per-transition, per-layer feature matching: n_stable, n_born, n_died, n_transformed |
| `feature_landscape` | Per-checkpoint, per-layer feature counts |
| `process_intensity` | Per-transition event counts: Ab-E, Tg-E, Di-E, As-E, De-E, plus churn rate |
| `hypotheses` | Verdict for each of 5 developmental hypotheses |
| `null_baseline` | Permutation-test results (SSI null distribution from 1,000 random class->superclass assignments) |
| `class_process_summary` | Per-class process breakdown across all transitions |
| `superclass_summary` | Per-superclass process breakdown |
| `discrimination_gradients` | Superclass-level and fine-class-level discrimination gradient evolution |
| `weight_matching` | Feature matching using SAE weight similarity |

### `consolidated_findings.json` (1.0 MB)

Master aggregation of 12 primary lanes (9 standard + 3 ablation/control). Structure:

```json
{
  "generated": "2026-03-21",
  "lanes": {
    "<lane_label>": {
      "metadata": { "architecture", "dataset", "seed", "epochs", "expansion", "adaptive_thresholds" },
      "process_intensity": [ { "transition", "ab_h", "tg_h", "di_h", "as_h", "de_h", "churn" }, ... ],
      "selectivity_evolution": [ ... ],
      "hypotheses": { "Ab-H", "Tg-H", "Di-H", "As-H", "De-H" },
      "abh_dih_ratios": [ { "ab_h", "tg_h", "di_h", "ratio", "churn" }, ... ],
      "first_transition_layer_stability": { ... }
    }
  },
  "cross_lane_statistics": {
    "f1_abh_dih_first_transition", "f1_abh_dih_last_transition",
    "f1_tg_fraction_first_transition", "f1_tg_fraction_by_architecture",
    "f2_churn_first_transition", "f2_churn_last_transition",
    "f3_layer_stability_first_transition",
    "f4_superclass_invariance",
    "hypothesis_verdicts"
  }
}
```

### `curriculum_switch_summary.json`

7-condition curriculum switch-point sweep. Parallel-array structure:

| Field | Description |
|-------|-------------|
| `conditions` | List of 7 condition names: `["standard", "switch_e05", ..., "switch_e30"]` |
| `accuracy.val_acc_terminal` | Terminal validation accuracy per condition (7 values) |
| `accuracy.overfit_gap_pp` | Train-val gap per condition (7 values) |
| `process_events.Di-H` | Total differentiation events per condition (7 values) |
| `process_events.Stable` | Total stable features per condition (7 values) |
| `selectivity.SSI/CSI` | Per-checkpoint selectivity trajectories per condition |
| `developmental_trajectories` | Per-transition Ab-H, Di-H, Tg-H breakdown per condition |

### `targeted_label_noise_summary_5seeds.json`

5-seed label noise experiment. Structure:

| Field | Description |
|-------|-------------|
| `conditions.<cond>.summary` | Per-condition mean/SEM for di_frac, ab_frac, tg_frac |
| `conditions.<cond>.per_seed.<seed>` | Per-seed data: lane_id, process_fractions, process_events_per_transition |

### `feature_survival_all_lanes.json`

Feature survival analysis for 12 lanes:

| Field | Description |
|-------|-------------|
| `<lane>.n_total_tracked` | Total features tracked across all layers |
| `<lane>.overall_survival_rate` | Fraction surviving all transitions |
| `<lane>.logrank_chi2`, `logrank_p` | Log-rank test: high-SSI vs low-SSI cohort |
| `<lane>.ssi_survival_corr`, `csi_survival_corr` | Spearman correlations: selectivity vs lifespan |

### `causal_intervention_summary.json`

Subspace-perturbation experiment:

| Field | Description |
|-------|-------------|
| `experiment` | Parameters: n_abh_features_targeted=9, intervention_checkpoint=6 |
| `conditions` | Terminal accuracy for 4 conditions: control, alpha=0.5, alpha=1.0, random_control |
| `trajectory_accuracy` | Per-milestone accuracy split by flowers/rest |

---

## Mapping: Data -> Figures

| Figure | Data Source | Key Fields |
|--------|-------------|------------|
| **Fig 1a** | N/A (schematic) | - |
| **Fig 1b** | `consolidated_findings.json` -> `lanes.*.process_intensity` | Ab-E, Tg-E, Di-E fractions |
| **Fig 1c** | `consolidated_findings.json` -> `lanes.*.abh_dih_ratios` | C/R ratio trajectories |
| **Fig 2a-c** | `consolidated_findings.json` -> `lanes.*.selectivity_evolution` | SSI, CSI, process intensity, churn |
| **Fig 2d** | `consolidated_findings.json` -> `lanes.*.feature_matching` | born/died/stable/transformed |
| **Fig 3a-i** | `consolidated_findings.json` -> `lanes.{ResNet18,ViTSmall,CCT7}-seed42` | Heatmaps: stability, SSI, CSI |
| **Fig 4a-d** | `targeted_label_noise_summary_5seeds.json` | Process fractions, paired differences |
| **Fig 5a-c** | `causal_intervention_summary.json` | Flower/rest accuracy trajectories, dose-response |
| **Fig 6a-d** | `curriculum_switch_summary.json` | Di-H, stable features, overfit gap, accuracy |
| **ED Fig 1** | `raw_lanes/*.json` -> `within_checkpoint_control` | False-positive rates |
| **ED Fig 2-4** | `consolidated_findings.json` -> cross-architecture | Replication across architectures |
| **ED Fig 5-6** | `consolidated_findings.json` -> ablation lanes | Extended training, 8x capacity |
| **ED Fig 7** | `consolidated_findings.json` -> `cross_lane_statistics.f4_superclass_invariance` | Cross-architecture SD |
| **ED Fig 8** | `raw_lanes/ResNet18_CIFAR100_independent_init_control.json` | Independent init C/R ratio |
| **ED Fig 9** | `targeted_label_noise_summary_5seeds.json` | Compensatory proliferation, alive features |
| **ED Fig 10** | Computed from 9 CIFAR-100 raw lanes | Activation magnitude vs selectivity |
| **ED Table 1** | All raw lanes + curriculum + noise per-seed files | Val accuracy for all 55 runs |

---

## Mapping: Data -> Key Claims

| Claim (Section) | Value | Data Source | Verification |
|-----------------|-------|-------------|-------------|
| C/R ratio at first transition (Results) | 39.4:1 | `consolidated_findings.json` -> ResNet18-seed42 abh_dih_ratios[0] | (4070+154637)/4024 = 39.4 |
| Terminal C/R ratio (Results) | 0.30:1 | Same -> abh_dih_ratios[-1] | (4850+0)/16190 = 0.30 |
| Sign test for ratio decline | p = 0.002 | All 9 CIFAR-100 lanes | 9/9 declining, binomial test |
| 6 of 9 full crossover | 6/9 < 1.0 | Terminal ratios: [0.30, 0.41, 0.55, 0.68, 0.76, 0.88, 1.61, 3.47, 4.39] | 6 below 1.0 |
| Feature churn | 81.5% | ResNet18-seed42 churn at first transition | 0.8151 |
| 6.7-fold Di-E reduction | 358816/53872 | `curriculum_switch_summary.json` | standard vs switch_e15 |
| 85% fewer Di-E | (358816-53872)/358816 | Same | 85.0% |
| Overfit gap halved | 37.1->18.5 pp | `curriculum_switch_summary.json` accuracy.overfit_gap_pp | standard vs switch_e30 |
| Di-E selective deficit | -7.8 +/- 2.2 pp | `targeted_label_noise_summary_5seeds.json` | between_sc_p03 minus random_p03, paired per seed |
| Cohen's d | -1.57 | Computed from 5 paired diffs | mean=-7.76, sd=4.96 |
| Flower suppression | -8.6 pp (alpha=1.0) | `causal_intervention_summary.json` | Terminal flower accuracy |
| Random control baseline | 17.67% | Same -> conditions.random_control | final_val_accuracy |
| SSI positive in 7/9 lanes | 7/9 | `feature_survival_all_lanes.json` ssi_survival_corr | CCT7-s137 (-0.006) and CCT7-s256 (-0.038) negative |
| CSI negative in 9/9 lanes | 9/9 | Same -> csi_survival_corr | All negative |
| Cosine similarity | 0.975 +/- 0.029 | `consolidated_findings.json` | Grand mean across 3 arch x 3 stages |
| 1,000 permutations | 1,000 | Raw sae_results.json on pipeline data | n_permutations=1000 |
| 55 training runs | 55 | 9+3+1+7+35 | All experimental categories |
| 2,420 SAEs | 2,420 | 600+420+1400 | Primary + curriculum + noise |

---

## Reproducibility

All figure generation scripts are in `figures/`:

| Script | Figure |
|--------|---------------|
| `plot_fig1.py` | Fig. 1 (three-phase coarse-to-fine sequence) |
| `plot_fig4.py` | Fig. 2 (cross-architecture replication) |
| `plot_fig2.py` | Fig. 3 (feature turnover) |
| `plot_fig5.py` | Fig. 4 (developmental persistence) |
| `plot_fig5_label_noise.py` | Fig. 5 (targeted label noise) |
| `plot_fig8.py` | Fig. 6 (curriculum sweep) |
| `plot_extended_data.py` | ED Figs. 1–8 |
| `plot_ed_table1.py` | ED Table 1 |
| `plot_activation_magnitude_selectivity.py` | ED Fig. 8 (selectivity vs magnitude) |

Statistical computation scripts in `analysis/`:

| Script | Purpose |
|--------|---------|
| `compute_feature_survival.py` | Kaplan-Meier and log-rank tests (Fig 3d-e) |
| `compute_feature_survival_tg.py` | Tg-H cohort survival analysis |
| `bootstrap_noise_ci.py` | BCa bootstrap CIs and Cohen's d for label noise |

---

## Validation

A comprehensive data validation script (`analysis/validate_manuscript.py`) verifies all 128 quantitative claims against the data files in this package. Run:

```bash
python3 analysis/validate_manuscript.py
```

Output: `data/data_validation_report.md` with per-claim verification status.

**Current status (2026-03-26):** 128 PASS, 0 FLAG, 0 FAIL.
