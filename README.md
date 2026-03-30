# Scaffolded representation learning in deep networks

Philipp Stecher<sup>1</sup>, Sandro Radovanovic<sup>2</sup>, Vlasta Sikimic<sup>3</sup>, Reinhard Kahle<sup>1,4</sup>

---

## Abstract

Deep networks learn coarse structure before fine-grained distinctions, yet whether coarse structure actively scaffolds later differentiation remains untested. Here we show that representations assemble through a load-bearing scaffold. Tracking features at per-sample resolution across 55 runs, three architecture families and two training datasets, we find a reproducible three-phase program: task-general features emerge and dominate first, superclass groupings form next, and class-level distinctions develop last. Selectively corrupting superclass boundaries impairs later differentiation whereas matched random noise does not, suggesting that fine-grained learning depends on the coherence of coarser representations. Conversely, a curriculum that pre-builds the scaffold reduces differentiation cost 6.7-fold while nearly preserving accuracy and halving overfitting. These findings connect critical learning periods, neural collapse, progressive differentiation, the lottery ticket hypotheses, and catastrophic forgetting within a single developmental account and provide training diagnostic insights relevant for curriculum design, transfer timing, and mechanistic interpretability.

---

## Pipeline Overview

The Representational Development Tracing (RDT) pipeline has six steps (Fig. 1a), implemented in `src/` and invocable via `run/run_pipeline.py`.

### Step 1: Training and Checkpoint Capture (`src/devtrain.py`)

A classifier (ResNet-18, ViT-Small, or CCT-7) is trained on CIFAR-100 or Tiny ImageNet. Checkpoints are spaced evenly by validation accuracy to capture key representational shifts — naturally dense early, sparse late in training. At each checkpoint, per-sample activations at every intermediate layer are recorded, producing 10-12 snapshots per run.

**Experimental manipulations** at this step:
- **Curriculum switching** (Fig. 6): Superclass labels for the first N epochs, then fine-class labels
- **Targeted label noise** (Fig. 5): Selective corruption of between-superclass, within-superclass, or random label structure

### Step 2: SAE Decomposition into Sparse Features (`src/sae.py`, `src/saeanalysis.py`)

At each checkpoint, a sparse autoencoder (SAE) with top-k sparsity (k=32) and unit-norm decoder columns is trained on the captured activations (X ≈ D · Z). This decomposes each layer's representation into ~160-210 interpretable features per layer. SAEs share initialization across checkpoints to enable feature tracking.

### Step 3: Feature Matching and Lifecycle Classification (`src/saeanalysis.py`)

Features are tracked across consecutive checkpoints via Hungarian matching on Pearson correlation of activation profiles (response magnitudes across all validation samples). Each feature is classified as **born** (newly active), **stable** (active in both), **transformed**, or **died** (no longer active). Stability threshold: r >= 0.5; death threshold: r < 0.2.

### Step 4: Process Identification via Selectivity Measurement (`src/saeanalysis.py`)

Three selectivity indices measure each feature's scope:
- **SAI** (Superclass Abstraction Index): task-general — feature fires uniformly across all classes and superclasses
- **SSI** (Superclass Selectivity Index): feature responds selectively to all classes in one superclass
- **CSI** (Class Selectivity Index): feature responds to one class within a superclass

### Step 5: Adaptive Threshold Calibration (`src/saeanalysis.py`)

To separate real selectivity from chance, thresholds are set at the 95th percentile of null distributions from 1,000 permutations of class-to-superclass assignments (SSI, SAI) and sample-to-class assignments (CSI).

### Step 6: Process Classification (`src/saeanalysis.py`)

Lifecycle state (born, stable, died) is combined with selectivity type (SAI, SSI, CSI) to label each feature event as one of five developmental processes: Assembly (As-E), Task-General (Tg-E), Abstraction (Ab-E), Differentiation (Di-E), or Decay (De-E). These are aggregated into developmental trajectories, tested for temporal ordering (Granger causality), and validated through experimental manipulations. All downstream analyses read from pre-computed JSON summary files in `data/`.

---

## Repository Structure

```
scaffolded-development/
├── src/                          Core ML library (Steps 1-6)
│   ├── devtrain.py               Training with periodic snapshot capture
│   ├── saeanalysis.py            Per-checkpoint SAE decomposition + event classification
│   ├── training.py               Data loading, optimizers, activation extraction
│   ├── models.py                 ResNet-18, ViT-Small architecture definitions
│   ├── cct.py                    CCT-7 (Compact Convolutional Transformer)
│   ├── sae.py                    Sparse autoencoder (top-k sparsity, unit-norm decoder)
│   ├── pathways.py               Attribution patching validation
│   ├── initialization.py         Weight initialization strategies
│   ├── stats.py                  Statistical utilities
│   ├── cifar100_hierarchy.py     CIFAR-100 superclass mapping (20 superclasses of 5)
│   ├── tiny_imagenet_hierarchy.py
│   └── inat_hierarchy.py         WordNet hierarchy for Tiny ImageNet superclasses
│
├── run/                          Standalone pipeline runner (no backend needed)
│   ├── run_pipeline.py           CLI: NETINIT -> DEVTRAIN -> SAEANALYSIS
│   └── configs/                  Pre-built JSON configs for all 27 conditions
│       ├── standard/             9 standard + 3 ablation + 1 cross-dataset (13 configs)
│       ├── curriculum_switch/    7 switch-point conditions (6 configs)
│       └── label_noise/          7 conditions x 5 seeds (35 configs)
│
├── analysis/                     Statistical analysis and validation
│   ├── validate_manuscript.py    Verifies all 128 quantitative claims
│   ├── granger_causality_*.py    Temporal precedence tests (Ab-E -> Di-E)
│   ├── compute_feature_survival*.py   Kaplan-Meier survival analysis
│   ├── predict_terminal_accuracy.py   Early-feature predictive models
│   └── bootstrap_noise_ci.py     BCa CIs and Cohen's d for noise experiments
│
├── figures/                      Figure generation (all read from data/)
│   ├── README.md                 Script-to-figure mapping (critical: script != figure numbers)
│   ├── plot_fig1.py              Fig. 1: Three-phase developmental program
│   ├── plot_fig4.py              Fig. 2: Cross-architecture replication
│   ├── plot_fig2.py              Fig. 3: Feature turnover window
│   ├── plot_fig5.py              Fig. 4: Developmental persistence
│   ├── plot_fig5_label_noise.py  Fig. 5: Targeted label noise
│   ├── plot_fig8.py          Fig. 6: Curriculum sweep
│   ├── plot_extended_data.py     ED Figs. 1-8
│   └── plot_ed_table1.py         ED Table 1
│
├── experiments/                  Original training orchestration (hyperparameter provenance)
├── data/                         Pre-computed analysis results (JSON)
│   ├── DATA_README.md            Complete data dictionary and figure mappings
│   └── raw_lanes/                Per-lane SAE results (on Zenodo)
├── demo/                         Lightweight demo (runs full pipeline on CIFAR-10 subset)
├── generate_source_data.py       Generates Source Data Excel files
└── tests/
```

## System Requirements

- **Python** >= 3.10
- **PyTorch** >= 2.2
- **GPU**: Not needed for figures, analysis, or validation. Required only for training replication (Step 1).
- **Typical install time**: < 5 minutes

## Installation

```bash
git clone https://github.com/<repo-url>/scaffolded-development.git
cd scaffolded-development
pip install -e .
```

---

## Reproducing Results

### Quick Start: Demo (~30 seconds)

```bash
python demo/demo_sae_analysis.py
```

Loads one pre-computed lane and reproduces the core finding: the construction/refinement ratio declines from 39.4:1 to 0.30:1 across training, demonstrating the construct-before-refine developmental program.

### Tier 1: Reproduce All Figures from Pre-computed Data (~5 min, CPU)

All figure scripts read from `data/` and write to `output/figures/`. No training or GPU required.

```bash
python figures/plot_fig1.py              # Fig. 1: Three-phase sequence
python figures/plot_fig4.py              # Fig. 2: Cross-architecture replication
python figures/plot_fig2.py              # Fig. 3: Feature turnover
python figures/plot_fig5.py              # Fig. 4: Developmental persistence
python figures/plot_fig5_label_noise.py  # Fig. 5: Targeted label noise
python figures/plot_fig8.py          # Fig. 6: Curriculum sweep
python figures/plot_extended_data.py     # ED Figs. 1-8
python figures/plot_ed_table1.py         # ED Table 1
```

> **Note:** Script filenames do not always match figure numbers. See `figures/README.md` for the definitive script-to-figure mapping.

### Tier 2: Validate All Quantitative Claims (~10 min, CPU)

```bash
# Verify all 128 quantitative claims against data files
python analysis/validate_manuscript.py    # Expect: 128 PASS, 0 FAIL

# Reproduce key statistics cited in the text
python analysis/bootstrap_noise_ci.py     # Cohen's d = -1.57, Di-E deficit = -7.8 pp
```

### Tier 3: Full Pipeline Reproduction (GPU, ~days)

This replicates the entire pipeline from scratch: dataset download, network training with snapshot capture, SAE decomposition, and event classification.

**Step 1 — Run a single lane:**

```bash
python run/run_pipeline.py --config run/configs/standard/resnet18_cifar100_seed42.json
```

This executes:
1. **NETINIT**: Creates a ResNet-18, initializes weights (kaiming_normal), saves `initial.pt`
2. **DEVTRAIN**: Trains for 50 epochs on CIFAR-100 with SGD (lr=0.1, momentum=0.9). Captures 10 uniformly-spaced snapshots with per-sample activations at all intermediate layers
3. **SAEANALYSIS**: At each of the 10 checkpoints, trains a 4x-expansion SAE (top-k, k=32) per layer. Matches features across checkpoints, computes SSI/CSI/SAI, classifies developmental events against 1,000-permutation null baseline

Output: `output/<experiment_name>/sae_analysis/sae_results.json`

**Step 2 — Reproduce all 55 lanes:**

Pre-built configs for all 27 experimental conditions are in `run/configs/`:

```bash
# Standard replication (9 lanes: 3 architectures x 3 seeds)
python run/run_pipeline.py --config run/configs/standard/resnet18_cifar100_seed42.json
python run/run_pipeline.py --config run/configs/standard/vitsmall_cifar100_seed42.json
python run/run_pipeline.py --config run/configs/standard/cct7_cifar100_seed42.json
# ... (13 configs total in standard/)

# Curriculum switch experiment (Fig. 6)
python run/run_pipeline.py --config run/configs/curriculum_switch/switch_e15.json
# ... (6 configs in curriculum_switch/)

# Targeted label noise experiment (Fig. 5)
python run/run_pipeline.py --config run/configs/label_noise/between_sc_p03_seed42.json
# ... (35 configs in label_noise/)
```

**Step 3 — Regenerate analysis data and figures:**

After training, run the analysis scripts to regenerate the summary JSONs in `data/`, then regenerate figures as in Tier 1.

**Dry run** (validates config without training):
```bash
python run/run_pipeline.py --config run/configs/standard/resnet18_cifar100_seed42.json --dry-run
```

---

## Data Availability

**In this repository** (`data/`): Pre-computed summary statistics (25 JSON files, ~3 MB) sufficient to reproduce all figures, extended data, and the 128-claim validation. See `data/DATA_README.md` for the complete data dictionary.

**On Zenodo** [DOI to be added upon acceptance]: Per-lane SAE decomposition results for all 55 training runs (~591 MB). Download into `data/raw_lanes/`.

**Public datasets**: CIFAR-100 and Tiny ImageNet are downloaded automatically by PyTorch during training.

## License

MIT License. See [LICENSE](LICENSE).

## Contact

Philipp Stecher — philippmstecher@gmail.com
