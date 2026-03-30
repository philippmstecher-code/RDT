# Model Card

Following the framework of Mitchell et al. (2019), *Model Cards for Model Reporting*.

## Model Details

This study trains classifiers to study how representations develop during training, not to propose models for deployment. Three architecture families are used:

| Architecture | Parameters | Type | Reference |
|---|---|---|---|
| ResNet-18 | 11.2M | Convolutional (residual) | He et al. 2016 |
| ViT-Small | 22.0M | Vision Transformer | Dosovitskiy et al. 2021 |
| CCT-7 | 3.7M | Compact Convolutional Transformer | Hassani et al. 2021 |

All classifiers output softmax probabilities over 100 classes (CIFAR-100) or 200 classes (Tiny ImageNet). Final classification layers are randomly initialized; backbone weights use Kaiming normal initialization (He et al. 2015).

**Sparse Autoencoders (SAEs)** are trained at each intermediate layer at each training checkpoint to decompose representations into interpretable features:

| Parameter | Value |
|---|---|
| Expansion factor | 4x input dimension (8x in ablation) |
| Sparsity | Top-k with k = 32 |
| Decoder constraint | Unit-norm columns |
| Training steps | 5,000 per SAE |
| Optimizer | Adam, lr = 10^-3 |
| Initialization | Shared across checkpoints (seed-deterministic) |
| Total SAEs trained | 2,420 |

## Training Data

| Dataset | Classes | Superclasses | Train size | Test size | Resolution |
|---|---|---|---|---|---|
| CIFAR-100 | 100 | 20 (5 classes each) | 50,000 | 10,000 | 32 x 32 |
| Tiny ImageNet | 200 | 27 (WordNet-derived) | 100,000 | 10,000 | 64 x 64 |

Both are publicly available benchmark datasets. No filtering or exclusion of samples.

**Preprocessing:**
- RandomCrop (32x32, padding=4) for CIFAR-100; (64x64, padding=8) for Tiny ImageNet
- RandomHorizontalFlip
- Normalize: CIFAR-100 mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2616]

**Data splits:** Standard train/test splits as distributed by the dataset providers. No custom validation split — the test set is used for validation metrics reported in the paper.

## Training Procedure

**Standard runs (9 lanes):**

| Parameter | ResNet-18 | ViT-Small | CCT-7 |
|---|---|---|---|
| Optimizer | SGD | AdamW | AdamW |
| Learning rate | 0.1 | 5 x 10^-4 | 5 x 10^-4 |
| Weight decay | 5 x 10^-4 | 0.03 | 0.03 |
| Momentum | 0.9 (Nesterov) | — | — |
| Batch size | 128 | 128 | 128 |
| Epochs | 50 | 50 | 50 |
| LR schedule | Cosine annealing | Cosine annealing | Cosine annealing |
| Seeds | 42, 137, 256 | 42, 137, 256 | 42, 137, 256 |

**Snapshot capture:** 10-12 uniformly spaced checkpoints per run. At each checkpoint, per-sample activations are extracted at all intermediate layers for the full validation set (~10,000 samples for CIFAR-100).

**Experimental manipulations:**
- *Curriculum switching* (7 conditions): Superclass labels for the first N epochs, then fine-class labels. N in {0, 5, 10, 15, 20, 25, 30}.
- *Targeted label noise* (7 conditions x 5 seeds = 35 runs): Between-superclass, within-superclass, or random label corruption at p = 0.1 or 0.3. Trained for 30 epochs with 8 checkpoints.
- *Ablations*: Extended training (200 epochs), higher SAE capacity (8x), independent SAE initialization, cross-dataset (Tiny ImageNet).

**Total: 55 training runs across 27 unique conditions.**

## Performance

Classification accuracy is not the goal of this study — models are trained to study representational dynamics, not to achieve state-of-the-art performance. Reported accuracies are for context only.

**Standard runs (CIFAR-100, 50 epochs):**

| Architecture | Seed 42 | Seed 137 | Seed 256 |
|---|---|---|---|
| ResNet-18 | 59.8% | 59.6% | 59.0% |
| ViT-Small | 49.2% | 48.1% | 48.2% |
| CCT-7 | 53.0% | 52.4% | 53.0% |

**Cross-dataset (Tiny ImageNet):** ResNet-18 seed 42: 53.0%

**SAE reconstruction quality:** Mean cosine similarity between original and reconstructed activations: 0.975 +/- 0.029 across all architectures, stages, and layers.

## Intended Use

These models are research artifacts for studying representational development in neural networks. They are **not intended for deployment** in any application. The classifiers serve as substrates for SAE-based feature decomposition; their absolute accuracy is secondary to the developmental trajectory they exhibit during training.

## Limitations

- Accuracies are below state-of-the-art for these datasets (the study prioritizes capturing training dynamics over final performance).
- Training hyperparameters follow standard practice but were not extensively tuned.
- The SAE decomposition assumes that top-k sparse features are a meaningful unit of analysis; this is supported by reconstruction quality (cosine sim > 0.97) but is a methodological choice.
- Superclass structure for CIFAR-100 uses the canonical 20-superclass grouping; Tiny ImageNet uses WordNet-derived groupings with unequal superclass sizes.

## Ethical Considerations

- No human subjects data.
- No personally identifiable information.
- CIFAR-100 and Tiny ImageNet are standard academic benchmarks with no known ethical concerns beyond general dataset bias considerations.
- The findings describe learning dynamics in artificial systems and do not make claims about biological neural development.

## Citation

```bibtex
@article{stecher2026scaffolded,
  title   = {Deep networks learn scaffolded representations},
  author  = {Stecher, Philipp and Radovanovic, Sandro and Sikimic, Vlasta and Kahle, Reinhard},
  journal = {Nature Machine Intelligence},
  year    = {2026}
}
```
