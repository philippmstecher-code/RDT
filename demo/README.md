# RDT Pipeline Demo

## What this demonstrates

This demo runs the **complete Representational Development Tracing (RDT) pipeline** end-to-end — the same code path used to produce the paper's 55 training lanes, scaled down for speed:

1. **NETINIT** — Creates a ResNet-18 and initializes weights (kaiming_normal, seed 42)
2. **DEVTRAIN** — Trains for 3 epochs on CIFAR-10 (10 classes), capturing 3 uniformly-spaced snapshots with per-sample activations at every intermediate layer
3. **SAEANALYSIS** — At each snapshot, trains a sparse autoencoder (4x expansion, top-k with k=32), matches features across checkpoints via Hungarian algorithm on Pearson correlation, computes selectivity indices (SSI, CSI, SAI), and classifies developmental events against a permutation null baseline

## Requirements

- Python >= 3.10
- PyTorch >= 2.2
- All dependencies from `requirements.txt`
- **No GPU required** (runs on CPU in ~2-5 minutes)
- CIFAR-10 is downloaded automatically on first run (~170 MB)

## Usage

```bash
python demo/demo_sae_analysis.py
```

## Expected output

The script prints progress for each pipeline stage, then a results summary:

- **SAE reconstruction quality**: mean cosine similarity across all checkpoints and layers
- **Developmental event counts**: Ab-E, Di-E, Tg-E, As-E, De-E per transition
- **Construction/Refinement ratio**: (Ab-E + Tg-E) / Di-E at each transition
- **Hypothesis verdicts**: whether each developmental hypothesis (Ab-H, Di-H, Tg-H, As-H, De-H) is supported

Full results are saved to `output/demo_pipeline/demo_cifar10_resnet18/sae_analysis/sae_results.json`.

## Differences from full reproduction

| Parameter | Demo | Paper |
|-----------|------|-------|
| Dataset | CIFAR-10 (10 classes) | CIFAR-100 (100 classes) |
| Epochs | 3 | 50 |
| Snapshots | 3 | 10-12 |
| SAE training steps | 1,000 | 5,000 |
| Null permutations | 10 | 1,000 |
| Runtime | ~2-5 min (CPU) | ~hours (GPU) |

For full reproduction, use:
```bash
python run/run_pipeline.py --config run/configs/standard/resnet18_cifar100_seed42.json
```
