# run/

Standalone training pipeline that replaces the original FastAPI backend for
clean, single-command reproducibility.

## Quick Start

```bash
# Full training run
python run/run_pipeline.py --config run/configs/standard/resnet18_cifar100_seed42.json

# Dry run (validates config, prints plan, no GPU work)
python run/run_pipeline.py --config run/configs/standard/resnet18_cifar100_seed42.json --dry-run
```

## Config Directory

Pre-built JSON configs are organised by experiment type:

```
run/configs/
  standard/          # Main replication runs (various seeds/architectures)
  curriculum_switch/  # Curriculum-ordering variants (Fig. 6)
  label_noise/        # Targeted label-noise experiments (Fig. 5)
```

## Config Format

Each JSON config specifies:

- `architecture` -- model name (e.g. `resnet18`, `cct7`)
- `dataset` / `num_classes` -- dataset identifier and class count
- `seed` -- random seed for full reproducibility
- `epochs` / `lr_schedule` -- training duration and learning-rate policy
- `sae` -- sparse autoencoder probe settings (width, sparsity, hook points)
- `save_every` -- snapshot frequency (epochs)

## Output Structure

Results are written to a timestamped directory under `output/`:

```
output/<run_id>/
  checkpoints/    # Model + SAE snapshots per epoch
  metrics/        # Per-epoch accuracy, loss, feature statistics
  config.json     # Copy of the config used
```

## Note

This pipeline is self-contained and does not require a running server or
RunPod instance. It replaces the orchestration scripts in `experiments/` for
anyone reproducing results locally or on a single GPU node.
