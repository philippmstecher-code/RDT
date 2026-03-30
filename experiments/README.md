# experiments/

Original RunPod orchestration scripts used to launch the 55 training runs
reported in the paper. These scripts call the FastAPI training backend via HTTP
API and are kept here for **hyperparameter provenance** -- they document the
exact configuration used for each experiment.

**For standalone reproduction**, use `run/run_pipeline.py` with the pre-built
JSON configs in `run/configs/` instead.

## Scripts

| Script | Manuscript runs |
|---|---|
| `run_seed256.py` | 3 standard replication runs (ResNet18, ViT-Small, CCT7, seed 256) |
| `run_resnet18_seed42_curriculum.py` | Standard baseline + curriculum baseline (seed 42) |
| `run_sae_expansion_comparison.py` | SAE expansion ablation (4x, 8x, 16x) |
| `run_independent_init_sae.py` | Independent SAE initialisation control |
| `run_resnet18_seed42_imagenet.py` | Cross-dataset validation (Tiny ImageNet) |
| `run_resnet18_seed42_curriculum_switch.py` | Curriculum-switch sweep, 7 runs (Fig. 6) |
| `run_resnet18_seed42_label_noise.py` | Label noise, 7 conditions, seed 42 (Fig. 5) |
| `run_resnet18_seed137_label_noise.py` | Label noise, 7 conditions each, seeds 137 & 256 |
| `run_resnet18_seed7_314_label_noise.py` | Label noise, 7 conditions each, seeds 7 & 314 |

## Note

These scripts depend on a running FastAPI backend and RunPod GPU instances.
They are **not** needed to reproduce results from cached data; use the
`analysis/` and `figures/` pipelines for that.
