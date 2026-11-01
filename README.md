# Scaffolded representation learning in deep networks: code

Philipp Stecher, Sandro Radovanović, Vlasta Sikimić, Reinhard Kahle

Code and derived data for the paper. Representational Development Tracing (RDT) decomposes the representations of a
network at every training checkpoint with sparse autoencoders, follows each feature across checkpoints and classifies
its developmental events (task-general, superclass and class-level). The repository contains the RDT pipeline, the
launchers of every experiment, the analyses, the derived data they produce, and the builders of every figure and
table of the paper and its Supplementary Information.

## Quick start

```
pip install -r requirements.txt          # or: make install
python run/run_pipeline.py --config run/configs/demo/demo_cifar100_4superclasses.json          # or: make demo
```

The demo runs the complete pipeline on a small problem: a ResNet-18 trained for six epochs on the 20 CIFAR-100 classes
of four superclasses, with checkpoints at initialisation, at four accuracy milestones and when validation accuracy
reaches 35%, an SAE per checkpoint and layer, feature matching, selectivity indices against permutation nulls and the
classification of developmental events. CIFAR-100 is downloaded into `data/experiments/` on first use. It takes about
five minutes on a laptop (GPU or Apple MPS used if present), writes 1.3 GB to `output/` and prints the events per
transition; the full results are in `output/demo_cifar100_4superclasses/sae_analysis/sae_results.json`. With only four
superclasses the superclass-selectivity threshold lies near its maximum, so the demo counts fewer superclass-level
events than runs on all 20 superclasses.

## Reproducing the figures and tables

```
export RDT_DATA=/path/to/rdt_per_run_data    # the unpacked figshare deposit (see "Data" below)
make figures                                  # or: bash reproduce/figures.sh [fig5 ed7 ...]
make check                                    # or: python reproduce/check.py
```

`reproduce/figures.sh` runs the builder of every figure and table of the paper and the Supplementary Information and
writes the figures, tables and their Source Data CSVs to `reproduce/out/` (one log per item), then lists what was
built. Without `RDT_DATA` it builds the items that need only this repository and skips the others.
`reproduce/check.py` compares every rebuilt Source Data table with the published Source Data workbooks in
`source_data/` (`reproduce/source_files.csv` names the file behind each table) and reports each as identical, same
values (formatting only) or different; with the package versions
of `requirements.txt` every table is identical. Supplementary Table 3 is rebuilt from the stored SAE activations
(1.1 TB, not deposited); the check compares its stored table.

## Layout

| Path | Contents |
|---|---|
| `packages/ml/src/` | RDT core: training with checkpoint capture (`devtrain.py`, `training.py`, `models.py`, `cct.py`), sparse autoencoders (`sae.py`), feature matching, selectivity indices, null thresholds and event classification (`saeanalysis.py`), interventions on the representation and the learning signal (`lesion.py`) |
| `packages/backend/src/` | parts of the job system through which the core runs were trained and analysed: the run configuration model (`netinit_models.py`, used by `run/run_pipeline.py`), storage and the API routes of the training and SAE analysis jobs (for reference; they need the job database and web service, which are not included) |
| `run/` | single-command pipeline runner, the demo and the configurations of the standard, ablation, curriculum-sweep and label-noise runs |
| `scripts/` | experiment orchestrators (`run_*.py`); scripts applied to the stored results of the core runs (`make_ab_tg_exclusive.py`, `reclassify_all_lanes.py`, `compute_feature_survival.py`, `trace_transformed_features.py`) |
| `scripts/granger/extractors/` | extraction of the per-superclass event series used by the Granger analyses |
| `scripts/gpu/core_runs/` | launchers of the core runs trained on RTX 4090 GPUs (they submit jobs to the job system's web service) |
| `scripts/paper/` | analyses and experiment launchers; `figures/` holds the figure and table builders |
| `analysis_outputs/paper/` | derived data the analyses and builders read |
| `source_data/` | the Source Data workbooks of the paper, one per figure, one sheet per panel |
| `variants/` | training code of specific experiments where it differs from `packages/ml/src` and `scripts/` |
| `reproduce/` | `figures.sh` rebuilds every figure and table; `check.py` compares the rebuilt Source Data with `source_data/` |
| `Makefile` | `make install`, `make demo`, `make figures`, `make check` |

## The RDT pipeline

1. Training and checkpoint capture: `packages/ml/src/devtrain.py` (checkpoints at accuracy milestones; per-sample
   activations at five or six layers).
2. SAE decomposition: `packages/ml/src/sae.py` (top-k, k = 32, expansion 4, unit-norm decoder), run per checkpoint and
   layer by `packages/ml/src/saeanalysis.py`.
3. Feature matching: Hungarian matching on activation-profile correlation (stable r ≥ 0.5, death r < 0.2).
4. Selectivity indices (SSI, CSI, SAI).
5. Permutation nulls and adaptive thresholds.
6. Event classification (Tg-E, Ab-E, Di-E, As-E, De-E). For the core runs, `scripts/make_ab_tg_exclusive.py` converts
   stored results to mutually exclusive Ab-E and Tg-E counts and `scripts/reclassify_all_lanes.py` applies the
   classification rules of the Methods to every run. In the code and the stored results the events are named Tg-H,
   Ab-H, Di-H, As-H and De-H (keys `tg_h` … `de_h`); the `hypotheses` block of `sae_results.json` is a summary of the
   job system that the paper does not use.

Steps 3-6 are in `packages/ml/src/saeanalysis.py`. `run/run_pipeline.py` runs steps 1-6 for one configuration.

## Figures and tables

| Item | Builder | Data |
|---|---|---|
| Fig. 1 | `scripts/paper/figures/fig1.py` | per-run data |
| Fig. 2 | `scripts/paper/figures/fig2.py` | per-run data |
| Fig. 3 | `scripts/paper/figures/fig3.py` | per-run data |
| Fig. 4 | `scripts/paper/figures/fig4.py` | this repository |
| Fig. 5 | `scripts/paper/figures/interventions/fig5.py` | this repository |
| Fig. 6 | `scripts/paper/figures/interventions/fig6.py` | per-run data |
| Extended Data Fig. 1 | `scripts/paper/figures/extended/ed1.py` | per-run data |
| Extended Data Fig. 2 | `scripts/paper/figures/interventions/ed2.py` | this repository |
| Extended Data Fig. 3 | `scripts/paper/figures/extended/ed3.py` | this repository |
| Extended Data Fig. 4 | `scripts/paper/figures/routing/ed4.py` | this repository |
| Extended Data Fig. 5 | `scripts/paper/figures/ed5.py` | this repository |
| Extended Data Fig. 6 | `scripts/paper/figures/interventions/ed6.py` | this repository |
| Extended Data Fig. 7 | `scripts/paper/figures/interventions/ed7.py` | per-run data |
| Extended Data Fig. 8 | `scripts/paper/figures/interventions/ed8.py` | per-run data |
| Extended Data Fig. 9 | `scripts/paper/figures/interventions/ed9.py` | per-run data |
| Extended Data Table 1 | `scripts/paper/edtable1_runs_all.py` | per-run data |
| Supplementary Fig. 1 | `scripts/paper/figures/supp1.py` | this repository |
| Supplementary Fig. 2 | `scripts/paper/figures/supp2.py` | this repository |
| Supplementary Fig. 3 | `scripts/paper/figures/extended/supp3.py` | per-run data |
| Supplementary Fig. 4 | `scripts/paper/figures/extended/supp4.py` | per-run data |
| Supplementary Fig. 5 | `scripts/paper/figures/supp5.py` | this repository |
| Supplementary Fig. 6 | `scripts/paper/figures/supp6.py` | this repository |
| Supplementary Fig. 7 | `scripts/paper/figures/interventions/supp7.py` | this repository |
| Supplementary Table 1 | `scripts/paper/supp_table1_ordering.py` | this repository |
| Supplementary Tables 2, 3 | `scripts/paper/si_source_files.py` | per-run data (Table 3 also needs the stored SAE activations, not deposited; the check compares its stored table) |

## Data

Builders marked "this repository" run from `analysis_outputs/paper/` alone. The others also read per-run results
(SAE results, training records and per-sample predictions of the runs), which are deposited in figshare with the paper
(`rdt_per_run_data.zip`, 3.9 GB unpacked). Unpack it anywhere and set `RDT_DATA` to the unpacked folder
(`rdt_per_run_data/`); the scripts read the core runs from `$RDT_DATA/core`, the intervention and control runs from
`$RDT_DATA/extended` (default `RDT_DATA=/data/rdt`), and `reproduce/figures.sh` links the runs of
`$RDT_DATA/repository/data/experiments` into `data/experiments`. Checkpoints and stored activations (1.1 TB) are
available from the corresponding author on request.

## Experiments

Every training run of the paper is listed in Extended Data Table 1. Each condition is launched as follows. The
configurations in `run/configs/` are the ones the job system stored for these runs, checked against each run's own
records (epochs trained, network, SAE settings). Tiny ImageNet is not downloaded automatically: unpack
tiny-imagenet-200 into `data/experiments/resnet18_tinyimagenet_seed42/datasets/tiny_imagenet/` (one folder per class
in `train/` and `val/`).

| Experiment | Launcher |
|---|---|
| Standard runs: ResNet-18, ViT (Tiny configuration for seeds 42 and 137, Small for seed 256; Methods), CCT-7 on CIFAR-100; 200 epochs; SAE expansion 8× and independent SAE initialisation (seed 256); Tiny ImageNet | `run/run_pipeline.py` with `run/configs/standard/*.json`; GPU runs `scripts/gpu/core_runs/` |
| Curriculum sweep | `run/configs/curriculum_sweep/*.json` |
| Label noise redrawn every epoch | `run/configs/label_noise/*.json` |
| Label noise with fixed labels; similarity-matched noise | `scripts/paper/queue_curriculum_noise.sh`, `scripts/paper/queue_noise_similarity_matched.sh` |
| Suppression of the superclass directions in the last block, random directions, random grouping | `scripts/paper/run_prevention.sh`, `scripts/paper/run_randgroup.sh` (orchestrator `scripts/run_scaffold_lesion_experiment.py`) |
| Sham control | `scripts/run_critical_sham.py`; per-epoch SAEs `scripts/paper/run_sham_perepoch_sae.py` |
| Acquisition blocking: superclass, random grouping, random directions, energy-matched, jointly matched; continuation to epoch 30 | `scripts/paper/run_gradblock.sh`, `run_gradenergy.sh`, `run_gradjoint.sh`, `run_gradcont.sh`; RDT `scripts/paper/run_blocking_rdt.py` |
| Controlled curriculum: superclasses, random grouping, fine classes only | `scripts/paper/run_curriculum.py`, `scripts/paper/queue_curriculum_k10_k30.sh`; RDT `scripts/paper/run_curriculum_sae.py` |
| Superclass-blind pre-training | `scripts/paper/queue_blind_pretrain.sh` |
| Five-label pre-training: five random groups of 20 classes, five groups of whole superclasses (Extended Data Fig. 9b) | `scripts/paper/queue_rand5_pretrain.sh`, `scripts/paper/queue_sup5_pretrain.sh` |
| Superclass-blind task alone (Supplementary Note 4) | `scripts/paper/queue_taskirr.sh` |
| Placement given by construction (Supplementary Note 3) | `scripts/paper/run_placement_given.sh` (seed 42), `scripts/paper/run_placement_given_full.sh` (other seeds) |
| Self-supervised control (SimCLR) | `scripts/paper/run_simclr.sh` |
| Per-epoch CCT-7 run | `scripts/paper/run_perepoch_cct.py` |
| Task-general transplant | `scripts/paper/run_tg_transplant.py` |

Some experiments ran from separate checkouts of this repository with modified training code; the files that differ
are in `variants/`, and `variants/experiments.json` lists which experiments used which. The launchers expect such a
checkout under `$RDT_DATA/checkouts/<name>` (a copy of this repository with `variants/<name>/` applied).
The pre-training tasks of the controlled curriculum are options of `packages/ml/src/devtrain.py`, all off by
default: `coarse_labels` set to `within_superclass_index` (superclass-blind task), `random_groups` or
`superclass_groups` (five-label tasks, with `n_coarse_groups` and `coarse_group_seed`).

The run reported as 11 superclass epochs (Fig. 6b,c; Extended Data Fig. 9e) was resumed from its epoch-5 state after a crash,
with the cosine schedule restarted; `run/configs/curriculum_sweep/switch_e15.json` holds its planned schedule
(15 + 35 epochs). Some seeds (SAE initialisation, random controls) are offset by Python's string hash of the layer
name; set `PYTHONHASHSEED=0` for repeatable reruns. The training record of the Tiny ImageNet run was not kept;
its configuration holds the planned schedule (100 epochs), and the paper reports the run as recorded with its results
(50 epochs, 53.0%).

Names in the code: a *lane* is one training run; *suppression* (also *lesion*, *prevention*, *critical*) projects
superclass directions out of the last block; the blocking arms are `gradsc` (coarse block), `gradrg` (random grouping),
`gradrand` (random directions), `gradenergy` (energy-matched) and `gradjoint` (jointly matched); `taskirr` is the
superclass-blind task; `sc_true`, `sc_rand` and `scratch` are the curriculum arms (superclasses, random grouping, fine
classes only), and `sc_blind`, `sc_rand5` and `sc_sup5` the superclass-blind, five-random-group and
five-superclass-group pre-training arms.

## Environment

Python 3.13.7 with the package versions in `requirements.txt` (analyses and the runs on an Apple M4); the runs on
NVIDIA RTX 4090 GPUs used the CUDA build of PyTorch 2.2.0. The experiment launchers in `scripts/paper/*.sh` are zsh
scripts. `RELEASE_NOTES.md` lists the edits made to the code for
this release; `PROVENANCE.md` traces every figure, table and number to the code and data that produced it.

## License

MIT (see `LICENSE`).
