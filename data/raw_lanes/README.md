# Raw Lane Data

This directory contains per-lane SAE analysis results for all 55 training runs (41 files, ~591 MB total).

These files are too large for a Git repository and are deposited on Zenodo:

**DOI**: [To be assigned at acceptance]

## To obtain the data

1. Download from the Zenodo archive (DOI above)
2. Extract into this directory (`data/raw_lanes/`)
3. The figure scripts `plot_ed_table1.py` and `plot_activation_magnitude_selectivity.py` read from this directory

## File listing

- 9 core lanes: `{ResNet18,ViTSmall,CCT7}_CIFAR100_seed{42,137,256}.json`
- 3 ablation/control: `ResNet18_CIFAR100_{200ep,8x,independent_init_control}_seed42.json`
- 1 cross-dataset: `ResNet18_TinyImageNet_seed42.json`
- 7 label noise aggregates: `ResNet18_CIFAR100_label_noise_{condition}.json` (~72 MB each)
- 21 label noise per-seed: `ResNet18_CIFAR100_label_noise_{condition}_seed{seed}.json` (~4.4 KB each)

See `../DATA_README.md` for complete file schemas and figure mappings.
