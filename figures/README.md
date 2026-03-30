# figures/

Plotting scripts that produce all main-text and extended-data figures.

**Important:** Script numbers do not always match figure numbers in the paper.
The table below is the definitive mapping.

## Script-to-Figure Mapping

| Figure | Script | Description |
|---|---|---|
| Fig. 1 | `plot_fig1.py` | Three-phase coarse-to-fine sequence |
| Fig. 2 | `plot_fig4.py` | Cross-architecture/seed/dataset replication |
| Fig. 3 | `plot_fig2.py` | Feature turnover window |
| Fig. 4 | `plot_fig5.py` | Developmental persistence |
| Fig. 5 | `plot_fig5_label_noise.py` | Targeted label noise |
| Fig. 6 | `plot_fig8.py` | Curriculum sweep |
| ED Figs. 1-8 | `plot_extended_data.py` | All extended data figures |
| ED Table 1 | `plot_ed_table1.py` | Overview of 55 runs |
| ED Fig. 8 | `plot_activation_magnitude_selectivity.py` | Selectivity vs magnitude |

## Data Flow

All scripts read pre-computed analysis results from `../data/` and write
rendered figures to `../output/figures/`.

## Helper Module

`epoch_labels.py` provides shared epoch-tick formatting used across plots.

## Usage

```bash
# Example: generate Fig. 1
python figures/plot_fig1.py
```
