# analysis/

Scripts that compute derived statistics, survival metrics, and causal tests
from raw training snapshots. Each script writes a JSON (or stdout) output that
is consumed by figure-plotting scripts or data validation.

## Script-to-Output Mapping

| Script | Output | Used by |
|---|---|---|
| `validate_manuscript.py` | stdout (128 PASS/FAIL) | Data validation |
| `compute_feature_survival.py` | `data/feature_survival_all_lanes.json` | validate_manuscript |
| `compute_feature_survival_tg.py` | `data/feature_survival_tg_expanded.json` | Fig. 4 |
| `granger_causality_superclass.py` | `data/granger_causality_results.json` | ED Fig. 5 |
| `granger_causality_onset.py` | `data/granger_causality_onset_results.json` | ED Fig. 5 |
| `granger_causality_tg.py` | `data/granger_causality_tg_results.json` | ED Fig. 5 |
| `granger_causality_tg_critical.py` | `data/granger_causality_tg_critical_results.json` | ED Fig. 5 |
| `predict_terminal_accuracy.py` | `data/predict_scatter_cache.json` | ED Fig. 8 (prediction panel) |
| `bootstrap_noise_ci.py` | stdout (Cohen's d, BCa CIs) | Methods text |
| `cumulative_scaffold_census.py` | `data/cumulative_scaffold_census.json` | Fig. 4 |

## Notes

- Scripts that read from the external training drive have **pre-computed
  outputs already committed in `data/`**, so figures can be regenerated without
  access to the original snapshots.
- `validate_manuscript.py` is the master check: it loads all cached JSON files
  and confirms every numerical claim in the paper.

## Usage

```bash
# Run all 128 data assertions
python analysis/validate_manuscript.py

# Recompute a specific analysis
python analysis/compute_feature_survival.py
```
