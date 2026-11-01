# Release notes

This repository holds the code that produced the paper's results. For the release, file and directory names were
standardized, comments and docstrings were edited for readability, and paths to the authors' machines were replaced
by the placeholders below. Every figure and table builder was rerun from this repository on the original data: each
figure is pixel-identical to the figure in the paper and the Supplementary Information, and each Source Data file
matches the published Source Data.

## Paths

- Per-run data: `$RDT_DATA/core` (core runs) and `$RDT_DATA/extended` (intervention and control runs), with
  `RDT_DATA` defaulting to `/data/rdt`, plus `data/experiments` inside the repository (curriculum sweep and the
  repository's own runs).
- Separate checkouts used by some launchers: `$RDT_DATA/checkouts/<name>` (this repository with `variants/<name>/`
  applied; `random_grouping` is identical to the main code).
- Launchers derive the repository root from their own location and call `$PYTHON` (default `python`); two caches are
  set by `$RDT_CACHE` and `$RDT_CACHE_DEPENDENCY`.

## Notes

- The core runs were created through the job system in `packages/backend`; `run/run_pipeline.py` runs the same
  stages from a JSON configuration.
- Launchers that call scripts not in this release say so in a comment.
- Glossary of names in the code: a *lane* is one training run, a *stove* or *job* is one stage of a run in the job
  system, and the job database (`rcx.db`) stores each run's configuration.
- A few totals in Extended Data Table 1 and the Methods (training hours, storage) were compiled from run logs; the
  per-run values are in `source_data/Source_Data_Extended_Data_Table1.xlsx`.
