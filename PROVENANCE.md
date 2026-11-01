# Provenance

Which code produced each figure, table and number of the paper and the SI. Paths are relative to this repository; inputs outside it (under `$RDT_DATA`) are per-run data, deposited in figshare with the paper (see README.md).

## Figures and tables

### Fig. 1
- builder: `scripts/paper/figures/fig1.py`
- helpers: `scripts/paper/figures/nmi_style.py`, `scripts/paper/figures/core_runs_lib.py`, `scripts/paper/figures/process_composition.py`
- `$RDT_DATA/core/experiments/abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb/lanes/<lane>/sae_analysis/sae_results.json` ← `packages/ml/src/saeanalysis.py`
- `$RDT_DATA/core/experiments/821eaadb-0160-48f8-9669-ec3d9185c428/lanes/45476b66-*/sae_analysis/sae_results.json` ← `packages/ml/src/saeanalysis.py`

### Fig. 2
- builder: `scripts/paper/figures/fig2.py`
- helpers: `scripts/paper/figures/nmi_style.py`, `scripts/paper/figures/core_runs_lib.py`, `scripts/paper/figures/process_composition.py`
- `$RDT_DATA/core/experiments/abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb/lanes/<lane>/sae_analysis/sae_results.json` ← `packages/ml/src/saeanalysis.py`
- `$RDT_DATA/core/experiments/821eaadb-0160-48f8-9669-ec3d9185c428/lanes/45476b66-*/sae_analysis/sae_results.json` ← `packages/ml/src/saeanalysis.py`
- `$RDT_DATA/core/experiments/abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb/lanes/<lane>/dev_snapshots/metrics/devtrain_snapshots.json` ← `packages/ml/src/devtrain.py`

### Fig. 3
- builder: `scripts/paper/figures/fig3.py`
- helpers: `scripts/paper/figures/nmi_style.py`, `scripts/paper/figures/turnover_lib.py`
- `$RDT_DATA/core/experiments/abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb/lanes/<lane>/sae_analysis/sae_results.json` ← `packages/ml/src/saeanalysis.py`
- `$RDT_DATA/core/experiments/abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb/lanes/<lane>/dev_snapshots/metrics/devtrain_snapshots.json` ← `packages/ml/src/devtrain.py`

### Fig. 4
- builder: `scripts/paper/figures/fig4.py`
- helpers: `scripts/paper/figures/nmi_style.py`, `scripts/paper/figures/core_runs_lib.py`, `scripts/paper/figures/process_composition.py`, `scripts/paper/figures/suppression_panels.py`, `scripts/paper/figures/fig_common.py`
- `analysis_outputs/paper/figures/fig4_lifespan_data.json` ← `scripts/paper/figures/fig4_lifespan.py`
- `analysis_outputs/paper/THRESHOLD_impact/fig4_stored_sai.json` ← `scripts/paper/fig4_stored_sai.py`
- `analysis_outputs/paper/CENSUS_exclusive/census_exclusive_per_checkpoint.csv` ← `scripts/paper/census_exclusive.py`
- `analysis_outputs/paper/CRITICAL_period/val_curves.csv` ← `scripts/paper/critical_period.py`
- `analysis_outputs/paper/XLAYER/val_acc_curves.csv` ← `scripts/paper/crosslayer_suppression.py`
- `analysis_outputs/paper/PSEUDO_superclass/lastblock_raw.csv` ← `scripts/paper/pseudo_superclass.py`
- `analysis_outputs/paper/PSEUDO_superclass/lastblock_per_network.csv` ← `scripts/paper/pseudo_superclass.py`

### Fig. 5
- builder: `scripts/paper/figures/interventions/fig5.py`
- helpers: `scripts/paper/figures/interventions/fig5_panels.py`, `scripts/paper/figures/nmi_style.py`
- `analysis_outputs/paper/RDT_pace/long.csv` ← `scripts/paper/rdt_pace.py`
- `analysis_outputs/paper/PLACEMENT_curve/long.csv` ← `scripts/paper/placement_curve.py`
- `analysis_outputs/paper/RDT_completion/rdt_completion_long.csv` ← `scripts/paper/rdt_completion_analysis.py`

### Fig. 6
- builder: `scripts/paper/figures/interventions/fig6.py`
- helpers: `scripts/paper/figures/nmi_style.py`, `scripts/paper/figures/interventions/fig6_panels.py`, `scripts/paper/figures/interventions/ed9.py`, `scripts/paper/figures/interventions/blind_arm.py`, `scripts/paper/figures/curriculum_lib.py`, `scripts/paper/figures/fig_common.py`
- `analysis_outputs/paper/figures/inputs/fig6_cifar_examples.npz` ← `scripts/paper/fig6_cifar_examples.py`
- `analysis_outputs/paper/DICTFREE/hierarchy/D_sae_census_timecourse.csv` ← `scripts/paper/hierarchy_report.py`
- `analysis_outputs/paper/SWEEP_RDT/census_K30.csv` ← `scripts/paper/rdt_census_k30.py`
- `analysis_outputs/paper/BLIND_PRETRAIN/census_timecourse.csv` ← `scripts/paper/blind_pretrain_measures.py`
- `analysis_outputs/paper/RDT_pace/curriculum_full_long.csv` ← `scripts/paper/rdt_pace_curriculum_full.py`
- `analysis_outputs/paper/CURRICULUM_sweep/tables/A_training_cost.csv` ← `scripts/paper/curriculum_training_cost.py`
- `data/experiments/7885dc97-1062-4b2f-9999-61d63b93d097/lanes/<lane_id>/dev_snapshots/metrics/devtrain_metrics.json` ← `packages/ml/src/devtrain.py`
- `$RDT_DATA/extended/experiments/curriculum/lanes/<lane>/dev_snapshots/metrics/devtrain_metrics.json` ← `packages/ml/src/devtrain.py`
- `$RDT_DATA/extended/experiments/blind_pretrain/lanes/<lane>/dev_snapshots/metrics/devtrain_metrics.json and .lane_done` ← `packages/ml/src/devtrain.py`

### Extended Data Fig. 1
- builder: `scripts/paper/figures/extended/ed1.py`
- helpers: `scripts/paper/figures/nmi_style.py`, `scripts/paper/figures/core_runs_lib.py`, `scripts/paper/figures/process_composition.py`, `scripts/paper/figures/checks_lib.py`, `scripts/paper/figures/validation_panels.py`
- `$RDT_DATA/core/experiments/abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb/lanes/<9 core lanes>/sae_analysis/sae_results.json` ← `packages/ml/src/saeanalysis.py`
- `$RDT_DATA/core/experiments/independent_sae/lanes/independent_sae_seed256*/sae_analysis/sae_results.json` ← `packages/ml/src/saeanalysis.py`
- `$RDT_DATA/core/experiments/578b5803-1f2f-49c4-99b8-bf7071f43917/lanes/abfc645c*/sae_analysis/sae_results.json` ← `packages/ml/src/saeanalysis.py`
- `$RDT_DATA/core/experiments/ce93d3f6-282b-4811-84f8-be6c589c0500/lanes/c645c39e*/sae_analysis/sae_results.json` ← `packages/ml/src/saeanalysis.py`
- `analysis_outputs/paper/figures/final/source_data_ed/ed1_c.csv` ← `scripts/paper/figures/validation_panels.py`
- `analysis_outputs/paper/CENSUS_exclusive/census_exclusive_per_checkpoint.csv` ← `scripts/paper/census_exclusive.py`
- `analysis_outputs/paper/CENSUS_exclusive/ordering_counts.csv` ← `scripts/paper/census_exclusive.py`
- `analysis_outputs/paper/DICTFREE/population/tables/population_per_checkpoint.csv` ← `scripts/paper/dictfree_analyze.py`
- `analysis_outputs/paper/DATA_order/transient_stage/share_sc_per_checkpoint.csv` ← `scripts/paper/transient_stage.py`
- `analysis_outputs/paper/SIMCLR/simclr_measures.csv` ← `scripts/paper/simclr_analysis.py`
- `analysis_outputs/paper/SIMCLR/simclr_summary.json` ← `scripts/paper/simclr_analysis.py`

### Extended Data Fig. 2
- builder: `scripts/paper/figures/interventions/ed2.py` (panels a-d drawn by `scripts/paper/figures/extended/ed2_panels.py`)
- helpers: `scripts/paper/figures/nmi_style.py`, `scripts/paper/figures/granger_stats.py`, `scripts/paper/figures/turnover_lib.py`, `scripts/paper/figures/granger_panels.py`
- `analysis_outputs/paper/SUPERCLASS_series/superclass_transition_series_cifar100.csv` ← `scripts/paper/figures/superclass_series.py`
- `analysis_outputs/paper/SUPERCLASS_series/superclass_transition_series_tiny.csv` ← `scripts/paper/figures/superclass_series.py`
- `analysis_outputs/paper/GRANGER_exclusive/series_exclusive.csv` ← `scripts/paper/granger_exclusive.py`
- `analysis_outputs/paper/figures/final/source_data_ed/ed2_network_level.csv, ed2_b.csv, ed2_d.csv, ed2_statistics.csv, ed2_d_runs.csv` ← `scripts/paper/figures/granger_panels.py`
- `analysis_outputs/paper/TG_dependency/proxy_long.csv` ← `scripts/paper/tg_dependency/substrate_proxy.py`
- `analysis_outputs/paper/TG_dependency/transplant_long.csv` ← `scripts/paper/tg_dependency/transplant_collect.py`

### Extended Data Fig. 3
- builder: `scripts/paper/figures/extended/ed3.py`
- helpers: `scripts/paper/figures/nmi_style.py`
- `analysis_outputs/paper/figures/final/source_data_ed/seed_events.csv` ← `scripts/paper/figures/seed_events.py`

### Extended Data Fig. 4
- builder: `scripts/paper/figures/routing/ed4.py`
- helpers: `scripts/paper/figures/architecture_spread.py`, `scripts/paper/figures/nmi_style.py`, `scripts/paper/figures/checks_lib.py`
- `analysis_outputs/paper/SI/superclass_process_fractions_arch_means.csv` ← `scripts/paper/si_source_files.py`
- `analysis_outputs/paper/SI/superclass_process_fractions_per_run.csv` ← `scripts/paper/si_source_files.py`

### Extended Data Fig. 5
- builder: `scripts/paper/figures/ed5.py`
- helpers: `scripts/paper/figures/nmi_style.py`, `scripts/paper/figures/checks_lib.py`
- `analysis_outputs/paper/DATA_order/mode_strength/saxe_superclass_units.csv` ← `scripts/paper/mode_strength.py`
- `analysis_outputs/paper/DATA_order/mode_strength/saxe_partition_test.csv` ← `scripts/paper/mode_strength.py`
- `analysis_outputs/paper/DATA_order/mode_strength/mode_strength_per_network.csv` ← `scripts/paper/mode_strength.py`
- `analysis_outputs/paper/DATA_order/mode_strength/mode_strength_results.json` ← `scripts/paper/mode_strength.py`

### Extended Data Fig. 6
- builder: `scripts/paper/figures/interventions/ed6.py`
- helpers: `scripts/paper/figures/nmi_style.py`, `scripts/paper/figures/routing/routing_panels.py`, `scripts/paper/figures/extended/blocking_routing_panels.py`, `scripts/paper/figures/fig4.py`, `scripts/paper/figures/core_runs_lib.py`, `scripts/paper/figures/process_composition.py`, `scripts/paper/figures/suppression_panels.py`, `scripts/paper/figures/intervention_panels.py`, `scripts/paper/figures/noise_lib.py`, `scripts/paper/figures/fig_common.py`
- `analysis_outputs/paper/DEPENDENCY/T1_T4_per_seed.csv` ← `scripts/paper/dependency_stats.py`
- `analysis_outputs/paper/DEPENDENCY/T1_ratio_by_target_category.csv` ← `scripts/paper/dependency_stats.py`
- `analysis_outputs/paper/CRITICAL_period/val_curves.csv` ← `scripts/paper/critical_period.py`
- `analysis_outputs/paper/XLAYER/val_acc_curves.csv` ← `scripts/paper/crosslayer_suppression.py`
- `analysis_outputs/paper/PSEUDO_superclass/lastblock_raw.csv` ← `scripts/paper/pseudo_superclass.py`
- `analysis_outputs/paper/PSEUDO_superclass/lastblock_per_network.csv` ← `scripts/paper/pseudo_superclass.py`
- `analysis_outputs/paper/RANDGROUP_control/specificity_summary.json and analysis_outputs/paper/RANDGROUP_control/dose_response.csv` ← `scripts/paper/randgroup_specificity.py`
- `analysis_outputs/paper/RANDGROUP_control/specificity_arms.csv` ← `scripts/paper/randgroup_specificity.py`
- `analysis_outputs/paper/RANDGROUP_control/fresh_readout.csv` ← `scripts/paper/fresh_readout.py`
- `analysis_outputs/paper/REGENERATION/contrasts_vs_sham_per_seed_epoch.csv` ← `scripts/paper/regeneration_aggregate.py`
- `analysis_outputs/paper/REGENERATION/key_numbers.json` ← `scripts/paper/regeneration_numbers.py`
- `analysis_outputs/paper/CLASS_level/per_class_long.csv and A_factorisation_per_seed.csv` ← `scripts/paper/class_level.py`
- `analysis_outputs/paper/ORACLE_within/oracle_long.csv` ← `scripts/paper/oracle_within.py`
- `analysis_outputs/paper/ROUTING/routing_resnet_clean_ep5-10-15-20-25.json, routing_resnet_clean_ep30.json` ← `scripts/paper/routing.py`
- `analysis_outputs/paper/ROUTING/routing_tf.json, routing_cct7.json, routing_resnet_projection.json` ← `scripts/paper/routing_transformers.py`, `scripts/paper/routing_cct7.py`, `scripts/paper/routing_resnet_projection.py`
- `analysis_outputs/paper/ROUTING/routing_within.json` ← `scripts/paper/routing_within.py`

### Extended Data Fig. 7
- builder: `scripts/paper/figures/interventions/ed7.py`
- helpers: `scripts/paper/figures/nmi_style.py`, `scripts/paper/figures/fig_common.py`, `scripts/paper/figures/intervention_panels.py`, `scripts/paper/figures/noise_lib.py`, `scripts/paper/figures/suppression_panels.py`, `scripts/paper/effect_units.py`
- `analysis_outputs/paper/GRADJOINT/long.csv` ← `scripts/paper/gradjoint_analysis.py`
- `analysis_outputs/paper/GRADBLOCK/final_long.csv` ← `scripts/paper/gradblock_complete.py`
- `analysis_outputs/paper/RANDGROUP_control/specificity_arms.csv` ← `scripts/paper/randgroup_specificity.py`
- `analysis_outputs/paper/ORACLE_within/oracle_summary.json` ← `scripts/paper/oracle_within.py`
- `analysis_outputs/paper/GRADBLOCK/superclass_gradblock.csv` ← `scripts/paper/gradblock_superclass.py`
- `analysis_outputs/paper/ORACLE_within/oracle_long.csv` ← `scripts/paper/oracle_within.py`
- `analysis_outputs/paper/NOISE_matched/noise_matched_per_seed.csv` ← `scripts/paper/matched_noise_q.py`
- `analysis_outputs/paper/STATIC_probe/static_probe.csv` ← `scripts/paper/static_probe.py`
- `$RDT_DATA/extended/experiments/static_noise_s<s>/lanes/{standard_resnet18,within_sc_resnet18_throughout,between_sc_resnet18_throughout,between_sc_matched_resnet18_throughout}/dev_snapshots/metrics/devtrain_metrics.json, devtrain_snapshots.json` ← `packages/ml/src/devtrain.py`
- `analysis_outputs/paper/CRITICAL_period/val_curves.csv` ← `scripts/paper/critical_period.py`
- `analysis_outputs/paper/RDT_completion/rdt_completion_long.csv` ← `scripts/paper/rdt_completion_analysis.py`

### Extended Data Fig. 8
- builder: `scripts/paper/figures/interventions/ed8.py`
- helpers: `scripts/paper/figures/routing/noise_routing_panels.py`, `scripts/paper/figures/noise_lib.py`, `scripts/paper/figures/fig_common.py`, `scripts/paper/figures/rdt_side_by_side.py`, `scripts/paper/figures/nmi_style.py`
- `analysis_outputs/paper/NOISE_scaffold/per_lane_metrics.csv` ← `scripts/paper/noise_scaffold_stats.py`
- `analysis_outputs/paper/NOISE_scaffold/die_trajectory_per_lane.csv` ← `scripts/paper/noise_scaffold_stats.py`
- `analysis_outputs/paper/STATIC_probe/static_probe.csv` ← `scripts/paper/static_probe.py`
- `$RDT_DATA/extended/experiments/static_noise_s*/lanes/*/dev_snapshots/metrics/{devtrain_metrics.json, devtrain_snapshots.json} and .lane_done` ← `packages/ml/src/devtrain.py`
- `analysis_outputs/paper/NOISE_matched/noise_matched_per_seed.csv` ← `scripts/paper/matched_noise_q.py`
- `analysis_outputs/paper/NOISE_matched/noise_matched_results.json` ← `scripts/paper/matched_noise_q.py`
- `analysis_outputs/paper/RDT_noise/per_lane/<arm>_seed<S>.json` ← `scripts/paper/noise_rdt_analysis.py`
- `analysis_outputs/paper/RDT_noise/nesting_per_lane.csv` ← `scripts/paper/noise_routing_nesting.py`
- `analysis_outputs/paper/PLACEMENT_curve/long.csv` ← `scripts/paper/placement_curve.py`
- `analysis_outputs/paper/RDT_noise_fit/features.csv` ← `scripts/paper/noise_fit_features.py`

### Extended Data Fig. 9
- builder: `scripts/paper/figures/interventions/ed9.py`
- helpers: `scripts/paper/figures/nmi_style.py`, `scripts/paper/figures/curriculum_lib.py`, `scripts/paper/figures/fig_common.py`, `scripts/paper/figures/interventions/blind_arm.py`, `scripts/paper/figures/interventions/fig6_panels.py`
- `analysis_outputs/paper/CURRICULUM_sweep/tables/A_training_cost.csv` ← `scripts/paper/curriculum_training_cost.py`
- `data/experiments/7885dc97-1062-4b2f-9999-61d63b93d097/lanes/<lane_id>/dev_snapshots/metrics/devtrain_metrics.json` ← `packages/ml/src/devtrain.py`
- `$RDT_DATA/extended/experiments/curriculum/lanes/<lane>/dev_snapshots/metrics/devtrain_metrics.json` ← `packages/ml/src/devtrain.py`
- `$RDT_DATA/extended/experiments/blind_pretrain/lanes/<lane>/dev_snapshots/metrics/devtrain_metrics.json and .lane_done` ← `packages/ml/src/devtrain.py`
- `analysis_outputs/paper/RAND5_PRETRAIN/summary.json`, `analysis_outputs/paper/SUP5_PRETRAIN/summary.json` ← `scripts/paper/rand5_pretrain_analysis.py`
- `analysis_outputs/paper/SAE_outcomes/events_EC.csv` ← `scripts/paper/sae_outcomes.py`
- `analysis_outputs/paper/BLIND_PRETRAIN/events_EC.csv` ← `scripts/paper/blind_pretrain_measures.py`
- `analysis_outputs/paper/BLIND_PRETRAIN/long.csv` ← `scripts/paper/blind_pretrain_analysis.py`
- `analysis_outputs/paper/figures/final/source_data_ed/ed9_event_composition.csv` ← `scripts/paper/figures/interventions/ed9.py`

### Extended Data Table 1
- builder: `scripts/paper/edtable1_runs_all.py`
- `analysis_outputs/paper/figures/final/source_data/edtable1_runs.csv` ← `scripts/paper/figures/edtable1_scan.py`
- `$RDT_DATA/extended/experiments/blind_pretrain/lanes/sc_blind_K20_seed*/dev_snapshots/metrics/devtrain_metrics.json` ← `packages/ml/src/devtrain.py`
- `$RDT_DATA/extended/experiments/placement_given/lanes/{lesion_gradsc_e00_seed*,lesion_gradjoint_e00_seed*,sham_e00_seed*}/dev_snapshots/metrics/devtrain_metrics.json` ← `packages/ml/src/devtrain.py`

### Supplementary Fig. 1
- builder: `scripts/paper/figures/supp1.py`
- helpers: `scripts/paper/figures/nmi_style.py`, `scripts/paper/figures/core_runs_lib.py`, `scripts/paper/figures/process_composition.py`, `scripts/paper/figures/checks_lib.py`, `scripts/paper/figures/seed_events.py`, `scripts/paper/figures/turnover_lib.py`
- `analysis_outputs/paper/SHAM_perepoch/results.json` ← `scripts/paper/sham_perepoch_analysis.py`
- `analysis_outputs/paper/SHAM_perepoch/feature_basis_summary.json` ← `scripts/paper/perepoch_feature_basis.py`

### Supplementary Fig. 2
- builder: `scripts/paper/figures/supp2.py`
- helpers: `scripts/paper/figures/nmi_style.py`, `scripts/paper/figures/core_runs_lib.py`, `scripts/paper/figures/process_composition.py`, `scripts/paper/figures/checks_lib.py`
- `analysis_outputs/paper/PEREPOCH_cct/results.json` ← `scripts/paper/perepoch_cct_analysis.py`
- `analysis_outputs/paper/PEREPOCH_cct/process_per_transition.csv` ← `scripts/paper/perepoch_cct_analysis.py`
- `analysis_outputs/paper/PEREPOCH_cct/sae_free_share_per_epoch.csv` ← `scripts/paper/perepoch_cct_analysis.py`
- `analysis_outputs/paper/PEREPOCH_cct/sparse_vs_dense.csv` ← `scripts/paper/perepoch_cct_analysis.py`
- `analysis_outputs/paper/PEREPOCH_cct/flicker_per_epoch.csv` ← `scripts/paper/perepoch_cct_analysis.py`
- `analysis_outputs/paper/PEREPOCH_cct/steps_per_layer.csv` ← `scripts/paper/perepoch_cct_analysis.py`

### Supplementary Fig. 3
- builder: `scripts/paper/figures/extended/supp3.py`
- helpers: `scripts/paper/figures/noise_lib.py`, `scripts/paper/figures/fig_common.py`, `scripts/paper/figures/nmi_style.py`
- `analysis_outputs/paper/NOISE_scaffold/per_lane_metrics.csv` ← `scripts/paper/noise_scaffold_stats.py`
- `analysis_outputs/paper/NOISE_scaffold/die_fraction_per_lane.csv` ← `scripts/paper/noise_scaffold_stats.py`
- `$RDT_DATA/core/experiments/0c28799c-ec9d-4e45-ac30-302dff810a8d/lanes/*/sae_analysis/sae_results.json (feature_landscape)` ← `packages/ml/src/saeanalysis.py`
- `analysis_outputs/paper/STATIC_probe/static_probe.csv` ← `scripts/paper/static_probe.py`
- `$RDT_DATA/extended/experiments/static_noise_s*/lanes/*/dev_snapshots/metrics/{devtrain_metrics.json, devtrain_snapshots.json}` ← `packages/ml/src/devtrain.py`

### Supplementary Fig. 4
- builder: `scripts/paper/figures/extended/supp4.py`
- helpers: `scripts/paper/figures/nmi_style.py`, `scripts/paper/figures/core_runs_lib.py`, `scripts/paper/figures/process_composition.py`, `scripts/paper/figures/validation_panels.py`, `scripts/paper/figures/checks_lib.py`, `scripts/paper/figures/magnitude_panels.py`
- `$RDT_DATA/core/experiments/abf16d20-2cb5-4fc0-a0eb-4f7e6ee4fbcb/lanes/<9 core lanes>/sae_analysis/sae_results.json (null_baseline, metadata.adaptive_thresholds)` ← `packages/ml/src/saeanalysis.py`
- `analysis_outputs/paper/figures/ed7_magnitude_records.csv.gz` ← `scripts/paper/figures/magnitude_panels.py`
- `$RDT_DATA/core/experiments/{abf16d20-.../lanes/935c13ae (4x, s256), 578b5803-1f2f-49c4-99b8-bf7071f43917/lanes/abfc645c (8x SAE, s256), abf16d20-.../lanes/7db28a9a (4x, s42), ce93d3f6-282b-4811-84f8-be6c589c0500/lanes/c645c39e (200 epochs, s42)}/sae_analysis/sae_results.json (process_intensity, selectivity_evolution)` ← `packages/ml/src/saeanalysis.py`

### Supplementary Fig. 5
- builder: `scripts/paper/figures/supp5.py`
- helpers: `scripts/paper/figures/nmi_style.py`, `scripts/paper/figures/validation_panels.py`, `scripts/paper/figures/checks_lib.py`
- `analysis_outputs/paper/NOISE_null/tables/resnet_dup_null_per_transition.csv` ← `scripts/paper/noise_null_analyze.py`
- `analysis_outputs/paper/MATCHING/process_ordering_by_rule.csv` ← `scripts/paper/matching_aggregate.py`
- `analysis_outputs/paper/TRANSFORMED/transformed_results.json` ← `scripts/paper/transformed_report.py`
- `analysis_outputs/paper/DICTFREE/population/tables/population_per_checkpoint.csv` ← `scripts/paper/dictfree_analyze.py`
- `analysis_outputs/paper/DATA_order/transient_stage/share_sc_per_checkpoint.csv` ← `scripts/paper/transient_stage.py`

### Supplementary Fig. 6
- builder: `scripts/paper/figures/supp6.py`
- helpers: `scripts/paper/figures/nmi_style.py`, `scripts/paper/figures/noise_lib.py`, `scripts/paper/figures/fig_common.py`, `scripts/paper/figures/core_runs_lib.py`, `scripts/paper/figures/process_composition.py`, `scripts/paper/figures/checks_lib.py`
- `analysis_outputs/paper/GRADBLOCK/final_long.csv` ← `scripts/paper/gradblock_complete.py`
- `analysis_outputs/paper/RDT_blocking/feature_routing_long.csv` ← `scripts/paper/blocking_feature_routing.py`
- `analysis_outputs/paper/RDT_blocking/rdt_blocking_long.csv (+ rdt_blocking_summary.json, asserted)` ← `scripts/paper/blocking_rdt_analysis.py`

### Supplementary Fig. 7
- builder: `scripts/paper/figures/interventions/supp7.py`

### Supplementary Table 1
- builder: `scripts/paper/supp_table1_ordering.py`
- `analysis_outputs/paper/MATCHING/process_ordering_by_rule.csv` ← `scripts/paper/matching_aggregate.py`

### Supplementary Table 2
- builder: `scripts/paper/si_source_files.py`
- `analysis_outputs/paper/THRESHOLD_impact/fig4_stored_sai.json` ← `scripts/paper/fig4_stored_sai.py`

### Supplementary Table 3
- builder: `scripts/paper/si_source_files.py`
- `analysis_outputs/paper/SI/transformed_treatments.csv` ← `scripts/paper/si_source_files.py`
- `analysis_outputs/paper/SI/supp_table3_transformed.csv` ← `scripts/paper/supp_table3_transformed.py`

## Numbers in the text

Traced: the value was located in the producer's output. Inferred: the producer was identified from its code and documentation, without re-locating the value.

| Section | Producer | Status |
|---|---|---|
| Abstract; Introduction; Methods (Training and checkpoint capture, last sentence) | `scripts/paper/figures/edtable1_scan.py` | traced |
| Abstract; Results: Damaging the scaffold slows fine-grained learning (specificity paragraph); Fig. 5 legend b | `scripts/paper/figures/interventions/fig5.py` | traced |
| Results: Networks construct before they refine (para 2); Discussion ("outnumbered 69:1"); Fig. 1 legend c | `scripts/paper/figures/fig1.py` | traced |
| Results: Networks construct before they refine (Granger paragraph); Extended Data Fig. 2 legend a | `scripts/paper/figures/extended/ed2_panels.py` | traced |
| Results: The developmental sequence is determined by the data (para 1); Extended Data Fig. 3 legend | `scripts/paper/figures/extended/ed3.py` | inferred |
| Results: The developmental sequence is determined by the data (Tiny ImageNet) | `scripts/paper/figures/fig2.py` | traced |
| Results: The developmental sequence is determined by the data (para 3); Methods (Cross-architecture, last sentence); SI Superclass share of between-class variance; ED 1 legend g,h | `scripts/paper/figures/extended/ed1.py` | traced |
| Results: The developmental sequence is determined by the data (timing paragraph); ED 5 legend; SI Mode strength and learning timing | `scripts/paper/figures/ed5.py` | traced |
| Fig. 2 legend b; Extended Data Fig. 4 legend | `scripts/paper/si_source_files.py` | inferred |
| Results: The scaffold begins to form during a brief window of maximal plasticity; Fig. 3 legend b | `scripts/paper/figures/fig3.py` | traced |
| Results: The scaffold outlives the features built on it; Fig. 4 legend; Methods (Feature survival analysis, Spearman/records) | `scripts/paper/figures/fig4.py` | traced |
| Results: The scaffold outlives the features built on it (birth-matched sentence); Supplementary Table 2; ED 1 legend c (50/200 epochs) | `scripts/paper/fig4_stored_sai.py` | traced |
| Methods: Feature survival analysis (log-rank) | `scripts/paper/survival_logrank.py` | traced |
| Results: Damaging the scaffold (design paragraph); Methods: Suppression / Acquisition blocking; SI Suppression, re-encoding and acquisition blocking (design) | `scripts/run_scaffold_lesion_experiment.py` | traced |
| Results: Damaging the scaffold (specificity paragraph); Fig. 5 legend a | `scripts/paper/figures/interventions/fig5.py` | traced |
| Results: Damaging the scaffold (specificity paragraph, "6.7 percentage points"); ED 7 legend d; SI Suppression/blocking (accuracy given the true superclass); Supplementary Note 3 (standard contrasts) | `scripts/paper/oracle_within.py` | traced |
| Results: Damaging the scaffold ("as much where the two blocks match exactly"); ED 7 legend c; SI Suppression/blocking (jointly matched and layer3+ pair, redrawn random directions) | `scripts/paper/gradjoint_analysis.py` | traced |
| Results: Damaging the scaffold (asymmetry paragraph, random grouping); Fig. 5 legend c | `scripts/paper/asymmetry_rdt.py` | traced |
| Fig. 5 legend c (random grouping share) | `scripts/paper/randgroup_specificity.py` | inferred |
| Results: Damaging the scaffold (asymmetry paragraph, SAE-free read-out); ED 7 legend a,e | `scripts/paper/figures/interventions/ed7.py` | inferred |
| Results: Damaging the scaffold (release sentence); ED 7 legend h | `scripts/paper/figures/interventions/ed7.py` | inferred |
| Results: Damaging the scaffold (nesting paragraph); Fig. 5 legend d; Methods: Controlled curriculum (confined definition); SI Epochs behind, confined class features and the clean curve | `scripts/paper/rdt_pace.py` | traced |
| Fig. 5 legend d (own-superclass activity) | `scripts/paper/rdt_completion_patterns.py` | traced |
| Results: Damaging the scaffold (label-noise sentence); ED 8 legend h; SI Suppression/blocking (Noise fitting of class-selective features) | `scripts/paper/noise_fit_features.py` | traced |
| Results: Damaging the scaffold (placement sentence); Supplementary Note 3 | `scripts/paper/placement_given_analysis.py` | traced |
| Results: Building the scaffold in advance (design, head start and dose-response paragraphs); Fig. 6 legends a–c | `scripts/paper/figures/interventions/fig6.py` | traced |
| Fig. 6 legend c (30-epoch census) | `scripts/paper/rdt_census_k30.py` | traced |
| Results: Building the scaffold in advance (specificity and source paragraphs); Discussion ("four epochs ahead ... under one"); Fig. 6 legend d; Supplementary Note 4 (head start) | `scripts/paper/figures/interventions/fig6.py` | traced |
| Extended Data Fig. 9b; Supplementary Note 4 (label count) | `scripts/paper/rand5_pretrain_analysis.py` | traced |
| Extended Data Fig. 9a (per-epoch reference); Supplementary Note 4 (blind-arm measures) | `scripts/paper/blind_pretrain_analysis.py` | traced |
| Supplementary Note 4 (frozen read-outs at the switch) | `scripts/paper/blind_frozen_readout.py` | traced |
| Results: Building the scaffold in advance (nesting paragraph); Fig. 6 legend e | `scripts/paper/figures/interventions/fig6.py` | traced |
| Fig. 6 legend e (after 30 pre-training epochs) | `scripts/paper/rdt_pace_curriculum_k30.py` | traced |
| Discussion; Supplementary Note 1 | `scripts/paper/simclr_analysis.py` | traced |
| Methods: Training and checkpoint capture; SI Activation extraction; SI Training, decomposition and event classification | `scripts/paper/figures/edtable1_scan.py` | traced |
| Methods: SAE decomposition into sparse features (settings); Methods: Null baseline/Process classification (definitions) | `packages/ml/src/saeanalysis.py` | traced |
| Methods: SAE decomposition (independent initialisation) | `scripts/paper/noise_null_lane.py` | traced |
| Methods: SAE decomposition (totals and reconstruction) | `scripts/paper/sae_totals.py` | traced |
| Methods: Feature mapping and lifecycle classification; Methods: Cross-architecture (transformed fates); Supplementary Table 3 | `scripts/paper/transformed_analysis.py` | traced |
| Supplementary Table 3 (CSI-lifespan columns); Results "two of the three treatments" | `scripts/paper/supp_table3_transformed.py` | traced |
| Methods: Feature mapping (within-checkpoint control); ED 1 legend a,b; Methods: Cross-architecture (census, 11 decompositions) | `scripts/paper/figures/extended/ed1.py` | traced |
| Methods: Process identification (magnitude invariance) | `scripts/paper/figures/extended/supp4.py` | traced |
| Methods: Null baseline (permuted-label SAI); SI Feature census and alternative definitions | `scripts/paper/robust_census_analyze_storedsai.py` | traced |
| Methods: Geometric nesting and cross-layer ablation; ED 6 legend a,b | `scripts/paper/dependency_stats.py` | traced |
| Methods: Cross-architecture and robustness controls (checkpoint density) | `scripts/paper/checkpoint_density.py` | traced |
| Methods: Cross-architecture and robustness controls (matching rule); Supplementary Fig. 5b | `scripts/paper/matching_aggregate.py` | traced |
| Methods: Targeted label noise experiment | `scripts/run_noise_curriculum_experiment.py` | traced |
| Methods: Fixed-label noise experiment | `scripts/paper/noise_matched_targets.py` | traced |
| Methods: Suppression of the superclass directions (held-out images) | `scripts/paper/heldout_blocking.py` | traced |
| Methods: Suppression (epochs 1-3 comparison); SI Suppression (accuracy given true superclass under suppression) | `scripts/paper/class_level.py` | partly inferred |
| Methods: Superclass-curriculum experiment; Fig. 6 legend b; ED 9 legend e; ED Table 1 sweep rows | `scripts/paper/figures/edtable1_scan.py` | traced |
| Methods: Controlled curriculum experiment | `scripts/paper/run_curriculum.py` | traced |
| Methods: Computational infrastructure; Data availability | `scripts/paper/figures/edtable1_scan.py` | partly compiled by hand (below) |
| Extended Data Table 1 (all cells) and its caption | `scripts/paper/figures/edtable1_scan.py` | partly compiled by hand (below) |
| SI Supplementary Methods: Thresholds of each run | `scripts/paper/si_source_files.py` | traced |
| SI Supplementary Methods: Thresholds of each run (all-features robustness) | `scripts/paper/core_events_allfeat.py` | traced |
| SI Supplementary Methods: Similarity-matched null for the nesting ratio | `scripts/paper/nesting_similarity_null.py` | traced |
| SI Supplementary Methods: Per-epoch checkpoints (ResNet-18 sham runs) | `scripts/paper/perepoch_stability.py` | traced |
| SI Supplementary Methods: Per-epoch checkpoints (centres of mass, feature basis); Supplementary Fig. 1 legend | `scripts/paper/figures/supp1.py` | traced |
| SI Supplementary Methods: Per-epoch checkpoints in a transformer; Supplementary Fig. 2 legend | `scripts/paper/perepoch_cct_analysis.py` | traced |
| SI Supplementary Methods: Suppression, re-encoding and acquisition blocking (blocked shares) | `scripts/paper/gradblock_dose.py` | traced |
| SI Supplementary Methods: Suppression, re-encoding and acquisition blocking (controls and end points) | `scripts/paper/gradblock_analysis.py` | traced |
| SI Supplementary Methods: Suppression, re-encoding and acquisition blocking (routing of the learning signal) | `scripts/paper/signal_routing.py` | traced |
| SI Supplementary Methods: Suppression, re-encoding and acquisition blocking (label-noise routing) | `scripts/paper/noise_routing_nesting.py` | traced |
| SI Supplementary Methods: Feature-level effects of blocking | `scripts/paper/blocking_feature_routing.py` | traced |
| SI Supplementary Methods: Feature-level effects of blocking (per-superclass breadth) | computed from the per-feature records of `scripts/paper/blocking_feature_routing.py` | inferred |
| SI Supplementary Methods: Feature-level effects of blocking (per-superclass dose-response) | `scripts/paper/blocking_superclass_doseresponse.py` | traced |
| SI Supplementary Methods: Removal of the superclass directions from trained networks | `scripts/paper/removal_matched_groupings.py` | traced |
| SI Supplementary Methods: Superclass assignment and accuracy given the true superclass | `scripts/paper/oracle_within.py` | traced |
| SI Supplementary Methods: Epochs behind, confined class features and the clean curve | `scripts/paper/placement_curve.py` | traced |
| SI Supplementary Methods: Targeted removal of superclass-selective features | `scripts/paper/routing_heldout.py` | traced |
| SI Supplementary Methods: Linear read-outs and withheld-class read-out | `scripts/paper/curriculum_probe.py` | traced (withheld-class read-out inferred) |
| SI Supplementary Note 2 (first paragraph) | `scripts/paper/feature_origins_stored.py` | traced |
| SI Supplementary Note 2 (redundancy) | `scripts/paper/redundancy.py` | traced |
| SI Supplementary Note 2 (epoch-by-epoch ancestry paragraph) | `scripts/paper/lineage/ancestry.py` | traced |
| SI Supplementary Note 4 (superclass-blind task, separability) | `scripts/paper/task_irrelevant_matched.py` | traced |
| SI Supplementary Table 1 | `scripts/paper/supp_table1_ordering.py` | traced |
| Supplementary Figure legends (SI) and Extended Data figure legends not listed above | figure builders | see Figures and tables |

## Values compiled by hand

- Methods and Data availability: run count (178), training hours and storage, compiled from run logs and disk usage.
- Extended Data Table 1 rows for redrawn random directions, placement given and superclass-blind pre-training: values from the runs' devtrain_metrics.json (`scripts/paper/edtable1_runs_all.py` reproduces the per-run values).
- Results, first section, sign test: P = 0.5⁹ (nine of nine runs, one-sided).
- Supplementary Table 2: χ²₁ and the hazard-ratio 95% CI derived from the stored Peto hazard ratio and P (χ² = χ²₁⁻¹(P), s.e. = |ln HR|/χ).
