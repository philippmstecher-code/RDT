# Frozen read-out at the switch, superclass-blind pre-training (ED Fig. 9d measure)

Written by scripts/paper/blind_frozen_readout.py. Blind-arm switch computed for 5 of 5 seeds (7, 42, 137, 256, 314); fine-class checkpoints per seed: 7: 7, 42: 7, 137: 7, 256: 7, 314: 7.

Read-out: 100-class linear read-out (fine100) of frozen pooled (avgpool) features of the 10,000 stored test images, StandardScaler + RidgeClassifier(alpha=10), 5-fold stratified CV (as in curriculum_probe.py). Switch = end of the 20 pre-training epochs, before any fine-class epoch (scratch, initialisation). Scratch, true superclasses and random grouping from CURRICULUM_probe/probe.csv; superclass-blind task from BLIND_PRETRAIN/frozen_readout.csv. Mean [t-based 95% CI over seeds], %.

## fine100 at the switch, per seed

| arm | seed 7 | seed 42 | seed 137 | seed 256 | seed 314 | mean [95% CI] |
|---|---|---|---|---|---|---|
| scratch (initialisation) | 13.1 | 12.7 | 12.9 | 12.7 | 12.9 | 12.8 [12.7, 13.0] (n=5) |
| true superclasses | 35.2 | 36.7 | 36.5 | 36.9 | 35.0 | 36.1 [35.0, 37.2] (n=5) |
| random grouping | 37.7 | 37.5 | 36.7 | 36.4 | 38.6 | 37.4 [36.3, 38.5] (n=5) |
| superclass-blind task | 17.6 | 17.3 | 13.5 | 16.7 | 14.5 | 15.9 [13.7, 18.2] (n=5) |

## Paired contrasts at the switch (seeds with the blind arm), percentage points

- blind - scratch (initialisation): 3.1 [0.8, 5.4] (n=5; per seed 7: +4.6, 42: +4.6, 137: +0.6, 256: +4.0, 314: +1.6)
- blind - true superclasses: -20.1 [-22.6, -17.7] (n=5; per seed 7: -17.6, 42: -19.4, 137: -23.0, 256: -20.2, 314: -20.5)
- blind - random grouping: -21.4 [-24.0, -18.9] (n=5; per seed 7: -20.1, 42: -20.2, 137: -23.2, 256: -19.7, 314: -24.1)

## Other read-outs at the switch, mean [95% CI] (%)

| arm | within_sc (5-way inside each superclass, chance 20) | super20 (20-way superclass, chance 5) |
|---|---|---|
| scratch (initialisation) | 35.3 [34.9, 35.8] | 22.9 [22.7, 23.1] |
| true superclasses | 49.8 [48.6, 50.9] | 56.4 [55.0, 57.7] |
| random grouping | 53.0 [52.0, 53.9] | 50.5 [49.0, 51.9] |
| superclass-blind task | 39.7 [38.3, 41.1] | 26.3 [23.4, 29.2] |

## fine100 over fine-class epochs (ED 9d checkpoints), mean [95% CI] (n seeds)

| fine epoch | scratch (initialisation) | true superclasses | random grouping | superclass-blind task |
|---|---|---|---|---|
| 0 | 12.8 [12.7, 13.0] (5) | 36.1 [35.0, 37.2] (5) | 37.4 [36.3, 38.5] (5) | 15.9 [13.7, 18.2] (5) |
| 1 | – | 34.6 [33.7, 35.4] (5) | 33.1 [32.3, 34.0] (5) | 22.3 [21.4, 23.2] (5) |
| 5 | 33.3 [32.7, 33.9] (5) | 40.8 [40.3, 41.3] (5) | 40.2 [39.1, 41.3] (5) | 33.5 [32.6, 34.4] (5) |
| 10 | 41.0 [40.6, 41.4] (5) | 44.7 [44.1, 45.3] (5) | 44.3 [43.1, 45.5] (5) | 39.7 [39.4, 40.0] (5) |
| 15 | 45.8 [45.3, 46.4] (5) | 47.7 [47.3, 48.1] (5) | 47.3 [46.2, 48.3] (5) | 43.2 [42.8, 43.6] (5) |
| 20 | 49.2 [48.9, 49.4] (5) | 50.1 [49.0, 51.1] (5) | 49.9 [49.0, 50.7] (5) | 47.0 [46.7, 47.3] (5) |
| 25 | 51.9 [51.2, 52.5] (5) | 52.8 [52.0, 53.6] (5) | 52.2 [51.7, 52.6] (5) | 49.7 [49.2, 50.3] (5) |
| 30 | 53.3 [52.7, 53.8] (5) | 53.9 [53.1, 54.8] (5) | 53.4 [52.9, 53.8] (5) | 51.2 [50.6, 51.7] (5) |

Initialisation check: blind-arm epoch-0 fine100 equals the curriculum arms' epoch 0 for the same seed (same initial weights): seed 7 yes, seed 42 yes, seed 137 yes, seed 256 yes, seed 314 yes.
