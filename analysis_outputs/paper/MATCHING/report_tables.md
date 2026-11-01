### Table 1. Reproduction of stored matching counts and process intensities

| Architecture | Seed | (layer, transition) cells with identical n_stable/n_born/n_died/n_transformed | transitions with identical process_intensity (Tg/Ab/Di/As/De, sample-weighted) |
|---|---|---|---|
| ResNet-18 | 42 | 55/55 | 11/11 |
| ResNet-18 | 137 | 55/55 | 11/11 |
| ResNet-18 | 256 | 54/55 | 10/11 |
| ViT-Small | 42 | 54/54 | 9/9 |
| ViT-Small | 137 | 54/54 | 9/9 |
| ViT-Small | 256 | 54/54 | 9/9 |
| CCT-7 | 42 | 60/60 | 10/10 |
| CCT-7 | 137 | 60/60 | 10/10 |
| CCT-7 | 256 | 60/60 | 10/10 |

**Total:** 506/507 cells, 89/90 transitions.

### Table 2. Hungarian diagnostics (mean over 3 seeds, range in parentheses; counts pooled over layers and transitions within a run; ResNet layer4 dropped)

| Metric | ResNet-18 real | ViT-Small real | CCT-7 real | ResNet-18 null | ViT-Small null | CCT-7 null |
|---|---|---|---|---|---|---|
| Stable (% alive A) | 45.4 (44.9–45.8) | 27.7 (27.4–27.8) | 36.1 (33.9–37.9) | 51.8 (51.1–52.3) | 25.3 (19.3–28.4) | 28.0 (26.1–29.6) |
| Transformed (% alive A) | 46.6 (46.3–47.1) | 47.9 (47.3–48.9) | 47.7 (47.2–48.3) | 39.9 (38.4–41.2) | 50.1 (49.4–50.9) | 44.7 (43.5–45.3) |
| Died (% alive A) | 8.0 (7.8–8.1) | 24.4 (23.3–25.0) | 16.1 (14.9–17.8) | 8.2 (6.7–9.3) | 24.5 (20.8–31.3) | 27.3 (25.1–28.6) |
| Born (% alive B) | 15.4 (15.0–15.9) | 41.4 (39.9–44.0) | 33.3 (30.6–36.2) | 8.3 (7.3–10.2) | 24.5 (21.2–31.0) | 27.2 (26.1–28.2) |
| Births with a backward partner r ≥ 0.5 (% births) | 8.2 (6.8–9.5) | 2.4 (2.0–2.6) | 2.5 (2.0–3.5) | 7.0 (3.6–10.3) | 0.8 (0.2–1.1) | 0.6 (0.4–0.7) |
| Births with a backward partner r ≥ 0.2 (% births) | 81.2 (75.9–85.5) | 47.8 (46.2–50.1) | 53.7 (49.1–58.6) | 90.7 (84.6–94.5) | 32.3 (30.8–33.8) | 38.0 (32.5–42.9) |
| Re-assignment scenario: newborn is a better match (r ≥ 0.5) for some A than A's assigned successor (% births) | 0.30 (0.26–0.37) | 0.03 (0.02–0.04) | 0.02 (0.00–0.05) | 1.63 (0.00–2.56) | 0.07 (0.00–0.14) | 0.04 (0.00–0.11) |
| Re-assignment scenario, count per run | 2.3 (2.0–3.0) | 1.7 (1.0–2.0) | 0.7 (0.0–2.0) | 0.7 (0.0–1.0) | 0.7 (0.0–1.0) | 0.3 (0.0–1.0) |
| Deaths with a forward partner r ≥ 0.5 (% deaths) | 18.0 (15.7–19.5) | 3.0 (1.9–4.5) | 3.1 (2.5–3.4) | 5.8 (5.7–5.9) | 0.4 (0.4–0.5) | 0.4 (0.3–0.5) |
| Deaths with a forward partner r ≥ 0.2 (% deaths) | 89.1 (87.9–89.8) | 39.4 (36.4–43.6) | 44.4 (38.9–48.7) | 89.1 (82.9–98.0) | 33.9 (31.7–35.9) | 36.6 (33.5–39.9) |
| Transformed A with some partner r ≥ 0.5 (% transformed) | 20.8 (20.6–21.2) | 11.2 (10.9–11.7) | 14.4 (14.0–14.6) | 16.7 (13.1–20.4) | 3.2 (2.4–3.8) | 5.0 (4.3–5.7) |
| Stable pairs that are mutual best matches (%) | 65.1 (64.3–66.6) | 70.4 (68.4–73.2) | 70.5 (69.6–71.0) | 87.9 (84.7–90.0) | 93.3 (92.0–94.2) | 93.8 (92.7–94.9) |
| Stable pairs with negative margin (a better alternative exists in ≥ 1 direction) (%) | 34.9 (33.4–35.7) | 29.6 (26.8–31.6) | 29.5 (29.0–30.4) | 12.1 (10.0–15.3) | 6.7 (5.8–8.0) | 6.2 (5.1–7.3) |
| Stable pairs with margin < 0.1 (%) | 64.4 (63.5–65.9) | 60.6 (58.4–63.7) | 56.7 (55.3–57.8) | 27.3 (24.3–30.7) | 17.3 (14.6–22.4) | 17.9 (16.3–20.4) |
| Stable pairs with an alternative partner r ≥ 0.5 (%) | 71.2 (69.9–72.7) | 68.2 (66.0–70.3) | 66.8 (64.3–69.2) | 43.5 (35.3–49.8) | 22.5 (19.6–25.9) | 24.7 (23.5–25.7) |
| Splits: A with ≥ 2 B partners r ≥ 0.35 (% alive A) | 70.0 (69.4–70.4) | 42.4 (41.4–43.5) | 49.7 (47.9–51.9) | 58.9 (55.2–61.0) | 15.6 (14.6–16.4) | 19.7 (17.4–22.4) |
| Merges: B with ≥ 2 A partners r ≥ 0.35 (% alive B) | 64.0 (63.6–64.2) | 32.9 (30.4–34.8) | 40.3 (37.6–43.7) | 58.1 (55.8–62.0) | 15.6 (15.2–16.1) | 19.8 (16.7–22.2) |
| Within-dictionary near-duplicates r ≥ 0.35 (% alive, same checkpoint) | 60.0 (59.4–60.4) | 31.4 (28.5–33.3) | 36.1 (33.5–38.9) |  |  |  |
| Within-dictionary near-duplicates r ≥ 0.5 (% alive, same checkpoint) | 27.5 (26.5–28.0) | 15.4 (14.1–16.4) | 15.6 (15.3–16.2) |  |  |  |

'null' = the k→terminal transition (same final model, SAE re-trained).

### Table 3. Event counts and Tg/Ab/Di label agreement under alternative matching rules (real transitions)

| Rule | Arch | Stable % A | Transformed / ambiguous % A | Died % A | Born % B | Ambiguous B % B | Label changed (% of features labelled under either rule) | Label changed (% of alive B) |
|---|---|---|---|---|---|---|---|---|
| Hungarian (pipeline) | ResNet-18 | 45.4 (44.9–45.8) | 46.6 (46.3–47.1) | 8.0 (7.8–8.1) | 15.4 (15.0–15.9) | 0.0 (0.0–0.0) | — | — |
| Hungarian (pipeline) | ViT-Small | 27.7 (27.4–27.8) | 47.9 (47.3–48.9) | 24.4 (23.3–25.0) | 41.4 (39.9–44.0) | 0.0 (0.0–0.0) | — | — |
| Hungarian (pipeline) | CCT-7 | 36.1 (33.9–37.9) | 47.7 (47.2–48.3) | 16.1 (14.9–17.8) | 33.3 (30.6–36.2) | 0.0 (0.0–0.0) | — | — |
| Mutual best | ResNet-18 | 35.6 (35.0–36.3) | 11.6 (11.3–12.1) | 52.8 (51.7–53.5) | 56.6 (55.5–57.3) | 0.0 (0.0–0.0) | 62.4 (58.9–67.2) | 6.2 (5.1–7.7) |
| Mutual best | ViT-Small | 22.7 (22.2–23.6) | 19.7 (17.7–21.7) | 57.6 (54.7–60.1) | 67.2 (66.5–68.3) | 0.0 (0.0–0.0) | 45.9 (38.9–49.6) | 6.2 (4.9–8.7) |
| Mutual best | CCT-7 | 29.8 (27.7–31.3) | 18.0 (17.0–18.8) | 52.3 (51.7–53.5) | 62.1 (60.6–63.9) | 0.0 (0.0–0.0) | 45.3 (43.9–47.9) | 6.1 (5.3–7.1) |
| Greedy | ResNet-18 | 45.3 (44.9–45.5) | 42.3 (42.1–42.6) | 12.4 (12.0–12.8) | 19.4 (18.9–20.3) | 0.0 (0.0–0.0) | 34.6 (33.9–35.6) | 1.8 (1.3–2.5) |
| Greedy | ViT-Small | 27.5 (27.1–27.8) | 45.5 (44.8–46.3) | 27.0 (25.9–27.6) | 43.5 (42.0–45.9) | 0.0 (0.0–0.0) | 19.2 (15.9–21.2) | 1.7 (1.3–2.5) |
| Greedy | CCT-7 | 36.0 (33.7–37.7) | 44.6 (44.1–45.2) | 19.4 (18.3–21.1) | 36.0 (33.3–38.8) | 0.0 (0.0–0.0) | 20.5 (17.9–22.8) | 1.8 (1.6–1.9) |
| Confirmed | ResNet-18 | 45.4 (44.9–45.8) | 53.7 (53.2–54.2) | 0.9 (0.8–0.9) | 2.9 (2.3–3.7) | 55.4 (54.2–56.6) | 23.5 (21.0–26.6) | 1.0 (0.6–1.5) |
| Confirmed | ViT-Small | 27.7 (27.4–27.8) | 57.5 (56.4–59.0) | 14.8 (13.1–15.9) | 21.6 (21.4–22.0) | 57.0 (56.5–57.7) | 49.3 (48.2–50.6) | 4.0 (2.5–6.7) |
| Confirmed | CCT-7 | 36.1 (33.9–37.9) | 54.9 (54.4–55.2) | 9.0 (7.6–10.9) | 15.5 (12.7–18.4) | 55.8 (55.2–56.4) | 46.3 (44.5–47.5) | 3.5 (2.8–4.1) |

A feature's label is its set of {Tg-E, Ab-E, Di-E} flags (Tg/Ab exclusive, as in the pipeline); 'changed' means the set differs between the rule and Hungarian.

### Table 4. Developmental program under each rule (9 runs; terminal excluded; layer4 dropped)

| Weighting | Rule | Runs with timing order Tg < Ab < Di (centre of mass) | Exceptions | Runs with C/R last < C/R first | Median Spearman ρ(C/R, t) | Runs with C/R < 1 at last real transition | Median ρ vs Hungarian per-transition counts (Tg / Ab / Di) |
|---|---|---|---|---|---|---|---|
| sample weighted | hungarian | 9/9 | none | 9/9 | -0.95 | 5/9 | — |
| sample weighted | mutual best | 9/9 | none | 9/9 | -0.98 | 0/9 | 0.48 / 0.76 / 0.95 |
| sample weighted | greedy | 9/9 | none | 9/9 | -0.95 | 5/9 | 0.87 / 0.93 / 0.98 |
| sample weighted | confirmed | 8/9 | CCT-7 s256 | 9/9 | -0.80 | 9/9 | 0.52 / 0.88 / 0.93 |
| feature level | hungarian | 9/9 | none | 9/9 | -0.93 | 9/9 | — |
| feature level | mutual best | 9/9 | none | 9/9 | -0.96 | 8/9 | 0.55 / 0.92 / 0.99 |
| feature level | greedy | 9/9 | none | 9/9 | -0.93 | 9/9 | 0.92 / 0.96 / 1.00 |
| feature level | confirmed | 8/9 | CCT-7 s256 | 9/9 | -0.90 | 8/9 | 0.71 / 0.91 / 0.98 |

Centre of mass = Σ t·count_t / Σ count_t on a 0–1 transition axis. Sample-weighted counts = Phase-6 per-sample totals (the quantity behind `process_intensity` and Fig. 2).

### Table 4b. Timing centre of mass (mean of 3 seeds) and first / last real C/R per seed (sample-weighted)

| Rule | Arch | COM Tg | COM Ab | COM Di | C/R first (s42; s137; s256) | C/R last real (s42; s137; s256) |
|---|---|---|---|---|---|---|
| hungarian | ResNet-18 | 0.18 | 0.45 | 0.62 | 55.1; ∞; ∞ | 2.84; 2.48; 0.21 |
| hungarian | ViT-Small | 0.40 | 0.57 | 0.77 | ∞; 69.4; 17.1 | 1.76; 0.53; 0.54 |
| hungarian | CCT-7 | 0.48 | 0.59 | 0.78 | 12.6; 84.3; 44.8 | 0.82; 1.57; 0.79 |
| mutual best | ResNet-18 | 0.38 | 0.50 | 0.63 | 84.8; ∞; ∞ | 5.37; 6.79; 16.10 |
| mutual best | ViT-Small | 0.32 | 0.62 | 0.80 | ∞; 455.7; 146.0 | 3.23; 2.88; 1.06 |
| mutual best | CCT-7 | 0.36 | 0.64 | 0.79 | 300.7; 382.6; 294.6 | 1.16; 1.89; 2.73 |
| greedy | ResNet-18 | 0.23 | 0.46 | 0.61 | 65.3; ∞; ∞ | 2.18; 0.39; 0.23 |
| greedy | ViT-Small | 0.40 | 0.57 | 0.77 | ∞; 74.8; 35.0 | 1.85; 1.20; 0.58 |
| greedy | CCT-7 | 0.48 | 0.60 | 0.78 | 21.4; 82.5; 62.6 | 0.82; 1.23; 0.88 |
| confirmed | ResNet-18 | 0.03 | 0.50 | 0.65 | 20.1; ∞; ∞ | 0.19; 0.40; 0.21 |
| confirmed | ViT-Small | 0.31 | 0.54 | 0.80 | ∞; 634.9; 12.6 | 0.63; 0.36; 0.28 |
| confirmed | CCT-7 | 0.56 | 0.58 | 0.78 | 38.5; 15.2; 1.0 | 0.49; 0.87; 0.56 |
### Table 5. Skip-one consistency (real checkpoints only)

| Metric | ResNet-18 | ViT-Small | CCT-7 | Per-epoch ResNet-18 lesion lanes (random L1–3; random avgpool; scaffold L1–3; scaffold avgpool) |
|---|---|---|---|---|
| Stable in chain k→k+1→k+2 (% alive at k) | 30.3 (30.0–30.7) | 23.3 (21.6–25.6) | 27.8 (27.0–28.4) | 40.5; 25.9; 39.1; 23.3 |
| Stable in direct k→k+2 (% alive at k) | 35.5 (34.9–36.0) | 27.3 (25.0–31.2) | 34.2 (33.6–34.6) | 50.6; 36.1; 52.0; 32.4 |
| Agreement on stable / not stable (%) | 76.6 (75.8–77.3) | 82.6 (80.5–84.0) | 78.1 (77.9–78.4) | 74.9; 74.8; 72.1; 77.2 |
| Cohen's κ | 0.47 (0.46–0.49) | 0.54 (0.52–0.55) | 0.49 (0.48–0.50) | 0.50; 0.42; 0.45; 0.44 |
| P(direct stable | chain stable) (%) | 70.0 (69.0–71.5) | 71.2 (69.7–72.9) | 72.3 (71.3–72.9) | 81.5; 71.2; 80.8; 70.7 |
| P(chain stable | direct stable) (%) | 59.8 (58.2–60.8) | 60.8 (59.8–61.3) | 58.6 (57.3–59.9) | 65.3; 51.0; 60.8; 50.8 |
| Same k+2 partner when both stable (%) | 49.6 (48.2–51.2) | 46.7 (46.5–47.0) | 49.9 (48.5–52.4) | 51.7; 48.0; 60.3; 52.8 |
| Direct-only: chain broken by a transformed link (%) | 91.8 (91.6–91.9) | 96.2 (94.6–97.5) | 96.6 (96.1–97.1) | 90.6; 97.1; 93.3; 90.7 |
| Direct-only: chain broken by a death (%) | 8.2 (8.1–8.4) | 3.8 (2.5–5.4) | 3.4 (2.9–3.9) | 9.4; 2.9; 6.7; 9.3 |
| Flicker: died k→k+1 but stable k→k+2 (% of deaths) | 8.9 (8.3–9.7) | 2.3 (0.9–4.5) | 3.4 (2.8–3.7) | 11.5; 4.5; 12.5; 8.3 |
| Flicker (% alive at k) | 0.7 (0.6–0.7) | 0.4 (0.1–0.6) | 0.3 (0.3–0.4) | 1.1; 0.4; 1.1; 1.0 |
| Flicker among features stable k−1→k (% of their deaths) | 24.1 (19.1–27.5) | 18.6 (9.5–30.0) | 22.1 (12.5–30.0) | 18.2; 22.2; 35.7; 36.4 |
| Births at k+1 that die at k+1→k+2 (%) | 15.8 (15.5–16.5) | 46.0 (40.3–50.7) | 37.6 (35.7–41.2) | 26.8; 13.4; 34.3; 17.1 |
### Table 6. Per-epoch (dense) vs 5-epoch (coarse) schedule, scaffold_lesion e00 lanes (epochs 0–15)

| Lane | Layers | Window (epochs) | Alive at start | Chain-stable all 5 steps (%) | Direct-stable (%) | Births: dense sum / dense surviving to end / coarse | Deaths: dense sum / coarse | Tg feats dense/coarse | Ab feats dense/coarse | Di feats dense/coarse | C/R dense (sum) | C/R coarse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | avgpool | 0->5 | 43 | 4.7 | 2.3 | 392 / 73 / 362 | 46 / 16 | 0/0 | 30/15 | 30/34 | 0.92 | 0.28 |
| random | avgpool | 5->10 | 389 | 5.7 | 15.7 | 87 / 19 / 3 | 245 / 161 | 0/0 | 40/15 | 43/8 | 0.79 | 2.08 |
| random | avgpool | 10->15 | 231 | 15.6 | 36.4 | 95 / 16 / 71 | 29 / 5 | 0/0 | 35/4 | 40/6 | 0.57 | 0.24 |
| random | layers1-3 | 0->5 | 112 | 8.0 | 30.4 | 124 / 45 / 60 | 82 / 18 | 32/8 | 0/0 | 0/0 | ∞ | ∞ |
| random | layers1-3 | 5->10 | 154 | 22.7 | 44.8 | 81 / 20 / 32 | 62 / 13 | 1/1 | 0/0 | 4/0 | 11.25 | ∞ |
| random | layers1-3 | 10->15 | 173 | 17.9 | 47.4 | 97 / 24 / 17 | 91 / 11 | 1/0 | 1/0 | 3/1 | 16.53 | 0.00 |
| scaffold | avgpool | 0->5 | 45 | 2.2 | 0.0 | 504 / 141 / 491 | 40 / 27 | 6/0 | 41/32 | 96/93 | 1.71 | 0.22 |
| scaffold | avgpool | 5->10 | 509 | 5.9 | 18.7 | 219 / 63 / 8 | 377 / 166 | 0/0 | 78/18 | 112/14 | 0.38 | 1.10 |
| scaffold | avgpool | 10->15 | 351 | 10.0 | 33.6 | 82 / 16 / 5 | 149 / 72 | 0/0 | 40/5 | 83/6 | 0.30 | 0.55 |
| scaffold | layers1-3 | 0->5 | 112 | 22.3 | 28.6 | 85 / 20 / 49 | 59 / 23 | 14/10 | 1/0 | 3/1 | 133.37 | 647.38 |
| scaffold | layers1-3 | 5->10 | 138 | 24.6 | 53.6 | 89 / 32 / 38 | 60 / 9 | 4/0 | 1/1 | 2/1 | 85.07 | 0.66 |
| scaffold | layers1-3 | 10->15 | 167 | 16.8 | 45.5 | 78 / 17 / 24 | 69 / 15 | 4/1 | 2/0 | 2/0 | 23.01 | ∞ |

### Table 7. Same-model null in the lesion lanes (checkpoint 18 = epoch 30 vs 19 = duplicate 'final')

| Lane | Layer | Alive A | Stable % | Transformed % | Died % | Born % B |
|---|---|---|---|---|---|---|
| random | avgpool | 264 | 58.3 | 38.6 | 3.0 | 7.2 |
| random | layer1 | 40 | 82.5 | 12.5 | 5.0 | 13.6 |
| random | layer2 | 74 | 48.6 | 41.9 | 9.5 | 10.7 |
| random | layer3 | 98 | 58.2 | 32.7 | 9.2 | 2.2 |
| random | layer4 | 269 | 63.6 | 34.9 | 1.5 | 3.3 |
| scaffold | avgpool | 255 | 61.2 | 36.9 | 2.0 | 5.7 |
| scaffold | layer1 | 39 | 84.6 | 10.3 | 5.1 | 9.8 |
| scaffold | layer2 | 77 | 54.5 | 31.2 | 14.3 | 12.0 |
| scaffold | layer3 | 100 | 55.0 | 34.0 | 11.0 | 3.3 |
| scaffold | layer4 | 265 | 57.7 | 37.4 | 4.9 | 3.1 |
