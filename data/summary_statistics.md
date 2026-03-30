# Summary Statistics for Paper

## Lane Overview
| Label | Architecture | Dataset | Seed | Epochs | Expansion |
|-------|-------------|---------|------|--------|-----------|
| ResNet18-CIFAR100-seed42 | ResNet-18 | CIFAR-100 | 42 | 50 | 4x |
| ResNet18-CIFAR100-seed137 | ResNet-18 | CIFAR-100 | 137 | 50 | 4x |
| ResNet18-CIFAR100-seed256 | ResNet-18 | CIFAR-100 | 256 | 50 | 4x |
| ViTSmall-CIFAR100-seed42 | ViT-Small | CIFAR-100 | 42 | 50 | 4x |
| ViTSmall-CIFAR100-seed137 | ViT-Small | CIFAR-100 | 137 | 50 | 4x |
| ViTSmall-CIFAR100-seed256 | ViT-Small | CIFAR-100 | 256 | 50 | 4x |
| CCT7-CIFAR100-seed42 | CCT-7 | CIFAR-100 | 42 | 50 | 4x |
| CCT7-CIFAR100-seed137 | CCT-7 | CIFAR-100 | 137 | 50 | 4x |
| CCT7-CIFAR100-seed256 | CCT-7 | CIFAR-100 | 256 | 50 | 4x |
| ResNet18-CIFAR100-200ep | ResNet-18 | CIFAR-100 | 42 | 200 | 4x |
| ResNet18-CIFAR100-8x | ResNet-18 | CIFAR-100 | 42 | 50 | 8x |
| ResNet18-TinyImageNet-seed42 | ResNet-18 | TinyImageNet | 42 | 50 | 4x |

## F1: Ab-H/Di-H Ratio Inversion
- **CCT7-CIFAR100-seed137**: 2.2:1 → 0.46:1
- **CCT7-CIFAR100-seed256**: 2.0:1 → 0.44:1
- **CCT7-CIFAR100-seed42**: 1.4:1 → 0.23:1
- **ResNet18-CIFAR100-200ep**: 2.1:1 → 0.11:1
- **ResNet18-CIFAR100-8x**: ∞:1 → 0.17:1
- **ResNet18-CIFAR100-seed137**: ∞:1 → 0.21:1
- **ResNet18-CIFAR100-seed256**: ∞:1 → 0.33:1
- **ResNet18-CIFAR100-seed42**: 1.0:1 → 0.30:1
- **ResNet18-TinyImageNet-seed42**: 3.5:1 → 0.04:1
- **ViTSmall-CIFAR100-seed137**: 0.9:1 → 0.37:1
- **ViTSmall-CIFAR100-seed256**: 0.4:1 → 0.30:1
- **ViTSmall-CIFAR100-seed42**: ∞:1 → 0.39:1

## F1: Tg-H Fraction of Superclass-Level Features at First Transition
(Tg-H / (Ab-H + Tg-H) — what fraction of superclass-level features are task-general)
- **CCT7-CIFAR100-seed137**: 97.4%
- **CCT7-CIFAR100-seed256**: 95.6%
- **CCT7-CIFAR100-seed42**: 88.6%
- **ResNet18-CIFAR100-200ep**: 93.8%
- **ResNet18-CIFAR100-8x**: 99.4%
- **ResNet18-CIFAR100-seed137**: 94.9%
- **ResNet18-CIFAR100-seed256**: 99.1%
- **ResNet18-CIFAR100-seed42**: 97.4%
- **ResNet18-TinyImageNet-seed42**: 89.9%
- **ViTSmall-CIFAR100-seed137**: 98.7%
- **ViTSmall-CIFAR100-seed256**: 97.5%
- **ViTSmall-CIFAR100-seed42**: 95.9%

## F1: Tg-H Fraction by Architecture (mean across seeds)
- **ResNet-18**: 80.9%
- **ViT-Small**: 85.8%
- **CCT-7**: 86.6%

## F2: Feature Churn
| Lane | First Transition | Last Transition |
|------|-----------------|-----------------|
| CCT7-CIFAR100-seed137 | 52.8% | 51.1% |
| CCT7-CIFAR100-seed256 | 46.6% | 47.1% |
| CCT7-CIFAR100-seed42 | 33.6% | 55.5% |
| ResNet18-CIFAR100-200ep | 79.7% | 24.5% |
| ResNet18-CIFAR100-8x | 77.9% | 24.3% |
| ResNet18-CIFAR100-seed137 | 83.9% | 23.2% |
| ResNet18-CIFAR100-seed256 | 76.7% | 24.0% |
| ResNet18-CIFAR100-seed42 | 81.5% | 21.2% |
| ResNet18-TinyImageNet-seed42 | 84.5% | 24.1% |
| ViTSmall-CIFAR100-seed137 | 80.7% | 46.0% |
| ViTSmall-CIFAR100-seed256 | 77.1% | 64.1% |
| ViTSmall-CIFAR100-seed42 | 64.6% | 48.3% |

## F3: Layer Stability at First Transition
- **ResNet18-CIFAR100-seed42**: avgpool: 2.3%, layer1: 35.3%, layer2: 18.1%, layer3: 4.2%, layer4: 2.5%
- **ResNet18-CIFAR100-seed137**: avgpool: 1.6%, layer1: 35.7%, layer2: 17.9%, layer3: 2.9%, layer4: 2.2%
- **ResNet18-CIFAR100-seed256**: avgpool: 3.7%, layer1: 53.1%, layer2: 17.9%, layer3: 10.0%, layer4: 5.1%
- **ViTSmall-CIFAR100-seed42**: encoder.layers.0: 10.8%, encoder.layers.11: 29.8%, encoder.layers.3: 21.4%, encoder.layers.6: 26.0%, encoder.layers.9: 25.2%, encoder.ln: 25.7%
- **ViTSmall-CIFAR100-seed137**: encoder.layers.0: 6.2%, encoder.layers.11: 15.7%, encoder.layers.3: 13.0%, encoder.layers.6: 13.9%, encoder.layers.9: 19.3%, encoder.ln: 17.5%
- **ViTSmall-CIFAR100-seed256**: encoder.layers.0: 21.7%, encoder.layers.11: 14.6%, encoder.layers.3: 23.0%, encoder.layers.6: 21.8%, encoder.layers.9: 13.8%, encoder.ln: 14.4%
- **CCT7-CIFAR100-seed42**: norm: 36.8%, transformer.layers.0: 63.2%, transformer.layers.1: 53.6%, transformer.layers.3: 48.1%, transformer.layers.5: 38.8%, transformer.layers.6: 33.6%
- **CCT7-CIFAR100-seed137**: norm: 26.6%, transformer.layers.0: 44.5%, transformer.layers.1: 38.5%, transformer.layers.3: 35.5%, transformer.layers.5: 30.0%, transformer.layers.6: 28.8%
- **CCT7-CIFAR100-seed256**: norm: 32.7%, transformer.layers.0: 51.9%, transformer.layers.1: 40.8%, transformer.layers.3: 38.2%, transformer.layers.5: 33.0%, transformer.layers.6: 36.5%
- **ResNet18-CIFAR100-200ep**: avgpool: 4.1%, layer1: 54.3%, layer2: 23.3%, layer3: 4.4%, layer4: 3.7%
- **ResNet18-CIFAR100-8x**: avgpool: 2.1%, layer1: 50.0%, layer2: 20.8%, layer3: 10.3%, layer4: 3.9%
- **ResNet18-TinyImageNet-seed42**: avgpool: 0.6%, layer1: 40.0%, layer2: 15.1%, layer3: 2.8%, layer4: 0.0%

## F4: Superclass Invariance (CIFAR-100)
- **ab_h**: mean=1.5%, range=4.3pp
- **di_h**: mean=2.9%, range=1.2pp
- **as_h**: mean=57.9%, range=3.0pp
- **de_h**: mean=29.2%, range=1.2pp

## Hypothesis Verdicts
| Lane | Ab-H | Tg-H | Di-H | As-H | De-H |
|------|------|------|------|------|------|
| CCT7-CIFAR100-seed137 | confirmed | partially_supported | confirmed | confirmed | partially_supported |
| CCT7-CIFAR100-seed256 | confirmed | partially_supported | confirmed | partially_supported | partially_supported |
| CCT7-CIFAR100-seed42 | confirmed | partially_supported | confirmed | confirmed | partially_supported |
| ResNet18-CIFAR100-200ep | confirmed | partially_supported | confirmed | partially_supported | partially_supported |
| ResNet18-CIFAR100-8x | confirmed | partially_supported | confirmed | partially_supported | partially_supported |
| ResNet18-CIFAR100-seed137 | confirmed | partially_supported | confirmed | confirmed | partially_supported |
| ResNet18-CIFAR100-seed256 | confirmed | partially_supported | confirmed | partially_supported | partially_supported |
| ResNet18-CIFAR100-seed42 | confirmed | partially_supported | confirmed | confirmed | partially_supported |
| ResNet18-TinyImageNet-seed42 | confirmed | partially_supported | confirmed | confirmed | partially_supported |
| ViTSmall-CIFAR100-seed137 | confirmed | partially_supported | confirmed | confirmed | partially_supported |
| ViTSmall-CIFAR100-seed256 | confirmed | partially_supported | confirmed | partially_supported | partially_supported |
| ViTSmall-CIFAR100-seed42 | confirmed | partially_supported | confirmed | confirmed | partially_supported |

## F5: Feature Survival (Log-Rank Tests)
| Lane | n_tracked | Survival Rate | Log-rank χ² | p | SSI-Surv ρ | CSI-Surv ρ |
|------|-----------|---------------|-------------|---|-----------|-----------|
| ResNet18-CIFAR100-seed42 | 10096 | 22.0% | 36.66 | 0.0000 | 0.057 | -0.072 |
| ResNet18-CIFAR100-seed137 | 10143 | 22.4% | 22.95 | 0.0000 | 0.060 | -0.055 |
| ResNet18-CIFAR100-seed256 | 9929 | 22.6% | 37.11 | 0.0000 | 0.063 | -0.058 |
| ViTSmall-CIFAR100-seed42 | 19997 | 26.9% | 25.88 | 0.0000 | 0.074 | -0.121 |
| ViTSmall-CIFAR100-seed137 | 20025 | 27.1% | 13.62 | 0.0002 | 0.060 | -0.118 |
| ViTSmall-CIFAR100-seed256 | 20149 | 31.9% | 103.27 | 0.0000 | 0.024 | -0.118 |
| CCT7-CIFAR100-seed42 | 21936 | 29.2% | 30.00 | 0.0000 | 0.018 | -0.135 |
| CCT7-CIFAR100-seed137 | 21987 | 29.7% | 12.26 | 0.0005 | -0.006 | -0.138 |
| CCT7-CIFAR100-seed256 | 21424 | 28.2% | 5.41 | 0.0200 | -0.038 | -0.183 |
| ResNet18-CIFAR100-200ep-seed42 | 9923 | 22.1% | 4.06 | 0.0438 | 0.018 | -0.092 |
| ResNet18-CIFAR100-8x-seed42 | 12903 | 21.9% | 42.87 | 0.0000 | 0.052 | -0.065 |
| ResNet18-TinyImageNet-seed42 | 7178 | 29.1% | 10.34 | 0.0013 | 0.042 | -0.076 |

## F5: Tg-H Expanded Survival
- **ResNet18-CIFAR100-seed42**: Tg-H survival=27.2%, non-Tg=20.1%, Log-rank χ²=71.42, p=<10⁻¹⁶
- **ResNet18-CIFAR100-seed137**: Tg-H survival=29.5%, non-Tg=19.7%, Log-rank χ²=96.01, p=<10⁻¹⁶
- **ViTSmall-CIFAR100-seed42**: Tg-H survival=27.3%, non-Tg=26.8%, Log-rank χ²=13.53, p=2.35e-04
- **ViTSmall-CIFAR100-seed137**: Tg-H survival=27.6%, non-Tg=26.9%, Log-rank χ²=8.48, p=3.59e-03
- **CCT7-CIFAR100-seed42**: Tg-H survival=23.6%, non-Tg=31.2%, Log-rank χ²=0.03, p=8.74e-01
- **CCT7-CIFAR100-seed137**: Tg-H survival=26.6%, non-Tg=30.8%, Log-rank χ²=13.49, p=2.40e-04
- **ResNet18-CIFAR100-200ep-seed42**: Tg-H survival=34.6%, non-Tg=16.0%, Log-rank χ²=286.41, p=<10⁻¹⁶
- **ResNet18-CIFAR100-8x-seed42**: Tg-H survival=27.1%, non-Tg=20.4%, Log-rank χ²=118.59, p=<10⁻¹⁶
- **ResNet18-TinyImageNet-seed42**: Tg-H survival=35.3%, non-Tg=24.6%, Log-rank χ²=63.82, p=<10⁻¹⁶