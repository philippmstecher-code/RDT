# Nature Machine Intelligence — Machine Learning Checklist (v1.1)

**Paper:** Deep networks learn scaffolded representations
**Authors:** Stecher, Radovanovic, Sikimic, Kahle

---

## Section 1: Availability and Reproducibility of Code and Data

*Select all that apply regarding the availability of the data and code used in the study.*

| Item | Response |
|------|----------|
| Code will be included in a CodeOcean capsule | **No.** |
| The source code is included in the submission or available in a public repository | **Yes.** GitHub: [URL to be inserted]. Archived on Zenodo: [DOI to be assigned at acceptance]. |
| A compiled standalone version of the software is included | **No.** The software is a Python library, not a compiled application. |
| A test dataset and instructions/scripts for replicating the results are included | **Yes.** Pre-computed analysis data (JSON) and all figure/validation scripts are included. A demo script (`demo/demo_sae_analysis.py`) runs the full pipeline on a CIFAR-10 subset in ~3 minutes. |
| A Readme file with instructions for installing and running the code is included | **Yes.** `README.md` documents installation, system requirements, and three-tier reproduction instructions. |
| Is code available? | **Yes.** The repository is public. |
| Pretrained models are used in the study and accessible through [URL] | **Not applicable.** All models are trained from scratch (Kaiming normal initialization). No pretrained weights are used. |
| Pretrained models are used in the study and are not accessible | **Not applicable.** See above. |
| The paper contains information on how to obtain code and data after publication | **Yes.** Code and Data Availability statements are included. Code is archived on Zenodo with a DOI. Pre-computed analysis data is in the repository; raw per-lane SAE results are deposited on Zenodo. |

---

## Section 2: Datasets

| Item | Response |
|------|----------|
| **2A.** All data sources are listed in the paper. | **Yes.** CIFAR-100 (Krizhevsky, 2009) and Tiny ImageNet (Le & Yang, 2015) are listed in Methods. |
| **2B.** The train, test and validation datasets are publicly available, and links/accession numbers have been provided. | **Yes.** CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html. Tiny ImageNet: https://image-net.org/download-images. Both are auto-downloaded by torchvision during training. |
| **2C.** We have reported and discussed potential dataset biases in the paper. | **Not applicable.** CIFAR-100 and Tiny ImageNet are established benchmarks with known properties. The study investigates universal training dynamics across architectures and datasets, not the content of specific categories. The cross-dataset replication (CIFAR-100 to Tiny ImageNet with different classes, resolution, and hierarchy) tests robustness to dataset-specific biases. |
| **2D.** The data cleaning and preprocessing steps are clearly and fully described. | **Yes.** Methods section specifies: RandomCrop (32x32, padding=4), RandomHorizontalFlip, Normalize (per-dataset mean/std). No data cleaning or filtering is applied — all samples are used. |
| **2E.** Instances of combining data from multiple sources are clearly identified. | **Not applicable.** Each experiment uses a single dataset. No data sources are combined. |

---

## Section 3: Model and Training

| Item | Response |
|------|----------|
| **3A.** What model architecture is the current model based on? | ResNet-18 (He et al. 2016), ViT-Small (Dosovitskiy et al. 2021), CCT-7 (Hassani et al. 2021). Sparse autoencoders follow Bricken et al. (2023) with top-k sparsity. All implementations in `src/models.py`, `src/cct.py`, `src/sae.py`. |
| **3B.** A Model Card is provided. | **Yes.** See `MODEL_CARD.md`. |
| **3C.** The model clearly splits data into different sets for training, validation, and testing. | **Yes.** Standard CIFAR-100 train/test split (50,000/10,000). Standard Tiny ImageNet train/val split (100,000/10,000). The test set is used exclusively for validation metrics; no hyperparameter tuning is performed on it. |
| **3D.** The method of data splitting is clearly stated. | **Yes.** The canonical train/test splits provided by the dataset distributors are used without modification. Stated in Methods. |
| **3E.** The data splitting mimics anticipated real-world applications. | **Not applicable.** This is a scientific study of training dynamics, not a deployment system. The standard benchmark splits are appropriate for the research question. |
| **3F.** The data splitting procedure has been chosen to avoid data leakage. | **Yes.** Standard non-overlapping train/test splits. No test-set information is used during training or model selection. |
| **3G.** The interpretability of the model has been studied and clearly validated. | **Yes.** The entire paper is an interpretability study. SAE-based feature decomposition is validated through: reconstruction quality (cosine sim 0.975), within-checkpoint controls (Extended Data Fig. 1), independent-initialization controls (Extended Data Fig. 1c), 1,000-permutation null baselines (Extended Data Fig. 1d-f), and causal manipulations (Figs. 5-6). |

---

## Section 4: Evaluation

| Item | Response |
|------|----------|
| **4A.** The performance metrics used are described and justified in the paper. | **Yes.** Selectivity indices (SSI, CSI, SAI), feature survival rate, churn rate, process fractions, Construction/Refinement ratio, Granger causality F-statistics, Cohen's d, log-rank test statistics, and BCa bootstrap confidence intervals. All defined in Methods. |
| **4B.** Cross-validation of the results is included. | **No formal cross-validation.** The study uses multi-seed replication (3 seeds for standard runs, 5 seeds for noise experiments) across 3 architectures and 2 datasets to assess reproducibility. This is more appropriate than cross-validation for studying training dynamics, where each run is an independent observation of the developmental process. |
| **4C.** Community-accepted benchmark datasets/tasks are used for comparisons. | **Yes.** CIFAR-100 and Tiny ImageNet are standard ML benchmarks. |
| **4D.** Baseline comparisons to simple/trivial models are provided. | **Not applicable.** The paper studies representational development during training, not proposing a classifier to be benchmarked. The relevant baselines are experimental controls: standard training vs. noise conditions vs. curriculum conditions. Null baselines (1,000-permutation tests) establish that observed selectivity indices exceed chance levels. |
| **4E.** Benchmarks with current state-of-the-art are provided. | **Not applicable.** The paper does not propose a new classification method or claim state-of-the-art accuracy. Classification accuracy is reported for completeness (Extended Data Table 1) but is not the contribution. |
| **4F.** Ablation experiments are included. | **Yes.** Extended training (200 epochs), higher SAE capacity (8x), independent SAE initialization, cross-dataset validation (Tiny ImageNet), and 7-condition curriculum sweep. See Extended Data Figs. 1, 4, and Fig. 6. |
| **4G.** The model has been tested on a fully independent dataset. | **Yes.** Core findings replicate on Tiny ImageNet (200 classes, 27 superclasses, 64x64 resolution) — a different dataset with different classes, image resolution, and superclass structure than the primary CIFAR-100 experiments. |

---

## Section 5: Computational Resources

| Item | Response |
|------|----------|
| **5A.** The paper contains information on hardware/computing resources that were used. | **Yes.** All training was conducted on NVIDIA RTX 4090 GPUs (24 GB VRAM) provisioned via RunPod cloud computing (12 vCPU, 41 GB RAM per instance). Storage: ~500 GB for checkpoints, activations, and SAE results across all 55 lanes. |
| **5B.** The paper includes information on the computational costs in terms of computation time, parallelization or carbon footprint estimates. | **Yes.** Total compute: ~200 GPU-hours on RTX 4090 across all 55 training runs (9 standard + 3 ablation + 1 cross-dataset + 7 curriculum + 35 label noise), including DEVTRAIN and SAEANALYSIS stages. Runs were executed sequentially or in small parallel batches (1-3 GPUs). |
