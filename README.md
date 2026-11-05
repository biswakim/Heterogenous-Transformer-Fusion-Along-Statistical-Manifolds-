# Heterogeneous Transformer Fusion along Statistical Manifolds: A Bures-Wasserstein Approach

This repository contains the implementation and experimental code for our research on fusing Vision Transformers with different architectural depths using Optimal Transport and Bures-Wasserstein geometry.

The work investigates how structural alignment, parameter alignment, and barycentric interpolation affect the quality of fused models, particularly when the parent Transformers have different numbers of blocks.

## Overview

Combining independently trained neural networks is challenging because their learned parameters may not have corresponding functional roles. This becomes more difficult when the networks differ in depth.

We study a multi-scale fusion framework that combines three approaches:

1. **Macro-scale alignment:** Fused Gromov-Wasserstein (FGW) transport for establishing cross-layer correspondences between Transformer blocks.
2. **Meso-scale synthesis:** Bures-Wasserstein barycentric interpolation for constructing intermediate Transformer blocks.
3. **Micro-scale alignment:** Sinkhorn transport and Hungarian assignment for aligning feed-forward channels and attention heads.

A central part of the work is investigating whether explicit permutation alignment is necessary when the final parameter synthesis is performed using Bures-Wasserstein interpolation.

## Research Questions

- How can Transformers with different depths be fused without joint retraining?
- Does graph-based structural alignment improve cross-layer correspondence?
- How does Bures-Wasserstein interpolation compare with Euclidean interpolation?
- Do hard permutation alignments survive subsequent barycentric synthesis?
- How do soft transport plans behave differently from hard permutations?
- How does fusion performance change under depth and width heterogeneity?

## Methodology

The framework is organised into three scales:

| Scale | Method | Function |
|---|---|---|
| Macro | Fused Gromov-Wasserstein | Cross-layer structural alignment |
| Meso | Bures-Wasserstein | Block-level parameter synthesis |
| Micro | Sinkhorn and Hungarian | Channel and attention-head alignment |

The implementation also includes alternative fusion strategies and ablation experiments to isolate the contribution of individual components.

## Theoretical Analysis

We investigate the permutation properties of Gaussian Bures-Wasserstein interpolation.

In particular, we establish:

- **Sample-axis invariance:** Bures-Wasserstein interpolation is invariant to hard permutations of the sample axis.
- **Feature-axis equivariance:** Feature-axis permutations transform the output correspondingly rather than leaving it unchanged.
- **Implications for model fusion:** Hard permutation alignment may have limited influence when followed by Bures-Wasserstein barycentric synthesis.

These properties are examined both analytically and through numerical experiments.

## Experiments

Experiments are conducted on:

- CIFAR-100
- Tiny-ImageNet

The study considers different Transformer depths, including 6-layer and 12-layer configurations, as well as additional depth and width heterogeneity settings.

### Fusion methods and baselines

- Full multi-scale OT fusion
- Bures-Wasserstein interpolation without alignment
- Euclidean interpolation
- Hungarian permutation alignment
- Sinkhorn transport alignment
- Identity padding
- CLAFusion-style dynamic programming alignment
- Naive parameter averaging
- Parent models and logit ensembles

Experiments include zero-shot evaluation, fine-tuning, depth-gap comparisons, width-heterogeneous settings, and ablation studies.

## Repository Structure

```text
.
├── src/
│   ├── baselines/
│   ├── experiments/
│   ├── fusion/
│   ├── models/
│   ├── stats/
│   ├── utils/
│   ├── viz/
│   ├── data/
│   └── __init__.py
│
├── make_figures.py
├── recovery_report.py
├── testgraph.py
│
├── README.md
└── setup/
```

The source directories contain the model implementations, fusion methods, alignment procedures, statistical utilities, experiment code, and visualisation components.

## Installation

The experiments use Python and PyTorch, along with the following libraries:

- PyTorch
- Torchvision
- SciPy
- Python Optimal Transport (POT)
- Matplotlib

CUDA support is used for GPU-based experiments.

Refer to the setup script for the dependency versions and CUDA configuration used in the experiments.

## Running Experiments

The repository includes experiment scripts and utilities for running evaluations, generating figures, and analysing experimental results.

The main utilities include:

- `make_figures.py`: Generates figures from experimental results.
- `recovery_report.py`: Produces recovery-related experimental reports.
- `testgraph.py`: Used for graph-related experiments and testing.

Refer to the individual scripts and experiment configurations for the corresponding execution arguments.

## Main Findings

Our experiments show that:

- Bures-Wasserstein interpolation provides substantially better zero-shot fusion performance than Euclidean interpolation.
- Hard permutation alignment produces no detectable improvement under the tested Bures interpolation settings.
- The graph-topology component of FGW does not produce a measurable improvement in the evaluated configurations.
- Soft transport alignment behaves differently from hard permutation alignment, particularly under width heterogeneity.
- Naive parameter averaging performs poorly immediately after fusion but can recover and surpass alignment-preserving methods with sufficient fine-tuning.

These findings suggest that distributional matching of weight statistics plays a larger role than explicit hard permutation alignment for the barycentric fusion operators studied here.

## Authors

**Biswadeep Chakraborty**  
Department of Electrical Engineering and Computer Science  
Indian Institute of Science Education and Research (IISER) Bhopal, India

**Tsung-Wei Chiang**  
National Chung Cheng University, Taiwan

Acknowledgements

Details will be added in a future update.

## Notes

Datasets, model checkpoints, and large experimental artifacts are not included in the source repository. Please refer to the manuscript for experimental details and evaluation protocols.
