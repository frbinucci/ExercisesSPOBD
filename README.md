# Labs: Signal Processing & Optimization for Big Data

Hands-on sessions in **MATLAB / Python** covering optimization, distributed optimization, big-data reduction, and feature extraction for communications and machine learning applications.

---

## Table of Contents
- [Lesson 1 — 15/10/2025 (3h)](#lesson-1--15102025-3h)
- [Lesson 2 — 22/10/2025 (3h)](#lesson-2--22102025-3h)
- [Lesson 3 — 31/10/2025 (3h)](#lesson-3--31102025-3h)
- [Lesson 4 — 12/11/2025 (3h)](#lesson-4--12112025-3h)

---

## Lesson 1

**Topic:** Coding and simulation exercises in MATLAB / Python for Optimization algorithms.

**Agenda:**
- Implement **LASSO** in MATLAB / Python:
  - **ISTA** (Iterative Soft-Thresholding Algorithm)
  - **ADMM** (Alternating Direction Method of Multipliers)
- Analyses:
  - Convergence
  - Sensitivity to the regularization parameter
  - Sparsity recovery
- Use of **synthetic** and **real** datasets
- **Water-Filling problem**:
  - Solution via **CVX**
  - Use of the **built-in MATLAB** water-filling function (when available)

---

## Lesson 2

**Topic:** Coding and simulation exercises in MATLAB / Python for **Distributed Optimization**.

**Planned topics:**
- LASSO on a **real dataset** (`MPG-DATASET`) with data split across multiple agents/processors:
  - Scenario with a **fusion center**
  - **Consensus algorithm** over a **connected bidirectional network** (undirected graph)
  - **Total Variation** formulation
- Coding and simulation of **regularized Support Vector Machine** model fitting on the **Breast Cancer** dataset

---

## Lesson 3

**Topic:** Coding and simulation exercises in MATLAB / Python for **Big Data Reduction**.

**Planned topics:**
- **Compressed sensing** for reconstruction of **undersampled images**:
  - **2D-DCT** dictionary
  - **Basis Pursuit** implemented with:
    - **ADMM**
    - **Iterative Shrinkage-Thresholding (IST/ISTA)**
- **Low-Rank + Sparse / Matrix Completion**:
  - Applications to **Internet traffic flow data**
  - **Denoising** of images corrupted by **salt-and-pepper** additive noise

---

## Lesson 4

**Topic:** Coding and simulation exercises in MATLAB / Python for **Feature Extraction** and **Goal-Oriented Communications**.

**Planned topics:**
- **Feature extraction** from image datasets for **classification**:
  - **PCA** (unsupervised)
  - **Information Bottleneck**
  - **Neural-network encoder** (autoencoder)
- Introductory notes on applications to **goal-oriented communications**

---

## Requirements

- Python 3.10+ (recommended)
- PyTorch (version: `xxx`)
- MATLAB R2024b (recommended)
- (Optional) `pip` / `venv`

## Setup

Clone the repository:

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
```

For Exercises 1–3, make sure to add the subfolders containing the MATLAB functions to the MATLAB path.

## Exercise 4

Exercise 4 is organized into three subfolders:

- **`GIB_REGRESSION/`**: MATLAB project to test Gaussian IB in a multivariate normal regression task
- **`NN_CLASSIFICATION/`**: Python project to test the Opportunistic Information Bottleneck in an image classification task (shallow neural networks on MNIST)
- **`AE_COMPARISONS/`**: Python project to test the Opportunistic Information Bottleneck in an image classification task (CNNs on GTSRB)

Both `NN_CLASSIFICATION` and `AE_COMPARISONS` require **PyTorch**. These folders include pre-trained inference models for the corresponding datasets.

### NN_CLASSIFICATION

> [!TIP]
> **NN_CLASSIFICATION — PCA/IB inference**
>
> ```bash
> python main.py --mode <ib|pca> --path_to_load <pca_ese|ib_ese>/network.pth
> ```
