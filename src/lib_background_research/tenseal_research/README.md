# TenSEAL Machine Learning Library Research

This repository explores the capabilities of the [`TenSEAL`](https://github.com/OpenMined/TenSEAL) library — a privacy-preserving machine learning framework using Fully Homomorphic Encryption (FHE). The goal is to evaluate performance and feasibility of encrypted inference using traditional ML models.

## Objectives

- Understand the features and limitations of the `TenSEAL` library.
- Implement and test core machine learning algorithms:
  - Logistic Regression
  - Linear Regression
  - Random Forest
  - K-Neighbors
  - Decision Tree
  - Neural Network
- Evaluate performance metrics including:
  - Execution time (clear vs FHE)
  - Accuracy / prediction similarity
  - Encryption overhead
  - Model compile time
  - Input size and scaling

## Getting Started

### 1. Prerequisites

- Python 3.8 to 3.11

### 2. Installation

```bash
# Access directory
cd .../src/ml_background_research/tenseal_research

# Create a virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install scikit-learn tabulate tenseal matplotlib
```

### 3. Directory structure

- .../data(MNIST)
- CNN_Model_Encrypted.ipynb
- CNN_Model_Plaintext.ipynb
- evaluation_baseline_classification.ipynb
- evaluation_baseline_regression.ipynb
- linear_regression.ipynb
- logistic_regression.ipynb
- README.md

### 4. TODO

- [x] Set baseline layout for .ipynb with FHE implementation and evaluation metrics
- [x] Implement and test Linear Regression compatibility
- [x] Implement and test Logistic Regression compatibility
- [x] Explore further ML models using TenSEAL
- [x] Compare performance across models and sample sizes
- [x] Document FHE-related and ML constraints

### 5. Issues with Models

- Library is mainly used for ML deployment instead of deep FHE deployment. Might have to figure out crossing over libraries to implement both FHE and ML
