# Concrete Machine Learning Library Research

This repository explores the capabilities of the [`concrete-ml`](https://github.com/zama-ai/concrete-ml) library — a privacy-preserving machine learning framework using Fully Homomorphic Encryption (FHE). The goal is to evaluate performance and feasibility of encrypted inference using traditional ML models.

## Objectives

- Understand the features and limitations of the `concrete.ml` library.
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
cd .../src/ml_background_research/concrete-ml

# Create a virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install scikit-learn tabulate concrete-ml matplotlib
```

### 3. Directory structure

- .../.artifacts
- decision_tree_classifier.ipynb
- decision_tree_regression.ipynb
- evaluation_baseline_classification.ipynb
- evaluation_baseline_regression.ipynb
- k_neighbors_classifier.ipynb (NOT COMPLETE)
- linear_regression.ipynb
- logistic_regression.ipynb
- neural_network_classifier.ipynb
- neural_network_regression.ipynb
- random_forest_classifier.ipynb
- random_forest_regression.ipynb
- README.md

### 4. TODO

- [x] Set baseline layout for .ipynb with FHE implementation and evaluation metrics
- [x] Implement and test Linear Regression compatibility
- [x] Implement and test Logistic Regression compatibility
- [x] Implement and test Random Forest Classifier and Regression compatibility
- [x] Implement and test K-Neighbors Classifier feasibility
- [x] Implement and test Decision Tree Classifier and Regression compatibility
- [x] Implement and test Random Forest Classifier and Regression compatibility
- [x] Compare performance across models and sample sizes
- [x] Document FHE-related and ML constraints

### 5. Issues with Models

- Decision Tree Classifier only ran up to 100,000 samples in the dataset before struggling. Accuracy also decreased significantly at the 10,000 sample range.
- Decision Tree Regressor only ran up to the 10000 sample range in the dataset before struggling. Accuracy was solid however, but clearly doesnt handle well to a higher sample size.
- K Neighbors Classifier -> oncrete.ml model is underdeveloped and does not support FHE compilation for KNN due to lack of learnable parameters. Due to being non-parametric, ONNX export lacks parameters, compiler requires something to quantize and compile.
- Linear Regression achieved the entire range of sample sizes. Accuracy remained consistent at 100% also.
- Logistic Regression achieved the entire range of sample sizes. Accuracy remained consistent at 100% across, however there was a drop in accuracy halfway through at the 10,000 range but however picked back up to 100% for the remaining higher sized sample datasets.
- Neural Network Classifier only ran up to 100,000 samples in the dataset before struggling. Accuracy also decreased significantly at the 10,000 sample range.
- Neural Network Regression only ran up to the 1,000,000 sample range, could have gone over if given more time to run. Accuracy remained consistent throughout (100%).
- Random Forest Classifier only ran up to the 100,000 samples in the dataset before struggling. Accuracy also decreased significantly at the 10,000 sample range.
- Random Forest Regression only ran up to the 10,000 samples in the dataset before struggling. Accuracy at the starting point was the lowest compared to the other models, and had a significant drop at the next sample size range also.
