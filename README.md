#

![Title](./docs/Images/title.png)

## Overview

With growing concerns around data privacy and collecting sensitive information, the ability to process data securely without exposing its underlying form has become a key engineering challenge. As machine learning (ML) models become increasingly integrated into business sectors, their reliance on raw data raises significant privacy and security risks. This project proposes developing a privacy-preserving ML system using Fully Homomorphic Encryption (FHE), a cryptographic method that enables computations to be performed directly on encrypted data. The aim is to investigate whether an end-to-end solution, where a selected ML model is trained and used for inference without decrypting input data, is feasible and practical under current constraints. The project will be evaluated by comparing the performance and accuracy of the encrypted model with an equivalent unencrypted version. Exploration into real-world use of FHE will be performed once requirements have been met, with a demonstration with a simple UI to show how this technology could be applied in practice.

## Project Motivation

Modern ML systems pose a privacy concern. Training and inference require access to raw data, exposing personally identifiable information (PII), health records, intellectual property, and more. Current practices lack transparency and consent, often scraping datasets without proper authorisation. Users of ML models often input private and confidential information, for example users in a business setting putting confidential business information into externally controlled models. Adversaries can exploit exposed data, leading to privacy breaches, model inversion attacks, and regulatory violation. FHE offers a promising solution by enabling computation directly on encrypted data, ensuring that user inputs are never exposed to third parties, service providers, or even the model itself.

## How to run

For information on how to run please refer to [instructions](readtheinstructions.md) file.

## Tools Implemented

Each tool has a non-encrypted and encrypted pipeline. Both of these are on the web application for comparison.

![Tools](./docs/Images/tools_implemented.png)

### Number Detector

The Number Detector allows users to hand-draw a digit (0-9) on a canvas and receive a prediction of the digit using a trained neural network. This tool demonstrates encrypted inference using FHE, where the digit is encrypted before being sent to the model, ensuring user privacy throughout the classification process.

### Face Detection

The Face Detection tool accepts an uploaded image and returns predicted bounding boxes around any detected human faces. The encrypted version of this pipeline performs face detection directly on encrypted image data using a CNN model adapted for FHE, showcasing priavacy preserving computer vision.

### Border Detection

Border Detection enhances an uploaded image by adding a stylised border based on encrypted model predictions. This task uses image-to-image encrypted inference and serves as an example of how FHE can be applied to secure image augmentation without exposing the original content to the server or service provider.

## Evaluation

The evaluation of this project will focus on both the technical performance and practical viability of the implemented system. Since the goal is to explore the feasibility of a privacy-preserving ML pipeline using encrypted data, the evaluation will assess how effectively the system preserves privacy while delivering meaningful and accurate model outputs. The evaluation will be structured around three key areas:

### 1. Correctness and Functionality

Ensure model predictions on encrypted inputs match those of plaintext models. Validate encrypted routines and round-trip correctness.

### 2. Performance and Trade-Off Analysis

Benchmark runtime (encryption, inference, decryption), memory usage, accuracy degradation. Experiment with FHE parameters (e.g. poly_modulus_degree, key sizes) to identify optimal trade-offs.

### 3. Usability and Real-world Demonstration

Evaluate the user interface and overall user experience. Consider deployment constraints for business or non-technical users. Present findings as a proof-of-concept for privacy-preserving ML as a service.

### 4. Explore different tools and libraries

Explore FHE libraries such as TenSEAl, PySEAL, Concrete-ML. Explore ML libraries such as scikit-learn, PyTorch, TensorFlow. Explore UI/Web libraries such as Node.js, Express, HTML/CSS/JS. Explore different datasets to see if it improves model inference.

The evaluation process will be iterative, with findings from each phase informing potential changes in direction. For instance, if performance is insufficient, the project may pivot to a simpler ML model, or adjust encryption parameters to improve speed while balancing privacy. This flexible and exploratory approach ensures the final outcome is not only technically sound but also provides a realistic assessment of the viability of FHE-based ML in practical settings.

## Future Work

Support encrypted training (not just inference) using leveled FHE or bootstrapped schemes. Add support for larger and more complex datasets, with tiling and batching. Explore federated learning + FHE for distributed privacy-preserving training. Consider integration with zero-knowledge proofs or differential privacy.
