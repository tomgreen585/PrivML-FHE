# Machine Learning Background Research

## What libraries have I looked into

### PySEAL

Is a fork of Microsoft Research's homomorphic encryption implementation, the Simple Encrypted Arithmetic Library (SEAL). This code wraps the SEAL build in a docker container and provides Python API's to the encryption library. Is more a pre-configured production library than exploratory.

### TenSEAL

Is a library for homomorphic encryption, enabling computations on encrypted data without decrypting it. While it doesn't directly provide a "image-to-image" transformation in the sense of style transfer or image manipulation tools, it can be used to perform operations on encrypted images, such as similarity calculations or certain parts of machine learning models like convolutions.

### ConcreteML

Concrete ML is an open-source, privacy-preserving, machine learning framework based on Fully Homomorphic Encryption (FHE). It simplifies the use of fully homomorphic encryption (FHE) for data scientists so that they can automatically turn machine learning models into their homomorphic equivalents, and use them without knowledge of cryptography. Concrete ML is designed with ease of use in mind. Data scientists can use models with APIs that are close to the frameworks they already know well, while additional options to those models allow them to run inference or training on encrypted data with FHE. The Concrete ML model classes are similar to those in scikit-learn and it is also possible to convert PyTorch models to FHE.

## Approach for investigating libraries

### concrete_ml_research

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

### pyseal_research

- Understand the features and limitations of the `PySEAL` library.
- Evaluate performance metrics including:
  - Execution time (clear vs FHE)
  - Accuracy / prediction similarity
  - Encryption overhead
  - Model compile time
  - Input size and scaling

### tenseal_research

- Understand the features and limitations of the `TenSEAL` library.
- Evaluate performance metrics including:
  - Execution time (clear vs FHE)
  - Accuracy / prediction similarity
  - Encryption overhead
  - Model compile time
  - Input size and scaling
