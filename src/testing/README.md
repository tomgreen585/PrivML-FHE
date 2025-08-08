# ML and Encryption Evaluation/Testing Scripts

## 1. How to run

Compile File:
Run File:
Run All Tests:

## 2. What each test evaluates

### Security Metrics

- Encryption key size -> size (in bits) of public and private keys (e.g. 128, 256)
- Noise Budget Management -> remaining noise after operations, important to ensure correct decryption
- Threshold Parameters (if multiple people decrypting) -> minimum number of participants needed to decrypt (e.g. (t, n) - threshold)
- Quantum resistance -> lattice problems / implementation resistant

### Performance Metrics

- Execution time -> time taken for encryption, inference, and decryption (total runtime)
- Latency -> time per prediction or per operation (for ML inference)
- Throughput -> number of encrypted operations per second
- Memory usage -> RAM needed for encrypted operations (during training or inference)
- Bootstrapping time -> time taken per bootstrapping oepration schemes

### Cryptographic Scheme Properties (FHE) Metrics

- Bootstrapping -> does the program avoid bootstrapping or how often is bootstrapping being performed?
- Approximate vs Exact Computations -> how much error is introduced due to approximation (CKKS or Concrete ML)
- Parameter tuning -> how aggressive is optimization on modulus sizes, scaling factors, etc.

### Machine Learning Metrics

- Model Accuracy (Plaintext vs Encrypted) -> drop in accuracy between plaintext and encrypted models (%)
- Model Depth -> number of layers supported under encryption (e.g. NN-10, NN-50)
- Activation Functions Support -> how many/which activations can be securely approximated (e.g. ReLU)
- Quantization Bits -> bit width used for encrypted computations (e.g. 6-bit, 8-bit)
- Error Tolerance -> acceptable probability of small error during encrypted inference

### Specifically for Classification Models (BASELINE)

#### Evaluate accuracy performed on the cleartext during prediction

- Accuracy measures the overall correctness which is useful for balanced datasets
- Precision tells us how many predicted postiives were actually correct and is important when false postivies are costly
- Recall caputres how many actual positives were correctly identified and are critical when false negatives matter
- F1 Score balances precision and recall and is espeically valuable for imbalanced classes
- AUC (ROC) evaluates how well the model separates classes across thresholds and is a robust measure of classification performance
- Elapsed Time (ML Prediction Time) reflects how fast the model can predict in cleartext. Useful for comparing with FHE execution time to assess performance trade-offs

#### Confusion Matrix

- Provides a visual breakdown of prediction results. Shows counts of true positives, false postives, true negatives, and false negatives and helps understand what kind of errors the model is making.
- Provides class-specific performance insight. Reveals which class is being misclassified and how often which would support better debugging or retraining decisions.

#### Classification Report

- Provides precision, recall, F1-score, and support for each class. Offers comprehensive look at how the model performs on each label.
- Helps assess if the model is biased toward one class by comparing metric values across classes.

#### Evalaution of Model

- Shows predictions made by the model running on the plaintext (clear)
- Shows predictions made by the model running on the encrypted text
- Similarity shows how often the FHE predictions match the plaintext one
- Shows total runtime of the process of training/test/FHE

### Specifically for Regression Models (BASELINE)

#### Evalaute accuracy performed on the cleartext during prediction

- Mean Squared Error (MSE): Measure saverage squared difference between predicted and actual values. Penalizes large errors more than small ones due to squaring. Lower is better, zero means perfect prediction.
- Mean Absolute Error (MAE): Measures average absolute difference between predicted and actual values. Less senstiive to outliers than MSE. Gives a more intuitive error magnitude in the same units as your target.
- Mean Absolute Percentage Error (MAPE): Measures average error as a percentage of actual values. Helps understand relative error. Useful when you care about percentage deviation, but can be problematic if actual values are close to zero
- R-Squared Score (R2): Indicates how much variance in the target variable is explained by the mode. Ranges from 0 to 1 -> 1 is perfect git, 0 is predicts mean only, and less than 0 is worse than mean model
- Explained Variance Score (EVS): Measures the proportion of the variance explained by the predictions. Similar to R2, but not penalized for model bias. Score of 1 means all variance is captured, 0 means none.

#### Prediction v Actual Plot

- Shows how well the predicted values align with the actual test values
- Points close to the red dashed line indicating high prediction accuracy
- Consistant pattern above or below the line may reveal bias in the model
- Wide spread of points suggests poor model fit or high error
- Tighter cluster around the line indicates strong model performance

#### Residual Plot

- Visualizes the errors (residuals) between predicted and actual values
- Points randomly scattered around the red line (y=0) indicating a good fit
- Clear pattern or curve in the residuals suggesting the model is missing some structure in the data
- Large spread in residuals implies high variance or inconsistent predictions
- Clustering or funnel shapes can signal uneven variance

#### Residual Histogram

- Shows the distribution of prediction errors
- Bell-shaped distribution indicates that the model's errors are random and unbiased
- Skewed or lopsided histograms suggest systematic error or bias in predictions
- Multiple peaks may indicate missing variables or mixed patterns in the data
- Long tails imply the presence of outliers or extreme prediction errors
