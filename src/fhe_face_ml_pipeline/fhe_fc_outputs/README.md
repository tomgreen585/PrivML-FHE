# Outputs

## CSV File

This is a continuously updated csv file, that updates each time the ML pipeline is run. This helps with keeping track of model performance throughout evaluation.

### Metrics are ordered by

- Timestamp (e.g. when the model was run)
- Model (Model Number)
- Process (Which stage of the ml pipeline)
- Metric (Metric type that is being recorded)
- Score (Metric Score)

## PDF Files

When the ML pipeline is run, a assoicated pdf file is generated for that specific model. This helps with keeping track of model outputs and performance throughout evaluation.

### PDF files contain

- Model number + timestamp
- Model configurations
- Model plots
- Model outputs

PDF outputs were too large due to model tuning performed. Have created a collated document that can be found here: <https://drive.google.com/file/d/1E8m3mDAtrP_gqLz0hxrG6zknLbRJvZjy/view?usp=sharing>

### Evaluation Performed

Performed parameter tuning over these hyperparameters used during training/validation/testing and encrypting workflows in ML+FHE pipelines

- Initial model (model 1)

#### Optimizer (baseline = Adam)

- Adadelta (model 2)
- Adagrad (model 3)
- AdamW (model 4) **this**
- Adamax (model 5)
- RAdam (model 6)
- Rprop (model 7)
- Adam (model 8) **this**
- SGD (model 9)
- RMSprop (model 10)

#### Epochs (baseline = 50)

- 30 (model 11)
- 50 (model 12)
- 75 (model 13)
- 100 (model 14) **this**
- 150 (model 15) **this**
- 200 (model 16)
- 300 (model 17)
- 400 (model 18)
- 500 (model 19)

#### Learning Rate (baseline = 0.001)

- 0.0001 (model 20) **this**
- 0.0005 (model 21) **this**
- 0.001 (model 22)
- 0.005 (model 23)
- 0.01 (model 24)
- 0.05 (model 25)

#### Batch Size (baseline = 16)

- 8 (model 26)
- 16 (model 27) **this**
- 32 (model 28) **this**
- 64 (model 29)
- 128 (model 30)

#### Loss Function (baseline = 0.001)

- MSELoss (model 31) **this**
- CrossEntropyLoss (model 32)
- L1Loss (model 33)
- KLDivLoss (model 34)

#### Dataset Size (baseline = 1600)

- 500 (model 35)
- 1600 (model 36)
- 3000 (model 37)
- 5000 (model 38)
- 7000 (model 39) **this**
- 10000 (model 40) **this**

#### Encrypted Loss Function (baseline = MSELoss)

- MSELoss (model 41) **this**
- CrossEntropyLoss (model 42)
- L1Loss (model 43)
- KLDivLoss (model 44)

#### Plaintext Model Hidden Dimensions (baseline = 256)

- 64 (model 45)
- 128 (model 46)
- 256 (model 47) **this**
- 512 (model 48) **this**
- 1024 (model 49)

#### Dropout (baseline = 0.5)

- 0.1 (model 51)
- 0.3 (model 52) **this**
- 0.5 (model 53) **this**
- 0.6 (model 54)

#### Seed (baseline = 500)

- 42 (model 55)
- 123 (model 56)
- 500 (model 57) **this**
- 999 (model 58) **this**
- 1337 (model 59)
