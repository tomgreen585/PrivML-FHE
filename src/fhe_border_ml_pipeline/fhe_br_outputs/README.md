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

PDF outputs were too large due to model tuning performed. Have created a collated document that can be found here: <https://drive.google.com/file/d/1dmYztyQF9z_9HcxfzXZoriu2o-SqEjbD/view?usp=sharing>

### Evaluation Performed

Performed parameter tuning over these hyperparameters used during training/validation/testing and encrypting workflows in ML+FHE pipelines

#### Optimizer (baseline = Adam)

- Adadelta (model 1)
- Adagrad (model 2)
- AdamW (model 3)
- Adamax (model 4) **this**
- RAdam (model 5)
- Rprop (model 6)
- Adam (model 7) **this**
- SGD (model 8)
- RMSprop (model 9)

#### Epochs (baseline = 50)

- 30 (model 10)
- 50 (model 11)
- 75 (model 12)
- 100 (model 13)
- 150 (model 14)
- 200 (model 15)
- 300 (model 16)
- 400 (model 17)
- 500 (model 18)

#### Learning Rate (baseline = 0.001)

- 0.0001 (model 19)
- 0.0005 (model 20)
- 0.001 (model 21) **this**
- 0.005 (model 22)
- 0.01 (model 23)
- 0.05 (model 24)

#### Batch Size (baseline = 8)

- 8 (model 25)
- 16 (model 26) **this**
- 32 (model 27)
- 64 (model 28)
- 128 (model 29)

#### Loss Function (baseline = MSELoss)

- MSELoss (model 30) **this**
- CrossEntropyLoss (model 31)
- L1Loss (model 32)
- KLDivLoss (model 33)

#### Dataset Size (baseline = 2500) <- **this**

- 500 (model 34)
- 1600 (model 35)
- 3000 (model 36)
- 5000 (model 37)
- 7000 (model 38)
- 10000 (model 39) **this**

#### Encrypted Loss Function (baseline = MSELoss)

- MSELoss (model 40) **this**
- CrossEntropyLoss (model 41)
- L1Loss (model 42)
- KLDivLoss (model 43)

#### Seed (baseline = 500)

- 42 (model 44)
- 123 (model 45)
- 500 (model 46) **this**
- 999 (model 47)
- 1337 (model 48)
