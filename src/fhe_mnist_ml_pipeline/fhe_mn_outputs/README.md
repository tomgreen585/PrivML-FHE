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

PDF outputs were too large due to model tuning performed. Have created a collated document that can be found here: <https://drive.google.com/file/d/1uP3i0XHT2fxFSdEQeBJlcs5ooGAXjfun/view?usp=sharing>

### Evaluation Performed

Performed parameter tuning over these hyperparameters used during training/validation/testing and encrypting workflows in ML+FHE pipelines

NOTE: Other evaluation parameters used in other models did not correlate to this model due to it being a classification model where the other two were regression.

#### Optimizer (baseline = Adam)

- Adadelta (model 1)
- Adagrad (model 2)
- Adamax (model 3) **this**
- RAdam (model 4) **this**
- Rprop (model 5)
- Adam (model 10) **this**
- SGD (model 6)
- RMSprop (model 7)

#### Epochs (baseline = 20)

- 30 (model 8)
- 50 (model 9)
- 75 (model 10) **this**
- 100 (model 11)
- 150 (model 12)
- 200 (model 13)
- 300 (model 14)
- 400 (model 15) **this**
- 500 (model 16)

#### Learning Rate (baseline = 0.001)

- 0.0001 (model 17) **this**
- 0.0005 (model 18) **this**
- 0.001 (model 19)
- 0.005 (model 20)
- 0.01 (model 21)
- 0.05 (model 22)

#### Batch Size (baseline = 64)

- 16 (model 23) **this**
- 32 (model 24)
- 64 (model 25) **this**
- 128 (model 26)

#### Loss Function (baseline = CrossEntropyLoss)

- CrossEntropyLoss (model 27)

#### Dataset Size (baseline = 6000)

- 1600 (model 28)
- 10000 (model 29) **this**

#### Encrypted Loss Function (baseline = CrossEntropyLoss)

- CrossEntropyLoss (model 30)

#### Plaintext Model Hidden Dimensions (baseline = 64)

- 64 (model 31)
- 128 (model 32) **this**
- 256 (model 33) **this**
- 512 (model 34)
- 1024 (model 35)

#### Dropout (baseline = 0.25)

- 0.1 (model 36)
- 0.3 (model 37) **this**
- 0.5 (model 38)
- 0.6 (model 39)

#### Seed (baseline = 500)

- 42 (model 40)
- 123 (model 41) **this**
- 500 (model 42) **this**
- 999 (model 43)
- 1337 (model 44)
