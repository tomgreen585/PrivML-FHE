# ML Pipeline

## Dataset

From Kaggle, MNIST is a subset of a larger set available from NIST. The MNIST database of handwritten digits was a training set of 60,000 examples, and a test set of 10,000 examples.

Dataset can be found here: <https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data>

## Model Ideas

Build a CNN that performs some image augmentation from a supplied image from the user. Want the user to draw number in web app and then:

1. A classification/regression model runs (over encrypted data via FHE), and
1. Based on the output, you apply classification/labeling.

### Hyprid Approach

- Run FHE-secure inference to protect the user's private image data.
- Use the result to control plaintext image augmentation.

### Current Approach

#### Build a CNN that

- Takes an input image (number drawn by user -> converted)
- Outputs what number that was drawn
- Using classification labels.

#### How This Works

- Input: Image (consisting of number)
- Output: Label of the number
- Loss: CrossEntropyLoss()

## Project Architecture

### ML Pipeline Side

#### Train model in PyTorch (plaintext)

- Use MNIST Dataset
- Preprocess (resize, normalize, grayscale)
- Model: CNN or MLP with exportable linear layers

#### Inference pipeline with TenSEAL

- User draws number - preprocessed -> encrypted (CKKS vector)
- Perform encrypted inference using saved weights
- Perform prediction and label number

## Directory Outline

### config.py

- Stores shared parameters (e.g. learning rate, FHE parameters, dataset path) so you don't hardcode them everywhere.

### data_loader.py

- Loads mnist datasets (plaintext format).
- Performs train/test split and returns data in usable format (e.g. NumPy arrays, tensors, encrytped vectors).
- Metrics display numpy arrays of shape (96, 96, 3) -> (Height, width, channel)

### evalaution.py

- Performs basic regression evaluation such as MSE, MAE
- Generates plots to visually evaluate model
- Append metrics to a continuously updated .csv -> continuously track performance
- Generates a new .pdf for each model run to visualize plots -> continuously track performance

### ml_main.py

- Simple orchestrator script that ties everything together
- Loads dataset
- Trains the model
- Evaluates it
- Runs encryption tests

### model.py

- Defines the machine learning model architecture (e.g. logistic regression, decision tree, neural net).
- Keeps it modular so you can plug the model into both training.py and evaluation.py
- Keep standard TenSEAL structure for ease of integration of FHE model

### preprocessing.py

- Generates target test data from original dataset (consisting of original images with red border set in order top, bottom, left, right)
- Applies image data augmentation techniques
- Performs dataset split to generate training, validation and testing datasets
- preprocesseng stacked to shape (N, 96, 96, 3) -> (N, Height, width, channel)
- Converted to tensor version -> still (N, Height, width, channel)

### save.py

- Saves the model created when Final_Model is specified during runtime
- Outputs the saved model.pth to directory

### testing.py

- For running predictions on new encrypted inputs after training
- Useful for demonstraing practical use of the encrypted model in deployment
- Sends performance metrics to evaluation.py to visualize/track

### training.py

- Handles model training on plaintext data (or encrypted data if supported by your FHE framework).
- Defines training loop, optimizer, loss functions, metrics, etc.
- Logs progress and saves trained models (serialized weights or encrypted parameters).
- Applies permute (0, 3, 1, 2) -> becomes (N, 3, 96, 96)
- Sends performance metrics to evaluation.py to visualize/track

## How To Run

### Testing

- If wanting to test with configurations and do not want to output a model then run it in "Testing" mode.
- Move to ml_pipeline directory: `cd engr489_project/src/mnist_ml_pipeline`
- Run in testing mode: `python3 mn_ml_main.py -t`

### Final

- If you are wanting to output a model then run it in "Final_Mode" and want to output/save the model.
- Move to ml_pipeline directory: `cd engr489_project/src/mnist_ml_pipeline`
- Run in final mode: `python3 mn_ml_main.py -f`
