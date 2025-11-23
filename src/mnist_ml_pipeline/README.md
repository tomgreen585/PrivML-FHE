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

Stores shared parameters (e.g. learning rate, FHE parameters, dataset path) so you don't hardcode them everywhere.

### data_loader.py

Loads and prepares the MNIST dataset for training and evaluation.Normalizes images and reshapes to (H, W, C). Supports configurable dataset truncation for faster prototyping. Returns arrays for direct conversion to PyTorch tensors or FHE encoding. Expected image output shape: (N, 96, 96, 3)

### evalaution.py

Performs basic regression evaluation such as MSE, MAE. Generates plots to visually evaluate model. Append metrics to a continuously updated .csv -> continuously track performance. Generates a new .pdf for each model run to visualize plots -> continuously track performance

### ml_main.py

Runs the main pipeline. Performs data loading, preprocessing, training, testing, evaluation and saving. Runs in two different modes: -t ("Testing" - train, test, but don't save model + visualise plots) and -f ("Final_Model" - train, test, saves model + doesnt visualize plots).

### model.py

Defines the machine learning model architecture (e.g. logistic regression, decision tree, neural net). Keeps it modular so you can plug the model into both training.py and evaluation.py. Keep standard TenSEAL structure for ease of integration of FHE model.

### preprocessing.py

Applies image data augmentation techniques. Performs dataset split to generate training, validation and testing datasets. Preprocesseng stacked to shape (N, 96, 96, 3) -> (N, Height, width, channel). Converted to tensor version -> still (N, Height, width, channel).

### save.py

Saves the model created when Final_Model is specified during runtime. Uses timestamped filename in the /models directory. Uses torch.save() to save state_dict.

### testing.py

For running predictions on new encrypted inputs after training. Useful for demonstraing practical use of the encrypted model in deployment. Sends performance metrics to evaluation.py to visualize/track. Permute: NHWC → NCHW,Unsqueeze: (B, H, W) -> (B, 1, H, W)

### training.py

Handles training of the ML model using plaintext data. Defines the full training loop with optimizer, loss computation, logging, and metric evalaution.

## mnist.py

Loads a model from the specified path. Converts a given image to a tensor. Performs inference and outputs the predicted label to a JSON file.

## How To Run

### Testing

- If wanting to test with configurations and do not want to output a model then run it in "Testing" mode.
- Move to ml_pipeline directory: `cd engr489_project/src/mnist_ml_pipeline`
- Run in testing mode: `python3 mn_ml_main.py -t`

### Final

- If you are wanting to output a model then run it in "Final_Mode" and want to output/save the model.
- Move to ml_pipeline directory: `cd engr489_project/src/mnist_ml_pipeline`
- Run in final mode: `python3 mn_ml_main.py -f`
