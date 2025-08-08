# ML Pipeline

## Dataset

Kaggle Human Faces (Web scraped dataset of human faces suggested for image processing). 7.2k images useful for multiple use cases, such image identifiers, classifier algorithms, etc. Thorough mix of all common creeds, races, age groups and profiles in an attempt to create a unbiased dataset with a few GAN generated images as well to aid the functionality of differentiating between real and generated faces.

Dataset can be found here: <https://www.kaggle.com/datasets/ashwingupta3012/human-faces?resource=download>

## Model Ideas

Build a CNN that performs some form of face identification from a supplied image from the user. Want the user to input an image of themselves and then:

1. A classification/regression model runs (over encrypted data via FHE), and
1. Based on the output, apply face identification.

### Hyprid Approach

- Run FHE-secure inference to protect the user's private image data.
- Use the result to control plaintext image augmentation.

### Current Approach

#### Build a CNN that

- Takes an input image (face)
- Outputs same image, but with green box around the face(s) in the image.
- Not using classification labels.
- The task is treated as image-to-image regression
- This is an Image-to-Image Regression task, where we are training a CNN to directly map input -> styled output, learning to draw the border/mask around the user's face.

#### How This Works

- Input: Image
- Output: Image of the same shape, but with a learned frame around the faces in the image
- Loss: MSE between the output and a target stylized image

#### What needs to be provided

- Source Dataset: Raw Face Images
- Target Dataset: Same Face Images but with the border/mask added (can generate this)

## Project Architecture

### ML Pipeline Side

#### Train model in PyTorch (plaintext)

- Use Kaggle face shape dataset
- Preprocess (resize, normalize, grayscale)
- Model: CNN or MLP with exportable linear layers

#### Inference pipeline with TenSEAL

- User image - preprocessed -> encrypted (CKKS vector)
- Perform encrypted inference using saved weights
- Perform prediction and face shape label

#### Limitations

- If no face is present, the model is designed to return a bounding box, so it will return some sort of box on the 'no face' image which is garbage. The model isn't trained to predict if nothing is there. This is something that can further be looked into.
- If there are multiple faces in the image, the model will not generalize to this. It is only trained to deliver one bounding box. This is maybe a possible further implementation to look into.

## Directory Outline

### config.py

- Stores shared parameters (e.g. learning rate, FHE parameters, dataset path) so you don't hardcode them everywhere.

### data_loader.py

- Loads and precprocesses datasets (plaintext format).
- Applies scaling/normalization and optionally converts to encrypted format.
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
- Move to ml_pipeline directory: `cd engr489_project/src/face_ml_pipeline`
- Run in testing mode: `python3 fc_ml_main.py -t`

### Final

- If you are wanting to output a model then run it in "Final_Mode" and want to output/save the model.
- Move to ml_pipeline directory: `cd engr489_project/src/face_ml_pipeline`
- Run in final mode: `python3 fc_ml_main.py -f`
