# ML Pipeline

## Dataset

Kaggle Human Faces (Web scraped dataset of human faces suggested for image processing). 7.2k images useful for multiple use cases, such image identifiers, classifier algorithms, etc. Thorough mix of all common creeds, races, age groups and profiles in an attempt to create a unbiased dataset with a few GAN generated images as well to aid the functionality of differentiating between real and generated faces.

Dataset can be found here: <https://www.kaggle.com/datasets/ashwingupta3012/human-faces?resource=download>

## Model Ideas

Build a CNN that performs some image augmentation from a supplied image from the user. Want the user to input an image of themselves and then:

1. A classification/regression model runs (over encrypted data via FHE), and
1. Based on the output, you apply image augmentation or transformation.

### Hyprid Approach

- Run FHE-secure inference to protect the user's private image data.
- Use the result to control plaintext image augmentation.

### Use Case Ideas

1. Border Color: ML model that adds a border or mask.
1. Smile Detection (Binary Classifier): Predict smile/no smile. Add some sort of overlay e.g. emojis.
1. Brightness Regression (Apply Filter): Predict average brightness of image. If too dark/light, auto-correct it using image filters.
1. Emotion Detection (Image Stylization): Predict emotion (happy, angry, neutral, etc.). Add a visual effect (e.g. greyscale)

### Current Approach

#### Build a CNN that

- Takes an input image (face)
- Outputs an augmented version of that image with a border or mask.
- You are not using classification labels.
- The task is treated as image-to-image regression
- This is an Image-to-Image Regression task, where we are training a CNN to directly map input -> styled output, learning to draw the border/mask itself.

#### How This Works

- Input: Image
- Output: Image of the same shape, but with a learned border/mask effect

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

#### Augment original image (plaintext)

- Add border or visual marker based on face shape prediction

## Directory Outline

### config.py

Stores shared parameters (e.g. learning rate, FHE parameters, dataset path) so you don't hardcode them everywhere.

### data_loader.py

Provides functionality to load grayscale images from the dataset directory, preprocess them, and generate input/output pairs for training the model. Each image is loaded, resized and normalized to simulate ground truth labels for a supervised learning task.

### enc_testing.py

For running predictions on new encrypted inputs after training. Useful for demonstraing practical use of the encrypted model in deployment. Sends performance metrics to evaluation.py to visualize/track.

### evalaution.py

Performs basic regression evaluation such as MSE, MAE. Generates plots to visually evaluate model. Append metrics to a continuously updated .csv -> continuously track performance. Generates a new .pdf for each model run to visualize plots -> continuously track performance

### fhe_model.py

Full FHE model implementation. Uses previous plaintext model as weight and bias inputs. Currently using CKKS Scheme. Does computations over two convolutions performing image to image.

### ml_main.py

Runs the main pipeline. Performs data loading, preprocessing, training, testing, evaluation and saving. Runs in two different modes: -t ("Testing" - train, test, but don't save model + visualise plots) and -f ("Final_Model" - train, test, saves model + doesnt visualize plots).

### model.py

Defines the machine learning model architecture (e.g. logistic regression, decision tree, neural net). Keeps it modular so you can plug the model into both training.py and evaluation.py. Keep standard TenSEAL structure for ease of integration of FHE model.

### preprocessing.py

Performs data processing. Splits dataset into training, validation, test and encryption sets using fixed sizes. Converts image data into torch tensors and visualizes input-output image pairs.

### save.py

Saves the model created when Final_Model is specified during runtime. Uses timestamped filename in the /models directory. Uses torch.save() to save state_dict.

### testing.py

For running predictions on new encrypted inputs after training. Useful for demonstraing practical use of the encrypted model in deployment. Sends performance metrics to evaluation.py to visualize/track.

### training.py

Handles training of the ML model using plaintext data. Defines the full training loop with optimizer, loss computation, logging, and metric evalaution.

### fhe_border.py

This script performs inference using a pre-trained border detection model with fhe wrapped model to add visual borders in input images.
It loads a PyTorch model, wraps with FHE model, processes an input image, runs the model, and saves the output.  

## How To Run

### Testing

- If wanting to test with configurations and do not want to output a model then run it in "Testing" mode.
- Move to ml_pipeline directory: `cd engr489_project/src/fhe_border_ml_pipeline`
- Run in testing mode: `python3 fhe_br_ml_main.py -t`

### Final

- If you are wanting to output a model then run it in "Final_Mode" and want to output/save the model.
- Move to ml_pipeline directory: `cd engr489_project/src/fhe_border_ml_pipeline`
- Run in final mode: `python3 fhe_br_ml_main.py -f`
