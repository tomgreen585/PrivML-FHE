import torch.nn as nn
import tenseal as ts
import torch.optim as optim

# config.py
# - Stores shared parameters (e.g. learning rate, FHE parameters, dataset path) so you don't hardcode them everywhere.

############################################## TRAINING.PY PARAMETERS ########################################
PLAIN_OPTIMIZER = optim.Adam

PLAIN_EPOCHS = 50

PLAIN_LEARNING_RATE = 0.001

PLAIN_BATCH_SIZE = 32

PLAIN_LOSS_FUNCTION = nn.CrossEntropyLoss()

############################################# PLAIN_TESTING.PY PARAMETERS ###################################

SAMPLE_OUTPUT_COUNT = 6

############################################# DATA_LOADER.PY PARAMETERS ###################################
#Size of dataset in testing mode (none specified for final -> complete dataset (num.7000))
DATASET_SIZE = 2000

DATASET_PATH = "data/mnist/"

TRAINING_IMAGE_FILE = "train-images.idx3-ubyte"

TRAINING_LABEL_FILE = "train-labels.idx1-ubyte"

TESTING_IMAGE_FILE = "t10k-images.idx3-ubyte"

TESTING_LABEL_FILE = "t10k-labels.idx1-ubyte"

############################################# PLAIN_MODEL.PY PARAMETERS ###################################

HIDDEN = 64

OUTPUT = 10

CONV_KERNEL_SIZE = 3

CONV_STRIDE_SIZE = 1

CONV_PADDING_SIZE = 1

DROPOUT = 0.25

POOL_KERNEL_SIZE = 2

POOL_STRIDE_SIZE = 2

############################################# PREPROCESSING.PY PARAMETERS ###################################

SEED = 500

AUGMENT_DATA = True

TRAINING_SIZE = 0.75

VALIDATION_SIZE = 0.15

TESTING_SIZE = 0.1

############################################# WEB APPLICATION PARAMETERS ###################################

CURRENT_MODEL = "20250826_223019.pth"