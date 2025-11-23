import torch.nn as nn
import tenseal as ts
import torch.optim as optim

# config.py
# - Stores shared parameters (e.g. learning rate, FHE parameters, dataset path) so you don't hardcode them everywhere.

############################################## TRAINING.PY PARAMETERS ########################################
PLAIN_OPTIMIZER = optim.Adam

PLAIN_EPOCHS = 50

PLAIN_LEARNING_RATE = 0.001

PLAIN_BATCH_SIZE = 16

PLAIN_LOSS_FUNCTION = nn.MSELoss()

############################################# PLAIN_TESTING.PY PARAMETERS ###################################

SAMPLE_OUTPUT_COUNT = 6 # also used in encrypted testing

############################################# DATA_LOADER.PY PARAMETERS ###################################
#Size of dataset in testing mode (none specified for final -> complete dataset (num.7000))
DATASET_SIZE = 1600

IMAGE_SIZE = 192

DATASET_PATH = "data/Humans"

############################################# PLAIN_MODEL.PY PARAMETERS ###################################

EN_KERNEL_SIZE = 3

EN_PADDING = 1

EN_ACT = nn.ReLU()

############################################# PREPROCESSING.PY PARAMETERS ###################################

SEED = 500

TRAINING_SIZE = 0.75

VALIDATION_SIZE = 0.15

TESTING_SIZE = 0.1

############################################# WEB APPLICATION PARAMETERS ###################################

CURRENT_MODEL = "20250826_211807.pth"