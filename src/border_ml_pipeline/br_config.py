import torch.nn as nn
import tenseal as ts
import torch.optim as optim

# config.py
# - Stores shared parameters (e.g. learning rate, FHE parameters, dataset path) so you don't hardcode them everywhere.

############################################## TRAINING.PY PARAMETERS ########################################
PLAIN_OPTIMIZER = optim.Adam

PLAIN_EPOCHS = 50

PLAIN_LEARNING_RATE = 0.001

PLAIN_BATCH_SIZE = 8

PLAIN_LOSS_FUNCTION = nn.MSELoss()

############################################# PLAIN_TESTING.PY PARAMETERS ###################################

SAMPLE_OUTPUT_COUNT = 6 # also used in encrypted testing

############################################# DATA_LOADER.PY PARAMETERS ###################################
#Size of dataset in testing mode (none specified for final -> complete dataset (num.7000))
DATASET_SIZE = 2500

IMAGE_SIZE = 192

DATASET_PATH = "data/Humans"

BORDER_COLOR = 0.0

BORDER_THICKNESS = 10

############################################# PLAIN_MODEL.PY PARAMETERS ###################################

EN_KERNEL_SIZE = 5

EN_STRIDE = 2

EN_PADDING = 2

DE_KERNEL_SIZE = 4

DE_STRIDE = 2

DE_PADDING = 1

############################################# PREPROCESSING.PY PARAMETERS ###################################

SEED = 500

TRAINING_SIZE = 0.75

VALIDATION_SIZE = 0.15

TESTING_SIZE = 0.1

############################################# WEB APPLICATION PARAMETERS ###################################

CURRENT_MODEL = "20250826_213156.pth"