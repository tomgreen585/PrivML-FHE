import torch.nn as nn
import tenseal as ts
import torch.optim as optim

# config.py
# - Stores shared parameters (e.g. learning rate, FHE parameters, dataset path) so you don't hardcode them everywhere.

############################################## TRAINING.PY PARAMETERS ########################################
PLAIN_OPTIMIZER = optim.Adam

PLAIN_EPOCHS = 150

PLAIN_LEARNING_RATE = 0.001

PLAIN_BATCH_SIZE = 16

PLAIN_LOSS_FUNCTION = nn.MSELoss()

############################################# PLAIN_TESTING.PY PARAMETERS ###################################

SAMPLE_OUTPUT_COUNT = 3 # also used in encrypted testing

############################################# DATA_LOADER.PY PARAMETERS ###################################
#Size of dataset in testing mode (none specified for final -> complete dataset (num.7000))
DATASET_SIZE = 3000

IMAGE_SIZE = 32

DATASET_PATH = "data/Humans"

BORDER_COLOR = 0.0

BORDER_THICKNESS = 10

############################################# ENCRYPTED_TESTING.PY PARAMETERS ###################################

SCHEME_TYPE = ts.SCHEME_TYPE.CKKS

ENCRYPTED_LOSS_FUNCTION = nn.MSELoss()

BITS_SCALE = 40

POLY_MODULUS_DEGREE = 32768

ST_KEY_GENERATION = 60

END_DECRYPTION_STABILITY = 60

COEFF_MOD_BIT_SIZES = [60, 40, 40, 40, 40, 60]

############################################# PLAIN_MODEL.PY PARAMETERS ###################################

PLAIN_CONV1_KERNEL_SIZE = 3

PLAIN_CONV2_KERNEL_SIZE = 1

PLAIN_CONV1_PADDING = 1

############################################# PREPROCESSING.PY PARAMETERS ###################################

SEED = 500

TRAINING_SIZE = 0.75

VALIDATION_SIZE = 0.15

TESTING_SIZE = 0.05

ENCRYPTED_SIZE = 0.05

############################################# WEB APPLICATION PARAMETERS ###################################

CURRENT_MODEL = "20250919_192546.pth"