import torch.nn as nn
import tenseal as ts
import torch.optim as optim

# config.py
# - Stores shared parameters (e.g. learning rate, FHE parameters, dataset path) so you don't hardcode them everywhere.

############################################## TRAINING.PY PARAMETERS ########################################
PLAIN_OPTIMIZER = optim.Adam

PLAIN_EPOCHS = 100

PLAIN_LEARNING_RATE = 0.001

PLAIN_BATCH_SIZE = 16

PLAIN_LOSS_FUNCTION = nn.MSELoss()

############################################# PLAIN_TESTING.PY PARAMETERS ###################################

SAMPLE_OUTPUT_COUNT = 6 # also used in encrypted testing

############################################# DATA_LOADER.PY PARAMETERS ###################################
#Size of dataset in testing mode (none specified for final -> complete dataset (num.7000))
DATASET_SIZE = 7000

IMAGE_SIZE = 128

DATASET_PATH = "data/Humans"

############################################# ENCRYPTED_TESTING.PY PARAMETERS ###################################

SCHEME_TYPE = ts.SCHEME_TYPE.CKKS

ENCRYPTED_LOSS_FUNCTION = nn.MSELoss()

BITS_SCALE = 26

POLY_MODULUS_DEGREE = 8192 #upscale possibly to 16384

ST_KEY_GENERATION = 31

END_DECRYPTION_STABILITY = 31

COEFF_MOD_BIT_SIZES = [31, 26, 26, 26, 26, 26, 26, 31]

############################################# PLAIN_MODEL.PY PARAMETERS ###################################

PLAIN_HIDDEN = 512

PLAIN_OUTPUT = 4

PLAIN_KERNEL_SIZE = 1

PLAIN_STRIDE = 2

PLAIN_PADDING = 0

PLAIN_DROPOUT = 0.3

############################################# PREPROCESSING.PY PARAMETERS ###################################

SEED = 999

TRAINING_SIZE = 0.75

VALIDATION_SIZE = 0.15

TESTING_SIZE = 0.05

ENCRYPTED_SIZE = 0.05

############################################# WEB APPLICATION PARAMETERS ###################################

CURRENT_MODEL = "20250919_194946.pth"