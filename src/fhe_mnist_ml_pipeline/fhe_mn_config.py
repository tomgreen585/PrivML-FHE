import torch.nn as nn
import tenseal as ts
import torch.optim as optim

# config.py
# - Stores shared parameters (e.g. learning rate, FHE parameters, dataset path) so you don't hardcode them everywhere.

############################################## TRAINING.PY PARAMETERS ########################################
PLAIN_OPTIMIZER = optim.Adam

PLAIN_EPOCHS = 75

PLAIN_LEARNING_RATE = 0.0005

PLAIN_BATCH_SIZE = 16

PLAIN_LOSS_FUNCTION = nn.CrossEntropyLoss()

############################################# PLAIN_TESTING.PY PARAMETERS ###################################

SAMPLE_OUTPUT_COUNT = 6

############################################# DATA_LOADER.PY PARAMETERS ###################################
#Size of dataset in testing mode (none specified for final -> complete dataset (num.7000))
DATASET_SIZE = 10000

DATASET_PATH = "data/mnist/"

TRAINING_IMAGE_FILE = "train-images.idx3-ubyte"

TRAINING_LABEL_FILE = "train-labels.idx1-ubyte"

TESTING_IMAGE_FILE = "t10k-images.idx3-ubyte"

TESTING_LABEL_FILE = "t10k-labels.idx1-ubyte"

############################################# ENCRYPTED_TESTING.PY PARAMETERS ###################################

SCHEME_TYPE = ts.SCHEME_TYPE.CKKS

ENCRYPTED_LOSS_FUNCTION = nn.CrossEntropyLoss()

BITS_SCALE = 26

POLY_MODULUS_DEGREE = 8192 #upscale possibly to 16384

ST_KEY_GENERATION = 31

END_DECRYPTION_STABILITY = 31

COEFF_MOD_BIT_SIZES = [31, 26, 26, 26, 26, 26, 26, 31]

############################################# PLAIN_MODEL.PY PARAMETERS ###################################

PLAIN_HIDDEN = 128

PLAIN_OUTPUT = 10

PLAIN_KERNEL_SIZE = 7

PLAIN_STRIDE = 3

PLAIN_PADDING = 0

PLAIN_DROPOUT = 0.3

############################################# PREPROCESSING.PY PARAMETERS ###################################

SEED = 123

AUGMENT_DATA = True

TRAINING_SIZE = 0.75

VALIDATION_SIZE = 0.15

TESTING_SIZE = 0.09

ENCRYPTED_SIZE = 0.01

############################################# WEB APPLICATION PARAMETERS ###################################

CURRENT_MODEL = "20250919_202045.pth"