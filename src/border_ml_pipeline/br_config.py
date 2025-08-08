import torch.nn as nn
import torch.optim as optim

# config.py
# - Stores shared parameters (e.g. learning rate, FHE parameters, dataset path) so you don't hardcode them everywhere.

########################################### ML_MAIN.PY PARAMETERS ####################################
#Dataset Path
DATASET_PATH = "data/Humans"

#Image Size (all images will be converted to this size)
IMAGE_SIZE = 192

#Training Set Ratio
TRAIN_RATIO = 0.75

#Validation Set Ratio
VAL_RATIO = 0.15

#Seed Size
SEED = 500

#Testing Output Sample Size
SAMPLE_OUTPUT_COUNT = 6

#Training Epochs
EPOCHS = 50

#Batch Size
BATCH_SIZE = 16

#Learning Rate
LEARNING_RATE = 0.001

############################################## TRAINING.PY PARAMETERS ########################################
# Model Optimizer
OPTIMIZER = optim.Adam

# Model Loss Function
LOSS_FUNCTION = nn.MSELoss()

############################################# DATA_LOADER.PY PARAMETERS ###################################
#Size of dataset in testing mode (none specified for final -> complete dataset (num.7000))
DATASET_SIZE = 1600

############################################# PREPROCESSING.PY PARAMETERS #################################
#Image border thickness
BORDER_THICKNESS = 5

#Image border color
BORDER_COLOR = (1.0, 0.0, 0.0)

############################################# MODEL.PY PARAMETERS ##########################################
#Encoding Kernel Size
EN_KERNEL_SIZE = 5

#Encoding Stride Num
EN_STRIDE = 2

#Encoding Padding
EN_PADDING=2

#Decoding Kernel Size
DE_KERNEL_SIZE = 4

#Decoding Stride Num
DE_STRIDE = 2

#Decoding Padding
DE_PADDING = 1