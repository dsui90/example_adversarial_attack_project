"""
Constants for the training script.
This file contains constants used in the training process, including
hyperparameters, normalization values, and available models.
"""

NORMALIZATION_RGB_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_RGB_STD  = [0.229, 0.224, 0.225]

NUM_CLASSES_CIFAR10 = 10

AVAILABLE_MODELS = [
    'mobilenetv2',
    'resnet50',
]

IMG_INPUT_SIZE = {
    'mobilenetv2': 224,
    'resnet50':    224,
}

# CONSTANT TRAINING PARAMETERS
NUM_EPOCHS = 10
LR = 0.01
MOMENTUM = 0.9
LR_STEP_SIZE = 7
LR_GAMMA = 0.1
BATCH_SIZE = 32
NUM_WORKERS = 1
