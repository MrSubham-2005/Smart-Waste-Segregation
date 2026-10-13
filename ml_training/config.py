"""
EcoSort AI - ML Training Configuration Module
Centralizes all hyperparameters, dataset paths, training settings, and model paths.
"""

import os
from pathlib import Path

# Base Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'val')
TEST_DIR = os.path.join(DATASET_DIR, 'test')

MODEL_DIR = os.path.join(ROOT_DIR, 'ai_engine', 'models')
MODEL_OUTPUT_PATH = os.path.join(MODEL_DIR, 'ecosort_mobilenetv2.keras')
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, 'class_indices.json')
RESULTS_DIR = os.path.join(ROOT_DIR, 'ml_training', 'results')

# The 4 target classes - directory names and index mapping
CLASS_NAMES_DIR = ['wet_waste', 'dry_waste', 'recyclable_waste', 'e_waste']
CLASS_DISPLAY_NAMES = ['Wet Waste', 'Dry Waste', 'Recyclable Waste', 'E-Waste']

CLASS_TO_INDEX = {
    'wet_waste': 0,
    'dry_waste': 1,
    'recyclable_waste': 2,
    'e_waste': 3
}

INDEX_TO_DISPLAY = {
    0: 'Wet Waste',
    1: 'Dry Waste',
    2: 'Recyclable Waste',
    3: 'E-Waste'
}

# Image & Model Hyperparameters
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 4
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 32))
EPOCHS_HEAD = int(os.environ.get('EPOCHS_HEAD', 15))
EPOCHS_FINE_TUNE = int(os.environ.get('EPOCHS_FINE_TUNE', 10))
LEARNING_RATE_HEAD = float(os.environ.get('LEARNING_RATE_HEAD', 0.0005))
LEARNING_RATE_FINE_TUNE = float(os.environ.get('LEARNING_RATE_FINE_TUNE', 0.00005))
DROPOUT_RATE = 0.3
CONFIDENCE_THRESHOLD = 0.50
RANDOM_SEED = 42

# Early Stopping & Checkpointing
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.2
MIN_LR = 1e-6
