# config.py

import os

# Base directory is the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'src', 'models')

# Ensure required directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Files to load for training
FILENAMES_TO_LOAD = [f'atp_{year}.csv' for year in range(2000, 2025)]

# Model directory
MODEL_DIR = os.path.join(BASE_DIR, 'src', 'models')