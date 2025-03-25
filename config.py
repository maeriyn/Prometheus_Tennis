# config.py

import os

# --- Path Configuration ---
# Get the absolute path of the directory where config.py is located (project root)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define data directories relative to the project root
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
# PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed') # Add later if needed
# MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')       # Add later if needed

# Create directories if they don't exist (optional, download_data.py might do this)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
# os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
# os.makedirs(MODELS_DIR, exist_ok=True)


# --- Data Loading Configuration ---
# Define the range of years for the ATP data files
START_YEAR = 2000
END_YEAR = 2024 # Inclusive

# Generate the list of filenames based on the year range
# This assumes your filenames are exactly like 'atp_2000.csv', 'atp_2001.csv', etc.
FILENAMES_TO_LOAD = [f'atp_{year}.csv' for year in range(END_YEAR, START_YEAR - 1, -1)]


# --- Other Configurations (Add as needed) ---
# RANDOM_SEED = 42
# TEST_SPLIT_DATE = '2023-01-01'
# TARGET_VARIABLE = 'winner' # Example