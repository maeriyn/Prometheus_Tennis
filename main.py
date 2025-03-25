# main.py (Modified)

import pandas as pd
import os
# from dotenv import load_dotenv # Uncomment if you use .env later
# load_dotenv() # Uncomment if you use .env later

# --- Configuration: Define where the local data is stored ---
DATA_DIR = os.path.join('data', 'raw')

# List of expected local filenames (should match keys/filenames in download_data.py)
LOCAL_FILES = {
    f'atp_{year}': f'atp_{year}'
    for year in range(2024, 1999, -1)
}
# --- End Configuration ---

def load_data_from_local(file_dict, data_directory):
    """Loads data from local files specified in a dictionary."""
    dataframes = {}
    print(f"Loading data from local directory: '{data_directory}'...")
    all_found = True
    for name, filename in file_dict.items():
        local_path = os.path.join(data_directory, filename)
        if not os.path.exists(local_path):
            print(f"   ERROR: File not found at {local_path}. Please run download_data.py first.")
            all_found = False
            continue # Skip to the next file if this one is missing
        try:
            print(f"-> Loading {filename}...")
            # You might need error handling or specific encoding
            dataframes[name] = pd.read_csv(local_path, encoding='iso-8859-1') # Example encoding
            print(f"   Loaded {name} successfully ({len(dataframes[name])} rows)")
        except Exception as e:
            print(f"   ERROR loading {filename}: {e}")
            all_found = False # Consider it a failure if loading fails

    if not all_found:
         print("ERROR: One or more files were missing or failed to load.")
         # Decide if you want to exit or continue with partial data
         # return None # Option to return nothing on failure

    print("Local data loading complete.")
    return dataframes

# --- Main Execution ---
if __name__ == "__main__":
    # Load the data from local files
    all_data = load_data_from_local(LOCAL_FILES, DATA_DIR)

    # Check if loading was successful (depends on error handling in load_data_from_local)
    if all_data:
        # Example: Access and print head of one DataFrame
        if 'atp_2023' in all_data:
            print("\n--- Head of ATP 2023 Data ---")
            print(all_data['atp_2023'].head())

        # --- Your analysis code will go here ---
        # Combine dataframes if needed:
        # combined_df = pd.concat(all_data.values(), ignore_index=True)
        # ...etc...
    else:
        print("\nExiting due to data loading errors.")