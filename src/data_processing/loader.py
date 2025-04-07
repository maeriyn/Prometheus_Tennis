# src/data_processing/loader.py

import pandas as pd
import os
import sys

def load_all_local_data(filenames_to_load, data_directory):
    """
    Loads multiple CSV files from a local directory into a single DataFrame.

    Args:
        filenames_to_load (list): A list of filenames (without directory path) to load.
        data_directory (str): The path to the directory containing the CSV files.

    Returns:
        pandas.DataFrame or None: A single DataFrame containing concatenated data
                                  from all successfully loaded files, or None if
                                  no files could be loaded.
    """
    all_dataframes = []
    print(f"Attempting to load data from local directory: '{data_directory}'...")
    files_found_count = 0
    files_loaded_count = 0
    files_missing = []
    files_load_errors = {}

    # Ensure filenames_to_load is a list
    if not isinstance(filenames_to_load, list):
         print("ERROR: filenames_to_load argument must be a list.", file=sys.stderr)
         return None

    filenames_to_load.sort() # Ensure consistent order (e.g., chronological)

    # --- First Pass: Check for missing files ---
    print("Checking for file existence...")
    for filename in filenames_to_load:
        local_path = os.path.join(data_directory, filename)
        if not os.path.exists(local_path):
            files_missing.append(filename)

    if files_missing:
        print(f"\nWARNING: The following {len(files_missing)} files were not found locally:", file=sys.stderr)
        # Print only a few missing files if the list is long
        display_limit = 10
        for i, missing_file in enumerate(files_missing):
             if i < display_limit:
                 print(f" - {missing_file}", file=sys.stderr)
             elif i == display_limit:
                 print(f" - ... and {len(files_missing) - display_limit} more.", file=sys.stderr)
                 break
        print("Suggestion: Run 'python download_data.py' to fetch missing files.\n", file=sys.stderr)
    else:
        print("All expected files found locally.")
    # --- End File Existence Check ---

    # --- Second Pass: Load existing files ---
    print("Loading existing files...")
    for filename in filenames_to_load:
        # Skip files already identified as missing
        if filename in files_missing:
            continue

        local_path = os.path.join(data_directory, filename)
        files_found_count += 1 # Increment here as we know it exists

        try:
            # Add data type optimization
            dtypes = {
                'winner_id': 'int32',
                'loser_id': 'int32',
                'winner_rank': 'float32',
                'loser_rank': 'float32',
                'winner_rank_points': 'float32',
                'loser_rank_points': 'float32'
            }
            # Use chunks for larger files
            df = pd.read_csv(local_path, 
                           encoding='iso-8859-1', 
                           low_memory=False,
                           dtype=dtypes,
                           usecols=lambda x: x not in ['winner_ioc', 'loser_ioc'])  # Skip unnecessary columns

            if df.empty:
                 print(f"   WARNING: Loaded file is empty: {filename}", file=sys.stderr)
                 continue # Skip empty files

            all_dataframes.append(df)
            files_loaded_count += 1
            # print(f"   Loaded {filename} successfully") # Less verbose

        except Exception as e:
            error_msg = f"Reason: {e}"
            print(f"   ERROR: Failed to load or process {filename}. {error_msg}", file=sys.stderr)
            files_load_errors[filename] = error_msg
    # --- End Loading Loop ---


    print(f"\n--- Loading Summary ---")
    print(f"Files expected: {len(filenames_to_load)}")
    print(f"Files found locally: {len(filenames_to_load) - len(files_missing)}")
    print(f"Files missing: {len(files_missing)}")
    print(f"Files successfully loaded (non-empty): {files_loaded_count}")
    print(f"Files with loading errors: {len(files_load_errors)}")
    if files_load_errors:
         for fname, err in files_load_errors.items():
              print(f" - Error for {fname}: {err}", file=sys.stderr)
    print("-----------------------")


    if not all_dataframes:
        print("\nERROR: No dataframes were loaded. Cannot proceed.", file=sys.stderr)
        return None

    print("\nConcatenating loaded dataframes...")
    try:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print("Concatenation complete.")
    except Exception as e:
        print(f"\nERROR: Failed to concatenate DataFrames. Reason: {e}", file=sys.stderr)
        return None

    return combined_df

# Add other data loading functions (e.g., load_processed_data) here later if needed.