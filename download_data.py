# main.py

import pandas as pd
import os
import sys

# --- Configuration - Imported from download_data.py ---
# This ensures consistency in directory and filenames
try:
    # Import the configuration variables directly from your download script
    from download_data import DATA_DIR, FILES_TO_DOWNLOAD
except ImportError:
    print("ERROR: Could not import configuration from download_data.py.", file=sys.stderr)
    print("Ensure download_data.py is in the same directory or accessible.", file=sys.stderr)
    # Define fallbacks or exit if the import fails
    DATA_DIR = os.path.join('data', 'raw')
    # You might need to manually list filenames here if import fails, but it's less ideal
    FILENAMES_TO_LOAD = [f'atp_{year}.csv' for year in range(2000, 2025)] # Manual fallback example
    print("\nWarning: Using fallback data directory and filenames.", file=sys.stderr)
    # sys.exit(1) # Optionally exit if config import is critical

# Get the list of base filenames from the imported dictionary keys
FILENAMES_TO_LOAD = list(FILES_TO_DOWNLOAD.keys())
# --- End Configuration ---


def load_all_local_data(filenames, data_directory):
    """
    Loads multiple CSV files from a local directory into a single DataFrame.

    Args:
        filenames (list): A list of filenames (without directory path) to load.
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

    # Sort filenames for consistent concatenation order (optional but good practice)
    filenames.sort()

    for filename in filenames:
        local_path = os.path.join(data_directory, filename)

        if not os.path.exists(local_path):
            files_missing.append(filename)
            continue # Skip to the next file

        files_found_count += 1
        try:
            # Use appropriate encoding - iso-8859-1 is common for Jeff Sackmann's data
            df = pd.read_csv(local_path, encoding='iso-8859-1', low_memory=False)
            # Optional: Add a column to know the source year/file if needed later
            # df['source_file'] = filename
            all_dataframes.append(df)
            files_loaded_count += 1
            # Print less verbose loading message
            # print(f"   Loaded {filename} successfully ({len(df)} rows)")
        except FileNotFoundError:
             # This case is handled by the os.path.exists check above, but included for robustness
             files_missing.append(filename)
        except pd.errors.EmptyDataError:
            print(f"   WARNING: Skipping empty file: {filename}", file=sys.stderr)
        except Exception as e:
            print(f"   ERROR: Failed to load or process {filename}. Reason: {e}", file=sys.stderr)

    print(f"\n--- Loading Summary ---")
    print(f"Files expected: {len(filenames)}")
    print(f"Files found locally: {files_found_count}")
    print(f"Files successfully loaded: {files_loaded_count}")

    if files_missing:
        print(f"Files missing ({len(files_missing)}): {', '.join(files_missing)}", file=sys.stderr)
        print("Suggestion: Run 'python download_data.py' to fetch missing files.", file=sys.stderr)

    if not all_dataframes:
        print("\nERROR: No dataframes were loaded. Cannot proceed.", file=sys.stderr)
        return None

    print("\nConcatenating loaded dataframes...")
    # Concatenate all loaded dataframes into a single one
    # ignore_index=True creates a new clean index for the combined DataFrame
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print("Concatenation complete.")
    print("-----------------------")

    return combined_df


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting main script...")

    # Load and combine all specified local data files
    master_df = load_all_local_data(FILENAMES_TO_LOAD, DATA_DIR)

    # Proceed only if data loading was successful
    if master_df is not None:
        print(f"\nSuccessfully loaded and combined data.")
        print(f"Total rows in combined DataFrame: {len(master_df):,}")
        print(f"Total columns: {len(master_df.columns)}")

        print("\n--- Head of Combined Data ---")
        print(master_df.head())

        print("\n--- Tail of Combined Data ---")
        print(master_df.tail())

        print("\n--- Basic Info ---")
        master_df.info()

        # --- Your analysis, preprocessing, and model training code goes below ---

        print("\nStarting data analysis / model training steps...")
        # Example: Convert tourney_date to datetime
        if 'tourney_date' in master_df.columns:
             try:
                 master_df['tourney_date'] = pd.to_datetime(master_df['tourney_date'], format='%Y%m%d')
                 print("\nConverted 'tourney_date' to datetime objects.")
             except Exception as e:
                 print(f"\nWarning: Could not convert 'tourney_date' to datetime. Error: {e}", file=sys.stderr)

        # Add your feature engineering, splitting, training etc. here
        # ...
        # ...

        print("\nMain script finished.")
        # ---------------------------------------------------------------------

    else:
        print("\nExiting script due to data loading failures.", file=sys.stderr)