# download_data.py

import requests
import os
import sys

# --- Configuration ---
# Define the target directory relative to where this script is run
# Saving inside 'data/raw' is a common convention
# os.path.dirname(__file__) gets the directory where download_data.py lives
# os.path.abspath gets the full path to avoid relative path issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

# Create the target directory if it doesn't exist
try:
    os.makedirs(DATA_DIR, exist_ok=True) # exist_ok=True prevents error if dir exists
    print(f"Ensured data directory exists: {DATA_DIR}")
except OSError as e:
    print(f"ERROR: Could not create data directory '{DATA_DIR}'. Reason: {e}", file=sys.stderr)
    sys.exit(1) # Exit if we can't create the directory

# Dictionary mapping desired local filenames to their RAW GitHub URLs
# Using the URLs provided previously
FILES_TO_DOWNLOAD = {
    'atp_2024.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2024.csv',
    'atp_2023.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2023.csv',
    'atp_2022.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2022.csv',
    'atp_2021.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2021.csv',
    'atp_2020.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2020.csv',
    'atp_2019.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2019.csv',
    'atp_2018.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2018.csv',
    'atp_2017.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2017.csv',
    'atp_2016.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2016.csv',
    'atp_2015.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2015.csv',
    'atp_2014.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2014.csv',
    'atp_2013.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2013.csv',
    'atp_2012.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2012.csv',
    'atp_2011.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2011.csv',
    'atp_2010.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2010.csv',
    'atp_2009.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2009.csv',
    'atp_2008.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2008.csv',
    'atp_2007.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2007.csv',
    'atp_2006.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2006.csv',
    'atp_2005.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2005.csv',
    'atp_2004.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2004.csv',
    'atp_2003.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2003.csv',
    'atp_2002.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2002.csv',
    'atp_2001.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2001.csv',
    'atp_2000.csv': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2000.csv'
    # Add other files like WTA or Futures here if needed, using the same pattern
}
# --- End Configuration ---

def download_file(url, local_path):
    """
    Downloads a file from a URL to a local path, only if it doesn't exist.

    Args:
        url (str): The URL to download from.
        local_path (str): The full local path to save the file.

    Returns:
        bool: True if the file exists locally (either skipped or downloaded),
              False if an error occurred during download.
    """
    # Check if file already exists
    if os.path.exists(local_path):
        print(f"  Skipping: File already exists at {os.path.basename(local_path)}")
        return True # Indicate success (already exists)

    # If file doesn't exist, proceed with download
    print(f"  Downloading: {os.path.basename(local_path)} from {url} ...")
    try:
        # Use stream=True for potentially larger files and better memory usage
        with requests.get(url, stream=True, timeout=30) as r: # Added timeout
            r.raise_for_status() # Check for download errors (like 404 Not Found)

            # Open the local file in binary write mode
            with open(local_path, 'wb') as f:
                # Write the content in chunks to handle potentially large files
                for chunk in r.iter_content(chunk_size=8192): # 8KB chunks
                    f.write(chunk)
        print(f"  Successfully downloaded {os.path.basename(local_path)}")
        return True
    except requests.exceptions.Timeout:
        print(f"ERROR: Timeout occurred while trying to download {url}", file=sys.stderr)
        # Clean up partially downloaded file if timeout occurs
        if os.path.exists(local_path): os.remove(local_path)
        return False
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to download {url}. Reason: {e}", file=sys.stderr)
        # Clean up partially downloaded file if error occurs
        if os.path.exists(local_path): os.remove(local_path)
        return False
    except Exception as e:
        print(f"ERROR: An unexpected error occurred for {url}: {e}", file=sys.stderr)
        if os.path.exists(local_path): os.remove(local_path)
        return False

# --- Main Download Logic ---
# This block runs only when the script is executed directly
if __name__ == "__main__":
    print(f"\nStarting data download process into '{DATA_DIR}'...")
    print(f"Checking {len(FILES_TO_DOWNLOAD)} files...")

    all_successful = True
    download_count = 0
    skip_count = 0
    error_count = 0

    # Loop through the dictionary of files to download
    for filename, url in FILES_TO_DOWNLOAD.items():
        # Construct the full local path using os.path.join for OS compatibility
        local_file_path = os.path.join(DATA_DIR, filename)

        # Check existence status before calling download function
        existed_before = os.path.exists(local_file_path)

        # Call the download function
        success = download_file(url, local_file_path)

        # Update counters based on outcome
        if success:
            if not existed_before:
                 download_count += 1 # Incremented only if newly downloaded
            else:
                 skip_count +=1 # Incremented if skipped
        else:
            all_successful = False # Mark failure if any download fails
            error_count += 1

    # --- Print Final Summary ---
    print("\n--- Download Summary ---")
    print(f"Total files checked: {len(FILES_TO_DOWNLOAD)}")
    print(f"Files newly downloaded: {download_count}")
    print(f"Files already existing (skipped): {skip_count}")
    print(f"Errors during download: {error_count}")

    if all_successful:
        print("\nDownload process completed successfully (or all files were present).")
    else:
        print("\nDownload process finished with errors. Check messages above.", file=sys.stderr)
    print("------------------------")

    # Exit with an error code if there were errors
    if not all_successful:
        sys.exit(1)