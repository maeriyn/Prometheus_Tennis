import pandas as pd
import os
# from dotenv import load_dotenv # Uncomment if you use .env later
# load_dotenv() # Uncomment if you use .env later

# --- Configuration: Replace with your actual RAW GitHub URLs ---
# Find the file on GitHub -> Click 'Raw' -> Copy the URL from browser bar
URLS = {
    'atp_2024': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/refs/heads/master/atp_matches_2024.csv',
    'atp_2023': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/refs/heads/master/atp_matches_2023.csv',
    'atp_2022': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/refs/heads/master/atp_matches_2022.csv',
    'atp_2021': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/refs/heads/master/atp_matches_2021.csv',
    'atp_2020': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/refs/heads/master/atp_matches_2020.csv',\
    'atp_2019': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/refs/heads/master/atp_matches_2019.csv',
    'atp_2018': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/refs/heads/master/atp_matches_2018.csv',
    'atp_2017': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/refs/heads/master/atp_matches_2017.csv',
    'atp_2016': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/refs/heads/master/atp_matches_2016.csv',
    'atp_2015': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/refs/heads/master/atp_matches_2015.csv',
    'atp_2014': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/refs/heads/master/atp_matches_2014.csv',
    'atp_2013': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/refs/heads/master/atp_matches_2013.csv',
    'atp_2012': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/refs/heads/master/atp_matches_2012.csv',
    'atp_2011': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/refs/heads/master/atp_matches_2011.csv',
    'atp_2010': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/refs/heads/master/atp_matches_2010.csv',
    # Add more files as needed
}

def load_data_from_urls(url_dict):
    """Loads data from a dictionary of URLs into pandas DataFrames."""
    dataframes = {}
    print("Loading data from URLs...")
    for name, url in url_dict.items():
        try:
            print(f"-> Loading {name} from {url[:50]}...") # Print truncated URL
            # You might need error handling or specific encoding
            dataframes[name] = pd.read_csv(url, encoding='iso-8859-1') # Example encoding, adjust if needed
            print(f"   Loaded {name} successfully ({len(dataframes[name])} rows)")
        except Exception as e:
            print(f"   ERROR loading {name}: {e}")
            # Decide how to handle errors: skip, stop, etc.
    print("Data loading complete.")
    return dataframes

# --- Main Execution ---
if __name__ == "__main__":
    # Load the data
    all_data = load_data_from_urls(URLS)

    # Example: Access and print head of one DataFrame
    if 'atp_2023' in all_data:
        print("\n--- Head of ATP 2023 Data ---")
        print(all_data['atp_2023'].head())

    # --- Your analysis code will go here ---
    # Combine dataframes if needed:
    # combined_df = pd.concat(all_data.values(), ignore_index=True)
    # print("\n--- Head of Combined Data ---")
    # print(combined_df.head())
    # print(f"Total combined rows: {len(combined_df)}")
    # ----------------------------------------