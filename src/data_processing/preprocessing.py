import pandas as pd
import numpy as np

def clean_data(df):
    """
    Clean the tennis match data by handling missing values, removing duplicates,
    and correcting common errors.
    
    Args:
        df (pd.DataFrame): Raw tennis matches DataFrame
    
    Returns:
        pd.DataFrame: Cleaned tennis matches DataFrame
    """
    print("Starting data cleaning process...")
    initial_rows = len(df)
    
    # Make a copy to avoid modifying the original
    cleaned = df.copy()
    
    # --- Handle Missing Values ---
    # For numeric columns, replace missing with median
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())
    
    # For categorical/string columns, replace missing with 'Unknown'
    categorical_cols = cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        cleaned[col] = cleaned[col].fillna('Unknown')
    
    # --- Remove Duplicates ---
    # Identify columns that should make a match unique
    key_columns = ['tourney_id', 'tourney_name', 'tourney_date', 
                  'match_num', 'winner_id', 'loser_id']
    cleaned = cleaned.drop_duplicates(subset=key_columns, keep='first')
    
    # --- Correct Common Errors ---
    # Ensure score format is consistent
    cleaned['score'] = cleaned['score'].replace('W/O', 'WO')
    cleaned['score'] = cleaned['score'].replace('DEF', 'WO')
    cleaned['score'] = cleaned['score'].replace('(RET)', 'RET')
    
    # Ensure tournament dates are valid
    cleaned['tourney_date'] = pd.to_datetime(cleaned['tourney_date'], format='%Y%m%d', errors='coerce')
    
    # Remove matches with invalid dates
    cleaned = cleaned.dropna(subset=['tourney_date'])
    
    # Ensure numerical stats are within valid ranges
    stat_columns = ['w_ace', 'w_df', 'l_ace', 'l_df', 'w_svpt', 'l_svpt']
    for col in stat_columns:
        if col in cleaned.columns:
            # Replace negative values with 0
            cleaned[col] = cleaned[col].clip(lower=0)
            # Replace unreasonably high values with median
            upper_limit = cleaned[col].quantile(0.99)  # 99th percentile
            cleaned.loc[cleaned[col] > upper_limit, col] = cleaned[col].median()
    
    # --- Final Cleanup ---
    # Remove rows where essential data is missing
    essential_cols = ['winner_id', 'loser_id', 'tourney_id']
    cleaned = cleaned.dropna(subset=essential_cols)
    
    # Print summary of changes
    final_rows = len(cleaned)
    rows_removed = initial_rows - final_rows
    print(f"\nCleaning Summary:")
    print(f"Initial rows: {initial_rows:,}")
    print(f"Rows removed: {rows_removed:,}")
    print(f"Final rows: {final_rows:,}")
    print(f"Percentage of data retained: {(final_rows/initial_rows)*100:.2f}%")
    
    return cleaned
