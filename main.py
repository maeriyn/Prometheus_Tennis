# main.py (or main_training.py)

import pandas as pd
import os
import sys

# --- Project Structure Imports ---
# Import the central configuration first
try:
    import config
except ModuleNotFoundError:
    print("CRITICAL ERROR: config.py not found in the project root.", file=sys.stderr)
    sys.exit(1) # Exit if config is missing

# Import your source code modules using 'src.' prefix
try:
    from src.data_processing import loader
    # from src.data_processing import preprocessing # Add later
    # from src.features import build_features     # Add later
    # ... other imports
except ModuleNotFoundError as e:
     print(f"CRITICAL ERROR: Failed to import module from 'src'. Error: {e}", file=sys.stderr)
     print("Ensure 'src' directory and '__init__.py' files exist.", file=sys.stderr)
     sys.exit(1)
# --- End Imports ---

def main():
    """Main function to run the training pipeline."""
    print("="*30)
    print(" Starting Prometheus_Tennis Training Pipeline ")
    print("="*30)

    # --- Step 1: Load Data ---
    print("\n--- Step 1: Loading Data ---")
    
    # Check if data directory exists and has files
    if not os.path.exists(config.RAW_DATA_DIR) or not os.listdir(config.RAW_DATA_DIR):
        print("\nERROR: Data directory empty or missing.", file=sys.stderr)
        print("Please run 'python src/data_processing/download_data.py' first.", file=sys.stderr)
        return

    # Load the data
    master_df = loader.load_all_local_data(
        config.FILENAMES_TO_LOAD,
        config.RAW_DATA_DIR
    )

    # Check if loading was successful before proceeding
    if master_df is None:
        print("\nCRITICAL ERROR: Data loading failed. Aborting pipeline.", file=sys.stderr)
        return # Exit the main function

    print(f"\nData loaded successfully.")
    print(f"Total rows in combined DataFrame: {len(master_df):,}")
    print(f"Total columns: {len(master_df.columns)}")
    print("\n--- Head of Combined Data ---")
    print(master_df.head()) # Show a preview
    print("--- End Step 1 ---")
    # --- End Step 1 ---


    # --- Step 2: Preprocessing ---
    print("\n--- Step 2: Preprocessing Data ---")
    try:
        from src.data_processing.preprocessing import clean_data
        cleaned_df = clean_data(master_df)
        
        if cleaned_df is None or len(cleaned_df) == 0:
            print("ERROR: Preprocessing resulted in empty dataset!", file=sys.stderr)
            return
            
        print("\n--- Data Quality Report ---")
        print(f"Missing values remaining:")
        print(cleaned_df.isnull().sum()[cleaned_df.isnull().sum() > 0])
        print("\nDataset shape after cleaning:", cleaned_df.shape)
        
    except Exception as e:
        print(f"ERROR: Preprocessing failed. Error: {e}", file=sys.stderr)
        return
    print("--- End Step 2 ---")


    # --- Step 3: Building Features ---
    print("\n--- Step 3: Building Features ---")
    try:
        from src.features.build_features import build_all_features
        
        h2h_features, player_stats, recent_stats = build_all_features(cleaned_df)
        
        print("\n--- Feature Statistics ---")
        print(f"Head-to-head pairs created: {len(h2h_features):,}")
        print(f"Career statistics created for {len(player_stats):,} players")
        print(f"Recent statistics created for {len(recent_stats):,} players")
        
        # Save features for later use
        h2h_features.to_csv(os.path.join(config.PROCESSED_DATA_DIR, 'head_to_head_with_names.csv'), index=False)
        player_stats.to_csv(os.path.join(config.PROCESSED_DATA_DIR, 'player_career_stats_with_names.csv'), index=False)
        recent_stats.to_csv(os.path.join(config.PROCESSED_DATA_DIR, 'player_recent_stats_with_names.csv'), index=False)
        
    except Exception as e:
        print(f"ERROR: Feature engineering failed. Error: {e}", file=sys.stderr)
        return
    print("--- End Step 3 ---")

    # --- Step 4: Model Training ---
    print("\n--- Step 4: Model Training ---")
    try:
        from src.models.train_model import prepare_training_data, train_gradient_boost, save_model
        
        # Prepare data
        X, y, feature_cols = prepare_training_data(h2h_features, recent_stats)
        print(f"\nPrepared dataset shape: {X.shape}")
        
        # Train model
        model_dir = os.path.join(config.MODEL_DIR, 'gradient_boost')
        model, scaler, metrics = train_gradient_boost(X, y, model_dir)
        
        # Print metrics
        print("\n--- Model Performance ---")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
        # Save model artifacts
        save_model(model, scaler, feature_cols, model_dir)
        
    except Exception as e:
        print(f"ERROR: Model training failed. Error: {e}", file=sys.stderr)
        return

    print("\n" + "="*30)
    print(" Prometheus_Tennis Training Pipeline Completed Successfully ")
    print("="*30)

if __name__ == "__main__":
    # This ensures the main function runs only when the script is executed directly
    # (not when imported as a module)
    main()