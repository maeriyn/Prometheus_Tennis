import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from .gradient_boost import GradientBoostModel

def prepare_training_data(h2h_df, recent_df, career_df):
    """
    Merges head-to-head matchups with player stats for ML training,
    keeping recent and career stats separate.

    Parameters:
        h2h_df (pd.DataFrame): Matchups table with columns ['player_id1', 'player_id2', ..., 'winner']  
        recent_df (pd.DataFrame): Recent stats per player
        career_df (pd.DataFrame): Career stats per player

    Returns:
        X (pd.DataFrame): Feature matrix 
        y (pd.Series): Target labels
        feature_cols (list): Names of feature columns used for prediction
    """
    # Validate dataframes have required columns
    for df, name in [(recent_df, 'recent_df'), (career_df, 'career_df')]:
        if 'player_id' not in df.columns:
            raise ValueError(f"Missing 'player_id' column in {name}. "
                           f"Available columns: {df.columns.tolist()}")

    # Keep stats separate by adding prefix to column names
    recent_features = recent_df.copy()
    career_features = career_df.copy()
    
    # Add prefixes to distinguish feature types
    recent_cols = [col for col in recent_features.columns if col != 'player_id']
    career_cols = [col for col in career_features.columns if col != 'player_id']
    
    recent_features.columns = ['recent_' + col if col != 'player_id' else col 
                             for col in recent_features.columns]
    career_features.columns = ['career_' + col if col != 'player_id' else col 
                             for col in career_features.columns]

    # Merge stats while keeping them separate
    X = recent_features.merge(career_features, on='player_id', how='outer')

    # Print data validation info
    print("\nData Merge Confirmation:")
    print(f"Shape of merged dataset: {X.shape}")
    print(f"Number of unique players: {X['player_id'].nunique()}")
    print("\nFeature groupings:")
    print(f"Recent stats columns: {recent_cols}")
    print(f"Career stats columns: {career_cols}")

    # Check for any missing values
    missing_cols = X.columns[X.isnull().any()].tolist()
    if missing_cols:
        print(f"\nWarning - Columns with missing values: {missing_cols}")
        print("Missing value counts:")
        print(X[missing_cols].isnull().sum())

    # For now, using a dummy y variable until h2h merging is implemented
    y = pd.Series(0, index=X.index)  # Placeholder
    
    # Get feature columns, excluding player_id
    feature_cols = [col for col in X.columns if col != 'player_id']

    return X, y, feature_cols

def train_model(X, y, model_type='gradient_boost', model_params=None):
    """Train the specified model type"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize model based on type
    if model_type == 'gradient_boost':
        model = GradientBoostModel(params=model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train and set scaler
    model.train(X_train_scaled, y_train)
    model.scaler = scaler
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    return model, metrics

def save_model(model, feature_cols, model_dir):
    """Save model and feature names"""
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir)
    
    with open(os.path.join(model_dir, 'feature_names.txt'), 'w') as f:
        f.write('\n'.join(feature_cols))

def train_gradient_boost(h2h_features, recent_stats, career_stats, model_params=None):
    """
    Train a gradient boosting model using the provided features and parameters.
    
    Parameters:
        h2h_features (pd.DataFrame): Head-to-head matchup features
        recent_stats (pd.DataFrame): Recent form statistics for players
        career_stats (pd.DataFrame): Career statistics for players
        model_params (dict, optional): Parameters for the gradient boosting model
    """
    # Prepare training data with all feature sets
    X, y, feature_cols = prepare_training_data(h2h_features, recent_stats, career_stats)
    
    # Train model
    model, metrics = train_model(X, y, model_type='gradient_boost', model_params=model_params)
    
    return model, metrics, feature_cols
