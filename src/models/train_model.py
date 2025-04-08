import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from .gradient_boost import GradientBoostModel

def prepare_training_data(h2h_features, player_stats, recent_stats):
    """Prepare data for model training"""
    # Combine features
    X = pd.merge(h2h_features, player_stats, on=['player1_id', 'player2_id'], how='left')
    X = pd.merge(X, recent_stats, on=['player1_id', 'player2_id'], how='left')
    
    # Define target variable
    y = X['player1_won'].astype(int)
    
    # Remove target and ID columns
    feature_cols = [col for col in X.columns if col not in ['player1_won', 'player1_id', 'player2_id', 'match_id']]
    X = X[feature_cols]
    
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
