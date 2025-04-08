import os
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from .base_model import BaseModel

class GradientBoostModel(BaseModel):
    def __init__(self, params=None):
        self.params = params or {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        }
        self.model = GradientBoostingClassifier(**self.params)
        self.scaler = None
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
        
    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, 'model.joblib'))
        if self.scaler:
            joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.joblib'))
            
    def load(self, model_dir):
        self.model = joblib.load(os.path.join(model_dir, 'model.joblib'))
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
