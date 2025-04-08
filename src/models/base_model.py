from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Base class for all models"""
    
    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """Get probability predictions"""
        pass
    
    @abstractmethod
    def save(self, model_dir):
        """Save model artifacts"""
        pass
    
    @abstractmethod
    def load(self, model_dir):
        """Load model artifacts"""
        pass
