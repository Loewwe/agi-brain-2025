"""
Model Prototypes for Research

Two model types:
1. LightGBMAlphaModel: Tabular baseline using gradient boosting
2. SeqAlphaModel: Simple sequence model (LSTM/GRU) on fixed windows

Both implement unified BaseAlphaModel interface.
"""

from typing import Protocol, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
from pydantic import BaseModel
import lightgbm as lgb


class TrainResult(BaseModel):
    """Training results."""
    auc: Optional[float] = None
    logloss: Optional[float] = None
    best_iteration: Optional[int] = None
    feature_importances: Optional[dict[str, float]] = None
    notes: Optional[str] = None


class BaseAlphaModel(Protocol):
    """Protocol for alpha models."""
    
    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray | pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> TrainResult:
        """Train the model."""
        ...
    
    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict probabilities P(y=1)."""
        ...
    
    def save(self, path: Path) -> None:
        """Save model to path."""
        ...
    
    @classmethod
    def load(cls, path: Path) -> "BaseAlphaModel":
        """Load model from path."""
        ...


class LightGBMAlphaModel:
    """
    LightGBM-based alpha model (tabular baseline).
    """
    
    def __init__(self, params: Optional[dict] = None, random_state: int = 42):
        """
        Args:
            params: LightGBM parameters
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        
        # Default params
        default_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': random_state,
        }
        
        if params:
            default_params.update(params)
        
        self.params = default_params
        self.model = None
        self.feature_names = None
        self.train_result = None
    
    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray | pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> TrainResult:
        """Train LightGBM model."""
        
        # Convert to DataFrame if needed
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
        if isinstance(X_val, np.ndarray) and X_val is not None:
            X_val = pd.DataFrame(X_val)
        
        self.feature_names = list(X_train.columns)
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = None
        
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train
        callbacks = []
        if valid_data:
            callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data] if valid_data else None,
            callbacks=callbacks if callbacks else None,
        )
        
        # Calculate metrics
        auc = None
        logloss = None
        
        if X_val is not None and y_val is not None:
            y_pred = self.model.predict(X_val)
            from sklearn.metrics import roc_auc_score, log_loss
            
            # Check if target is degenerate (only one class)
            if len(np.unique(y_val)) > 1:
                try:
                    auc = roc_auc_score(y_val, y_pred)
                    logloss = log_loss(y_val, y_pred)
                except ValueError:
                    # Handle edge cases gracefully
                    pass
        
        # Feature importances (ensure string keys)
        importances = {
            str(name): float(imp)
            for name, imp in zip(
                self.feature_names,
                self.model.feature_importance(importance_type='gain').tolist()
            )
        }
        
        self.train_result = TrainResult(
            auc=auc,
            logloss=logloss,
            best_iteration=self.model.best_iteration,
            feature_importances=importances,
        )
        
        return self.train_result
    
    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        return self.model.predict(X)
    
    def save(self, path: Path) -> None:
        """Save model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_model(str(path / "model.txt"))
        
        # Save metadata
        metadata = {
            'params': self.params,
            'feature_names': self.feature_names,
            'random_state': self.random_state,
            'train_result': self.train_result.model_dump() if self.train_result else None,
        }
        
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "LightGBMAlphaModel":
        """Load model."""
        path = Path(path)
        
        # Load metadata
        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls(
            params=metadata['params'],
            random_state=metadata['random_state'],
        )
        
        # Load model
        instance.model = lgb.Booster(model_file=str(path / "model.txt"))
        instance.feature_names = metadata['feature_names']
        
        if metadata['train_result']:
            instance.train_result = TrainResult(**metadata['train_result'])
        
        return instance


class SeqAlphaModel:
    """
    Simple sequence model using LSTM/GRU.
    
    Input: (batch, window_size, n_features)
    Output: P(y=1)
    """
    
    def __init__(
        self,
        window_size: int = 32,
        n_features: Optional[int] = None,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        random_state: int = 42,
    ):
        """
        Args:
            window_size: Length of input sequences
            n_features: Number of features (inferred from data if None)
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            random_state: Random seed
        """
        self.window_size = window_size
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.random_state = random_state
        
        self.model = None
        self.train_result = None
        
        # Set random seeds
        np.random.seed(random_state)
        try:
            import torch
            torch.manual_seed(random_state)
        except ImportError:
            pass
    
    def _build_model(self, n_features: int):
        """Build PyTorch model."""
        import torch
        import torch.nn as nn
        
        class LSTMClassifier(nn.Module):
            def __init__(self, n_features, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    n_features,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                )
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size, 1)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                lstm_out, _ = self.lstm(x)
                # Take last output
                last_output = lstm_out[:, -1, :]
                dropped = self.dropout(last_output)
                logits = self.fc(dropped)
                output = self.sigmoid(logits)
                # Squeeze but preserve at least 1D
                if output.dim() > 1:
                    return output.squeeze(-1)
                return output
        
        return LSTMClassifier(n_features, self.hidden_size, self.num_layers, self.dropout)
    
    def _prepare_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Convert tabular data to sequences."""
        n_samples = len(X) - self.window_size + 1
        
        if n_samples <= 0:
            raise ValueError(f"Not enough samples for window_size={self.window_size}")
        
        X_seq = np.array([
            X[i:i+self.window_size]
            for i in range(n_samples)
        ])
        
        if y is not None:
            # Target is for the last bar in the window
            y_seq = y[self.window_size-1:]
            return X_seq, y_seq
        
        return X_seq
    
    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray | pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> TrainResult:
        """Train sequence model."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.metrics import roc_auc_score, log_loss
        
        # Convert to numpy
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(X_val, pd.DataFrame) and X_val is not None:
            X_val = X_val.values
        
        # Infer features
        if self.n_features is None:
            self.n_features = X_train.shape[1]
        
        # Build model
        self.model = self._build_model(self.n_features)
        
        # Prepare sequences
        X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train)
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train_seq)
        y_train_t = torch.FloatTensor(y_train_seq)
        
        # Training
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        batch_size = 32
        n_epochs = 20
        
        for epoch in range(n_epochs):
            self.model.train()
            
            # Simple batching
            indices = np.arange(len(X_train_t))
            np.random.shuffle(indices)
            
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = X_train_t[batch_idx]
                y_batch = y_train_t[batch_idx]
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        auc = None
        logloss_val = None
        
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val)
            X_val_t = torch.FloatTensor(X_val_seq)
            
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(X_val_t).numpy()
            
            auc = roc_auc_score(y_val_seq, y_pred)
            logloss_val = log_loss(y_val_seq, y_pred)
        
        self.train_result = TrainResult(
            auc=auc,
            logloss=logloss_val,
            notes=f"Trained for {n_epochs} epochs",
        )
        
        return self.train_result
    
    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        import torch
        
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_seq = self._prepare_sequences(X)
        X_t = torch.FloatTensor(X_seq)
        
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_t).numpy()
        
        return y_pred
    
    def save(self, path: Path) -> None:
        """Save model."""
        import torch
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), path / "model.pt")
        
        # Save metadata
        metadata = {
            'window_size': self.window_size,
            'n_features': self.n_features,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'random_state': self.random_state,
            'train_result': self.train_result.model_dump() if self.train_result else None,
        }
        
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "SeqAlphaModel":
        """Load model."""
        import torch
        
        path = Path(path)
        
        # Load metadata
        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls(
            window_size=metadata['window_size'],
            n_features=metadata['n_features'],
            hidden_size=metadata['hidden_size'],
            num_layers=metadata['num_layers'],
            dropout=metadata['dropout'],
            random_state=metadata['random_state'],
        )
        
        # Build and load model
        instance.model = instance._build_model(instance.n_features)
        instance.model.load_state_dict(torch.load(path / "model.pt"))
        instance.model.eval()
        
        if metadata['train_result']:
            instance.train_result = TrainResult(**metadata['train_result'])
        
        return instance
