"""
Data Preprocessing for CTR Models
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class DataProcessor:
    """
    Data preprocessing for CTR models.
    
    Features:
    - Automatic feature type detection
    - High-cardinality handling
    - Feature crossing
    - Label encoding
    """
    
    def __init__(self):
        self.encoders = {}
        self.feature_names = []
        self.sparse_features = []
        self.dense_features = []
        self.vocab_sizes = {}
        self._fitted = False
    
    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame = None,
        y_val: np.ndarray = None,
        sparse_features: List[str] = None,
        dense_features: List[str] = None,
        cross_features: List[Tuple[str, str]] = None,
        high_cardinality_threshold: int = 100,
        val_ratio: float = 0.2,
    ) -> dict:
        """
        Fit and transform training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (if None, split from train)
            y_val: Validation labels
            sparse_features: Categorical feature names
            dense_features: Numerical feature names
            cross_features: Feature pairs for crossing
            high_cardinality_threshold: Min frequency for high-cardinality handling
            val_ratio: Validation split ratio if X_val is None
        
        Returns:
            Dict with processed data
        """
        # Train-val split if needed
        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_ratio, random_state=42
            )
        
        X_train = X_train.copy()
        X_val = X_val.copy()
        
        # Auto-detect feature types if not provided
        if sparse_features is None and dense_features is None:
            sparse_features, dense_features = self._detect_feature_types(X_train)
        
        sparse_features = sparse_features or []
        dense_features = dense_features or []
        
        self.sparse_features = sparse_features.copy()
        self.dense_features = dense_features.copy()
        
        # Create cross features
        if cross_features:
            for col1, col2 in cross_features:
                cross_name = f'{col1}_x_{col2}'
                X_train[cross_name] = X_train[col1].astype(str) + '_' + X_train[col2].astype(str)
                X_val[cross_name] = X_val[col1].astype(str) + '_' + X_val[col2].astype(str)
                self.sparse_features.append(cross_name)
        
        # Select features
        all_features = self.sparse_features + self.dense_features
        X_train = X_train[all_features]
        X_val = X_val[all_features]
        
        # Handle high-cardinality
        for col in self.sparse_features:
            vc = X_train[col].value_counts()
            valid_values = set(vc[vc >= high_cardinality_threshold].index)
            X_train[col] = X_train[col].apply(lambda x: x if x in valid_values else 'OTHER')
            X_val[col] = X_val[col].apply(lambda x: x if x in valid_values else 'OTHER')
        
        # Label encode sparse features
        for col in self.sparse_features:
            le = LabelEncoder()
            all_vals = pd.concat([X_train[col], X_val[col]]).astype(str).unique()
            le.fit(all_vals)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_val[col] = le.transform(X_val[col].astype(str))
            self.encoders[col] = le
            self.vocab_sizes[col] = len(le.classes_)
        
        self.feature_names = all_features
        self._fitted = True
        
        # Prepare output
        train_input = {name: X_train[name].values for name in self.feature_names}
        val_input = {name: X_val[name].values for name in self.feature_names}
        
        return {
            'X_train': X_train,
            'y_train': np.array(y_train, dtype=np.float32),
            'X_val': X_val,
            'y_val': np.array(y_val, dtype=np.float32),
            'train_input': train_input,
            'val_input': val_input,
            'feature_names': self.feature_names,
            'sparse_features': self.sparse_features,
            'dense_features': self.dense_features,
            'vocab_sizes': self.vocab_sizes,
        }
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted encoders."""
        if not self._fitted:
            raise ValueError("DataProcessor not fitted. Call fit_transform() first.")
        
        X = X.copy()
        
        # Select features
        X = X[self.feature_names]
        
        # Encode sparse features
        for col in self.sparse_features:
            le = self.encoders[col]
            # Handle unseen values
            X[col] = X[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )
        
        return X
    
    def _detect_feature_types(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Auto-detect sparse (categorical) and dense (numerical) features."""
        sparse = []
        dense = []
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                sparse.append(col)
            elif X[col].nunique() < 20:  # Treat low-cardinality int as categorical
                sparse.append(col)
            else:
                dense.append(col)
        
        return sparse, dense
