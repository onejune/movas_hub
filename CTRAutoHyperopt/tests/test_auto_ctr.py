"""
Unit tests for CTR Auto HyperOpt
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctr_auto_hyperopt.data.preprocessing import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """Test DataProcessor class."""
    
    def setUp(self):
        """Create sample data."""
        np.random.seed(42)
        n = 1000
        self.X = pd.DataFrame({
            'user_id': np.random.randint(0, 100, n),
            'item_id': np.random.randint(0, 50, n),
            'category': np.random.choice(['A', 'B', 'C', 'D'], n),
            'price': np.random.uniform(1, 100, n),
        })
        self.y = np.random.randint(0, 2, n)
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        processor = DataProcessor()
        data = processor.fit_transform(
            self.X, self.y,
            sparse_features=['user_id', 'item_id', 'category'],
            dense_features=['price'],
        )
        
        self.assertIn('X_train', data)
        self.assertIn('y_train', data)
        self.assertIn('X_val', data)
        self.assertIn('y_val', data)
        self.assertIn('feature_names', data)
        self.assertIn('vocab_sizes', data)
        
        # Check shapes
        self.assertEqual(len(data['X_train']) + len(data['X_val']), len(self.X))
        self.assertEqual(len(data['feature_names']), 4)
    
    def test_cross_features(self):
        """Test feature crossing."""
        processor = DataProcessor()
        data = processor.fit_transform(
            self.X, self.y,
            sparse_features=['user_id', 'item_id', 'category'],
            cross_features=[('user_id', 'category')],
        )
        
        self.assertIn('user_id_x_category', data['feature_names'])
    
    def test_high_cardinality(self):
        """Test high cardinality handling."""
        processor = DataProcessor()
        data = processor.fit_transform(
            self.X, self.y,
            sparse_features=['user_id', 'item_id', 'category'],
            high_cardinality_threshold=10,  # Low threshold to trigger OTHER
        )
        
        # Some values should be encoded as OTHER
        self.assertTrue(processor._fitted)
    
    def test_transform(self):
        """Test transform method."""
        processor = DataProcessor()
        data = processor.fit_transform(
            self.X, self.y,
            sparse_features=['user_id', 'item_id', 'category'],
        )
        
        # Transform new data
        X_new = self.X.iloc[:10].copy()
        X_transformed = processor.transform(X_new)
        
        self.assertEqual(len(X_transformed), 10)


class TestSearchModules(unittest.TestCase):
    """Test search modules (smoke tests)."""
    
    def test_import_ml_search(self):
        """Test MLSearch import."""
        from ctr_auto_hyperopt.search.ml_search import MLSearch
        searcher = MLSearch(models=['lgbm'], time_budget=10)
        self.assertIsNotNone(searcher)
    
    def test_import_dl_search(self):
        """Test DLSearch import."""
        from ctr_auto_hyperopt.search.dl_search import DLSearch
        searcher = DLSearch(models=['DeepFM'], time_budget=10)
        self.assertIsNotNone(searcher)
    
    def test_import_mlgb_search(self):
        """Test MLGBSearch import."""
        try:
            from ctr_auto_hyperopt.search.mlgb_search import MLGBSearch
            searcher = MLGBSearch(models=['MLP'], time_budget=10)
            self.assertIsNotNone(searcher)
        except ImportError:
            self.skipTest("MLGB not available")


class TestAutoCTR(unittest.TestCase):
    """Test AutoCTR main class."""
    
    def test_init(self):
        """Test AutoCTR initialization."""
        from ctr_auto_hyperopt import AutoCTR
        
        auto_ctr = AutoCTR(
            task='binary',
            metric='auc',
            time_budget=60,
            search_ml=True,
            search_dl=False,
        )
        
        self.assertEqual(auto_ctr.task, 'binary')
        self.assertEqual(auto_ctr.metric, 'auc')
        self.assertIn('lgbm', auto_ctr.ml_models)
    
    def test_model_pool_filter(self):
        """Test include/exclude model filtering."""
        from ctr_auto_hyperopt import AutoCTR
        
        auto_ctr = AutoCTR(
            include_models=['lgbm', 'xgboost'],
            exclude_models=['xgboost'],
        )
        
        self.assertIn('lgbm', auto_ctr.ml_models)
        self.assertNotIn('xgboost', auto_ctr.ml_models)


if __name__ == '__main__':
    unittest.main(verbosity=2)
