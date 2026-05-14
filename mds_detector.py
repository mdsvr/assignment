"""
ML-Based MDS Detection Model

Based on the methodology in "Creating Static Data for MDS Analysis"
Implements machine learning models for MDS attack detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')


class MDSDetector:
    """
    Machine learning-based MDS attack detector.
    
    Implements multiple ML approaches for binary and multi-class
    MDS detection following the methodology in the research document.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the MDS detector.
        
        Args:
            model_type: Type of ML model to use
                Options: 'random_forest', 'gradient_boosting', 'logistic_regression',
                        'svm', 'neural_network'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the selected ML model."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        elif self.model_type == 'neural_network':
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_data(self, df: pd.DataFrame, label_col: str = 'label',
                    feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training/testing.
        
        Args:
            df: DataFrame with the dataset
            label_col: Name of the label column
            feature_cols: List of feature columns to use (if None, auto-detect)
            
        Returns:
            Tuple of (features, labels)
        """
        # Create binary labels (attack vs benign)
        attack_labels = ['msbds', 'mfbds', 'mlpds', 'mdsum']
        df_prep = df.copy()
        df_prep['is_attack'] = df_prep[label_col].isin(attack_labels).astype(int)
        
        # Select feature columns
        if feature_cols is None:
            numeric_cols = df_prep.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols 
                          if col not in ['sample_id', 'run_id', 'timestamp', 'is_attack']]
        
        self.feature_names = feature_cols
        
        X = df_prep[feature_cols].fillna(0)
        y = df_prep['is_attack']
        
        return X, y
    
    def split_by_run(self, df: pd.DataFrame, label_col: str = 'label',
                    train_ratio: float = 0.6, val_ratio: float = 0.2,
                    feature_cols: Optional[List[str]] = None) -> Tuple:
        """
        Split data by run_id to prevent data leakage (PDF Section 6.3).
        
        Args:
            df: DataFrame with the dataset
            label_col: Name of the label column
            train_ratio: Ratio of runs for training
            val_ratio: Ratio of runs for validation
            feature_cols: List of feature columns to use
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if 'run_id' not in df.columns:
            print("Warning: No run_id column found, using random split instead")
            return self.random_split(df, label_col, train_ratio, val_ratio, feature_cols)
        
        # Get unique run IDs
        unique_runs = df['run_id'].unique()
        np.random.shuffle(unique_runs)
        
        # Calculate split indices
        n_runs = len(unique_runs)
        train_end = int(n_runs * train_ratio)
        val_end = int(n_runs * (train_ratio + val_ratio))
        
        train_runs = unique_runs[:train_end]
        val_runs = unique_runs[train_end:val_end]
        test_runs = unique_runs[val_end:]
        
        # Split data by runs
        train_df = df[df['run_id'].isin(train_runs)]
        val_df = df[df['run_id'].isin(val_runs)]
        test_df = df[df['run_id'].isin(test_runs)]
        
        print(f"Split by run_id:")
        print(f"  Train: {len(train_runs)} runs ({len(train_df)} samples)")
        print(f"  Val: {len(val_runs)} runs ({len(val_df)} samples)")
        print(f"  Test: {len(test_runs)} runs ({len(test_df)} samples)")
        
        # Prepare data
        X_train, y_train = self.prepare_data(train_df, label_col, feature_cols)
        X_val, y_val = self.prepare_data(val_df, label_col, feature_cols)
        X_test, y_test = self.prepare_data(test_df, label_col, feature_cols)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def random_split(self, df: pd.DataFrame, label_col: str = 'label',
                    train_ratio: float = 0.6, val_ratio: float = 0.2,
                    feature_cols: Optional[List[str]] = None) -> Tuple:
        """
        Random split of data (use only if run_id not available).
        
        Args:
            df: DataFrame with the dataset
            label_col: Name of the label column
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            feature_cols: List of feature columns to use
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Prepare data first
        X, y = self.prepare_data(df, label_col, feature_cols)
        
        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=1-train_ratio-val_ratio, random_state=42, stratify=y
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_ratio/(train_ratio+val_ratio),
            random_state=42, stratify=y_trainval
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
             X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        Train the MDS detection model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        # Scale features (except for tree-based models)
        if self.model_type in ['logistic_regression', 'svm', 'neural_network']:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            
            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                X_val = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
        
        # Train the model
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            print(f"Validation accuracy: {val_acc:.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions (0 = benign, 1 = attack)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        # Scale if needed
        if self.model_type in ['logistic_regression', 'svm', 'neural_network']:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        # Scale if needed
        if self.model_type in ['logistic_regression', 'svm', 'neural_network']:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Add AUC if probabilities available
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
            except:
                metrics['roc_auc'] = None
        
        # Add classification report
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      n_folds: int = 5, run_ids: Optional[np.ndarray] = None) -> Dict:
        """
        Perform cross-validation following PDF Section 9.3.
        
        Args:
            X: Features
            y: Labels
            n_folds: Number of folds
            run_ids: Optional run IDs for run-level splitting
            
        Returns:
            Dictionary containing cross-validation results
        """
        if run_ids is not None:
            # Run-level CV (GroupKFold-like)
            unique_runs = np.unique(run_ids)
            np.random.shuffle(unique_runs)
            
            fold_size = len(unique_runs) // n_folds
            cv_scores = []
            
            for fold in range(n_folds):
                start_idx = fold * fold_size
                end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else len(unique_runs)
                test_runs = unique_runs[start_idx:end_idx]
                train_runs = np.setdiff1d(unique_runs, test_runs)
                
                train_mask = np.isin(run_ids, train_runs)
                test_mask = np.isin(run_ids, test_runs)
                
                X_train, X_test_fold = X[train_mask], X[test_mask]
                y_train, y_test_fold = y[train_mask], y[test_mask]
                
                # Scale if needed
                if self.model_type in ['logistic_regression', 'svm', 'neural_network']:
                    X_train_scaled = self.scaler.fit_transform(X_train)
                    X_test_scaled = self.scaler.transform(X_test_fold)
                    X_train, X_test_fold = X_train_scaled, X_test_scaled
                
                # Train and evaluate
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test_fold)
                score = accuracy_score(y_test_fold, y_pred)
                cv_scores.append(score)
            
            return {
                'cv_scores': cv_scores,
                'mean_score': np.mean(cv_scores),
                'std_score': np.std(cv_scores),
                'type': 'run_level'
            }
        else:
            # Standard stratified K-fold
            if self.model_type in ['logistic_regression', 'svm', 'neural_network']:
                X = self.scaler.fit_transform(X)
            
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.model, X, y, cv=skf, scoring='accuracy')
            
            return {
                'cv_scores': cv_scores.tolist(),
                'mean_score': np.mean(cv_scores),
                'std_score': np.std(cv_scores),
                'type': 'stratified'
            }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting feature importance")
        
        if self.feature_names is None:
            raise ValueError("Feature names not available")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names, importances))
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
            return dict(zip(self.feature_names, importances))
        else:
            return {}
    
    def print_evaluation_report(self, metrics: Dict):
        """
        Print a formatted evaluation report.
        
        Args:
            metrics: Dictionary containing evaluation metrics
        """
        print("=" * 70)
        print("MDS Detection Model Evaluation Report")
        print("=" * 70)
        print(f"\nModel Type: {self.model_type}")
        print("\nPerformance Metrics:")
        print("-" * 70)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        if metrics.get('roc_auc'):
            print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print("-" * 70)
        cm = metrics['confusion_matrix']
        print(f"                Predicted")
        print(f"                Benign    Attack")
        print(f"Actual Benign  {cm[0][0]:6d}  {cm[0][1]:6d}")
        print(f"Actual Attack  {cm[1][0]:6d}  {cm[1][1]:6d}")
        
        print("\nDetailed Classification Report:")
        print("-" * 70)
        report = metrics['classification_report']
        for label in ['0', '1']:
            print(f"Class {label}:")
            print(f"  Precision: {report[label]['precision']:.4f}")
            print(f"  Recall:    {report[label]['recall']:.4f}")
            print(f"  F1-Score:  {report[label]['f1-score']:.4f}")
        
        print("=" * 70)


class MultiClassMDSDetector(MDSDetector):
    """
    Multi-class MDS detector that distinguishes between attack variants.
    """
    
    def prepare_data(self, df: pd.DataFrame, label_col: str = 'label',
                    feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for multi-class classification.
        
        Args:
            df: DataFrame with the dataset
            label_col: Name of the label column
            feature_cols: List of feature columns to use
            
        Returns:
            Tuple of (features, labels)
        """
        # Use original labels (multi-class)
        df_prep = df.copy()
        
        # Select feature columns
        if feature_cols is None:
            numeric_cols = df_prep.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols 
                          if col not in ['sample_id', 'run_id', 'timestamp']]
        
        self.feature_names = feature_cols
        
        X = df_prep[feature_cols].fillna(0)
        y = df_prep[label_col]
        
        return X, y


if __name__ == "__main__":
    # Example usage
    from synthetic_data_generator import SyntheticDataGenerator
    
    print("Generating synthetic dataset...")
    generator = SyntheticDataGenerator()
    df = generator.generate_default_dataset(n_samples=10000)
    
    print("Training MDS detector...")
    detector = MDSDetector(model_type='random_forest')
    
    # Split by run_id
    X_train, X_val, X_test, y_train, y_val, y_test = detector.split_by_run(df)
    
    # Train
    detector.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    metrics = detector.evaluate(X_test, y_test)
    detector.print_evaluation_report(metrics)
    
    # Feature importance
    print("\nTop 10 Feature Importance:")
    print("-" * 70)
    importance = detector.get_feature_importance()
    for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{feat:30s}: {imp:.4f}")
