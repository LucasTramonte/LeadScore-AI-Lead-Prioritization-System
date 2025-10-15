"""
Model training module for the LeadScore AI system.
Implements the training pipeline based on notebook analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, classification_report
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path

from ..data.feature_engineering import SimpleOutlierCapper, FeatureEngineer

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles model training and selection based on notebook analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ModelTrainer.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or {}
        self.feature_engineer = FeatureEngineer(config)
        self.best_model = None
        self.best_pipeline = None
        self.training_stats = None
        
    def _get_model_configurations(self) -> Dict[str, Any]:
        """
        Get model configurations based on notebook analysis.
        
        Returns:
            Dictionary with model configurations
        """
        return {
            'CatBoost': CatBoostClassifier(
                iterations=500,
                learning_rate=0.1,
                depth=6,
                loss_function='Logloss',
                eval_metric='AUC',
                random_seed=42,
                verbose=False
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced',
                C=1.0
            )
        }
    
    def _create_preprocessing_pipeline(self, feature_sets: Dict[str, List[str]]) -> ColumnTransformer:
        """
        Create preprocessing pipeline based on feature types.
        
        Args:
            feature_sets: Dictionary with feature categories
            
        Returns:
            ColumnTransformer for preprocessing
        """
        numeric_features = feature_sets['numeric_features']
        categorical_features = feature_sets['categorical_features']
        binary_features = feature_sets['binary_features']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('outlier_capper', SimpleOutlierCapper(multiplier=1.5)),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline([
                    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
                ]), categorical_features),
                ('bin', 'passthrough', binary_features)
            ]
        )
        
        return preprocessor
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'converted') -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Prepare data for training with feature engineering.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            
        Returns:
            Tuple of (features DataFrame, target Series, training statistics)
        """
        logger.info("Starting data preparation...")
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Apply feature engineering
        X_enhanced, reference_stats = self.feature_engineer.fit_transform(X)
        
        # Get final feature set
        feature_sets = self.feature_engineer.get_final_feature_set()
        all_features = (
            feature_sets['numeric_features'] + 
            feature_sets['categorical_features'] + 
            feature_sets['binary_features']
        )
        
        # Select final features
        X_final = X_enhanced[all_features]
        
        training_stats = {
            'total_samples': len(X_final),
            'total_features': len(all_features),
            'conversion_rate': y.mean(),
            'feature_sets': feature_sets,
            'reference_stats': reference_stats
        }
        
        logger.info(f"Data preparation complete. Shape: {X_final.shape}, Conversion rate: {y.mean():.1%}")
        return X_final, y, training_stats
    
    def train_and_compare_models(self, X: pd.DataFrame, y: pd.Series, 
                                test_size: float = 0.2) -> Dict[str, Dict[str, Any]]:
        """
        Train and compare multiple models.
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Test set size for evaluation
            
        Returns:
            Dictionary with model comparison results
        """
        logger.info("Starting model training and comparison...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Get feature sets and create preprocessor
        feature_sets = self.feature_engineer.get_final_feature_set()
        preprocessor = self._create_preprocessing_pipeline(feature_sets)
        
        # Get model configurations
        models = self._get_model_configurations()
        
        # Train and evaluate each model
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Evaluate model
            train_pred_proba = pipeline.predict_proba(X_train)[:, 1]
            test_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            
            train_auc = roc_auc_score(y_train, train_pred_proba)
            test_auc = roc_auc_score(y_test, test_pred_proba)
            overfitting_gap = train_auc - test_auc
            
            # Cross-validation
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1
            )
            
            # Store results
            results[model_name] = {
                'pipeline': pipeline,
                'train_auc': train_auc,
                'test_auc': test_auc,
                'overfitting_gap': overfitting_gap,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_predictions': test_pred_proba
            }
            
            logger.info(f"{model_name} - Test AUC: {test_auc:.4f}, CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return results
    
    def select_best_model(self, model_results: Dict[str, Dict[str, Any]]) -> Tuple[str, Pipeline]:
        """
        Select the best model based on test AUC score.
        
        Args:
            model_results: Results from model comparison
            
        Returns:
            Tuple of (best model name, best pipeline)
        """
        # Find best model by test AUC
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['test_auc'])
        best_pipeline = model_results[best_model_name]['pipeline']
        
        logger.info(f"Best model selected: {best_model_name}")
        logger.info(f"Best model test AUC: {model_results[best_model_name]['test_auc']:.4f}")
        
        return best_model_name, best_pipeline
    
    def optimize_thresholds(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[float, float]:
        """
        Optimize classification thresholds for High/Medium/Low priority.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Tuple of (high_threshold, medium_threshold)
        """
        logger.info("Optimizing classification thresholds...")
        
        thresholds_to_test = np.arange(0.1, 0.9, 0.05)
        best_score = 0
        best_thresholds = (0.7, 0.4)  # Default
        
        for high_thresh in thresholds_to_test:
            for med_thresh in thresholds_to_test:
                if med_thresh >= high_thresh:
                    continue
                
                # Assign priorities
                priorities = np.where(y_pred_proba >= high_thresh, 'High',
                             np.where(y_pred_proba >= med_thresh, 'Medium', 'Low'))
                
                # Calculate high priority metrics
                high_mask = priorities == 'High'
                if high_mask.sum() > 0:
                    high_conv_rate = y_true[high_mask].mean()
                    high_volume_pct = high_mask.sum() / len(y_true) * 100
                    
                    # Score based on conversion rate and reasonable volume
                    if 10 <= high_volume_pct <= 30:  # Reasonable volume range
                        score = high_conv_rate
                        if score > best_score:
                            best_score = score
                            best_thresholds = (high_thresh, med_thresh)
        
        logger.info(f"Optimal thresholds: High={best_thresholds[0]:.3f}, Medium={best_thresholds[1]:.3f}")
        return best_thresholds
    
    def train_final_model(self, df: pd.DataFrame, target_column: str = 'converted') -> Dict[str, Any]:
        """
        Train the final model with full pipeline.
        
        Args:
            df: Full training DataFrame
            target_column: Name of target column
            
        Returns:
            Dictionary with training results and model artifacts
        """
        logger.info("Starting final model training...")
        
        # Prepare data
        X, y, training_stats = self.prepare_data(df, target_column)
        
        # Train and compare models
        model_results = self.train_and_compare_models(X, y)
        
        # Select best model
        best_model_name, best_pipeline = self.select_best_model(model_results)
        
        # Get test predictions for threshold optimization
        test_predictions = model_results[best_model_name]['test_predictions']
        
        # Split data to get test labels for threshold optimization
        _, _, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Optimize thresholds
        optimal_thresholds = self.optimize_thresholds(y_test.values, test_predictions)
        
        # Store results
        self.best_model = best_model_name
        self.best_pipeline = best_pipeline
        self.training_stats = training_stats
        
        # Create final results
        final_results = {
            'best_model_name': best_model_name,
            'best_pipeline': best_pipeline,
            'model_comparison': {
                name: {
                    'train_auc': results['train_auc'],
                    'test_auc': results['test_auc'],
                    'cv_mean': results['cv_mean'],
                    'cv_std': results['cv_std'],
                    'overfitting_gap': results['overfitting_gap']
                }
                for name, results in model_results.items()
            },
            'optimal_thresholds': {
                'high': optimal_thresholds[0],
                'medium': optimal_thresholds[1]
            },
            'training_stats': training_stats,
            'feature_engineer': self.feature_engineer
        }
        
        logger.info("Final model training completed successfully")
        return final_results
    
    def get_feature_importance(self, pipeline: Pipeline, feature_sets: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Extract feature importance from trained model.
        
        Args:
            pipeline: Trained pipeline
            feature_sets: Dictionary with feature categories
            
        Returns:
            DataFrame with feature importance
        """
        try:
            # Get feature names after preprocessing
            numeric_features = feature_sets['numeric_features']
            categorical_features = feature_sets['categorical_features']
            binary_features = feature_sets['binary_features']
            
            # Get categorical feature names after one-hot encoding
            cat_feature_names = (
                pipeline.named_steps['preprocessor']
                .named_transformers_['cat']
                .named_steps['onehot']
                .get_feature_names_out(categorical_features)
            )
            
            all_feature_names = numeric_features + list(cat_feature_names) + binary_features
            
            # Get feature importance
            if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
                importance_values = pipeline.named_steps['classifier'].feature_importances_
            else:
                # For linear models, use absolute coefficients
                importance_values = np.abs(pipeline.named_steps['classifier'].coef_[0])
            
            # Create DataFrame
            feature_importance = pd.DataFrame({
                'feature': all_feature_names,
                'importance': importance_values
            }).sort_values('importance', ascending=False)
            
            return feature_importance
            
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
            return pd.DataFrame()
