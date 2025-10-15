"""
Training service for LeadScore AI system.
Orchestrates the complete training pipeline with hyperparameter optimization.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import optuna
from optuna.samplers import TPESampler

from ..core.config_manager import get_config
from ..data.data_loader import DataLoader
from ..data.data_validator import DataValidator
from ..model.model_trainer import ModelTrainer
from ..model.model_persistence import ModelPersistence

logger = logging.getLogger(__name__)


class TrainingService:
    """
    High-level training service that orchestrates the complete training pipeline with optimization.
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize training service.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager or get_config()
        self.data_loader = DataLoader(self.config.get_data_config())
        self.data_validator = DataValidator(self.config.get_data_config())
        self.model_trainer = ModelTrainer(self.config.to_dict())
        self.model_persistence = ModelPersistence(self.config.get_model_config())
        
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                feature_sets: Dict[str, List[str]], n_trials: int = 30) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        logger.info(f"Optimizing hyperparameters with {n_trials} trials...")
        
        # Use the model trainer's preprocessing pipeline
        preprocessor = self.model_trainer._create_preprocessing_pipeline(feature_sets)
        
        def objective(trial):
            # Suggest model type
            model_type = trial.suggest_categorical('model_type', ['RandomForest', 'GradientBoosting', 'CatBoost'])
            
            if model_type == 'RandomForest':
                model = RandomForestClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                    max_depth=trial.suggest_int('max_depth', 3, 15),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                    max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    random_state=self.config.get('model.random_state', 42),
                    class_weight='balanced'
                )
            elif model_type == 'GradientBoosting':
                model = GradientBoostingClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    max_depth=trial.suggest_int('max_depth', 3, 10),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                    random_state=self.config.get('model.random_state', 42)
                )
            else:  # CatBoost
                model = CatBoostClassifier(
                    iterations=trial.suggest_int('iterations', 100, 500),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    depth=trial.suggest_int('depth', 3, 10),
                    l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1, 10),
                    random_seed=self.config.get('model.random_state', 42),
                    verbose=False
                )
            
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # Cross-validation
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, 
                cv=self.config.get('model.cv_folds', 5), 
                scoring='roc_auc', n_jobs=-1
            )
            return cv_scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best hyperparameters found:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"Best CV AUC: {study.best_value:.4f}")
        
        return study.best_params
    
    def run_training_pipeline(self, model_name: Optional[str] = None, 
                            n_trials: int = 50) -> Dict[str, Any]:
        """
        Run the complete training pipeline with hyperparameter optimization.
        
        Args:
            model_name: Name for the saved model
            n_trials: Number of optimization trials
            
        Returns:
            Training results summary
        """
        logger.info("Starting LeadScore AI optimized model training...")
        
        try:
            # Load and validate data
            logger.info("Loading training data...")
            df = self.data_loader.load_default_data()
            logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
            
            is_valid, validation_errors = self.data_validator.validate_all(df)
            if not is_valid:
                logger.warning("Data validation issues found:")
                for category, errors in validation_errors.items():
                    if errors:
                        logger.warning(f"  {category}: {errors}")
            
            # Use ModelTrainer for basic training
            logger.info("Training baseline models...")
            baseline_results = self.model_trainer.train_final_model(df)
            
            # Get prepared data for hyperparameter optimization
            X, y, training_stats = self.model_trainer.prepare_data(df)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Get feature sets from the trained model
            feature_sets = self.model_trainer.feature_engineer.get_final_feature_set()
            
            # Optimize hyperparameters
            logger.info("Optimizing hyperparameters...")
            best_params = self.optimize_hyperparameters(X_train, y_train, feature_sets, n_trials)
            
            # Create optimized model
            model_type = best_params['model_type']
            model_params = {k: v for k, v in best_params.items() if k != 'model_type'}
            
            if model_type == 'RandomForest':
                optimized_model = RandomForestClassifier(
                    **model_params, 
                    random_state=42, 
                    class_weight='balanced'
                )
            elif model_type == 'GradientBoosting':
                optimized_model = GradientBoostingClassifier(
                    **model_params, 
                    random_state=42
                )
            else:  # CatBoost
                optimized_model = CatBoostClassifier(
                    **model_params, 
                    random_seed=42, 
                    verbose=False
                )
            
            # Create optimized pipeline
            preprocessor = self.model_trainer._create_preprocessing_pipeline(feature_sets)
            optimized_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', optimized_model)
            ])
            
            # Train and evaluate optimized model
            optimized_pipeline.fit(X_train, y_train)
            optimized_test_proba = optimized_pipeline.predict_proba(X_test)[:, 1]
            optimized_auc = roc_auc_score(y_test, optimized_test_proba)
            
            logger.info(f"Optimized {model_type} AUC: {optimized_auc:.4f}")
            
            # Compare with baseline
            best_baseline_auc = max([result['test_auc'] for result in baseline_results['model_comparison'].values()])
            
            # Use optimized model if it's better
            if optimized_auc > best_baseline_auc:
                logger.info(f"Optimized model is better ({optimized_auc:.4f} vs {best_baseline_auc:.4f})")
                final_pipeline = optimized_pipeline
                final_model_name = f"Optimized_{model_type}"
                final_auc = optimized_auc
            else:
                logger.info(f"Baseline model is better ({best_baseline_auc:.4f} vs {optimized_auc:.4f})")
                final_pipeline = baseline_results['best_pipeline']
                final_model_name = baseline_results['best_model_name']
                final_auc = best_baseline_auc
            
            # Save the best model
            if model_name is None:
                model_name = f"leadscore_{final_model_name.lower().replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Prepare final results for saving
            final_results = {
                'best_model_name': final_model_name,
                'best_pipeline': final_pipeline,
                'model_comparison': baseline_results['model_comparison'],
                'optimal_thresholds': baseline_results['optimal_thresholds'],
                'training_stats': baseline_results['training_stats'],
                'feature_engineer': self.model_trainer.feature_engineer
            }
            
            # Add optimized model results if used
            if optimized_auc > best_baseline_auc:
                final_results['model_comparison'][final_model_name] = {
                    'test_auc': optimized_auc,
                    'hyperparameters': best_params
                }
            
            # Save model
            model_path = self.model_persistence.save_model_artifacts(final_results, model_name)
            
            logger.info(f"Training completed successfully. Model saved as: {model_name}")
            
            return {
                'best_model_name': final_model_name,
                'model_saved_as': model_name,
                'final_auc': final_auc,
                'baseline_auc': best_baseline_auc,
                'optimized_auc': optimized_auc if optimized_auc > best_baseline_auc else None,
                'best_params': best_params,
                'model_path': model_path,
                'training_stats': baseline_results['training_stats']
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise
