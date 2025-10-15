"""
Training service for LeadScore AI system.
Orchestrates the complete training pipeline without code duplication.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import optuna
from optuna.samplers import TPESampler

from ..core.config_manager import get_config
from ..data.data_loader import DataLoader
from ..data.data_validator import DataValidator
from ..data.feature_engineering import FeatureEngineer, SimpleOutlierCapper
from ..model.model_persistence import ModelPersistence

logger = logging.getLogger(__name__)


class TrainingService:
    """
    High-level training service that orchestrates the complete training pipeline.
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
        self.feature_engineer = FeatureEngineer(self.config.get('feature_engineering', {}))
        self.model_persistence = ModelPersistence(self.config.get_model_config())
        
        self.best_features = None
        self.feature_importance_scores = {}
        
    def get_feature_sets(self) -> Dict[str, Dict[str, List[str]]]:
        """Define different feature sets to test."""
        
        # Original notebook features (baseline)
        baseline_features = {
            'numeric': ['days_since_first_touch', 'engagement_score', 'lead_quality_score'],
            'categorical': ['lead_source'],
            'binary': ['is_recent_lead']
        }
        
        # Extended feature set (more features)
        extended_features = {
            'numeric': [
                'days_since_first_touch', 'email_open_rate', 'advanced_engagement_score',
                'company_quality_score', 'lead_quality_score', 'engagement_score',
                'company_size_score', 'time_decay_factor'
            ],
            'categorical': ['segmento', 'contact_role', 'lead_source', 'crm_stage'],
            'binary': [
                'exporta', 'download_whitepaper', 'demo_solicitada', 'urgencia_projeto',
                'is_engaged_prospect', 'is_recent_lead', 'is_warm_lead',
                'high_value_decision_maker', 'warm_lead_high_engagement', 'is_urgent_lead'
            ]
        }
        
        # Minimal feature set (most important only)
        minimal_features = {
            'numeric': ['lead_quality_score', 'days_since_first_touch', 'advanced_engagement_score'],
            'categorical': ['segmento', 'lead_source'],
            'binary': ['is_engaged_prospect', 'exporta']
        }
        
        # Engagement-focused features
        engagement_features = {
            'numeric': [
                'email_open_rate', 'email_response_rate', 'meeting_conversion_rate',
                'advanced_engagement_score', 'engagement_score', 'total_touchpoints',
                'proactive_signals', 'days_since_first_touch'
            ],
            'categorical': ['lead_source', 'contact_role'],
            'binary': [
                'download_whitepaper', 'demo_solicitada', 'is_engaged_prospect',
                'warm_lead_high_engagement', 'is_urgent_lead'
            ]
        }
        
        return {
            'baseline': baseline_features,
            'extended': extended_features,
            'minimal': minimal_features,
            'engagement': engagement_features
        }
    
    def create_preprocessor(self, features: Dict[str, List[str]]) -> ColumnTransformer:
        """Create preprocessing pipeline for given features."""
        return ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('outlier_capper', SimpleOutlierCapper(
                        multiplier=self.config.get('model.outlier_multiplier', 1.5)
                    )),
                    ('scaler', StandardScaler())
                ]), features['numeric']),
                ('cat', Pipeline([
                    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
                ]), features['categorical']),
                ('bin', 'passthrough', features['binary'])
            ]
        )
    
    def evaluate_model_comprehensive(self, pipeline: Pipeline, model_name: str, 
                                   X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                   y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        pipeline.fit(X_train, y_train)
        
        train_pred_proba = pipeline.predict_proba(X_train)[:, 1]
        test_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        train_auc = roc_auc_score(y_train, train_pred_proba)
        test_auc = roc_auc_score(y_test, test_pred_proba)
        overfitting_gap = train_auc - test_auc

        cv_scores = cross_val_score(
            pipeline, X_train, y_train, 
            cv=self.config.get('model.cv_folds', 5), 
            scoring='roc_auc', n_jobs=-1
        )
        
        return {
            'model_name': model_name,
            'pipeline': pipeline,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'overfitting_gap': overfitting_gap,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_pred_proba': test_pred_proba
        }
    
    def test_feature_sets(self, X_train_enhanced: pd.DataFrame, X_test_enhanced: pd.DataFrame, 
                         y_train: pd.Series, y_test: pd.Series) -> Tuple[Dict[str, Any], str]:
        """Test different feature combinations."""
        logger.info("Testing different feature sets...")
        
        feature_sets = self.get_feature_sets()
        results = {}
        
        for set_name, features in feature_sets.items():
            logger.info(f"Testing {set_name} feature set...")
            
            # Combine all features
            all_features = features['numeric'] + features['categorical'] + features['binary']
            
            # Select features
            X_train_subset = X_train_enhanced[all_features]
            X_test_subset = X_test_enhanced[all_features]
            
            # Create preprocessor and pipeline
            preprocessor = self.create_preprocessor(features)
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(
                    n_estimators=50, max_depth=6, 
                    random_state=self.config.get('model.random_state', 42), 
                    class_weight='balanced'
                ))
            ])
            
            result = self.evaluate_model_comprehensive(
                pipeline, f'RF_{set_name}', X_train_subset, X_test_subset, y_train, y_test
            )
            
            results[set_name] = {
                'features': features,
                'all_features': all_features,
                'n_features': len(all_features),
                'test_auc': result['test_auc'],
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std'],
                'overfitting_gap': result['overfitting_gap']
            }
            
            logger.info(f"  {set_name}: {len(all_features)} features, AUC: {result['test_auc']:.4f}")
        
        # Find best feature set
        best_set = max(results.keys(), key=lambda x: results[x]['test_auc'])
        logger.info(f"Best feature set: {best_set} with AUC: {results[best_set]['test_auc']:.4f}")
        
        return results, best_set
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                y_train: pd.Series, y_test: pd.Series, 
                                features: Dict[str, List[str]], n_trials: int = 30) -> Tuple[Dict[str, Any], float]:
        """Optimize hyperparameters using Optuna."""
        logger.info(f"Optimizing hyperparameters with {n_trials} trials...")
        
        preprocessor = self.create_preprocessor(features)
        
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
        
        return study.best_params, study.best_value
    
    def train_final_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                          y_train: pd.Series, y_test: pd.Series, 
                          features: Dict[str, List[str]], best_params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Train final models with optimized parameters."""
        logger.info("Training final optimized models...")
        
        preprocessor = self.create_preprocessor(features)
        
        # Create optimized model
        model_type = best_params['model_type']
        model_params = {k: v for k, v in best_params.items() if k != 'model_type'}
        
        if model_type == 'RandomForest':
            model = RandomForestClassifier(
                **model_params, 
                random_state=self.config.get('model.random_state', 42), 
                class_weight='balanced'
            )
        elif model_type == 'GradientBoosting':
            model = GradientBoostingClassifier(
                **model_params, 
                random_state=self.config.get('model.random_state', 42)
            )
        else:  # CatBoost
            model = CatBoostClassifier(
                **model_params, 
                random_seed=self.config.get('model.random_state', 42), 
                verbose=False
            )
        
        # Create and train pipeline
        optimized_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Evaluate optimized model
        optimized_result = self.evaluate_model_comprehensive(
            optimized_pipeline, f'Optimized_{model_type}', X_train, X_test, y_train, y_test
        )
        
        # Also train baseline models for comparison
        baseline_models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=6, 
                random_state=self.config.get('model.random_state', 42), 
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6, 
                random_state=self.config.get('model.random_state', 42)
            ),
            'CatBoost': CatBoostClassifier(
                iterations=500, learning_rate=0.1, depth=6, 
                random_seed=self.config.get('model.random_state', 42), 
                verbose=False
            ),
            'Logistic Regression': LogisticRegression(
                random_state=self.config.get('model.random_state', 42), 
                max_iter=1000, class_weight='balanced'
            )
        }
        
        baseline_results = {}
        for name, model in baseline_models.items():
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            result = self.evaluate_model_comprehensive(
                pipeline, name, X_train, X_test, y_train, y_test
            )
            baseline_results[name] = result
        
        return optimized_result, baseline_results
    
    def run_training_pipeline(self, model_name: Optional[str] = None, 
                            n_trials: int = 50) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            model_name: Name for the saved model
            n_trials: Number of optimization trials
            
        Returns:
            Training results summary
        """
        logger.info("Starting LeadScore AI optimized model training...")
        
        try:
            # Load data
            logger.info("Loading training data...")
            df = self.data_loader.load_default_data()
            logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
            
            # Validate data
            logger.info("Validating data quality...")
            is_valid, validation_errors = self.data_validator.validate_all(df)
            
            if not is_valid:
                logger.warning("Data validation issues found:")
                for category, errors in validation_errors.items():
                    if errors:
                        logger.warning(f"  {category}: {errors}")
            
            # Prepare data
            X_initial = df.drop('converted', axis=1)
            y = df['converted']
            
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X_initial, y, 
                test_size=self.config.get('model.test_size', 0.2), 
                random_state=self.config.get('model.random_state', 42), 
                stratify=y
            )
            
            logger.info(f"Dataset summary:")
            logger.info(f"  - Total records: {len(df):,}")
            logger.info(f"  - Training set: {X_train_raw.shape[0]} samples")
            logger.info(f"  - Test set: {X_test_raw.shape[0]} samples")
            logger.info(f"  - Conversion rate: {y.mean():.1%}")
            
            # Apply feature engineering
            X_train_enhanced, train_stats = self.feature_engineer.fit_transform(X_train_raw)
            X_test_enhanced = self.feature_engineer.transform(X_test_raw)
            
            logger.info(f"Enhanced features created. New shape: {X_train_enhanced.shape}")
            
            # Step 1: Test different feature sets
            feature_results, best_feature_set = self.test_feature_sets(
                X_train_enhanced, X_test_enhanced, y_train, y_test
            )
            
            # Step 2: Get best features
            best_features = feature_results[best_feature_set]['features']
            all_features = feature_results[best_feature_set]['all_features']
            
            # Prepare data with best features
            X_train_final = X_train_enhanced[all_features]
            X_test_final = X_test_enhanced[all_features]
            
            # Step 3: Optimize hyperparameters
            best_params, best_cv_score = self.optimize_hyperparameters(
                X_train_final, X_test_final, y_train, y_test, best_features, n_trials
            )
            
            # Step 4: Train final models
            optimized_result, baseline_results = self.train_final_models(
                X_train_final, X_test_final, y_train, y_test, best_features, best_params
            )
            
            # Find best overall model
            all_results = {optimized_result['model_name']: optimized_result}
            all_results.update(baseline_results)
            
            best_overall = max(all_results.keys(), key=lambda x: all_results[x]['test_auc'])
            best_result = all_results[best_overall]
            
            # Save the best model
            if model_name is None:
                model_name = f"leadscore_{best_overall.lower().replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Prepare training results for saving
            training_results_for_save = {
                'best_model_name': best_overall,
                'best_pipeline': best_result['pipeline'],
                'feature_engineer': self.feature_engineer,
                'model_comparison': {name: result for name, result in all_results.items()},
                'optimal_thresholds': {'high': 0.65, 'medium': 0.10},  # Default thresholds
                'training_stats': {
                    'total_samples': len(df),
                    'total_features': len(all_features),
                    'conversion_rate': y.mean()
                }
            }
            
            model_path = self.model_persistence.save_model_artifacts(training_results_for_save, model_name)
            
            # Return comprehensive results
            return {
                'best_model_name': best_overall,
                'model_saved_as': model_name,
                'final_auc': best_result['test_auc'],
                'selected_feature_set': best_feature_set,
                'selected_features': best_features,
                'all_features': all_features,
                'best_params': best_params,
                'feature_set_results': feature_results,
                'model_results': all_results,
                'training_stats': {
                    'total_samples': len(df),
                    'training_samples': len(X_train_raw),
                    'test_samples': len(X_test_raw),
                    'conversion_rate': y.mean(),
                    'n_features': len(all_features)
                }
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise
