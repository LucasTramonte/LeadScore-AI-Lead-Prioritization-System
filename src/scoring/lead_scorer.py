"""
Lead scoring engine for the LeadScore AI system.
Provides real-time lead scoring and priority classification.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
from pathlib import Path

from ..data.feature_engineering import FeatureEngineer
from ..data.data_validator import DataValidator
from ..model.model_persistence import ModelPersistence

logger = logging.getLogger(__name__)


class LeadScorer:
    """
    Real-time lead scoring engine that loads a trained model and scores individual leads.
    """
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LeadScorer.
        
        Args:
            model_path: Path to trained model or model name
            config: Configuration dictionary
        """
        self.config = config or {}
        self.model_artifacts = None
        self.feature_engineer = None
        self.validator = DataValidator(config)
        self.is_loaded = False
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load trained model and associated artifacts.
        
        Args:
            model_path: Path to model directory or model name
        """
        logger.info(f"Loading model from {model_path}")
        
        try:
            # Load model artifacts
            persistence = ModelPersistence(self.config)
            self.model_artifacts = persistence.load_model_artifacts(model_path)
            
            # Initialize feature engineer with loaded reference stats
            self.feature_engineer = FeatureEngineer(self.config)
            self.feature_engineer.reference_stats = self.model_artifacts['feature_artifacts']['reference_stats']
            
            self.is_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def score_lead(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single lead and return probability and priority.
        
        Args:
            lead_data: Dictionary with lead information
            
        Returns:
            Dictionary with scoring results
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert to DataFrame
        df = pd.DataFrame([lead_data])
        
        # Validate data
        is_valid, validation_errors = self.validator.validate_all(df)
        if not is_valid:
            logger.warning(f"Data validation issues: {validation_errors}")
        
        # Apply feature engineering
        try:
            df_enhanced = self.feature_engineer.transform(df)
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise ValueError(f"Feature engineering failed: {str(e)}")
        
        # Select features for prediction
        feature_sets = self.model_artifacts['feature_artifacts']['feature_sets']
        all_features = (
            feature_sets['numeric_features'] + 
            feature_sets['categorical_features'] + 
            feature_sets['binary_features']
        )
        
        # Check for missing features
        missing_features = [f for f in all_features if f not in df_enhanced.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        X = df_enhanced[all_features]
        
        # Get prediction
        pipeline = self.model_artifacts['pipeline']
        probability = pipeline.predict_proba(X)[0, 1]
        
        # Assign priority
        thresholds = self.model_artifacts['metadata']['optimal_thresholds']
        priority = self._assign_priority(probability, thresholds)
        
        # Create result
        result = {
            'lead_id': lead_data.get('lead_id', 'unknown'),
            'conversion_probability': float(probability),
            'priority': priority,
            'confidence_score': self._calculate_confidence(probability, thresholds),
            'thresholds_used': thresholds,
            'model_info': {
                'model_type': self.model_artifacts['metadata'].get('best_model_type', 'Unknown'),
                'model_version': self.model_artifacts['metadata'].get('version', '1.0.0'),
                'training_date': self.model_artifacts['metadata'].get('training_timestamp', 'Unknown')
            },
            'feature_contributions': self._get_feature_contributions(X, pipeline, all_features),
            'validation_warnings': validation_errors if not is_valid else []
        }
        
        return result
    
    def score_leads_batch(self, leads_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score multiple leads efficiently.
        
        Args:
            leads_data: List of lead dictionaries
            
        Returns:
            List of scoring results
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not leads_data:
            return []
        
        logger.info(f"Scoring {len(leads_data)} leads")
        
        # Convert to DataFrame
        df = pd.DataFrame(leads_data)
        
        # Validate data
        is_valid, validation_errors = self.validator.validate_all(df)
        if not is_valid:
            logger.warning(f"Batch data validation issues: {validation_errors}")
        
        # Apply feature engineering
        try:
            df_enhanced = self.feature_engineer.transform(df)
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise ValueError(f"Feature engineering failed: {str(e)}")
        
        # Select features for prediction
        feature_sets = self.model_artifacts['feature_artifacts']['feature_sets']
        all_features = (
            feature_sets['numeric_features'] + 
            feature_sets['categorical_features'] + 
            feature_sets['binary_features']
        )
        
        X = df_enhanced[all_features]
        
        # Get predictions
        pipeline = self.model_artifacts['pipeline']
        probabilities = pipeline.predict_proba(X)[:, 1]
        
        # Assign priorities
        thresholds = self.model_artifacts['metadata']['optimal_thresholds']
        priorities = [self._assign_priority(prob, thresholds) for prob in probabilities]
        
        # Create results
        results = []
        for i, (lead_data, probability, priority) in enumerate(zip(leads_data, probabilities, priorities)):
            result = {
                'lead_id': lead_data.get('lead_id', f'lead_{i}'),
                'conversion_probability': float(probability),
                'priority': priority,
                'confidence_score': self._calculate_confidence(probability, thresholds),
                'thresholds_used': thresholds,
                'model_info': {
                    'model_type': self.model_artifacts['metadata'].get('best_model_type', 'Unknown'),
                    'model_version': self.model_artifacts['metadata'].get('version', '1.0.0'),
                    'training_date': self.model_artifacts['metadata'].get('training_timestamp', 'Unknown')
                }
            }
            results.append(result)
        
        return results
    
    def get_feature_requirements(self) -> Dict[str, Any]:
        """
        Get information about required features for scoring.
        
        Returns:
            Dictionary with feature requirements
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        feature_sets = self.model_artifacts['feature_artifacts']['feature_sets']
        validation_rules = self.validator.validation_rules
        
        requirements = {
            'required_features': {
                'numeric': feature_sets['numeric_features'],
                'categorical': feature_sets['categorical_features'],
                'binary': feature_sets['binary_features']
            },
            'feature_descriptions': {
                'segmento': 'Industry segment (e.g., Energia & Utilities, Químicos & Plásticos)',
                'faturamento_anual_milhoes': 'Annual revenue in millions of Brazilian Reais',
                'numero_SKUs': 'Number of SKUs (product complexity indicator)',
                'exporta': 'Export status (0/1)',
                'margem_media_setor': 'Average sector margin percentage',
                'contact_role': 'Contact person role (e.g., Diretor de Operações)',
                'lead_source': 'Lead source (e.g., Evento Setorial, Indicação de Cliente)',
                'crm_stage': 'CRM pipeline stage',
                'emails_enviados': 'Number of emails sent',
                'emails_abertos': 'Number of emails opened',
                'emails_respondidos': 'Number of emails responded to',
                'reunioes_realizadas': 'Number of meetings held',
                'download_whitepaper': 'Whitepaper download (0/1)',
                'demo_solicitada': 'Demo requested (0/1)',
                'problemas_reportados_precificacao': 'Pricing problems reported (0/1)',
                'urgencia_projeto': 'Project urgency (0/1)',
                'days_since_first_touch': 'Days since first contact'
            },
            'validation_rules': {
                'categorical_values': validation_rules['categorical_values'],
                'numeric_ranges': validation_rules['numeric_ranges'],
                'binary_columns': validation_rules['binary_columns']
            },
            'example_lead': self._get_example_lead()
        }
        
        return requirements
    
    def _assign_priority(self, probability: float, thresholds: Dict[str, float]) -> str:
        """
        Assign priority based on probability and thresholds.
        
        Args:
            probability: Conversion probability
            thresholds: Dictionary with high and medium thresholds
            
        Returns:
            Priority level (High/Medium/Low)
        """
        if probability >= thresholds['high']:
            return 'High'
        elif probability >= thresholds['medium']:
            return 'Medium'
        else:
            return 'Low'
    
    def _calculate_confidence(self, probability: float, thresholds: Dict[str, float]) -> float:
        """
        Calculate confidence score based on how far the probability is from thresholds.
        
        Args:
            probability: Conversion probability
            thresholds: Dictionary with thresholds
            
        Returns:
            Confidence score between 0 and 1
        """
        high_thresh = thresholds['high']
        medium_thresh = thresholds['medium']
        
        if probability >= high_thresh:
            # High priority: confidence based on distance from high threshold
            max_distance = 1.0 - high_thresh
            distance = probability - high_thresh
            confidence = 0.7 + 0.3 * (distance / max_distance) if max_distance > 0 else 1.0
        elif probability >= medium_thresh:
            # Medium priority: confidence based on position between thresholds
            range_size = high_thresh - medium_thresh
            distance_from_medium = probability - medium_thresh
            confidence = 0.4 + 0.3 * (distance_from_medium / range_size) if range_size > 0 else 0.55
        else:
            # Low priority: confidence based on distance from medium threshold
            distance_from_medium = medium_thresh - probability
            confidence = max(0.1, 0.4 - 0.3 * (distance_from_medium / medium_thresh)) if medium_thresh > 0 else 0.25
        
        return min(1.0, max(0.0, confidence))
    
    def _get_feature_contributions(self, X: pd.DataFrame, pipeline, feature_names: List[str]) -> Dict[str, float]:
        """
        Get feature contributions for interpretability (simplified version).
        
        Args:
            X: Feature matrix
            pipeline: Trained pipeline
            feature_names: List of feature names
            
        Returns:
            Dictionary with feature contributions
        """
        try:
            # For tree-based models, we can get feature importance
            if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
                # Get feature names after preprocessing
                preprocessor = pipeline.named_steps['preprocessor']
                
                # Transform the data to get the actual feature names
                X_transformed = preprocessor.transform(X)
                
                # Get feature importance
                importance = pipeline.named_steps['classifier'].feature_importances_
                
                # Map back to original features (simplified)
                contributions = {}
                for i, feature in enumerate(feature_names[:len(importance)]):
                    contributions[feature] = float(importance[i]) if i < len(importance) else 0.0
                
                return contributions
            else:
                # For linear models, return empty dict for now
                return {}
                
        except Exception as e:
            logger.warning(f"Could not calculate feature contributions: {str(e)}")
            return {}
    
    def _get_example_lead(self) -> Dict[str, Any]:
        """
        Get an example lead for documentation.
        
        Returns:
            Dictionary with example lead data
        """
        return {
            'lead_id': 'example_001',
            'segmento': 'Energia & Utilities',
            'faturamento_anual_milhoes': 75.0,
            'numero_SKUs': 150,
            'exporta': 1,
            'margem_media_setor': 18.5,
            'contact_role': 'Diretor de Operações',
            'lead_source': 'Indicação de Cliente',
            'crm_stage': 'Qualificado Marketing',
            'emails_enviados': 5,
            'emails_abertos': 4,
            'emails_respondidos': 2,
            'reunioes_realizadas': 2,
            'download_whitepaper': 1,
            'demo_solicitada': 1,
            'problemas_reportados_precificacao': 1,
            'urgencia_projeto': 1,
            'days_since_first_touch': 45
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        metadata = self.model_artifacts['metadata']
        
        return {
            'model_name': metadata.get('model_name', 'Unknown'),
            'model_type': metadata.get('best_model_type', 'Unknown'),
            'version': metadata.get('version', '1.0.0'),
            'training_date': metadata.get('training_timestamp', 'Unknown'),
            'thresholds': metadata.get('optimal_thresholds', {'high': 0.7, 'medium': 0.4}),
            'training_stats': metadata.get('training_stats', {}),
            'model_comparison': metadata.get('model_comparison', {}),
            'framework_versions': metadata.get('framework_versions', {})
        }
