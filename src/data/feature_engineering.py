"""
Feature engineering module for the LeadScore AI system.
Implements the advanced feature engineering pipeline based on notebook analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from ..improvements.advanced_features import AdvancedFeatureEngineer

logger = logging.getLogger(__name__)


class SimpleOutlierCapper(BaseEstimator, TransformerMixin):
    """
    Custom transformer for handling outliers using the IQR method.
    """
    
    def __init__(self, multiplier: float = 1.5):
        """
        Initialize outlier capper.
        
        Args:
            multiplier: IQR multiplier for outlier detection (default: 1.5)
        """
        self.multiplier = multiplier
        self.bounds_ = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the outlier bounds on the training data.
        
        Args:
            X: Training data
            y: Target variable (unused)
            
        Returns:
            self
        """
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.multiplier * IQR
            upper_bound = Q3 + self.multiplier * IQR
            self.bounds_[col] = {'lower': lower_bound, 'upper': upper_bound}
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by capping outliers.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data with outliers capped
        """
        X_transformed = X.copy()
        for col in X.columns:
            if col in self.bounds_:
                lower = self.bounds_[col]['lower']
                upper = self.bounds_[col]['upper']
                X_transformed[col] = X_transformed[col].clip(lower=lower, upper=upper)
        return X_transformed


class FeatureEngineer:
    """
    Advanced feature engineering pipeline based on notebook analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize FeatureEngineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.reference_stats = None
        self.advanced_engineer = AdvancedFeatureEngineer()
        
    def create_enhanced_features(self, df: pd.DataFrame, 
                                reference_stats: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Create enhanced features with interaction terms and advanced ratios.
        Based on the feature engineering from the model development notebook.
        
        Args:
            df: Input DataFrame
            reference_stats: Reference statistics for normalization (from training set)
            
        Returns:
            Tuple of (enhanced DataFrame, reference statistics)
        """
        df_enhanced = df.copy()
        
        # Calculate reference stats if not provided (for training set)
        if reference_stats is None:
            reference_stats = {
                'max_revenue': df_enhanced['faturamento_anual_milhoes'].max(),
                'max_skus': df_enhanced['numero_SKUs'].max(),
                'max_margin': df_enhanced['margem_media_setor'].max(),
                'max_meetings': df_enhanced['reunioes_realizadas'].max(),
                'max_emails_sent': df_enhanced['emails_enviados'].max(),
                'max_days': df_enhanced['days_since_first_touch'].max()
            }
        
        # Basic engagement ratios
        df_enhanced['email_open_rate'] = np.where(
            df_enhanced['emails_enviados'] > 0, 
            df_enhanced['emails_abertos'] / df_enhanced['emails_enviados'], 
            0
        )
        
        df_enhanced['email_response_rate'] = np.where(
            df_enhanced['emails_abertos'] > 0, 
            df_enhanced['emails_respondidos'] / df_enhanced['emails_abertos'], 
            0
        )
        
        df_enhanced['meeting_conversion_rate'] = np.where(
            df_enhanced['emails_enviados'] > 0,
            df_enhanced['reunioes_realizadas'] / df_enhanced['emails_enviados'], 
            0
        )
        
        # Basic engagement score (for backward compatibility)
        df_enhanced['engagement_score'] = (
            (df_enhanced['reunioes_realizadas'] / reference_stats['max_meetings']) * 0.6 +
            df_enhanced['download_whitepaper'] * 0.2 +
            df_enhanced['demo_solicitada'] * 0.2
        )
        
        # Proactive signals
        df_enhanced['proactive_signals'] = (
            df_enhanced['download_whitepaper'] + 
            df_enhanced['demo_solicitada'] + 
            df_enhanced['problemas_reportados_precificacao'] + 
            df_enhanced['urgencia_projeto']
        )
        
        # Total touchpoints
        df_enhanced['total_touchpoints'] = (
            df_enhanced['emails_enviados'] + 
            df_enhanced['reunioes_realizadas'] + 
            df_enhanced['download_whitepaper'] + 
            df_enhanced['demo_solicitada']
        )
        
        # Binary features
        df_enhanced['is_recent_lead'] = (df_enhanced['days_since_first_touch'] <= 30).astype(int)
        df_enhanced['is_warm_lead'] = df_enhanced['lead_source'].isin(['Indicação de Cliente', 'Evento Setorial']).astype(int)
        df_enhanced['is_engaged_prospect'] = (
            (df_enhanced['emails_respondidos'] > 0) | 
            (df_enhanced['reunioes_realizadas'] > 0) |
            (df_enhanced['demo_solicitada'] == 1)
        ).astype(int)
        
        # Company size score
        df_enhanced['company_size_score'] = (
            (df_enhanced['faturamento_anual_milhoes'] / reference_stats['max_revenue']) * 0.4 +
            (df_enhanced['numero_SKUs'] / reference_stats['max_skus']) * 0.3 +
            (df_enhanced['margem_media_setor'] / reference_stats['max_margin']) * 0.3
        )
        
        # High-value interaction features
        df_enhanced['high_value_decision_maker'] = (
            df_enhanced['segmento'].isin(['Energia & Utilities', 'Químicos & Plásticos']) &
            df_enhanced['contact_role'].isin(['Diretor de Operações', 'Diretor Financeiro (CFO)'])
        ).astype(int)
        
        df_enhanced['warm_lead_high_engagement'] = (
            df_enhanced['lead_source'].isin(['Indicação de Cliente', 'Evento Setorial']) &
            (df_enhanced['reunioes_realizadas'] > 0)
        ).astype(int)
        
        # Time-based urgency features
        df_enhanced['is_urgent_lead'] = (
            (df_enhanced['days_since_first_touch'] <= 30) & 
            (df_enhanced['urgencia_projeto'] == 1)
        ).astype(int)
        
        # Company quality score
        df_enhanced['company_quality_score'] = (
            (df_enhanced['faturamento_anual_milhoes'] / reference_stats['max_revenue']) * 0.3 +
            (df_enhanced['numero_SKUs'] / reference_stats['max_skus']) * 0.2 +
            (df_enhanced['margem_media_setor'] / reference_stats['max_margin']) * 0.2 +
            df_enhanced['exporta'] * 0.1 +
            df_enhanced['segmento'].isin(['Energia & Utilities', 'Químicos & Plásticos']).astype(int) * 0.2
        )
        
        # Advanced engagement score
        df_enhanced['advanced_engagement_score'] = (
            df_enhanced['email_open_rate'] * 0.15 +
            df_enhanced['email_response_rate'] * 0.25 +
            df_enhanced['meeting_conversion_rate'] * 0.3 +
            df_enhanced['download_whitepaper'] * 0.1 +
            df_enhanced['demo_solicitada'] * 0.1 +
            df_enhanced['problemas_reportados_precificacao'] * 0.05 +
            df_enhanced['urgencia_projeto'] * 0.05
        )
        
        # Time decay factor
        df_enhanced['time_decay_factor'] = 1 - (df_enhanced['days_since_first_touch'] / reference_stats['max_days'])
        
        # Lead quality composite score (most important feature from notebook)
        df_enhanced['lead_quality_score'] = (
            df_enhanced['advanced_engagement_score'] * 0.35 +
            df_enhanced['company_quality_score'] * 0.25 +
            df_enhanced['contact_role'].isin(['Diretor de Operações']).astype(int) * 0.15 +
            df_enhanced['lead_source'].isin(['Indicação de Cliente', 'Evento Setorial']).astype(int) * 0.15 +
            df_enhanced['time_decay_factor'] * 0.1
        )
        
        logger.info(f"Enhanced features created. Dataset shape: {df_enhanced.shape}")
        return df_enhanced, reference_stats
    
    def get_final_feature_set(self) -> Dict[str, list]:
        """
        Get the final feature set based on notebook analysis.
        
        Returns:
            Dictionary with feature categories
        """
        # Get advanced features and merge with existing ones
        advanced_features = self.advanced_engineer.get_feature_set()
        
        # Base features from original feature engineering
        base_numeric = [
            'days_since_first_touch',
            'email_open_rate',
            'advanced_engagement_score',
            'company_quality_score'
        ]
        
        # Advanced features (excluding duplicates)
        advanced_numeric = [f for f in advanced_features['numeric_features'] if f not in base_numeric]
        
        # Base binary features
        base_binary = [
            'exporta',
            'download_whitepaper',
            'is_engaged_prospect'
        ]
        
        # Advanced binary features (excluding duplicates)
        advanced_binary = [f for f in advanced_features['binary_features'] if f not in base_binary]
        
        return {
            'numeric_features': base_numeric + advanced_numeric,
            'categorical_features': advanced_features['categorical_features'],
            'binary_features': base_binary + advanced_binary
        }
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Fit the feature engineer on training data and transform.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Tuple of (transformed DataFrame, reference statistics)
        """
        # Apply advanced features first
        df_advanced = self.advanced_engineer.fit_transform(df)
        
        # Then apply existing features
        enhanced_df, self.reference_stats = self.create_enhanced_features(df_advanced)
        return enhanced_df, self.reference_stats
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted reference statistics.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if self.reference_stats is None:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        # Apply advanced features first
        df_advanced = self.advanced_engineer.fit_transform(df)
        
        # Then apply existing features
        enhanced_df, _ = self.create_enhanced_features(df_advanced, self.reference_stats)
        return enhanced_df
