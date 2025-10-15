"""
Advanced feature engineering for better lead scoring performance.
Addresses the poor model performance by creating more predictive features.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler

class AdvancedFeatureEngineer:
    """Advanced feature engineering to boost model performance."""
    
    def __init__(self):
        self.industry_stats = {}
        self.source_stats = {}
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features that capture engagement patterns."""
        df = df.copy()
        
        # Create total_touchpoints if it doesn't exist
        if 'total_touchpoints' not in df.columns:
            df['total_touchpoints'] = (
                df['emails_enviados'] + 
                df['reunioes_realizadas'] + 
                df['download_whitepaper'] + 
                df['demo_solicitada']
            )
        
        # Engagement velocity (interactions per day)
        df['engagement_velocity'] = np.where(
            df['days_since_first_touch'] > 0,
            df['total_touchpoints'] / df['days_since_first_touch'],
            0
        )
        
        # Time decay factor (recent leads are more valuable)
        df['recency_score'] = np.exp(-df['days_since_first_touch'] / 30)
        
        return df
    
    def create_industry_intelligence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add industry-specific intelligence features."""
        df = df.copy()
        
        # Use predefined industry conversion rates based on domain knowledge
        # These would typically come from historical data analysis
        industry_rates = {
            'Energia & Utilities': 0.65,
            'Químicos & Plásticos': 0.58,
            'Alimentos & Bebidas': 0.45,
            'Metalurgia': 0.42,
            'Máquinas & Equipamentos': 0.48,
            'Construção': 0.38,
            'Bens de Consumo': 0.35
        }
        
        df['industry_conversion_rate'] = df['segmento'].map(
            lambda x: industry_rates.get(x, 0.48)
        )
        
        # Lead source quality based on domain knowledge
        source_rates = {
            'Evento Setorial': 0.62,
            'Indicação de Cliente': 0.68,
            'Inbound (Site)': 0.35,
            'Prospecção Ativa': 0.28,
            'Conteúdo Técnico': 0.45
        }
        
        df['source_quality_score'] = df['lead_source'].map(
            lambda x: source_rates.get(x, 0.48)
        )
        
        return df
    
    def create_rfm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create RFM (Recency, Frequency, Monetary) analysis features."""
        df = df.copy()
        
        # Recency (inverted days since first touch)
        df['rfm_recency'] = 1 / (1 + df['days_since_first_touch'] / 30)
        
        # Frequency (total touchpoints normalized)
        max_touchpoints = df['total_touchpoints'].max()
        df['rfm_frequency'] = df['total_touchpoints'] / max_touchpoints if max_touchpoints > 0 else 0
        
        # Monetary (company size proxy)
        max_revenue = df['faturamento_anual_milhoes'].max()
        df['rfm_monetary'] = df['faturamento_anual_milhoes'] / max_revenue if max_revenue > 0 else 0
        
        # Combined RFM score
        df['rfm_score'] = (
            df['rfm_recency'] * 0.4 + 
            df['rfm_frequency'] * 0.3 + 
            df['rfm_monetary'] * 0.3
        )
        
        return df
    
    def create_composite_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite scoring features."""
        df = df.copy()
        
        # Advanced lead quality score
        df['lead_quality_score'] = (
            df['rfm_score'] * 0.30 +
            df['engagement_velocity'] * 0.25 +
            df['industry_conversion_rate'] * 0.20 +
            df['source_quality_score'] * 0.15 +
            df['recency_score'] * 0.10
        )
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all advanced feature engineering."""
        df = self.create_temporal_features(df)
        df = self.create_industry_intelligence(df)
        df = self.create_rfm_features(df)
        df = self.create_composite_score(df)
        
        return df
    
    def get_feature_set(self) -> Dict[str, list]:
        """Return the advanced feature set for model training."""
        return {
            'numeric_features': [
                'lead_quality_score',
                'rfm_score',
                'engagement_velocity',
                'recency_score',
                'industry_conversion_rate',
                'source_quality_score'
            ],
            'categorical_features': [
                'segmento',
                'contact_role',
                'lead_source',
                'crm_stage'
            ],
            'binary_features': [
                'exporta',
                'download_whitepaper',
                'demo_solicitada',
                'urgencia_projeto'
            ]
        }

def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Comprehensive data quality validation."""
    return {
        'sample_size': len(df),
        'conversion_rate': df['converted'].mean(),
        'missing_values': df.isnull().sum().sum(),
        'high_correlation_features': df.corr()['converted'].abs().nlargest(10).to_dict()
    }
