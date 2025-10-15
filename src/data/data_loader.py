"""
Data loading module for the LeadScore AI system.
Handles loading and initial processing of lead data from various sources.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and initial processing of lead data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config: Configuration dictionary with data paths and settings
        """
        self.config = config or {}
        self.data_path = self.config.get('data_path', 'data/raw/')
        
    def load_excel_data(self, file_path: str, sheet_name: int = 1) -> pd.DataFrame:
        """
        Load data from Excel file with proper handling of datetime conversion.
        
        Args:
            file_path: Path to the Excel file
            sheet_name: Sheet number or name to load (default: 1)
            
        Returns:
            DataFrame with loaded and initially processed data
        """
        try:
            logger.info(f"Loading data from {file_path}, sheet {sheet_name}")
            
            # Load the Excel file
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Handle the datetime conversion issue for revenue column
            if 'faturamento_anual_milhoes' in df.columns:
                df['faturamento_anual_milhoes'] = df['faturamento_anual_milhoes'].apply(
                    lambda x: float(x.day + x.month/10) if isinstance(x, datetime) else float(x)
                )
            
            # Convert created_at to datetime if it exists
            if 'created_at' in df.columns:
                df['created_at'] = pd.to_datetime(df['created_at'])
            
            logger.info(f"Successfully loaded {len(df)} records with {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def load_default_data(self) -> pd.DataFrame:
        """
        Load the default dataset from the configured data path.
        
        Returns:
            DataFrame with the default lead dataset
        """
        default_path = Path(self.data_path) / 'data.xlsx'
        return self.load_excel_data(str(default_path))
    
    def validate_required_columns(self, df: pd.DataFrame) -> bool:
        """
        Validate that the DataFrame contains all required columns.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if all required columns are present
        """
        required_columns = [
            'segmento', 'faturamento_anual_milhoes', 'numero_SKUs', 'exporta',
            'margem_media_setor', 'contact_role', 'lead_source', 'crm_stage',
            'emails_enviados', 'emails_abertos', 'emails_respondidos',
            'reunioes_realizadas', 'download_whitepaper', 'demo_solicitada',
            'problemas_reportados_precificacao', 'urgencia_projeto',
            'days_since_first_touch'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        logger.info("All required columns are present")
        return True
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of the loaded data.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with data summary statistics
        """
        summary = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'conversion_rate': df.get('converted', pd.Series()).mean() if 'converted' in df.columns else None,
            'date_range': {
                'min_date': df['created_at'].min() if 'created_at' in df.columns else None,
                'max_date': df['created_at'].max() if 'created_at' in df.columns else None
            }
        }
        
        return summary
