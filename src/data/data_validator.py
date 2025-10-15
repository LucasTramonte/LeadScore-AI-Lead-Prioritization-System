"""
Data validation module for the LeadScore AI system.
Ensures data quality and consistency before processing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates data quality and consistency for lead scoring.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataValidator.
        
        Args:
            config: Configuration dictionary with validation rules
        """
        self.config = config or {}
        self.validation_rules = self._get_default_validation_rules()
        
    def _get_default_validation_rules(self) -> Dict[str, Any]:
        """
        Get default validation rules for lead data.
        
        Returns:
            Dictionary with validation rules
        """
        return {
            'required_columns': [
                'segmento', 'faturamento_anual_milhoes', 'numero_SKUs', 'exporta',
                'margem_media_setor', 'contact_role', 'lead_source', 'crm_stage',
                'emails_enviados', 'emails_abertos', 'emails_respondidos',
                'reunioes_realizadas', 'download_whitepaper', 'demo_solicitada',
                'problemas_reportados_precificacao', 'urgencia_projeto',
                'days_since_first_touch'
            ],
            'numeric_ranges': {
                'faturamento_anual_milhoes': {'min': 0, 'max': 1000},
                'numero_SKUs': {'min': 0, 'max': 10000},
                'margem_media_setor': {'min': 0, 'max': 100},
                'emails_enviados': {'min': 0, 'max': 100},
                'emails_abertos': {'min': 0, 'max': 100},
                'emails_respondidos': {'min': 0, 'max': 100},
                'reunioes_realizadas': {'min': 0, 'max': 50},
                'days_since_first_touch': {'min': 0, 'max': 1000}
            },
            'categorical_values': {
                'segmento': [
                    'Alimentos & Bebidas', 'Químicos & Plásticos', 'Metalurgia',
                    'Máquinas & Equipamentos', 'Construção Civil', 'Energia & Utilities',
                    'Bens de Consumo'
                ],
                'contact_role': [
                    'Analista de Preços', 'Diretor Financeiro (CFO)', 'Diretor de Operações',
                    'Gerente Financeiro', 'Coordenador de Custos'
                ],
                'lead_source': [
                    'Evento Setorial', 'Indicação de Cliente', 'Inbound (Site)',
                    'Prospecção Ativa'
                ],
                'crm_stage': [
                    'Novo', 'Qualificado Marketing', 'Qualificado Vendas', 'Proposta',
                    'Negociação'
                ]
            },
            'binary_columns': [
                'exporta', 'download_whitepaper', 'demo_solicitada',
                'problemas_reportados_precificacao', 'urgencia_projeto'
            ]
        }
    
    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame schema against required columns.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        required_columns = self.validation_rules['required_columns']
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for extra columns that might indicate data issues
        extra_columns = [col for col in df.columns if col not in required_columns and col != 'converted']
        if extra_columns:
            logger.warning(f"Extra columns found (will be ignored): {extra_columns}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_data_types(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate data types for each column.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check numeric columns
        numeric_columns = list(self.validation_rules['numeric_ranges'].keys())
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Column '{col}' should be numeric but found {df[col].dtype}")
        
        # Check binary columns
        binary_columns = self.validation_rules['binary_columns']
        for col in binary_columns:
            if col in df.columns:
                unique_values = df[col].dropna().unique()
                if not all(val in [0, 1] for val in unique_values):
                    errors.append(f"Column '{col}' should be binary (0/1) but found values: {unique_values}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_data_ranges(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate numeric data ranges.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        for col, ranges in self.validation_rules['numeric_ranges'].items():
            if col in df.columns:
                min_val = ranges['min']
                max_val = ranges['max']
                
                # Check for values outside expected range
                out_of_range = df[(df[col] < min_val) | (df[col] > max_val)][col]
                if len(out_of_range) > 0:
                    errors.append(
                        f"Column '{col}' has {len(out_of_range)} values outside range "
                        f"[{min_val}, {max_val}]. Range: {out_of_range.min():.2f} to {out_of_range.max():.2f}"
                    )
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_categorical_values(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate categorical column values.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        for col, expected_values in self.validation_rules['categorical_values'].items():
            if col in df.columns:
                unique_values = df[col].dropna().unique()
                unexpected_values = [val for val in unique_values if val not in expected_values]
                
                if unexpected_values:
                    errors.append(
                        f"Column '{col}' has unexpected values: {unexpected_values}. "
                        f"Expected: {expected_values}"
                    )
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_business_logic(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate business logic constraints.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Email logic validation
        if all(col in df.columns for col in ['emails_enviados', 'emails_abertos', 'emails_respondidos']):
            # Emails opened cannot exceed emails sent
            invalid_opened = df[df['emails_abertos'] > df['emails_enviados']]
            if len(invalid_opened) > 0:
                errors.append(f"{len(invalid_opened)} records have more emails opened than sent")
            
            # Emails responded cannot exceed emails opened
            invalid_responded = df[df['emails_respondidos'] > df['emails_abertos']]
            if len(invalid_responded) > 0:
                errors.append(f"{len(invalid_responded)} records have more emails responded than opened")
        
        # Revenue and SKU relationship validation
        if all(col in df.columns for col in ['faturamento_anual_milhoes', 'numero_SKUs']):
            # Very small companies shouldn't have too many SKUs
            suspicious_small = df[
                (df['faturamento_anual_milhoes'] < 5) & (df['numero_SKUs'] > 500)
            ]
            if len(suspicious_small) > 0:
                logger.warning(f"{len(suspicious_small)} small companies have unusually high SKU counts")
        
        # Days since first touch validation
        if 'days_since_first_touch' in df.columns:
            # Negative days don't make sense
            negative_days = df[df['days_since_first_touch'] < 0]
            if len(negative_days) > 0:
                errors.append(f"{len(negative_days)} records have negative days since first touch")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with data quality metrics
        """
        report = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'duplicate_records': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'numeric_summaries': {},
            'categorical_summaries': {}
        }
        
        # Missing values analysis
        missing_counts = df.isnull().sum()
        report['missing_values'] = {
            col: {'count': int(count), 'percentage': float(count / len(df) * 100)}
            for col, count in missing_counts.items() if count > 0
        }
        
        # Numeric summaries
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            report['numeric_summaries'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median()),
                'outliers_count': int(self._count_outliers(df[col]))
            }
        
        # Categorical summaries
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            value_counts = df[col].value_counts()
            report['categorical_summaries'][col] = {
                'unique_values': int(df[col].nunique()),
                'most_common': value_counts.head(5).to_dict(),
                'least_common': value_counts.tail(5).to_dict()
            }
        
        return report
    
    def _count_outliers(self, series: pd.Series) -> int:
        """
        Count outliers using IQR method.
        
        Args:
            series: Pandas Series to analyze
            
        Returns:
            Number of outliers
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return len(outliers)
    
    def validate_all(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Run all validation checks.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        validation_results = {}
        overall_valid = True
        
        # Schema validation
        schema_valid, schema_errors = self.validate_schema(df)
        validation_results['schema'] = schema_errors
        overall_valid &= schema_valid
        
        # Data type validation
        dtype_valid, dtype_errors = self.validate_data_types(df)
        validation_results['data_types'] = dtype_errors
        overall_valid &= dtype_valid
        
        # Range validation
        range_valid, range_errors = self.validate_data_ranges(df)
        validation_results['ranges'] = range_errors
        overall_valid &= range_valid
        
        # Categorical validation
        cat_valid, cat_errors = self.validate_categorical_values(df)
        validation_results['categorical'] = cat_errors
        overall_valid &= cat_valid
        
        # Business logic validation
        logic_valid, logic_errors = self.validate_business_logic(df)
        validation_results['business_logic'] = logic_errors
        overall_valid &= logic_valid
        
        if overall_valid:
            logger.info("All validation checks passed")
        else:
            logger.warning("Some validation checks failed")
            
        return overall_valid, validation_results
