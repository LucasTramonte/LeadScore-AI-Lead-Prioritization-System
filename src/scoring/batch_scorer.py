"""
Batch scoring module for the LeadScore AI system.
Handles large-scale lead scoring operations with performance optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .lead_scorer import LeadScorer
from ..data.data_loader import DataLoader

logger = logging.getLogger(__name__)


class BatchScorer:
    """
    Handles batch scoring operations for large datasets with performance optimization.
    """
    
    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize BatchScorer.
        
        Args:
            model_path: Path to trained model
            config: Configuration dictionary
        """
        self.config = config or {}
        self.lead_scorer = LeadScorer(model_path, config)
        self.data_loader = DataLoader(config)
        
    def score_from_file(self, input_file: str, output_file: Optional[str] = None,
                       batch_size: int = 1000, include_details: bool = False) -> Dict[str, Any]:
        """
        Score leads from a file and save results.
        
        Args:
            input_file: Path to input file (Excel or CSV)
            output_file: Path to output file (optional)
            batch_size: Number of leads to process at once
            include_details: Whether to include detailed scoring information
            
        Returns:
            Dictionary with scoring results and statistics
        """
        logger.info(f"Starting batch scoring from {input_file}")
        start_time = time.time()
        
        # Load data
        if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
            df = self.data_loader.load_excel_data(input_file)
        else:
            df = pd.read_csv(input_file)
        
        logger.info(f"Loaded {len(df)} leads from {input_file}")
        
        # Score in batches
        results = []
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for i in range(0, len(df), batch_size):
            batch_num = i // batch_size + 1
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            batch_df = df.iloc[i:i+batch_size]
            batch_leads = batch_df.to_dict('records')
            
            # Score batch
            batch_results = self.lead_scorer.score_leads_batch(batch_leads)
            results.extend(batch_results)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Add original data if requested
        if include_details:
            # Merge with original data
            original_df = df.copy()
            if 'lead_id' not in original_df.columns:
                original_df['lead_id'] = [f'lead_{i}' for i in range(len(original_df))]
            
            results_df = results_df.merge(original_df, on='lead_id', how='left')
        
        # Save results
        if output_file:
            if output_file.endswith('.xlsx'):
                results_df.to_excel(output_file, index=False)
            else:
                results_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        
        # Calculate statistics
        end_time = time.time()
        processing_time = end_time - start_time
        
        stats = self._calculate_batch_statistics(results_df, processing_time)
        
        return {
            'results': results_df,
            'statistics': stats,
            'processing_time': processing_time,
            'total_leads': len(df),
            'output_file': output_file
        }
    
    def score_dataframe(self, df: pd.DataFrame, batch_size: int = 1000) -> pd.DataFrame:
        """
        Score leads from a DataFrame.
        
        Args:
            df: Input DataFrame with lead data
            batch_size: Number of leads to process at once
            
        Returns:
            DataFrame with scoring results
        """
        logger.info(f"Scoring {len(df)} leads from DataFrame")
        
        # Convert to list of dictionaries
        leads_data = df.to_dict('records')
        
        # Add lead_id if not present
        for i, lead in enumerate(leads_data):
            if 'lead_id' not in lead:
                lead['lead_id'] = f'lead_{i}'
        
        # Score in batches
        results = []
        total_batches = (len(leads_data) + batch_size - 1) // batch_size
        
        for i in range(0, len(leads_data), batch_size):
            batch_num = i // batch_size + 1
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            batch_leads = leads_data[i:i+batch_size]
            batch_results = self.lead_scorer.score_leads_batch(batch_leads)
            results.extend(batch_results)
        
        return pd.DataFrame(results)
    
    def score_with_parallel_processing(self, leads_data: List[Dict[str, Any]], 
                                     max_workers: int = 4, batch_size: int = 250) -> List[Dict[str, Any]]:
        """
        Score leads using parallel processing for better performance.
        
        Args:
            leads_data: List of lead dictionaries
            max_workers: Maximum number of worker threads
            batch_size: Size of each batch for parallel processing
            
        Returns:
            List of scoring results
        """
        logger.info(f"Scoring {len(leads_data)} leads with {max_workers} workers")
        
        # Split data into batches
        batches = [leads_data[i:i+batch_size] for i in range(0, len(leads_data), batch_size)]
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self.lead_scorer.score_leads_batch, batch): i 
                for i, batch in enumerate(batches)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    logger.info(f"Completed batch {batch_idx + 1}/{len(batches)}")
                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {str(e)}")
        
        return results
    
    def generate_priority_report(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive priority distribution report.
        
        Args:
            results_df: DataFrame with scoring results
            
        Returns:
            Dictionary with priority analysis
        """
        if 'priority' not in results_df.columns:
            raise ValueError("Results DataFrame must contain 'priority' column")
        
        # Priority distribution
        priority_counts = results_df['priority'].value_counts()
        priority_percentages = results_df['priority'].value_counts(normalize=True) * 100
        
        # Probability statistics by priority
        priority_stats = results_df.groupby('priority')['conversion_probability'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(4)
        
        # Confidence score statistics
        confidence_stats = results_df.groupby('priority')['confidence_score'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(4)
        
        # Top leads by priority
        top_leads = {}
        for priority in ['High', 'Medium', 'Low']:
            priority_leads = results_df[results_df['priority'] == priority]
            if len(priority_leads) > 0:
                top_leads[priority] = priority_leads.nlargest(5, 'conversion_probability')[
                    ['lead_id', 'conversion_probability', 'confidence_score']
                ].to_dict('records')
        
        # Probability distribution
        prob_ranges = {
            '0.9-1.0': len(results_df[results_df['conversion_probability'] >= 0.9]),
            '0.8-0.9': len(results_df[(results_df['conversion_probability'] >= 0.8) & 
                                    (results_df['conversion_probability'] < 0.9)]),
            '0.7-0.8': len(results_df[(results_df['conversion_probability'] >= 0.7) & 
                                    (results_df['conversion_probability'] < 0.8)]),
            '0.6-0.7': len(results_df[(results_df['conversion_probability'] >= 0.6) & 
                                    (results_df['conversion_probability'] < 0.7)]),
            '0.5-0.6': len(results_df[(results_df['conversion_probability'] >= 0.5) & 
                                    (results_df['conversion_probability'] < 0.6)]),
            '0.0-0.5': len(results_df[results_df['conversion_probability'] < 0.5])
        }
        
        return {
            'total_leads': len(results_df),
            'priority_distribution': {
                'counts': priority_counts.to_dict(),
                'percentages': priority_percentages.to_dict()
            },
            'priority_statistics': priority_stats.to_dict(),
            'confidence_statistics': confidence_stats.to_dict(),
            'top_leads_by_priority': top_leads,
            'probability_distribution': prob_ranges,
            'summary': {
                'high_priority_leads': int(priority_counts.get('High', 0)),
                'medium_priority_leads': int(priority_counts.get('Medium', 0)),
                'low_priority_leads': int(priority_counts.get('Low', 0)),
                'avg_probability': float(results_df['conversion_probability'].mean()),
                'avg_confidence': float(results_df['confidence_score'].mean())
            }
        }
    
    def export_prioritized_lists(self, results_df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
        """
        Export separate files for each priority level.
        
        Args:
            results_df: DataFrame with scoring results
            output_dir: Directory to save priority lists
            
        Returns:
            Dictionary with file paths for each priority
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_paths = {}
        
        for priority in ['High', 'Medium', 'Low']:
            priority_df = results_df[results_df['priority'] == priority].copy()
            
            if len(priority_df) > 0:
                # Sort by probability descending
                priority_df = priority_df.sort_values('conversion_probability', ascending=False)
                
                # Save to file
                filename = f"{priority.lower()}_priority_leads.xlsx"
                file_path = output_path / filename
                priority_df.to_excel(file_path, index=False)
                
                file_paths[priority] = str(file_path)
                logger.info(f"Exported {len(priority_df)} {priority} priority leads to {file_path}")
        
        return file_paths
    
    def _calculate_batch_statistics(self, results_df: pd.DataFrame, processing_time: float) -> Dict[str, Any]:
        """
        Calculate comprehensive batch processing statistics.
        
        Args:
            results_df: DataFrame with results
            processing_time: Total processing time in seconds
            
        Returns:
            Dictionary with statistics
        """
        total_leads = len(results_df)
        
        stats = {
            'processing_performance': {
                'total_processing_time': round(processing_time, 2),
                'leads_per_second': round(total_leads / processing_time, 2),
                'average_time_per_lead': round(processing_time / total_leads * 1000, 2)  # milliseconds
            },
            'data_quality': {
                'total_leads_processed': total_leads,
                'leads_with_warnings': len(results_df[results_df['validation_warnings'].astype(str) != '[]']),
                'success_rate': 100.0  # All leads processed successfully if we reach this point
            },
            'scoring_distribution': {
                'min_probability': float(results_df['conversion_probability'].min()),
                'max_probability': float(results_df['conversion_probability'].max()),
                'mean_probability': float(results_df['conversion_probability'].mean()),
                'median_probability': float(results_df['conversion_probability'].median()),
                'std_probability': float(results_df['conversion_probability'].std())
            },
            'priority_summary': results_df['priority'].value_counts().to_dict(),
            'confidence_summary': {
                'mean_confidence': float(results_df['confidence_score'].mean()),
                'min_confidence': float(results_df['confidence_score'].min()),
                'max_confidence': float(results_df['confidence_score'].max())
            }
        }
        
        return stats
    
    def validate_input_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate input data before batch scoring.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'total_records': len(df),
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check required columns
        feature_requirements = self.lead_scorer.get_feature_requirements()
        required_features = (
            feature_requirements['required_features']['numeric'] +
            feature_requirements['required_features']['categorical'] +
            feature_requirements['required_features']['binary']
        )
        
        missing_columns = [col for col in required_features if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_columns}")
        
        # Check data types and ranges
        validator = self.lead_scorer.validator
        is_valid, validation_errors = validator.validate_all(df)
        
        if not is_valid:
            validation_results['warnings'].extend([
                f"{category}: {errors}" for category, errors in validation_errors.items() if errors
            ])
        
        # Check for duplicates
        if df.duplicated().any():
            duplicate_count = df.duplicated().sum()
            validation_results['warnings'].append(f"Found {duplicate_count} duplicate records")
            validation_results['recommendations'].append("Consider removing duplicates before scoring")
        
        # Check data completeness
        missing_data = df.isnull().sum()
        high_missing = missing_data[missing_data > len(df) * 0.1]  # More than 10% missing
        
        if len(high_missing) > 0:
            validation_results['warnings'].append(f"High missing data in columns: {high_missing.to_dict()}")
            validation_results['recommendations'].append("Consider data imputation for columns with high missing rates")
        
        return validation_results
