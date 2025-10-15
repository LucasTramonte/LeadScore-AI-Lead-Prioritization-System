"""
Main CLI interface for the LeadScore AI system.
Provides command-line access to all system functionality.
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from ..data.data_loader import DataLoader
from ..data.data_validator import DataValidator
from ..model.model_persistence import ModelPersistence
from ..scoring.lead_scorer import LeadScorer
from ..scoring.batch_scorer import BatchScorer
from ..services.training_service import TrainingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data_file(file_path: str, data_loader: DataLoader = None) -> pd.DataFrame:
    """
    Helper function to load data from Excel or CSV files.
    
    Args:
        file_path: Path to the data file
        data_loader: Optional DataLoader instance
        
    Returns:
        DataFrame with loaded data
    """
    if data_loader is None:
        data_loader = DataLoader()
    
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return data_loader.load_excel_data(file_path)
    else:
        return pd.read_csv(file_path)


def setup_train_parser(subparsers):
    """Set up the train command parser."""
    train_parser = subparsers.add_parser('train', help='Train a new lead scoring model')
    train_parser.add_argument('--data', required=True, help='Path to training data file')
    train_parser.add_argument('--config', help='Path to configuration file')
    train_parser.add_argument('--output', help='Directory to save trained model')
    train_parser.add_argument('--model-name', help='Custom name for the model')
    train_parser.add_argument('--target-column', default='converted', help='Name of target column')
    train_parser.set_defaults(func=train_model)


def setup_score_parser(subparsers):
    """Set up the score command parser."""
    score_parser = subparsers.add_parser('score', help='Score leads using a trained model')
    score_parser.add_argument('--model', required=True, help='Path to trained model or model name')
    score_parser.add_argument('--input', required=True, help='Path to input data file')
    score_parser.add_argument('--output', help='Path to output file')
    score_parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for processing')
    score_parser.add_argument('--include-details', action='store_true', help='Include detailed scoring information')
    score_parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    score_parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    score_parser.set_defaults(func=score_leads)


def setup_evaluate_parser(subparsers):
    """Set up the evaluate command parser."""
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model', required=True, help='Path to trained model or model name')
    eval_parser.add_argument('--data', required=True, help='Path to evaluation data file')
    eval_parser.add_argument('--output', help='Path to save evaluation report')
    eval_parser.add_argument('--target-column', default='converted', help='Name of target column')
    eval_parser.set_defaults(func=evaluate_model)


def setup_list_parser(subparsers):
    """Set up the list command parser."""
    list_parser = subparsers.add_parser('list', help='List saved models')
    list_parser.set_defaults(func=list_models)


def setup_info_parser(subparsers):
    """Set up the info command parser."""
    info_parser = subparsers.add_parser('info', help='Get information about a model')
    info_parser.add_argument('--model', required=True, help='Path to trained model or model name')
    info_parser.set_defaults(func=model_info)


def setup_validate_parser(subparsers):
    """Set up the validate command parser."""
    validate_parser = subparsers.add_parser('validate', help='Validate data quality')
    validate_parser.add_argument('--data', required=True, help='Path to data file')
    validate_parser.add_argument('--output', help='Path to save validation report')
    validate_parser.set_defaults(func=validate_data)


def setup_export_parser(subparsers):
    """Set up the export command parser."""
    export_parser = subparsers.add_parser('export', help='Export model for deployment')
    export_parser.add_argument('--model', required=True, help='Model name to export')
    export_parser.add_argument('--output', required=True, help='Export directory')
    export_parser.set_defaults(func=export_model)


def train_model(args):
    """Train a new lead scoring model."""
    logger.info("Starting model training...")
    
    try:
        # Load configuration if provided
        config = {}
        if args.config:
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        
        # Load data
        data_loader = DataLoader(config)
        df = data_loader.load_excel_data(args.data)
        logger.info(f"Loaded {len(df)} records for training")
        
        # Validate data
        validator = DataValidator(config)
        is_valid, validation_errors = validator.validate_all(df)
        if not is_valid:
            logger.warning(f"Data validation issues: {validation_errors}")
        
        # Train model using TrainingService
        from ..core.config_manager import get_config
        config_manager = get_config() if not config else config
        training_service = TrainingService(config_manager)
        training_results = training_service.run_training_pipeline(args.model_name)
        
        model_path = f"models/{training_results['model_saved_as']}"
        
        logger.info(f"Model training completed successfully!")
        logger.info(f"Best model: {training_results['best_model_name']}")
        logger.info(f"Test AUC: {training_results['model_comparison'][training_results['best_model_name']]['test_auc']:.4f}")
        logger.info(f"Model saved to: {model_path}")
        
        # Print model comparison
        print("\n=== MODEL COMPARISON ===")
        for model_name, metrics in training_results['model_comparison'].items():
            print(f"{model_name:20} | Test AUC: {metrics['test_auc']:.4f} | CV AUC: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
        
        print(f"\n=== OPTIMAL THRESHOLDS ===")
        print(f"High Priority: ≥ {training_results['optimal_thresholds']['high']:.3f}")
        print(f"Medium Priority: ≥ {training_results['optimal_thresholds']['medium']:.3f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


def score_leads(args):
    """Score leads using a trained model."""
    logger.info("Starting lead scoring...")
    
    try:
        # Initialize batch scorer
        batch_scorer = BatchScorer(args.model)
        
        # Load input data
        df = load_data_file(args.input)
        
        validation_results = batch_scorer.validate_input_data(df)
        if not validation_results['is_valid']:
            logger.error(f"Input data validation failed: {validation_results['issues']}")
            sys.exit(1)
        
        if validation_results['warnings']:
            logger.warning(f"Data validation warnings: {validation_results['warnings']}")
        
        # Score leads
        if args.parallel:
            logger.info(f"Using parallel processing with {args.workers} workers")
            leads_data = df.to_dict('records')
            results = batch_scorer.score_with_parallel_processing(
                leads_data, max_workers=args.workers, batch_size=args.batch_size
            )
            results_df = pd.DataFrame(results)
        else:
            scoring_results = batch_scorer.score_from_file(
                args.input, args.output, args.batch_size, args.include_details
            )
            results_df = scoring_results['results']
        
        # Generate priority report
        priority_report = batch_scorer.generate_priority_report(results_df)
        
        logger.info("Lead scoring completed successfully!")
        print(f"\n=== SCORING RESULTS ===")
        print(f"Total leads scored: {len(results_df)}")
        print(f"High priority leads: {priority_report['summary']['high_priority_leads']}")
        print(f"Medium priority leads: {priority_report['summary']['medium_priority_leads']}")
        print(f"Low priority leads: {priority_report['summary']['low_priority_leads']}")
        print(f"Average probability: {priority_report['summary']['avg_probability']:.3f}")
        print(f"Average confidence: {priority_report['summary']['avg_confidence']:.3f}")
        
        if args.output:
            logger.info(f"Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Scoring failed: {str(e)}")
        sys.exit(1)


def evaluate_model(args):
    """Evaluate a trained model."""
    logger.info("Starting model evaluation...")
    
    try:
        # Load model
        scorer = LeadScorer(args.model)
        model_info = scorer.get_model_info()
        
        # Load evaluation data
        data_loader = DataLoader()
        df = data_loader.load_excel_data(args.data)
        logger.info(f"Loaded {len(df)} records for evaluation")
        
        # Prepare data
        if args.target_column not in df.columns:
            logger.error(f"Target column '{args.target_column}' not found in data")
            sys.exit(1)
        
        # Score the data
        batch_scorer = BatchScorer(args.model)
        results_df = batch_scorer.score_dataframe(df)
        
        # Simple evaluation using sklearn metrics
        y_true = df[args.target_column].values
        y_pred_proba = results_df['conversion_probability'].values
        
        # Calculate basic metrics
        auc_score = roc_auc_score(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        # Priority-based analysis
        results_with_truth = results_df.copy()
        results_with_truth['actual_converted'] = y_true
        
        priority_analysis = {}
        for priority in ['High', 'Medium', 'Low']:
            priority_data = results_with_truth[results_with_truth['priority'] == priority]
            if len(priority_data) > 0:
                conversion_rate = priority_data['actual_converted'].mean()
                count = len(priority_data)
                priority_analysis[priority] = {
                    'count': count,
                    'conversion_rate': conversion_rate,
                    'percentage': count / len(results_with_truth) * 100
                }
        
        # Print results
        print(f"\n=== MODEL EVALUATION RESULTS ===")
        print(f"Model: {model_info['model_type']}")
        print(f"AUC Score: {auc_score:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        
        print(f"\n=== PRIORITY PERFORMANCE ===")
        for priority, metrics in priority_analysis.items():
            print(f"{priority:6} Priority: {metrics['count']:4d} leads ({metrics['percentage']:5.1f}%) | "
                  f"Conversion: {metrics['conversion_rate']:.1%}")
        
        # Save report if requested
        if args.output:
            evaluation_report = {
                'model_type': model_info['model_type'],
                'auc_score': float(auc_score),
                'average_precision': float(avg_precision),
                'priority_analysis': priority_analysis,
                'total_samples': len(results_with_truth),
                'overall_conversion_rate': float(y_true.mean())
            }
            with open(args.output, 'w') as f:
                json.dump(evaluation_report, f, indent=2, default=str)
            logger.info(f"Evaluation report saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)


def list_models(args):
    """List all saved models."""
    try:
        persistence = ModelPersistence()
        models_df = persistence.list_saved_models()
        
        if len(models_df) == 0:
            print("No saved models found.")
            return
        
        print(f"\n=== SAVED MODELS ({len(models_df)} found) ===")
        print(f"{'Model Name':<30} {'Type':<15} {'Test AUC':<10} {'Training Date':<20}")
        print("-" * 80)
        
        for _, model in models_df.iterrows():
            print(f"{model['model_name']:<30} {model['model_type']:<15} "
                  f"{model['test_auc']:<10} {str(model['training_date'])[:19]:<20}")
        
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        sys.exit(1)


def model_info(args):
    """Get detailed information about a model."""
    try:
        scorer = LeadScorer(args.model)
        model_info = scorer.get_model_info()
        feature_requirements = scorer.get_feature_requirements()
        
        print(f"\n=== MODEL INFORMATION ===")
        print(f"Model Name: {model_info['model_name']}")
        print(f"Model Type: {model_info['model_type']}")
        print(f"Version: {model_info['version']}")
        print(f"Training Date: {model_info['training_date']}")
        
        print(f"\n=== THRESHOLDS ===")
        print(f"High Priority: ≥ {model_info['thresholds']['high']:.3f}")
        print(f"Medium Priority: ≥ {model_info['thresholds']['medium']:.3f}")
        
        print(f"\n=== TRAINING STATISTICS ===")
        stats = model_info['training_stats']
        print(f"Total Samples: {stats.get('total_samples', 'Unknown'):,}")
        print(f"Total Features: {stats.get('total_features', 'Unknown')}")
        print(f"Conversion Rate: {stats.get('conversion_rate', 0):.1%}")
        
        print(f"\n=== MODEL PERFORMANCE ===")
        if model_info['model_comparison']:
            best_model = model_info['model_type']
            if best_model in model_info['model_comparison']:
                metrics = model_info['model_comparison'][best_model]
                print(f"Test AUC: {metrics.get('test_auc', 'Unknown'):.4f}")
                print(f"CV AUC: {metrics.get('cv_mean', 'Unknown'):.4f} ± {metrics.get('cv_std', 0):.4f}")
        
        print(f"\n=== REQUIRED FEATURES ===")
        print(f"Numeric Features ({len(feature_requirements['required_features']['numeric'])}):")
        for feature in feature_requirements['required_features']['numeric']:
            print(f"  - {feature}")
        
        print(f"Categorical Features ({len(feature_requirements['required_features']['categorical'])}):")
        for feature in feature_requirements['required_features']['categorical']:
            print(f"  - {feature}")
        
        print(f"Binary Features ({len(feature_requirements['required_features']['binary'])}):")
        for feature in feature_requirements['required_features']['binary']:
            print(f"  - {feature}")
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        sys.exit(1)


def validate_data(args):
    """Validate data quality."""
    try:
        # Load data
        df = load_data_file(args.data)
        
        # Validate data
        validator = DataValidator()
        is_valid, validation_errors = validator.validate_all(df)
        quality_report = validator.get_data_quality_report(df)
        
        print(f"\n=== DATA VALIDATION RESULTS ===")
        print(f"File: {args.data}")
        print(f"Total Records: {len(df):,}")
        print(f"Total Columns: {len(df.columns)}")
        print(f"Overall Valid: {'Yes' if is_valid else 'No'}")
        
        if not is_valid:
            print(f"\n=== VALIDATION ERRORS ===")
            for category, errors in validation_errors.items():
                if errors:
                    print(f"{category.upper()}:")
                    for error in errors:
                        print(f"  - {error}")
        
        print(f"\n=== DATA QUALITY SUMMARY ===")
        print(f"Duplicate Records: {quality_report['duplicate_records']}")
        
        if quality_report['missing_values']:
            print(f"\nMissing Values:")
            for col, info in quality_report['missing_values'].items():
                print(f"  {col}: {info['count']} ({info['percentage']:.1f}%)")
        
        # Save report if requested
        if args.output:
            report = {
                'validation_results': {'is_valid': is_valid, 'errors': validation_errors},
                'quality_report': quality_report
            }
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Validation report saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        sys.exit(1)


def export_model(args):
    """Export model for deployment."""
    try:
        persistence = ModelPersistence()
        export_path = persistence.export_model_for_deployment(args.model, args.output)
        
        print(f"\n=== MODEL EXPORT SUCCESSFUL ===")
        print(f"Model: {args.model}")
        print(f"Export Path: {export_path}")
        print(f"Deployment package ready for production use.")
        
    except Exception as e:
        logger.error(f"Model export failed: {str(e)}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='LeadScore AI - Lead Prioritization System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  leadscore train --data data/raw/data.xlsx --model-name my_model

  # Score leads
  leadscore score --model my_model --input new_leads.xlsx --output scored_leads.xlsx

  # Evaluate model performance
  leadscore evaluate --model my_model --data test_data.xlsx

  # List all saved models
  leadscore list

  # Get model information
  leadscore info --model my_model

  # Validate data quality
  leadscore validate --data data.xlsx

  # Export model for deployment
  leadscore export --model my_model --output deployment/
        """
    )
    
    parser.add_argument('--version', action='version', version='LeadScore AI v1.0.0')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Set up command parsers
    setup_train_parser(subparsers)
    setup_score_parser(subparsers)
    setup_evaluate_parser(subparsers)
    setup_list_parser(subparsers)
    setup_info_parser(subparsers)
    setup_validate_parser(subparsers)
    setup_export_parser(subparsers)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Execute command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
