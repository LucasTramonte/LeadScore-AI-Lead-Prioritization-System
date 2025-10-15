"""
Model persistence module for the LeadScore AI system.
Handles saving and loading of trained models and associated artifacts.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
import joblib
import json
import pickle
from pathlib import Path
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)


class ModelPersistence:
    """
    Handles model persistence operations including saving and loading models,
    feature engineering artifacts, and metadata.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ModelPersistence.
        
        Args:
            config: Configuration dictionary with paths and settings
        """
        self.config = config or {}
        self.models_dir = Path(self.config.get('models_dir', 'models'))
        self.models_dir.mkdir(exist_ok=True)
        
    def save_model_artifacts(self, training_results: Dict[str, Any], 
                           model_name: str = None) -> str:
        """
        Save complete model artifacts including pipeline, metadata, and feature engineering.
        
        Args:
            training_results: Results from model training
            model_name: Optional custom model name
            
        Returns:
            Path to saved model directory
        """
        # Generate model name if not provided
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_name = training_results['best_model_name'].replace(' ', '_')
            model_name = f"leadscore_{best_model_name.lower()}_{timestamp}"
        
        # Create model directory
        model_dir = self.models_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        logger.info(f"Saving model artifacts to {model_dir}")
        
        # Save the trained pipeline
        pipeline_path = model_dir / "pipeline.joblib"
        joblib.dump(training_results['best_pipeline'], pipeline_path)
        
        # Save feature engineering artifacts
        feature_engineer = training_results['feature_engineer']
        feature_artifacts = {
            'reference_stats': feature_engineer.reference_stats,
            'feature_sets': feature_engineer.get_final_feature_set()
        }
        
        feature_path = model_dir / "feature_artifacts.joblib"
        joblib.dump(feature_artifacts, feature_path)
        
        # Save model metadata
        metadata = {
            'model_name': model_name,
            'best_model_type': training_results['best_model_name'],
            'training_timestamp': datetime.now().isoformat(),
            'model_comparison': training_results['model_comparison'],
            'optimal_thresholds': training_results['optimal_thresholds'],
            'training_stats': training_results['training_stats'],
            'version': '1.0.0',
            'framework_versions': self._get_framework_versions()
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save configuration
        config_path = model_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # Create model info summary
        self._create_model_summary(model_dir, training_results)
        
        logger.info(f"Model artifacts saved successfully to {model_dir}")
        return str(model_dir)
    
    def load_model_artifacts(self, model_path: str) -> Dict[str, Any]:
        """
        Load complete model artifacts.
        
        Args:
            model_path: Path to model directory or specific model name
            
        Returns:
            Dictionary with loaded model artifacts
        """
        # Handle both full paths and model names
        if Path(model_path).is_absolute() or '/' in model_path:
            model_dir = Path(model_path)
        else:
            model_dir = self.models_dir / model_path
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        logger.info(f"Loading model artifacts from {model_dir}")
        
        # Load pipeline
        pipeline_path = model_dir / "pipeline.joblib"
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
        
        pipeline = joblib.load(pipeline_path)
        
        # Load feature artifacts
        feature_path = model_dir / "feature_artifacts.joblib"
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature artifacts file not found: {feature_path}")
        
        feature_artifacts = joblib.load(feature_path)
        
        # Load metadata
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            logger.warning("Metadata file not found, using defaults")
            metadata = {}
        
        # Load configuration
        config_path = model_dir / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            logger.warning("Config file not found, using defaults")
            config = {}
        
        artifacts = {
            'pipeline': pipeline,
            'feature_artifacts': feature_artifacts,
            'metadata': metadata,
            'config': config,
            'model_dir': str(model_dir)
        }
        
        logger.info("Model artifacts loaded successfully")
        return artifacts
    
    def list_saved_models(self) -> pd.DataFrame:
        """
        List all saved models with their metadata.
        
        Returns:
            DataFrame with model information
        """
        models_info = []
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                metadata_path = model_dir / "metadata.json"
                
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Extract key information
                        model_info = {
                            'model_name': model_dir.name,
                            'model_type': metadata.get('best_model_type', 'Unknown'),
                            'training_date': metadata.get('training_timestamp', 'Unknown'),
                            'test_auc': metadata.get('model_comparison', {}).get(
                                metadata.get('best_model_type', ''), {}
                            ).get('test_auc', 'Unknown'),
                            'cv_auc': metadata.get('model_comparison', {}).get(
                                metadata.get('best_model_type', ''), {}
                            ).get('cv_mean', 'Unknown'),
                            'high_threshold': metadata.get('optimal_thresholds', {}).get('high', 'Unknown'),
                            'medium_threshold': metadata.get('optimal_thresholds', {}).get('medium', 'Unknown'),
                            'total_samples': metadata.get('training_stats', {}).get('total_samples', 'Unknown'),
                            'conversion_rate': metadata.get('training_stats', {}).get('conversion_rate', 'Unknown'),
                            'model_path': str(model_dir)
                        }
                        
                        models_info.append(model_info)
                        
                    except Exception as e:
                        logger.warning(f"Could not read metadata for {model_dir.name}: {str(e)}")
                        models_info.append({
                            'model_name': model_dir.name,
                            'model_type': 'Unknown',
                            'training_date': 'Unknown',
                            'test_auc': 'Unknown',
                            'cv_auc': 'Unknown',
                            'high_threshold': 'Unknown',
                            'medium_threshold': 'Unknown',
                            'total_samples': 'Unknown',
                            'conversion_rate': 'Unknown',
                            'model_path': str(model_dir)
                        })
        
        if models_info:
            df = pd.DataFrame(models_info)
            # Sort by training date (most recent first)
            df = df.sort_values('training_date', ascending=False)
            return df
        else:
            return pd.DataFrame()
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a saved model and all its artifacts.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            True if deletion was successful
        """
        model_dir = self.models_dir / model_name
        
        if not model_dir.exists():
            logger.warning(f"Model {model_name} not found")
            return False
        
        try:
            import shutil
            shutil.rmtree(model_dir)
            logger.info(f"Model {model_name} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {str(e)}")
            return False
    
    def export_model_for_deployment(self, model_name: str, export_path: str) -> str:
        """
        Export model in a deployment-ready format.
        
        Args:
            model_name: Name of the model to export
            export_path: Path where to export the model
            
        Returns:
            Path to exported model
        """
        # Load model artifacts
        artifacts = self.load_model_artifacts(model_name)
        
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Create deployment package
        deployment_package = {
            'pipeline': artifacts['pipeline'],
            'feature_artifacts': artifacts['feature_artifacts'],
            'thresholds': artifacts['metadata'].get('optimal_thresholds', {'high': 0.7, 'medium': 0.4}),
            'model_type': artifacts['metadata'].get('best_model_type', 'Unknown'),
            'version': artifacts['metadata'].get('version', '1.0.0'),
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Save deployment package
        deployment_path = export_dir / f"{model_name}_deployment.joblib"
        joblib.dump(deployment_package, deployment_path)
        
        # Create deployment info
        deployment_info = {
            'model_name': model_name,
            'export_path': str(deployment_path),
            'export_timestamp': datetime.now().isoformat(),
            'model_type': artifacts['metadata'].get('best_model_type', 'Unknown'),
            'thresholds': artifacts['metadata'].get('optimal_thresholds', {'high': 0.7, 'medium': 0.4}),
            'usage_instructions': {
                'load': "joblib.load('path_to_model')",
                'predict': "model['pipeline'].predict_proba(X)[:, 1]",
                'classify': "Use thresholds to assign High/Medium/Low priority"
            }
        }
        
        info_path = export_dir / f"{model_name}_deployment_info.json"
        with open(info_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        logger.info(f"Model exported for deployment to {deployment_path}")
        return str(deployment_path)
    
    def _get_framework_versions(self) -> Dict[str, str]:
        """
        Get versions of key frameworks used.
        
        Returns:
            Dictionary with framework versions
        """
        versions = {}
        
        try:
            import sklearn
            versions['scikit-learn'] = sklearn.__version__
        except ImportError:
            pass
        
        try:
            import catboost
            versions['catboost'] = catboost.__version__
        except ImportError:
            pass
        
        try:
            import pandas
            versions['pandas'] = pandas.__version__
        except ImportError:
            pass
        
        try:
            import numpy
            versions['numpy'] = numpy.__version__
        except ImportError:
            pass
        
        return versions
    
    def _create_model_summary(self, model_dir: Path, training_results: Dict[str, Any]) -> None:
        """
        Create a human-readable model summary.
        
        Args:
            model_dir: Directory where model is saved
            training_results: Training results
        """
        summary_lines = [
            "# LeadScore AI Model Summary",
            f"**Model Type:** {training_results['best_model_name']}",
            f"**Training Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Performance Metrics",
        ]
        
        # Add model comparison
        best_model = training_results['best_model_name']
        if best_model in training_results['model_comparison']:
            metrics = training_results['model_comparison'][best_model]
            summary_lines.extend([
                f"- **Test AUC:** {metrics['test_auc']:.4f}",
                f"- **CV AUC:** {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}",
                f"- **Overfitting Gap:** {metrics['overfitting_gap']:.4f}",
            ])
        
        # Add thresholds
        thresholds = training_results['optimal_thresholds']
        summary_lines.extend([
            "",
            "## Classification Thresholds",
            f"- **High Priority:** ≥ {thresholds['high']:.3f}",
            f"- **Medium Priority:** ≥ {thresholds['medium']:.3f}",
            f"- **Low Priority:** < {thresholds['medium']:.3f}",
        ])
        
        # Add training stats
        stats = training_results['training_stats']
        summary_lines.extend([
            "",
            "## Training Dataset",
            f"- **Total Samples:** {stats['total_samples']:,}",
            f"- **Total Features:** {stats['total_features']}",
            f"- **Conversion Rate:** {stats['conversion_rate']:.1%}",
        ])
        
        # Add model comparison table
        summary_lines.extend([
            "",
            "## Model Comparison",
            "| Model | Test AUC | CV AUC | Overfitting Gap |",
            "|-------|----------|--------|-----------------|",
        ])
        
        for model_name, metrics in training_results['model_comparison'].items():
            summary_lines.append(
                f"| {model_name} | {metrics['test_auc']:.4f} | "
                f"{metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f} | "
                f"{metrics['overfitting_gap']:.4f} |"
            )
        
        # Save summary
        summary_path = model_dir / "README.md"
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))
