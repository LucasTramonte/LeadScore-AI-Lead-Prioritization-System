#!/usr/bin/env python3
"""
Aprix Lead Scoring System - Model Training Script

This script trains machine learning models for lead scoring and prioritization.
It includes data loading, feature engineering, model training, and evaluation.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.services.training_service import TrainingService
from src.core.config_manager import get_config

def main():
    """Main training function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Aprix Lead Scoring System model training...")
    
    try:
        # Load configuration
        config_manager = get_config()
        
        # Initialize training service
        training_service = TrainingService(config_manager)
        
        # Run training
        results = training_service.run_training_pipeline()
        
        logger.info("Training completed successfully!")
        logger.info(f"Best model: {results['best_model_name']}")
        logger.info(f"Test AUC: {results['final_auc']:.4f}")
        logger.info(f"Model saved as: {results['model_saved_as']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
