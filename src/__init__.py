"""
LeadScore AI - Lead Prioritization System

A comprehensive machine learning system for automated lead scoring and prioritization
in industrial software companies.
"""

__version__ = "1.0.0"
__author__ = "LeadScore AI Team"
__description__ = "AI-powered lead scoring and prioritization system"

# Core modules
from . import data
from . import model
from . import scoring
from . import cli

# Main classes for easy import
from .data import DataLoader, FeatureEngineer, DataValidator
from .model import ModelTrainer, ModelEvaluator, ModelPersistence
from .scoring import LeadScorer, BatchScorer

__all__ = [
    # Modules
    'data',
    'model', 
    'scoring',
    'cli',
    # Classes
    'DataLoader',
    'FeatureEngineer', 
    'DataValidator',
    'ModelTrainer',
    'ModelEvaluator',
    'ModelPersistence',
    'LeadScorer',
    'BatchScorer'
]
