"""
Model evaluation module for the LeadScore AI system.
Provides comprehensive model evaluation and performance analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, learning_curve
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation and performance analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ModelEvaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
    def evaluate_binary_classification(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                     y_pred: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Comprehensive binary classification evaluation.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            y_pred: Predicted labels (optional, will be computed if not provided)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if y_pred is None:
            y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Basic metrics
        auc_score = roc_auc_score(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # ROC curve data
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        
        # Precision-Recall curve data
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        return {
            'auc_score': auc_score,
            'average_precision': avg_precision,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds},
            'pr_curve': {'precision': precision, 'recall': recall, 'thresholds': pr_thresholds}
        }
    
    def evaluate_lead_scoring_performance(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                        thresholds: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluate lead scoring performance with priority levels.
        
        Args:
            y_true: True conversion labels
            y_pred_proba: Predicted conversion probabilities
            thresholds: Dictionary with 'high' and 'medium' thresholds
            
        Returns:
            Dictionary with lead scoring performance metrics
        """
        high_threshold = thresholds['high']
        medium_threshold = thresholds['medium']
        
        # Assign priorities
        priorities = np.where(y_pred_proba >= high_threshold, 'High',
                     np.where(y_pred_proba >= medium_threshold, 'Medium', 'Low'))
        
        # Calculate metrics for each priority level
        priority_metrics = {}
        
        for priority in ['High', 'Medium', 'Low']:
            mask = priorities == priority
            if mask.sum() > 0:
                priority_metrics[priority] = {
                    'count': int(mask.sum()),
                    'percentage': float(mask.sum() / len(y_true) * 100),
                    'conversion_rate': float(y_true[mask].mean()),
                    'total_conversions': int(y_true[mask].sum())
                }
            else:
                priority_metrics[priority] = {
                    'count': 0,
                    'percentage': 0.0,
                    'conversion_rate': 0.0,
                    'total_conversions': 0
                }
        
        # Calculate lift metrics
        overall_conversion_rate = y_true.mean()
        
        for priority in priority_metrics:
            if priority_metrics[priority]['count'] > 0:
                priority_metrics[priority]['lift'] = (
                    priority_metrics[priority]['conversion_rate'] / overall_conversion_rate
                )
            else:
                priority_metrics[priority]['lift'] = 0.0
        
        # Calculate cumulative metrics
        sorted_indices = np.argsort(y_pred_proba)[::-1]  # Sort by probability descending
        sorted_true = y_true[sorted_indices]
        
        # Top percentiles performance
        percentiles = [5, 10, 20, 30, 50]
        percentile_metrics = {}
        
        for p in percentiles:
            n_samples = int(len(y_true) * p / 100)
            if n_samples > 0:
                top_p_conversion = sorted_true[:n_samples].mean()
                percentile_metrics[f'top_{p}%'] = {
                    'conversion_rate': float(top_p_conversion),
                    'lift': float(top_p_conversion / overall_conversion_rate),
                    'sample_count': n_samples
                }
        
        return {
            'priority_metrics': priority_metrics,
            'percentile_metrics': percentile_metrics,
            'overall_conversion_rate': float(overall_conversion_rate),
            'thresholds_used': thresholds
        }
    
    def calculate_business_impact(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                thresholds: Dict[str, float], 
                                revenue_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate business impact metrics for lead scoring.
        
        Args:
            y_true: True conversion labels
            y_pred_proba: Predicted conversion probabilities
            thresholds: Dictionary with 'high' and 'medium' thresholds
            revenue_data: Optional revenue data for each lead
            
        Returns:
            Dictionary with business impact metrics
        """
        high_threshold = thresholds['high']
        medium_threshold = thresholds['medium']
        
        # Assign priorities
        priorities = np.where(y_pred_proba >= high_threshold, 'High',
                     np.where(y_pred_proba >= medium_threshold, 'Medium', 'Low'))
        
        # Calculate efficiency metrics
        total_conversions = y_true.sum()
        high_priority_mask = priorities == 'High'
        high_priority_conversions = y_true[high_priority_mask].sum()
        
        # Efficiency: what percentage of conversions are captured in high priority
        conversion_capture_rate = high_priority_conversions / total_conversions if total_conversions > 0 else 0
        
        # Efficiency: what percentage of leads need to be processed to get X% of conversions
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        sorted_true = y_true[sorted_indices]
        cumulative_conversions = np.cumsum(sorted_true)
        
        # Find percentage of leads needed to capture 50%, 80% of conversions
        efficiency_metrics = {}
        for target_pct in [50, 80]:
            target_conversions = total_conversions * target_pct / 100
            if target_conversions > 0:
                idx = np.where(cumulative_conversions >= target_conversions)[0]
                if len(idx) > 0:
                    leads_needed_pct = (idx[0] + 1) / len(y_true) * 100
                    efficiency_metrics[f'leads_for_{target_pct}%_conversions'] = leads_needed_pct
        
        # Revenue impact (if revenue data provided)
        revenue_impact = {}
        if revenue_data is not None:
            for priority in ['High', 'Medium', 'Low']:
                mask = priorities == priority
                if mask.sum() > 0:
                    priority_revenue = revenue_data[mask]
                    converted_revenue = revenue_data[mask & (y_true == 1)]
                    
                    revenue_impact[priority] = {
                        'total_potential_revenue': float(priority_revenue.sum()),
                        'converted_revenue': float(converted_revenue.sum()),
                        'avg_revenue_per_lead': float(priority_revenue.mean()),
                        'avg_revenue_per_conversion': float(converted_revenue.mean()) if len(converted_revenue) > 0 else 0
                    }
        
        return {
            'conversion_capture_rate': float(conversion_capture_rate),
            'efficiency_metrics': efficiency_metrics,
            'revenue_impact': revenue_impact,
            'total_conversions': int(total_conversions),
            'high_priority_conversions': int(high_priority_conversions),
            'high_priority_leads': int(high_priority_mask.sum())
        }
    
    def cross_validate_model(self, pipeline, X: pd.DataFrame, y: pd.Series,
                           cv_folds: int = 5, scoring: str = 'roc_auc') -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.
        
        Args:
            pipeline: Trained pipeline
            X: Features
            y: Target
            cv_folds: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with CV results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
        
        return {
            'cv_scores': cv_scores.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'cv_min': float(cv_scores.min()),
            'cv_max': float(cv_scores.max()),
            'scoring_metric': scoring,
            'cv_folds': cv_folds
        }
    
    def analyze_model_stability(self, pipeline, X: pd.DataFrame, y: pd.Series,
                              n_iterations: int = 10) -> Dict[str, Any]:
        """
        Analyze model stability across multiple random splits.
        
        Args:
            pipeline: Model pipeline
            X: Features
            y: Target
            n_iterations: Number of random splits to test
            
        Returns:
            Dictionary with stability metrics
        """
        logger.info(f"Analyzing model stability across {n_iterations} random splits...")
        
        auc_scores = []
        
        for i in range(n_iterations):
            # Random split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=i, stratify=y
            )
            
            # Train and evaluate
            pipeline.fit(X_train, y_train)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            auc_scores.append(auc)
        
        auc_scores = np.array(auc_scores)
        
        return {
            'auc_scores': auc_scores.tolist(),
            'mean_auc': float(auc_scores.mean()),
            'std_auc': float(auc_scores.std()),
            'min_auc': float(auc_scores.min()),
            'max_auc': float(auc_scores.max()),
            'auc_range': float(auc_scores.max() - auc_scores.min()),
            'coefficient_of_variation': float(auc_scores.std() / auc_scores.mean()),
            'n_iterations': n_iterations
        }
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                 thresholds: Dict[str, float],
                                 model_name: str = "Model") -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            thresholds: Classification thresholds
            model_name: Name of the model
            
        Returns:
            Comprehensive evaluation report
        """
        logger.info(f"Generating evaluation report for {model_name}...")
        
        # Binary classification metrics
        binary_metrics = self.evaluate_binary_classification(y_true, y_pred_proba)
        
        # Lead scoring performance
        scoring_performance = self.evaluate_lead_scoring_performance(y_true, y_pred_proba, thresholds)
        
        # Business impact
        business_impact = self.calculate_business_impact(y_true, y_pred_proba, thresholds)
        
        # Compile report
        report = {
            'model_name': model_name,
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'dataset_info': {
                'total_samples': len(y_true),
                'positive_samples': int(y_true.sum()),
                'negative_samples': int(len(y_true) - y_true.sum()),
                'positive_rate': float(y_true.mean())
            },
            'binary_classification_metrics': {
                'auc_score': binary_metrics['auc_score'],
                'average_precision': binary_metrics['average_precision'],
                'classification_report': binary_metrics['classification_report']
            },
            'lead_scoring_performance': scoring_performance,
            'business_impact': business_impact,
            'thresholds': thresholds
        }
        
        return report
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple models performance.
        
        Args:
            model_results: Dictionary with model results
            
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            comparison_data.append({
                'Model': model_name,
                'Train_AUC': results.get('train_auc', 0),
                'Test_AUC': results.get('test_auc', 0),
                'CV_AUC_Mean': results.get('cv_mean', 0),
                'CV_AUC_Std': results.get('cv_std', 0),
                'Overfitting_Gap': results.get('overfitting_gap', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test_AUC', ascending=False)
        
        return comparison_df
    
    def plot_model_performance(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                             model_name: str = "Model", save_path: Optional[str] = None) -> None:
        """
        Plot model performance visualizations.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Optional path to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} Performance Analysis', fontsize=16, fontweight='bold')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.6)
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        axes[0, 1].plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {avg_precision:.3f})')
        axes[0, 1].axhline(y=y_true.mean(), color='k', linestyle='--', alpha=0.6, label='Baseline')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Probability Distribution
        axes[1, 0].hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7, label='Non-converted', density=True)
        axes[1, 0].hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7, label='Converted', density=True)
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Probability Distribution by Class')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Confusion Matrix
        y_pred = (y_pred_proba >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        axes[1, 1].set_title('Confusion Matrix (threshold=0.5)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plots saved to {save_path}")
        
        plt.show()
