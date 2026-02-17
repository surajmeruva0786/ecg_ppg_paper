"""
Model comparison and ranking utilities.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def compare_models(results_dict: Dict) -> pd.DataFrame:
    """
    Compare multiple models based on their metrics.
    
    Args:
        results_dict: Dictionary of model results
        
    Returns:
        DataFrame with comparison results
    """
    logger.info("Comparing models...")
    
    comparison_data = []
    
    for model_name, result in results_dict.items():
        if 'metrics' in result:
            metrics = result['metrics']
            row = {'Model': model_name}
            row.update(metrics)
            
            # Add training time if available
            if 'train_time' in result:
                row['Training Time (s)'] = result['train_time']
            
            comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by F1 score (descending)
    if 'f1' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('f1', ascending=False)
    
    return comparison_df


def create_comparison_table(comparison_df: pd.DataFrame, save_path=None) -> pd.DataFrame:
    """
    Create a formatted comparison table.
    
    Args:
        comparison_df: DataFrame with model comparisons
        save_path: Optional path to save the table
        
    Returns:
        Formatted DataFrame
    """
    # Round numeric columns
    numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        comparison_df[col] = comparison_df[col].round(4)
    
    if save_path:
        comparison_df.to_csv(save_path, index=False)
        logger.info(f"Comparison table saved to {save_path}")
    
    return comparison_df


def rank_models(comparison_df: pd.DataFrame, metrics=['f1', 'roc_auc', 'accuracy']) -> pd.DataFrame:
    """
    Rank models based on multiple metrics.
    
    Args:
        comparison_df: DataFrame with model comparisons
        metrics: List of metrics to consider for ranking
        
    Returns:
        DataFrame with rankings
    """
    ranking_df = comparison_df.copy()
    
    # Calculate average rank across metrics
    ranks = []
    for metric in metrics:
        if metric in ranking_df.columns:
            ranking_df[f'{metric}_rank'] = ranking_df[metric].rank(ascending=False)
            ranks.append(f'{metric}_rank')
    
    if ranks:
        ranking_df['average_rank'] = ranking_df[ranks].mean(axis=1)
        ranking_df = ranking_df.sort_values('average_rank')
    
    return ranking_df


def statistical_comparison(results1, results2, metric='f1', alpha=0.05):
    """
    Perform statistical comparison between two models.
    
    Args:
        results1: Results from model 1 (including CV scores)
        results2: Results from model 2 (including CV scores)
        metric: Metric to compare
        alpha: Significance level
        
    Returns:
        Dictionary with comparison results
    """
    if 'cv_scores' in results1 and 'cv_scores' in results2:
        scores1 = results1['cv_scores']
        scores2 = results2['cv_scores']
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        is_significant = p_value < alpha
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': is_significant,
            'mean_diff': np.mean(scores1) - np.mean(scores2)
        }
    else:
        logger.warning("Cross-validation scores not available for statistical comparison")
        return None


if __name__ == "__main__":
    # Test comparison functions
    print("Testing Model Comparison Module")
    
    # Create dummy results
    results = {
        'Model A': {
            'metrics': {
                'accuracy': 0.85,
                'precision': 0.83,
                'recall': 0.87,
                'f1': 0.85,
                'roc_auc': 0.90
            },
            'train_time': 10.5
        },
        'Model B': {
            'metrics': {
                'accuracy': 0.88,
                'precision': 0.86,
                'recall': 0.89,
                'f1': 0.87,
                'roc_auc': 0.92
            },
            'train_time': 25.3
        },
        'Model C': {
            'metrics': {
                'accuracy': 0.82,
                'precision': 0.80,
                'recall': 0.84,
                'f1': 0.82,
                'roc_auc': 0.88
            },
            'train_time': 5.2
        }
    }
    
    # Compare models
    comparison_df = compare_models(results)
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Rank models
    ranked_df = rank_models(comparison_df)
    print("\nModel Rankings:")
    print(ranked_df[['Model', 'f1', 'roc_auc', 'average_rank']])
