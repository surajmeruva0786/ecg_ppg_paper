"""
Metrics calculation and model evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef, cohen_kappa_score, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
    }
    
    # Add ROC-AUC and PR-AUC if probabilities are provided
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_proba)
        except:
            metrics['roc_auc'] = 0
            metrics['pr_auc'] = 0
    
    # Calculate specificity and sensitivity for binary classification
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics


def cross_validate_model(model, X, y, cv=5, scoring='f1') -> Dict[str, Any]:
    """
    Perform cross-validation on a model.
    
    Args:
        model: Scikit-learn compatible model
        X: Feature matrix
        y: Target vector
        cv: Number of folds
        scoring: Scoring metric
        
    Returns:
        Dictionary with cross-validation results
    """
    logger.info(f"Performing {cv}-fold cross-validation...")
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    
    results = {
        'cv_scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'min_score': np.min(scores),
        'max_score': np.max(scores)
    }
    
    logger.info(f"CV {scoring}: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
    
    return results


def evaluate_model(model, X_test, y_test, model_name='Model') -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating {model_name}...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Probabilities (if available)
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba_full = model.predict_proba(X_test)
        if len(y_proba_full.shape) > 1 and y_proba_full.shape[1] > 1:
            y_proba = y_proba_full[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    results = {
        'model_name': model_name,
        'metrics': metrics,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred,
        'probabilities': y_proba
    }
    
    logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    
    return results


if __name__ == "__main__":
    # Test metrics calculation
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    print("Testing Metrics Module")
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    results = evaluate_model(model, X_test, y_test, 'Random Forest')
    
    print("\nMetrics:")
    for k, v in results['metrics'].items():
        print(f"  {k}: {v:.4f}")
    
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
