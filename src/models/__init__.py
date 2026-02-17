"""
Models module for ECG and PPG classification.
"""

from .traditional_ml import (
    train_logistic_regression,
    train_decision_tree,
    train_random_forest,
    train_svm,
    train_knn,
    train_xgboost,
    train_lightgbm,
    train_catboost,
    train_all_ml_models
)

__all__ = [
    'train_logistic_regression',
    'train_decision_tree',
    'train_random_forest',
    'train_svm',
    'train_knn',
    'train_xgboost',
    'train_lightgbm',
    'train_catboost',
    'train_all_ml_models'
]
