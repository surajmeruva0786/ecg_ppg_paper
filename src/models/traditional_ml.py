"""
Traditional Machine Learning models for heart attack prediction.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from pathlib import Path
from typing import Dict, Tuple, Any
import logging
import time

logger = logging.getLogger(__name__)


def train_logistic_regression(
    X_train, y_train, X_val, y_val,
    param_grid=None,
    cv=5,
    random_state=42
) -> Tuple[Any, Dict]:
    """
    Train Logistic Regression with hyperparameter tuning.
    """
    logger.info("Training Logistic Regression...")
    
    if param_grid is None:
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000]
        }
    
    start_time = time.time()
    
    model = LogisticRegression(random_state=random_state)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'model_name': 'Logistic Regression',
        'best_params': grid_search.best_params_,
        'train_time': train_time,
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, average='weighted'),
        'recall': recall_score(y_val, y_pred, average='weighted'),
        'f1': f1_score(y_val, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) == 2 else 0
    }
    
    logger.info(f"Logistic Regression - F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return best_model, metrics


def train_decision_tree(
    X_train, y_train, X_val, y_val,
    param_grid=None,
    cv=5,
    random_state=42
) -> Tuple[Any, Dict]:
    """
    Train Decision Tree with hyperparameter tuning.
    """
    logger.info("Training Decision Tree...")
    
    if param_grid is None:
        param_grid = {
            'max_depth': [3, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
    
    start_time = time.time()
    
    model = DecisionTreeClassifier(random_state=random_state)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'model_name': 'Decision Tree',
        'best_params': grid_search.best_params_,
        'train_time': train_time,
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, average='weighted'),
        'recall': recall_score(y_val, y_pred, average='weighted'),
        'f1': f1_score(y_val, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) == 2 else 0
    }
    
    logger.info(f"Decision Tree - F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return best_model, metrics


def train_random_forest(
    X_train, y_train, X_val, y_val,
    param_grid=None,
    cv=5,
    random_state=42
) -> Tuple[Any, Dict]:
    """
    Train Random Forest with hyperparameter tuning.
    """
    logger.info("Training Random Forest...")
    
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    
    start_time = time.time()
    
    model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'model_name': 'Random Forest',
        'best_params': grid_search.best_params_,
        'train_time': train_time,
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, average='weighted'),
        'recall': recall_score(y_val, y_pred, average='weighted'),
        'f1': f1_score(y_val, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) == 2 else 0
    }
    
    logger.info(f"Random Forest - F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return best_model, metrics


def train_svm(
    X_train, y_train, X_val, y_val,
    param_grid=None,
    cv=5,
    random_state=42
) -> Tuple[Any, Dict]:
    """
    Train SVM with hyperparameter tuning.
    """
    logger.info("Training SVM...")
    
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    
    start_time = time.time()
    
    model = SVC(probability=True, random_state=random_state)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'model_name': 'SVM',
        'best_params': grid_search.best_params_,
        'train_time': train_time,
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, average='weighted'),
        'recall': recall_score(y_val, y_pred, average='weighted'),
        'f1': f1_score(y_val, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) == 2 else 0
    }
    
    logger.info(f"SVM - F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return best_model, metrics


def train_knn(
    X_train, y_train, X_val, y_val,
    param_grid=None,
    cv=5,
    random_state=42
) -> Tuple[Any, Dict]:
    """
    Train K-Nearest Neighbors with hyperparameter tuning.
    """
    logger.info("Training KNN...")
    
    if param_grid is None:
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    
    start_time = time.time()
    
    model = KNeighborsClassifier(n_jobs=-1)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'model_name': 'KNN',
        'best_params': grid_search.best_params_,
        'train_time': train_time,
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, average='weighted'),
        'recall': recall_score(y_val, y_pred, average='weighted'),
        'f1': f1_score(y_val, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) == 2 else 0
    }
    
    logger.info(f"KNN - F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return best_model, metrics


def train_xgboost(
    X_train, y_train, X_val, y_val,
    param_grid=None,
    cv=5,
    random_state=42
) -> Tuple[Any, Dict]:
    """
    Train XGBoost with hyperparameter tuning.
    """
    logger.info("Training XGBoost...")
    
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0]
        }
    
    start_time = time.time()
    
    model = XGBClassifier(random_state=random_state, n_jobs=-1, eval_metric='logloss')
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'model_name': 'XGBoost',
        'best_params': grid_search.best_params_,
        'train_time': train_time,
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, average='weighted'),
        'recall': recall_score(y_val, y_pred, average='weighted'),
        'f1': f1_score(y_val, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) == 2 else 0
    }
    
    logger.info(f"XGBoost - F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return best_model, metrics


def train_lightgbm(
    X_train, y_train, X_val, y_val,
    param_grid=None,
    cv=5,
    random_state=42
) -> Tuple[Any, Dict]:
    """
    Train LightGBM with hyperparameter tuning.
    """
    logger.info("Training LightGBM...")
    
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'num_leaves': [31, 50]
        }
    
    start_time = time.time()
    
    model = LGBMClassifier(random_state=random_state, n_jobs=-1, verbose=-1)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'model_name': 'LightGBM',
        'best_params': grid_search.best_params_,
        'train_time': train_time,
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, average='weighted'),
        'recall': recall_score(y_val, y_pred, average='weighted'),
        'f1': f1_score(y_val, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) == 2 else 0
    }
    
    logger.info(f"LightGBM - F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return best_model, metrics


def train_catboost(
    X_train, y_train, X_val, y_val,
    param_grid=None,
    cv=5,
    random_state=42
) -> Tuple[Any, Dict]:
    """
    Train CatBoost with hyperparameter tuning.
    """
    logger.info("Training CatBoost...")
    
    if param_grid is None:
        param_grid = {
            'iterations': [100, 200],
            'depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3]
        }
    
    start_time = time.time()
    
    model = CatBoostClassifier(random_state=random_state, verbose=0)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'model_name': 'CatBoost',
        'best_params': grid_search.best_params_,
        'train_time': train_time,
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, average='weighted'),
        'recall': recall_score(y_val, y_pred, average='weighted'),
        'f1': f1_score(y_val, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) == 2 else 0
    }
    
    logger.info(f"CatBoost - F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return best_model, metrics


def train_all_ml_models(
    X_train, y_train, X_val, y_val,
    models_to_train=None,
    save_dir='results/models',
    cv=5,
    random_state=42
) -> Dict:
    """
    Train all traditional ML models and return results.
    """
    if models_to_train is None:
        models_to_train = ['logistic_regression', 'decision_tree', 'random_forest', 
                          'svm', 'knn', 'xgboost', 'lightgbm', 'catboost']
    
    results = {}
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    model_functions = {
        'logistic_regression': train_logistic_regression,
        'decision_tree': train_decision_tree,
        'random_forest': train_random_forest,
        'svm': train_svm,
        'knn': train_knn,
        'xgboost': train_xgboost,
        'lightgbm': train_lightgbm,
        'catboost': train_catboost
    }
    
    for model_name in models_to_train:
        if model_name in model_functions:
            try:
                model, metrics = model_functions[model_name](
                    X_train, y_train, X_val, y_val, cv=cv, random_state=random_state
                )
                results[model_name] = {
                    'model': model,
                    'metrics': metrics
                }
                
                # Save model
                model_file = save_path / f'{model_name}.pkl'
                joblib.dump(model, model_file)
                logger.info(f"Saved {model_name} to {model_file}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Test with synthetic data
    from sklearn.datasets import make_classification
    
    print("Testing Traditional ML Models")
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a single model
    model, metrics = train_random_forest(X_train, y_train, X_val, y_val)
    
    print("\nRandom Forest Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
