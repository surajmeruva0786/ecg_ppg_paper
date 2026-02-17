"""
Data splitting and augmentation module.

This module provides functions for splitting data into train/val/test sets,
handling class imbalance, and data augmentation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        train_size: Proportion of training data
        val_size: Proportion of validation data
        test_size: Proportion of test data
        random_state: Random seed for reproducibility
        stratify: Whether to use stratified splitting
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "train_size + val_size + test_size must equal 1.0"
    
    logger.info(f"Splitting data: train={train_size}, val={val_size}, test={test_size}")
    
    # First split: separate test set
    stratify_param = y if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    # Second split: separate train and validation
    val_ratio = val_size / (train_size + val_size)
    stratify_param = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=stratify_param
    )
    
    logger.info(f"Train set: {X_train.shape}, Val set: {X_val.shape}, Test set: {X_test.shape}")
    logger.info(f"Train labels: {y_train.value_counts().to_dict()}")
    logger.info(f"Val labels: {y_val.value_counts().to_dict()}")
    logger.info(f"Test labels: {y_test.value_counts().to_dict()}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: str = 'auto',
    random_state: int = 42,
    k_neighbors: int = 5
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to balance training data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        sampling_strategy: Sampling strategy for SMOTE
        random_state: Random seed
        k_neighbors: Number of nearest neighbors for SMOTE
        
    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    logger.info("Applying SMOTE for class balancing")
    logger.info(f"Original class distribution: {y_train.value_counts().to_dict()}")
    
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=k_neighbors
    )
    
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Convert back to DataFrame/Series
    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled, name=y_train.name)
    
    logger.info(f"Resampled class distribution: {y_resampled.value_counts().to_dict()}")
    logger.info(f"Original size: {len(y_train)}, Resampled size: {len(y_resampled)}")
    
    return X_resampled, y_resampled


def apply_adasyn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: str = 'auto',
    random_state: int = 42,
    n_neighbors: int = 5
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply ADASYN to balance training data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        sampling_strategy: Sampling strategy
        random_state: Random seed
        n_neighbors: Number of nearest neighbors
        
    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    logger.info("Applying ADASYN for class balancing")
    logger.info(f"Original class distribution: {y_train.value_counts().to_dict()}")
    
    adasyn = ADASYN(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        n_neighbors=n_neighbors
    )
    
    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
    
    # Convert back to DataFrame/Series
    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled, name=y_train.name)
    
    logger.info(f"Resampled class distribution: {y_resampled.value_counts().to_dict()}")
    
    return X_resampled, y_resampled


def apply_smote_tomek(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: str = 'auto',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE + Tomek links for class balancing.
    
    Args:
        X_train: Training features
        y_train: Training labels
        sampling_strategy: Sampling strategy
        random_state: Random seed
        
    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    logger.info("Applying SMOTE + Tomek for class balancing")
    logger.info(f"Original class distribution: {y_train.value_counts().to_dict()}")
    
    smote_tomek = SMOTETomek(
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )
    
    X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
    
    # Convert back to DataFrame/Series
    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled, name=y_train.name)
    
    logger.info(f"Resampled class distribution: {y_resampled.value_counts().to_dict()}")
    
    return X_resampled, y_resampled


def augment_timeseries(
    X: np.ndarray,
    y: np.ndarray,
    augmentation_factor: int = 2,
    jitter_std: float = 0.01,
    scaling_factor: float = 0.1,
    rotation_angle: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment time-series data with jittering, scaling, and rotation.
    
    Args:
        X: Input data (samples, features)
        y: Labels
        augmentation_factor: How many augmented copies to create
        jitter_std: Standard deviation for jittering
        scaling_factor: Factor for scaling augmentation
        rotation_angle: Angle for rotation augmentation
        
    Returns:
        Tuple of (X_augmented, y_augmented)
    """
    logger.info(f"Augmenting time-series data with factor {augmentation_factor}")
    
    X_aug_list = [X]
    y_aug_list = [y]
    
    for i in range(augmentation_factor - 1):
        # Jittering: add random noise
        X_jitter = X + np.random.normal(0, jitter_std, X.shape)
        
        # Scaling: multiply by random factor
        scale = 1 + np.random.uniform(-scaling_factor, scaling_factor)
        X_scaled = X * scale
        
        # Rotation: apply rotation matrix (simplified for 1D signals)
        X_rotated = X + rotation_angle * np.random.randn(*X.shape)
        
        # Randomly choose augmentation type
        aug_type = np.random.choice(['jitter', 'scale', 'rotate'])
        if aug_type == 'jitter':
            X_aug_list.append(X_jitter)
        elif aug_type == 'scale':
            X_aug_list.append(X_scaled)
        else:
            X_aug_list.append(X_rotated)
        
        y_aug_list.append(y)
    
    X_augmented = np.vstack(X_aug_list)
    y_augmented = np.hstack(y_aug_list)
    
    logger.info(f"Original size: {X.shape}, Augmented size: {X_augmented.shape}")
    
    return X_augmented, y_augmented


def get_class_weights(y: pd.Series) -> dict:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y: Target labels
        
    Returns:
        Dictionary of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, weights))
    
    logger.info(f"Class weights: {class_weights}")
    
    return class_weights


if __name__ == "__main__":
    # Test data splitting
    print("Testing Data Splitting Module")
    
    # Generate synthetic data
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=42
    )
    
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y, name='target')
    
    # Test splitting
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X_df, y_series,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15
    )
    
    # Test SMOTE
    X_smote, y_smote = apply_smote(X_train, y_train)
    
    # Test class weights
    weights = get_class_weights(y_train)
