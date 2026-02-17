"""
Data loading module for ECG and PPG datasets.

This module provides functions to load, validate, and prepare the ECG and PPG datasets
for further processing.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_ecg(file_path: str = None, config: Dict = None) -> pd.DataFrame:
    """
    Load ECG dataset.
    
    Args:
        file_path: Path to ECG CSV file
        config: Configuration dictionary
        
    Returns:
        DataFrame containing ECG data
    """
    if config is None:
        config = load_config()
    
    if file_path is None:
        file_path = config['data']['ecg_path']
    
    logger.info(f"Loading ECG dataset from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"ECG dataset loaded successfully: {df.shape}")
        logger.info(f"Columns: {df.shape[1]}, Samples: {df.shape[0]}")
        
        # Basic validation
        if df.isnull().sum().sum() > 0:
            logger.warning(f"Found {df.isnull().sum().sum()} missing values in ECG dataset")
        
        # Check for label column (assuming last column is label)
        label_col = df.columns[-1]
        logger.info(f"Label column: {label_col}")
        logger.info(f"Label distribution:\n{df[label_col].value_counts()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading ECG dataset: {e}")
        raise


def load_ppg(file_path: str = None, config: Dict = None) -> pd.DataFrame:
    """
    Load PPG dataset.
    
    Args:
        file_path: Path to PPG CSV file
        config: Configuration dictionary
        
    Returns:
        DataFrame containing PPG data
    """
    if config is None:
        config = load_config()
    
    if file_path is None:
        file_path = config['data']['ppg_path']
    
    logger.info(f"Loading PPG dataset from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"PPG dataset loaded successfully: {df.shape}")
        logger.info(f"Columns: {df.shape[1]}, Samples: {df.shape[0]}")
        
        # Basic validation
        if df.isnull().sum().sum() > 0:
            logger.warning(f"Found {df.isnull().sum().sum()} missing values in PPG dataset")
        
        # Check for label column (assuming last column is label)
        label_col = df.columns[-1]
        logger.info(f"Label column: {label_col}")
        logger.info(f"Label distribution:\n{df[label_col].value_counts()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading PPG dataset: {e}")
        raise


def load_datasets(config_path: str = "config.yaml") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both ECG and PPG datasets.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (ecg_df, ppg_df)
    """
    config = load_config(config_path)
    
    ecg_df = load_ecg(config=config)
    ppg_df = load_ppg(config=config)
    
    return ecg_df, ppg_df


def get_data_info(df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict[str, Any]:
    """
    Get comprehensive information about a dataset.
    
    Args:
        df: Input DataFrame
        dataset_name: Name of the dataset for logging
        
    Returns:
        Dictionary containing dataset statistics
    """
    info = {
        'name': dataset_name,
        'shape': df.shape,
        'n_samples': df.shape[0],
        'n_features': df.shape[1] - 1,  # Excluding label column
        'missing_values': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'dtypes': df.dtypes.value_counts().to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
    }
    
    # Label information (assuming last column is label)
    label_col = df.columns[-1]
    info['label_column'] = label_col
    info['label_distribution'] = df[label_col].value_counts().to_dict()
    info['n_classes'] = df[label_col].nunique()
    
    # Feature statistics
    feature_cols = df.columns[:-1]
    info['feature_stats'] = {
        'mean': df[feature_cols].mean().mean(),
        'std': df[feature_cols].std().mean(),
        'min': df[feature_cols].min().min(),
        'max': df[feature_cols].max().max(),
    }
    
    return info


def validate_dataset(df: pd.DataFrame, dataset_name: str = "Dataset") -> bool:
    """
    Validate dataset for common issues.
    
    Args:
        df: Input DataFrame
        dataset_name: Name of the dataset
        
    Returns:
        True if validation passes, False otherwise
    """
    logger.info(f"Validating {dataset_name}...")
    
    issues = []
    
    # Check for empty dataset
    if df.empty:
        issues.append("Dataset is empty")
    
    # Check for duplicate rows
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        issues.append(f"Found {n_duplicates} duplicate rows")
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        issues.append(f"Found {len(constant_cols)} constant columns: {constant_cols[:5]}")
    
    # Check for high missing value columns (>50%)
    high_missing = df.columns[df.isnull().sum() / len(df) > 0.5].tolist()
    if high_missing:
        issues.append(f"Found {len(high_missing)} columns with >50% missing values")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_cols = [col for col in numeric_cols if np.isinf(df[col]).any()]
    if inf_cols:
        issues.append(f"Found infinite values in {len(inf_cols)} columns")
    
    if issues:
        logger.warning(f"Validation issues found in {dataset_name}:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    else:
        logger.info(f"{dataset_name} validation passed!")
        return True


if __name__ == "__main__":
    # Test data loading
    print("="*80)
    print("Testing Data Loading Module")
    print("="*80)
    
    # Load datasets
    ecg_df, ppg_df = load_datasets()
    
    # Get dataset info
    print("\n" + "="*80)
    print("ECG Dataset Information")
    print("="*80)
    ecg_info = get_data_info(ecg_df, "ECG")
    for key, value in ecg_info.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*80)
    print("PPG Dataset Information")
    print("="*80)
    ppg_info = get_data_info(ppg_df, "PPG")
    for key, value in ppg_info.items():
        print(f"{key}: {value}")
    
    # Validate datasets
    print("\n" + "="*80)
    print("Dataset Validation")
    print("="*80)
    validate_dataset(ecg_df, "ECG")
    validate_dataset(ppg_df, "PPG")
