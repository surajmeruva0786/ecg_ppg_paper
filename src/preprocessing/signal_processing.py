"""
Signal processing module for ECG and PPG data.

This module provides functions for filtering, normalization, outlier removal,
and missing value handling.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from typing import Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def normalize_signal(
    data: Union[pd.DataFrame, np.ndarray],
    method: str = 'standard',
    feature_range: Tuple[float, float] = (0, 1)
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Normalize signal data.
    
    Args:
        data: Input data (DataFrame or array)
        method: Normalization method ('standard', 'minmax', 'robust')
        feature_range: Range for minmax scaling
        
    Returns:
        Normalized data in same format as input
    """
    is_dataframe = isinstance(data, pd.DataFrame)
    
    if is_dataframe:
        columns = data.columns
        index = data.index
        values = data.values
    else:
        values = data
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler(feature_range=feature_range)
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    normalized = scaler.fit_transform(values)
    
    if is_dataframe:
        return pd.DataFrame(normalized, columns=columns, index=index)
    else:
        return normalized


def filter_signal(
    signal_data: np.ndarray,
    fs: float = 100.0,
    lowcut: float = 0.5,
    highcut: float = 50.0,
    filter_type: str = 'butterworth',
    order: int = 4
) -> np.ndarray:
    """
    Apply bandpass filter to signal.
    
    Args:
        signal_data: Input signal (1D array)
        fs: Sampling frequency in Hz
        lowcut: Low cutoff frequency
        highcut: High cutoff frequency
        filter_type: Type of filter ('butterworth', 'savgol')
        order: Filter order
        
    Returns:
        Filtered signal
    """
    if filter_type == 'butterworth':
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = signal.butter(order, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, signal_data)
        
    elif filter_type == 'savgol':
        window_length = min(51, len(signal_data) if len(signal_data) % 2 == 1 else len(signal_data) - 1)
        filtered = signal.savgol_filter(signal_data, window_length, order)
        
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    return filtered


def apply_filtering_to_dataframe(
    df: pd.DataFrame,
    fs: float = 100.0,
    lowcut: float = 0.5,
    highcut: float = 50.0,
    filter_type: str = 'butterworth',
    order: int = 4,
    exclude_label: bool = True
) -> pd.DataFrame:
    """
    Apply filtering to all columns in a DataFrame.
    
    Args:
        df: Input DataFrame
        fs: Sampling frequency
        lowcut: Low cutoff frequency
        highcut: High cutoff frequency
        filter_type: Type of filter
        order: Filter order
        exclude_label: Whether to exclude the last column (assumed to be label)
        
    Returns:
        Filtered DataFrame
    """
    logger.info(f"Applying {filter_type} filter to dataset...")
    
    df_filtered = df.copy()
    
    # Determine columns to filter
    if exclude_label:
        feature_cols = df.columns[:-1]
        label_col = df.columns[-1]
    else:
        feature_cols = df.columns
    
    # Apply filtering to each column
    for col in feature_cols:
        try:
            df_filtered[col] = filter_signal(
                df[col].values,
                fs=fs,
                lowcut=lowcut,
                highcut=highcut,
                filter_type=filter_type,
                order=order
            )
        except Exception as e:
            logger.warning(f"Could not filter column {col}: {e}")
    
    logger.info("Filtering completed")
    return df_filtered


def remove_outliers(
    df: pd.DataFrame,
    method: str = 'iqr',
    threshold: float = 1.5,
    contamination: float = 0.1,
    exclude_label: bool = True
) -> pd.DataFrame:
    """
    Remove or cap outliers from dataset.
    
    Args:
        df: Input DataFrame
        method: Outlier detection method ('iqr', 'isolation_forest', 'zscore')
        threshold: Threshold for IQR or z-score method
        contamination: Contamination parameter for isolation forest
        exclude_label: Whether to exclude the last column
        
    Returns:
        DataFrame with outliers handled
    """
    logger.info(f"Removing outliers using {method} method...")
    
    df_clean = df.copy()
    
    if exclude_label:
        feature_cols = df.columns[:-1]
    else:
        feature_cols = df.columns
    
    if method == 'iqr':
        for col in feature_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Cap outliers instead of removing
            df_clean[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    elif method == 'zscore':
        for col in feature_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df_clean[col] = df[col].where(z_scores <= threshold, df[col].median())
    
    elif method == 'isolation_forest':
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_mask = iso_forest.fit_predict(df[feature_cols]) == -1
        
        logger.info(f"Found {outlier_mask.sum()} outlier samples")
        
        # Replace outlier rows with median values
        for col in feature_cols:
            df_clean.loc[outlier_mask, col] = df[col].median()
    
    else:
        raise ValueError(f"Unknown outlier method: {method}")
    
    logger.info("Outlier removal completed")
    return df_clean


def handle_missing_values(
    df: pd.DataFrame,
    method: str = 'mean',
    exclude_label: bool = True
) -> pd.DataFrame:
    """
    Handle missing values in dataset.
    
    Args:
        df: Input DataFrame
        method: Imputation method ('mean', 'median', 'forward_fill', 'interpolate', 'drop')
        exclude_label: Whether to exclude the last column
        
    Returns:
        DataFrame with missing values handled
    """
    n_missing = df.isnull().sum().sum()
    
    if n_missing == 0:
        logger.info("No missing values found")
        return df
    
    logger.info(f"Handling {n_missing} missing values using {method} method...")
    
    df_imputed = df.copy()
    
    if exclude_label:
        feature_cols = df.columns[:-1]
    else:
        feature_cols = df.columns
    
    if method == 'mean':
        df_imputed[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())
    
    elif method == 'median':
        df_imputed[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    
    elif method == 'forward_fill':
        df_imputed[feature_cols] = df[feature_cols].fillna(method='ffill')
        # Handle any remaining NaNs at the beginning
        df_imputed[feature_cols] = df_imputed[feature_cols].fillna(method='bfill')
    
    elif method == 'interpolate':
        df_imputed[feature_cols] = df[feature_cols].interpolate(method='linear', axis=0)
        # Handle any remaining NaNs
        df_imputed[feature_cols] = df_imputed[feature_cols].fillna(df[feature_cols].mean())
    
    elif method == 'drop':
        df_imputed = df.dropna()
        logger.info(f"Dropped {len(df) - len(df_imputed)} rows with missing values")
    
    else:
        raise ValueError(f"Unknown imputation method: {method}")
    
    logger.info("Missing value handling completed")
    return df_imputed


def remove_baseline_wander(signal_data: np.ndarray, window_size: int = 200) -> np.ndarray:
    """
    Remove baseline wander from signal using moving average.
    
    Args:
        signal_data: Input signal
        window_size: Window size for moving average
        
    Returns:
        Signal with baseline removed
    """
    baseline = pd.Series(signal_data).rolling(window=window_size, center=True).mean()
    baseline = baseline.fillna(method='bfill').fillna(method='ffill')
    
    return signal_data - baseline.values


def preprocess_pipeline(
    df: pd.DataFrame,
    config: dict,
    dataset_name: str = "Dataset"
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        dataset_name: Name of dataset for logging
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Starting preprocessing pipeline for {dataset_name}")
    logger.info(f"Initial shape: {df.shape}")
    
    df_processed = df.copy()
    
    # 1. Handle missing values
    if config['preprocessing']['handle_missing']:
        df_processed = handle_missing_values(
            df_processed,
            method=config['preprocessing']['missing_method']
        )
    
    # 2. Remove outliers
    if config['preprocessing']['handle_outliers']:
        df_processed = remove_outliers(
            df_processed,
            method=config['preprocessing']['outlier_method']
        )
    
    # 3. Apply signal filtering
    if config['signal_processing']['apply_filtering']:
        df_processed = apply_filtering_to_dataframe(
            df_processed,
            lowcut=config['signal_processing']['lowcut'],
            highcut=config['signal_processing']['highcut'],
            filter_type=config['signal_processing']['filter_type'],
            order=config['signal_processing']['order']
        )
    
    # 4. Normalize
    label_col = df_processed.columns[-1]
    features = df_processed.iloc[:, :-1]
    labels = df_processed[label_col]
    
    features_normalized = normalize_signal(
        features,
        method=config['preprocessing']['normalization']
    )
    
    df_processed = pd.concat([features_normalized, labels], axis=1)
    
    logger.info(f"Preprocessing completed. Final shape: {df_processed.shape}")
    
    return df_processed


if __name__ == "__main__":
    # Test signal processing functions
    print("Testing Signal Processing Module")
    
    # Generate test signal
    t = np.linspace(0, 1, 1000)
    test_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(1000)
    
    # Test filtering
    filtered = filter_signal(test_signal, fs=1000, lowcut=1, highcut=10)
    print(f"Original signal shape: {test_signal.shape}")
    print(f"Filtered signal shape: {filtered.shape}")
    
    # Test normalization
    test_df = pd.DataFrame(np.random.randn(100, 5))
    normalized = normalize_signal(test_df, method='standard')
    print(f"\nNormalized data mean: {normalized.mean().mean():.6f}")
    print(f"Normalized data std: {normalized.std().mean():.6f}")
