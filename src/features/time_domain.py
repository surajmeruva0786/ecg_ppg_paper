"""
Time-domain feature extraction for ECG and PPG signals.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from typing import Union, Dict
import logging

logger = logging.getLogger(__name__)


def extract_statistical_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Extract statistical features from signal.
    
    Args:
        signal: Input signal (1D array)
        
    Returns:
        Dictionary of statistical features
    """
    features = {
        'mean': np.mean(signal),
        'median': np.median(signal),
        'std': np.std(signal),
        'var': np.var(signal),
        'min': np.min(signal),
        'max': np.max(signal),
        'range': np.ptp(signal),
        'rms': np.sqrt(np.mean(signal**2)),
        'skewness': stats.skew(signal),
        'kurtosis': stats.kurtosis(signal),
        'q25': np.percentile(signal, 25),
        'q75': np.percentile(signal, 75),
        'iqr': stats.iqr(signal),
        'mad': np.median(np.abs(signal - np.median(signal))),  # Median Absolute Deviation
    }
    
    return features


def extract_peak_features(signal: np.ndarray, height=None, distance=None) -> Dict[str, float]:
    """
    Extract peak-related features from signal.
    
    Args:
        signal: Input signal (1D array)
        height: Minimum peak height
        distance: Minimum distance between peaks
        
    Returns:
        Dictionary of peak features
    """
    # Find peaks
    peaks, properties = find_peaks(signal, height=height, distance=distance)
    
    features = {
        'n_peaks': len(peaks),
        'peak_rate': len(peaks) / len(signal),  # Peaks per sample
    }
    
    if len(peaks) > 0:
        peak_heights = signal[peaks]
        features.update({
            'mean_peak_height': np.mean(peak_heights),
            'std_peak_height': np.std(peak_heights),
            'max_peak_height': np.max(peak_heights),
            'min_peak_height': np.min(peak_heights),
        })
        
        # Peak-to-peak intervals
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks)
            features.update({
                'mean_peak_interval': np.mean(peak_intervals),
                'std_peak_interval': np.std(peak_intervals),
                'cv_peak_interval': np.std(peak_intervals) / np.mean(peak_intervals) if np.mean(peak_intervals) > 0 else 0,
            })
    else:
        features.update({
            'mean_peak_height': 0,
            'std_peak_height': 0,
            'max_peak_height': 0,
            'min_peak_height': 0,
            'mean_peak_interval': 0,
            'std_peak_interval': 0,
            'cv_peak_interval': 0,
        })
    
    return features


def extract_morphological_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Extract morphological features from signal.
    
    Args:
        signal: Input signal (1D array)
        
    Returns:
        Dictionary of morphological features
    """
    # Zero crossing rate
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    zcr = len(zero_crossings) / len(signal)
    
    # Signal energy
    energy = np.sum(signal**2)
    
    # Signal power
    power = energy / len(signal)
    
    # Area under curve (absolute)
    auc = np.trapz(np.abs(signal))
    
    # First derivative (slope)
    first_derivative = np.diff(signal)
    mean_slope = np.mean(np.abs(first_derivative))
    max_slope = np.max(np.abs(first_derivative))
    
    # Second derivative (curvature)
    second_derivative = np.diff(first_derivative)
    mean_curvature = np.mean(np.abs(second_derivative))
    
    features = {
        'zero_crossing_rate': zcr,
        'signal_energy': energy,
        'signal_power': power,
        'auc': auc,
        'mean_slope': mean_slope,
        'max_slope': max_slope,
        'mean_curvature': mean_curvature,
    }
    
    return features


def extract_all_time_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Extract all time-domain features from signal.
    
    Args:
        signal: Input signal (1D array)
        
    Returns:
        Dictionary of all time-domain features
    """
    features = {}
    
    # Statistical features
    stat_features = extract_statistical_features(signal)
    features.update({f'stat_{k}': v for k, v in stat_features.items()})
    
    # Peak features
    peak_features = extract_peak_features(signal)
    features.update({f'peak_{k}': v for k, v in peak_features.items()})
    
    # Morphological features
    morph_features = extract_morphological_features(signal)
    features.update({f'morph_{k}': v for k, v in morph_features.items()})
    
    return features


def extract_time_features_from_dataframe(
    df: pd.DataFrame,
    exclude_label: bool = True
) -> pd.DataFrame:
    """
    Extract time-domain features from all rows in a DataFrame.
    
    Args:
        df: Input DataFrame where each row is a signal
        exclude_label: Whether to exclude the last column (assumed to be label)
        
    Returns:
        DataFrame with extracted features
    """
    logger.info("Extracting time-domain features from DataFrame")
    
    if exclude_label:
        signals = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values
    else:
        signals = df.values
        labels = None
    
    feature_list = []
    
    for i, signal in enumerate(signals):
        if i % 100 == 0:
            logger.info(f"Processing signal {i}/{len(signals)}")
        
        features = extract_all_time_features(signal)
        feature_list.append(features)
    
    feature_df = pd.DataFrame(feature_list)
    
    if labels is not None:
        feature_df['label'] = labels
    
    logger.info(f"Extracted {feature_df.shape[1]} time-domain features from {feature_df.shape[0]} signals")
    
    return feature_df


if __name__ == "__main__":
    # Test time-domain feature extraction
    print("Testing Time-Domain Feature Extraction")
    
    # Generate test signal
    t = np.linspace(0, 1, 1000)
    test_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(1000)
    
    # Extract features
    stat_features = extract_statistical_features(test_signal)
    print("\nStatistical Features:")
    for k, v in stat_features.items():
        print(f"  {k}: {v:.4f}")
    
    peak_features = extract_peak_features(test_signal)
    print("\nPeak Features:")
    for k, v in peak_features.items():
        print(f"  {k}: {v:.4f}")
    
    morph_features = extract_morphological_features(test_signal)
    print("\nMorphological Features:")
    for k, v in morph_features.items():
        print(f"  {k}: {v:.4f}")
    
    # Test on DataFrame
    test_df = pd.DataFrame(np.random.randn(10, 100))
    test_df['label'] = np.random.randint(0, 2, 10)
    
    feature_df = extract_time_features_from_dataframe(test_df)
    print(f"\nExtracted features shape: {feature_df.shape}")
    print(f"Feature columns: {feature_df.columns.tolist()}")
