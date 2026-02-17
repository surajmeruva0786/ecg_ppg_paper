"""
Heart Rate Variability (HRV) feature extraction for ECG signals.
"""

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import entropy
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def detect_r_peaks(ecg_signal: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """
    Detect R-peaks in ECG signal.
    
    Args:
        ecg_signal: ECG signal (1D array)
        fs: Sampling frequency in Hz
        
    Returns:
        Array of R-peak indices
    """
    # Simple R-peak detection using find_peaks
    # In practice, you might want to use more sophisticated methods
    # like Pan-Tompkins algorithm or neurokit2
    
    # Normalize signal
    signal_norm = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
    
    # Find peaks with appropriate parameters
    # Height threshold and minimum distance between peaks
    height_threshold = np.mean(signal_norm) + 0.5 * np.std(signal_norm)
    min_distance = int(0.6 * fs)  # Minimum 600ms between beats (100 bpm max)
    
    peaks, _ = scipy_signal.find_peaks(
        signal_norm,
        height=height_threshold,
        distance=min_distance
    )
    
    return peaks


def extract_rr_intervals(r_peaks: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """
    Extract RR intervals from R-peak indices.
    
    Args:
        r_peaks: Array of R-peak indices
        fs: Sampling frequency in Hz
        
    Returns:
        Array of RR intervals in milliseconds
    """
    if len(r_peaks) < 2:
        return np.array([])
    
    # Calculate RR intervals in samples
    rr_intervals_samples = np.diff(r_peaks)
    
    # Convert to milliseconds
    rr_intervals_ms = (rr_intervals_samples / fs) * 1000
    
    return rr_intervals_ms


def extract_hrv_time_features(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Extract time-domain HRV features from RR intervals.
    
    Args:
        rr_intervals: Array of RR intervals in milliseconds
        
    Returns:
        Dictionary of time-domain HRV features
    """
    if len(rr_intervals) < 2:
        return {
            'hrv_mean_rr': 0,
            'hrv_sdnn': 0,
            'hrv_rmssd': 0,
            'hrv_sdsd': 0,
            'hrv_nn50': 0,
            'hrv_pnn50': 0,
            'hrv_nn20': 0,
            'hrv_pnn20': 0,
            'hrv_cv': 0,
        }
    
    # Mean RR interval
    mean_rr = np.mean(rr_intervals)
    
    # SDNN: Standard deviation of RR intervals
    sdnn = np.std(rr_intervals, ddof=1)
    
    # Successive differences
    diff_rr = np.diff(rr_intervals)
    
    # RMSSD: Root mean square of successive differences
    rmssd = np.sqrt(np.mean(diff_rr**2))
    
    # SDSD: Standard deviation of successive differences
    sdsd = np.std(diff_rr, ddof=1)
    
    # NN50: Number of successive differences > 50ms
    nn50 = np.sum(np.abs(diff_rr) > 50)
    
    # pNN50: Percentage of NN50
    pnn50 = (nn50 / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0
    
    # NN20: Number of successive differences > 20ms
    nn20 = np.sum(np.abs(diff_rr) > 20)
    
    # pNN20: Percentage of NN20
    pnn20 = (nn20 / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0
    
    # Coefficient of variation
    cv = (sdnn / mean_rr) * 100 if mean_rr > 0 else 0
    
    features = {
        'hrv_mean_rr': mean_rr,
        'hrv_sdnn': sdnn,
        'hrv_rmssd': rmssd,
        'hrv_sdsd': sdsd,
        'hrv_nn50': nn50,
        'hrv_pnn50': pnn50,
        'hrv_nn20': nn20,
        'hrv_pnn20': pnn20,
        'hrv_cv': cv,
    }
    
    return features


def extract_hrv_freq_features(rr_intervals: np.ndarray, fs: float = 4.0) -> Dict[str, float]:
    """
    Extract frequency-domain HRV features from RR intervals.
    
    Args:
        rr_intervals: Array of RR intervals in milliseconds
        fs: Resampling frequency for RR intervals (typically 4 Hz)
        
    Returns:
        Dictionary of frequency-domain HRV features
    """
    if len(rr_intervals) < 10:
        return {
            'hrv_vlf': 0,
            'hrv_lf': 0,
            'hrv_hf': 0,
            'hrv_lf_hf_ratio': 0,
            'hrv_total_power': 0,
            'hrv_lf_norm': 0,
            'hrv_hf_norm': 0,
        }
    
    # Interpolate RR intervals to get evenly sampled signal
    time_rr = np.cumsum(rr_intervals) / 1000  # Convert to seconds
    time_rr = np.insert(time_rr, 0, 0)
    rr_with_first = np.insert(rr_intervals, 0, rr_intervals[0])
    
    # Create evenly sampled time vector
    time_interp = np.arange(0, time_rr[-1], 1/fs)
    
    # Interpolate
    rr_interp = np.interp(time_interp, time_rr, rr_with_first)
    
    # Compute PSD
    freqs, psd = scipy_signal.welch(rr_interp, fs=fs, nperseg=min(256, len(rr_interp)))
    
    # Define frequency bands (Hz)
    vlf_band = (0.003, 0.04)
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)
    
    # Calculate band powers
    vlf_power = np.trapz(psd[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])],
                         freqs[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])])
    lf_power = np.trapz(psd[(freqs >= lf_band[0]) & (freqs < lf_band[1])],
                        freqs[(freqs >= lf_band[0]) & (freqs < lf_band[1])])
    hf_power = np.trapz(psd[(freqs >= hf_band[0]) & (freqs < hf_band[1])],
                        freqs[(freqs >= hf_band[0]) & (freqs < hf_band[1])])
    
    total_power = vlf_power + lf_power + hf_power
    
    # LF/HF ratio
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
    
    # Normalized powers
    lf_norm = lf_power / (lf_power + hf_power) if (lf_power + hf_power) > 0 else 0
    hf_norm = hf_power / (lf_power + hf_power) if (lf_power + hf_power) > 0 else 0
    
    features = {
        'hrv_vlf': vlf_power,
        'hrv_lf': lf_power,
        'hrv_hf': hf_power,
        'hrv_lf_hf_ratio': lf_hf_ratio,
        'hrv_total_power': total_power,
        'hrv_lf_norm': lf_norm,
        'hrv_hf_norm': hf_norm,
    }
    
    return features


def extract_hrv_nonlinear_features(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Extract non-linear HRV features from RR intervals.
    
    Args:
        rr_intervals: Array of RR intervals in milliseconds
        
    Returns:
        Dictionary of non-linear HRV features
    """
    if len(rr_intervals) < 3:
        return {
            'hrv_sd1': 0,
            'hrv_sd2': 0,
            'hrv_sd_ratio': 0,
            'hrv_sampen': 0,
        }
    
    # Poincaré plot features (SD1, SD2)
    diff_rr = np.diff(rr_intervals)
    
    # SD1: Standard deviation perpendicular to line of identity
    sd1 = np.sqrt(np.var(diff_rr) / 2)
    
    # SD2: Standard deviation along line of identity
    sd2 = np.sqrt(2 * np.var(rr_intervals) - np.var(diff_rr) / 2)
    
    # SD1/SD2 ratio
    sd_ratio = sd1 / sd2 if sd2 > 0 else 0
    
    # Sample Entropy (simplified version)
    # This is a basic implementation; for production use, consider more robust libraries
    def sample_entropy(signal, m=2, r=0.2):
        """Calculate sample entropy."""
        N = len(signal)
        r = r * np.std(signal)
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            x = [[signal[j] for j in range(i, i + m)] for i in range(N - m + 1)]
            C = [len([1 for xj in x if _maxdist(xi, xj, m) <= r]) - 1 for xi in x]
            return sum(C) / (N - m + 1)
        
        try:
            return -np.log(_phi(m + 1) / _phi(m))
        except:
            return 0
    
    sampen = sample_entropy(rr_intervals)
    
    features = {
        'hrv_sd1': sd1,
        'hrv_sd2': sd2,
        'hrv_sd_ratio': sd_ratio,
        'hrv_sampen': sampen,
    }
    
    return features


def extract_all_hrv_features(ecg_signal: np.ndarray, fs: float = 100.0) -> Dict[str, float]:
    """
    Extract all HRV features from ECG signal.
    
    Args:
        ecg_signal: ECG signal (1D array)
        fs: Sampling frequency in Hz
        
    Returns:
        Dictionary of all HRV features
    """
    # Detect R-peaks
    r_peaks = detect_r_peaks(ecg_signal, fs)
    
    # Extract RR intervals
    rr_intervals = extract_rr_intervals(r_peaks, fs)
    
    if len(rr_intervals) < 2:
        logger.warning("Not enough R-peaks detected for HRV analysis")
        # Return zero features
        features = {}
        features.update(extract_hrv_time_features(np.array([])))
        features.update(extract_hrv_freq_features(np.array([])))
        features.update(extract_hrv_nonlinear_features(np.array([])))
        return features
    
    # Extract features
    features = {}
    
    # Time-domain features
    time_features = extract_hrv_time_features(rr_intervals)
    features.update(time_features)
    
    # Frequency-domain features
    freq_features = extract_hrv_freq_features(rr_intervals)
    features.update(freq_features)
    
    # Non-linear features
    nonlinear_features = extract_hrv_nonlinear_features(rr_intervals)
    features.update(nonlinear_features)
    
    return features


def extract_hrv_features_from_dataframe(
    df: pd.DataFrame,
    fs: float = 100.0,
    exclude_label: bool = True
) -> pd.DataFrame:
    """
    Extract HRV features from all rows in a DataFrame.
    
    Args:
        df: Input DataFrame where each row is an ECG signal
        fs: Sampling frequency
        exclude_label: Whether to exclude the last column (assumed to be label)
        
    Returns:
        DataFrame with extracted HRV features
    """
    logger.info("Extracting HRV features from DataFrame")
    
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
        
        features = extract_all_hrv_features(signal, fs)
        feature_list.append(features)
    
    feature_df = pd.DataFrame(feature_list)
    
    if labels is not None:
        feature_df['label'] = labels
    
    logger.info(f"Extracted {feature_df.shape[1]} HRV features from {feature_df.shape[0]} signals")
    
    return feature_df


if __name__ == "__main__":
    # Test HRV feature extraction
    print("Testing HRV Feature Extraction")
    
    # Generate synthetic ECG signal
    fs = 100.0
    t = np.linspace(0, 10, int(10 * fs))
    
    # Simple synthetic ECG with periodic R-peaks
    ecg_signal = np.zeros_like(t)
    heart_rate = 75  # bpm
    rr_interval_samples = int(60 / heart_rate * fs)
    
    for i in range(0, len(t), rr_interval_samples):
        if i < len(ecg_signal):
            ecg_signal[i] = 1.0
    
    # Add some noise
    ecg_signal += 0.1 * np.random.randn(len(t))
    
    # Extract HRV features
    hrv_features = extract_all_hrv_features(ecg_signal, fs)
    
    print("\nHRV Features:")
    for k, v in hrv_features.items():
        print(f"  {k}: {v:.4f}")
