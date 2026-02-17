"""
Frequency-domain feature extraction for ECG and PPG signals.
"""

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def extract_fft_features(signal: np.ndarray, fs: float = 100.0) -> Dict[str, float]:
    """
    Extract FFT-based features from signal.
    
    Args:
        signal: Input signal (1D array)
        fs: Sampling frequency in Hz
        
    Returns:
        Dictionary of FFT features
    """
    # Compute FFT
    n = len(signal)
    fft_vals = fft(signal)
    fft_freqs = fftfreq(n, 1/fs)
    
    # Get positive frequencies only
    positive_freqs = fft_freqs[:n//2]
    fft_magnitude = np.abs(fft_vals[:n//2])
    
    # Dominant frequency
    dominant_freq_idx = np.argmax(fft_magnitude)
    dominant_freq = positive_freqs[dominant_freq_idx]
    
    # Mean and std of frequency magnitudes
    mean_magnitude = np.mean(fft_magnitude)
    std_magnitude = np.std(fft_magnitude)
    
    # Spectral centroid
    spectral_centroid = np.sum(positive_freqs * fft_magnitude) / np.sum(fft_magnitude)
    
    # Spectral spread
    spectral_spread = np.sqrt(np.sum(((positive_freqs - spectral_centroid)**2) * fft_magnitude) / np.sum(fft_magnitude))
    
    # Spectral entropy
    normalized_magnitude = fft_magnitude / np.sum(fft_magnitude)
    spectral_entropy = entropy(normalized_magnitude + 1e-10)
    
    features = {
        'dominant_frequency': dominant_freq,
        'fft_mean_magnitude': mean_magnitude,
        'fft_std_magnitude': std_magnitude,
        'spectral_centroid': spectral_centroid,
        'spectral_spread': spectral_spread,
        'spectral_entropy': spectral_entropy,
    }
    
    return features


def extract_psd_features(signal: np.ndarray, fs: float = 100.0) -> Dict[str, float]:
    """
    Extract Power Spectral Density features.
    
    Args:
        signal: Input signal (1D array)
        fs: Sampling frequency in Hz
        
    Returns:
        Dictionary of PSD features
    """
    # Compute PSD using Welch's method
    freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=min(256, len(signal)))
    
    # Total power
    total_power = np.trapz(psd, freqs)
    
    # Define frequency bands (typical for HRV analysis)
    vlf_band = (0.003, 0.04)  # Very Low Frequency
    lf_band = (0.04, 0.15)    # Low Frequency
    hf_band = (0.15, 0.4)     # High Frequency
    
    # Calculate band powers
    vlf_power = np.trapz(psd[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])],
                         freqs[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])])
    lf_power = np.trapz(psd[(freqs >= lf_band[0]) & (freqs < lf_band[1])],
                        freqs[(freqs >= lf_band[0]) & (freqs < lf_band[1])])
    hf_power = np.trapz(psd[(freqs >= hf_band[0]) & (freqs < hf_band[1])],
                        freqs[(freqs >= hf_band[0]) & (freqs < hf_band[1])])
    
    # LF/HF ratio
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
    
    # Normalized powers
    lf_norm = lf_power / (lf_power + hf_power) if (lf_power + hf_power) > 0 else 0
    hf_norm = hf_power / (lf_power + hf_power) if (lf_power + hf_power) > 0 else 0
    
    features = {
        'total_power': total_power,
        'vlf_power': vlf_power,
        'lf_power': lf_power,
        'hf_power': hf_power,
        'lf_hf_ratio': lf_hf_ratio,
        'lf_norm': lf_norm,
        'hf_norm': hf_norm,
    }
    
    return features


def extract_spectral_features(signal: np.ndarray, fs: float = 100.0) -> Dict[str, float]:
    """
    Extract additional spectral features.
    
    Args:
        signal: Input signal (1D array)
        fs: Sampling frequency in Hz
        
    Returns:
        Dictionary of spectral features
    """
    # Compute spectrogram
    freqs, times, Sxx = scipy_signal.spectrogram(signal, fs=fs)
    
    # Mean and std across time
    mean_spectrogram = np.mean(Sxx, axis=1)
    std_spectrogram = np.std(Sxx, axis=1)
    
    # Spectral flatness (measure of noisiness)
    spectral_flatness = np.exp(np.mean(np.log(mean_spectrogram + 1e-10))) / (np.mean(mean_spectrogram) + 1e-10)
    
    # Spectral rolloff (frequency below which 85% of energy is contained)
    cumsum_spectrum = np.cumsum(mean_spectrogram)
    rolloff_idx = np.where(cumsum_spectrum >= 0.85 * cumsum_spectrum[-1])[0]
    spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
    
    # Spectral bandwidth
    spectral_centroid = np.sum(freqs * mean_spectrogram) / np.sum(mean_spectrogram)
    spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * mean_spectrogram) / np.sum(mean_spectrogram))
    
    features = {
        'spectral_flatness': spectral_flatness,
        'spectral_rolloff': spectral_rolloff,
        'spectral_bandwidth': spectral_bandwidth,
        'mean_spectrogram_energy': np.mean(mean_spectrogram),
        'std_spectrogram_energy': np.mean(std_spectrogram),
    }
    
    return features


def extract_all_frequency_features(signal: np.ndarray, fs: float = 100.0) -> Dict[str, float]:
    """
    Extract all frequency-domain features from signal.
    
    Args:
        signal: Input signal (1D array)
        fs: Sampling frequency in Hz
        
    Returns:
        Dictionary of all frequency-domain features
    """
    features = {}
    
    # FFT features
    fft_features = extract_fft_features(signal, fs)
    features.update({f'fft_{k}': v for k, v in fft_features.items()})
    
    # PSD features
    psd_features = extract_psd_features(signal, fs)
    features.update({f'psd_{k}': v for k, v in psd_features.items()})
    
    # Spectral features
    spectral_features = extract_spectral_features(signal, fs)
    features.update({f'spec_{k}': v for k, v in spectral_features.items()})
    
    return features


def extract_frequency_features_from_dataframe(
    df: pd.DataFrame,
    fs: float = 100.0,
    exclude_label: bool = True
) -> pd.DataFrame:
    """
    Extract frequency-domain features from all rows in a DataFrame.
    
    Args:
        df: Input DataFrame where each row is a signal
        fs: Sampling frequency
        exclude_label: Whether to exclude the last column (assumed to be label)
        
    Returns:
        DataFrame with extracted features
    """
    logger.info("Extracting frequency-domain features from DataFrame")
    
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
        
        features = extract_all_frequency_features(signal, fs)
        feature_list.append(features)
    
    feature_df = pd.DataFrame(feature_list)
    
    if labels is not None:
        feature_df['label'] = labels
    
    logger.info(f"Extracted {feature_df.shape[1]} frequency-domain features from {feature_df.shape[0]} signals")
    
    return feature_df


if __name__ == "__main__":
    # Test frequency-domain feature extraction
    print("Testing Frequency-Domain Feature Extraction")
    
    # Generate test signal
    fs = 100.0
    t = np.linspace(0, 10, int(10 * fs))
    test_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t) + 0.2 * np.random.randn(len(t))
    
    # Extract features
    fft_features = extract_fft_features(test_signal, fs)
    print("\nFFT Features:")
    for k, v in fft_features.items():
        print(f"  {k}: {v:.4f}")
    
    psd_features = extract_psd_features(test_signal, fs)
    print("\nPSD Features:")
    for k, v in psd_features.items():
        print(f"  {k}: {v:.4f}")
    
    spectral_features = extract_spectral_features(test_signal, fs)
    print("\nSpectral Features:")
    for k, v in spectral_features.items():
        print(f"  {k}: {v:.4f}")
