"""
Wavelet-based feature extraction for ECG and PPG signals.
"""

import numpy as np
import pandas as pd
import pywt
from scipy.stats import entropy
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def extract_wavelet_features(
    signal: np.ndarray,
    wavelet: str = 'db4',
    level: int = 5
) -> Dict[str, float]:
    """
    Extract wavelet decomposition features.
    
    Args:
        signal: Input signal (1D array)
        wavelet: Wavelet type (e.g., 'db4', 'sym5', 'coif3')
        level: Decomposition level
        
    Returns:
        Dictionary of wavelet features
    """
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    features = {}
    
    # Extract features from each level
    for i, coeff in enumerate(coeffs):
        prefix = f'wavelet_level_{i}'
        
        features[f'{prefix}_mean'] = np.mean(coeff)
        features[f'{prefix}_std'] = np.std(coeff)
        features[f'{prefix}_energy'] = np.sum(coeff**2)
        features[f'{prefix}_max'] = np.max(np.abs(coeff))
    
    return features


def extract_wavelet_energy(
    signal: np.ndarray,
    wavelet: str = 'db4',
    level: int = 5
) -> Dict[str, float]:
    """
    Extract wavelet energy features.
    
    Args:
        signal: Input signal (1D array)
        wavelet: Wavelet type
        level: Decomposition level
        
    Returns:
        Dictionary of wavelet energy features
    """
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Calculate energy for each level
    energies = [np.sum(c**2) for c in coeffs]
    total_energy = np.sum(energies)
    
    # Relative energies
    relative_energies = [e / total_energy if total_energy > 0 else 0 for e in energies]
    
    features = {
        'wavelet_total_energy': total_energy,
    }
    
    for i, (energy, rel_energy) in enumerate(zip(energies, relative_energies)):
        features[f'wavelet_energy_level_{i}'] = energy
        features[f'wavelet_rel_energy_level_{i}'] = rel_energy
    
    return features


def extract_wavelet_entropy(
    signal: np.ndarray,
    wavelet: str = 'db4',
    level: int = 5
) -> Dict[str, float]:
    """
    Extract wavelet entropy features.
    
    Args:
        signal: Input signal (1D array)
        wavelet: Wavelet type
        level: Decomposition level
        
    Returns:
        Dictionary of wavelet entropy features
    """
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    features = {}
    
    # Calculate entropy for each level
    for i, coeff in enumerate(coeffs):
        # Normalize coefficients to create probability distribution
        coeff_abs = np.abs(coeff)
        coeff_norm = coeff_abs / (np.sum(coeff_abs) + 1e-10)
        
        # Calculate Shannon entropy
        wavelet_entropy = entropy(coeff_norm + 1e-10)
        
        features[f'wavelet_entropy_level_{i}'] = wavelet_entropy
    
    return features


def extract_wavelet_packet_features(
    signal: np.ndarray,
    wavelet: str = 'db4',
    level: int = 3,
    max_nodes: int = 8
) -> Dict[str, float]:
    """
    Extract wavelet packet decomposition features.
    
    Args:
        signal: Input signal (1D array)
        wavelet: Wavelet type
        level: Decomposition level
        max_nodes: Maximum number of nodes to extract features from
        
    Returns:
        Dictionary of wavelet packet features
    """
    # Perform wavelet packet decomposition
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=level)
    
    # Get all nodes at the maximum level
    nodes = [node.path for node in wp.get_level(level, 'freq')]
    
    features = {}
    
    # Extract features from each node (limit to max_nodes)
    for i, node_path in enumerate(nodes[:max_nodes]):
        node = wp[node_path]
        coeff = node.data
        
        prefix = f'wp_node_{i}'
        
        features[f'{prefix}_energy'] = np.sum(coeff**2)
        features[f'{prefix}_mean'] = np.mean(coeff)
        features[f'{prefix}_std'] = np.std(coeff)
    
    return features


def extract_all_wavelet_features(
    signal: np.ndarray,
    wavelet: str = 'db4',
    level: int = 5
) -> Dict[str, float]:
    """
    Extract all wavelet-based features from signal.
    
    Args:
        signal: Input signal (1D array)
        wavelet: Wavelet type
        level: Decomposition level
        
    Returns:
        Dictionary of all wavelet features
    """
    features = {}
    
    # Basic wavelet features
    wavelet_features = extract_wavelet_features(signal, wavelet, level)
    features.update(wavelet_features)
    
    # Wavelet energy features
    energy_features = extract_wavelet_energy(signal, wavelet, level)
    features.update(energy_features)
    
    # Wavelet entropy features
    entropy_features = extract_wavelet_entropy(signal, wavelet, level)
    features.update(entropy_features)
    
    # Wavelet packet features (use lower level to avoid too many features)
    wp_features = extract_wavelet_packet_features(signal, wavelet, level=min(3, level))
    features.update(wp_features)
    
    return features


def extract_wavelet_features_from_dataframe(
    df: pd.DataFrame,
    wavelet: str = 'db4',
    level: int = 5,
    exclude_label: bool = True
) -> pd.DataFrame:
    """
    Extract wavelet features from all rows in a DataFrame.
    
    Args:
        df: Input DataFrame where each row is a signal
        wavelet: Wavelet type
        level: Decomposition level
        exclude_label: Whether to exclude the last column (assumed to be label)
        
    Returns:
        DataFrame with extracted features
    """
    logger.info(f"Extracting wavelet features from DataFrame (wavelet={wavelet}, level={level})")
    
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
        
        features = extract_all_wavelet_features(signal, wavelet, level)
        feature_list.append(features)
    
    feature_df = pd.DataFrame(feature_list)
    
    if labels is not None:
        feature_df['label'] = labels
    
    logger.info(f"Extracted {feature_df.shape[1]} wavelet features from {feature_df.shape[0]} signals")
    
    return feature_df


if __name__ == "__main__":
    # Test wavelet feature extraction
    print("Testing Wavelet Feature Extraction")
    
    # Generate test signal
    t = np.linspace(0, 1, 1000)
    test_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.2 * np.random.randn(1000)
    
    # Extract features
    wavelet_features = extract_wavelet_features(test_signal)
    print(f"\nWavelet Features: {len(wavelet_features)} features extracted")
    
    energy_features = extract_wavelet_energy(test_signal)
    print(f"Wavelet Energy Features: {len(energy_features)} features extracted")
    
    entropy_features = extract_wavelet_entropy(test_signal)
    print(f"Wavelet Entropy Features: {len(entropy_features)} features extracted")
    
    all_features = extract_all_wavelet_features(test_signal)
    print(f"\nTotal Wavelet Features: {len(all_features)} features")
    
    # Show some examples
    print("\nExample features:")
    for i, (k, v) in enumerate(list(all_features.items())[:10]):
        print(f"  {k}: {v:.4f}")
