"""
Traditional ML-only pipeline (bypasses PyTorch CUDA issues).

This script runs the complete pipeline using only traditional ML models,
avoiding PyTorch/CUDA dependency issues.
"""

import sys
from pathlib import Path
import yaml
import logging
import argparse
import pandas as pd
import numpy as np

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from preprocessing.data_loader import load_config
from preprocessing.signal_processing import preprocess_pipeline
from preprocessing.data_splitting import split_data, apply_smote
from features.time_domain import extract_time_features_from_dataframe
from features.frequency_domain import extract_frequency_features_from_dataframe
from features.wavelet_features import extract_wavelet_features_from_dataframe
from features.hrv_features import extract_hrv_features_from_dataframe
from models.traditional_ml import train_all_ml_models
from evaluation.comparison import compare_models, create_comparison_table

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_ml_only.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_dataset(config, dataset_type='ecg'):
    """Load a single dataset."""
    if dataset_type == 'ecg':
        ecg_path = config['data']['ecg_path']
        logger.info(f"Loading ECG dataset from {ecg_path}")
        df = pd.read_csv(ecg_path)
        logger.info(f"ECG dataset loaded: {df.shape}")
        return df
    elif dataset_type == 'ppg':
        ppg_path = config['data']['ppg_path']
        logger.info(f"Loading PPG dataset from {ppg_path}")
        df = pd.read_csv(ppg_path)
        logger.info(f"PPG dataset loaded: {df.shape}")
        return df
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def extract_all_features(df, config, dataset_name='Dataset'):
    """Extract all features from a dataset."""
    logger.info(f"Extracting features from {dataset_name}...")
    
    all_features = []
    
    # Time-domain features
    if config['features']['time_domain']:
        logger.info("Extracting time-domain features...")
        time_features = extract_time_features_from_dataframe(df)
        all_features.append(time_features)
    
    # Frequency-domain features
    if config['features']['frequency_domain']:
        logger.info("Extracting frequency-domain features...")
        freq_features = extract_frequency_features_from_dataframe(df)
        all_features.append(freq_features)
    
    # Wavelet features
    if config['features']['wavelet']:
        logger.info("Extracting wavelet features...")
        wavelet_features = extract_wavelet_features_from_dataframe(df)
        all_features.append(wavelet_features)
    
    # HRV features (for ECG only)
    if config['features']['hrv'] and 'ecg' in dataset_name.lower():
        logger.info("Extracting HRV features...")
        hrv_features = extract_hrv_features_from_dataframe(df)
        all_features.append(hrv_features)
    
    # Combine all features
    if len(all_features) > 0:
        combined_features = all_features[0]
        for feat_df in all_features[1:]:
            feat_df_no_label = feat_df.drop(columns=['label'], errors='ignore')
            combined_features = pd.concat([combined_features, feat_df_no_label], axis=1)
        
        logger.info(f"Total features extracted: {combined_features.shape[1]}")
        return combined_features
    else:
        logger.warning("No features extracted!")
        return df


def run_ml_pipeline(config_path='config.yaml', dataset_type='ecg', quick_test=False):
    """Run the ML-only pipeline."""
    logger.info("="*80)
    logger.info("Starting Heart Attack Prediction Pipeline (Traditional ML Only)")
    logger.info("="*80)
    
    # Load configuration
    config = load_config(config_path)
    
    # Process dataset
    logger.info(f"\nProcessing {dataset_type.upper()} Dataset")
    logger.info("="*80)
    
    # Step 1: Load data
    logger.info("\nStep 1: Loading Data")
    df = load_dataset(config, dataset_type)
    
    # Step 2: Preprocessing
    logger.info("\nStep 2: Preprocessing")
    df_processed = preprocess_pipeline(df, config, dataset_type)
    
    # Step 3: Feature Extraction
    logger.info("\nStep 3: Feature Extraction")
    df_features = extract_all_features(df_processed, config, dataset_type)
    
    # Separate features and labels
    X = df_features.drop(columns=['label'], errors='ignore')
    y = df_features['label'] if 'label' in df_features.columns else df_processed.iloc[:, -1]
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Label distribution:\n{y.value_counts()}")
    
    # Step 4: Data Splitting
    logger.info("\nStep 4: Data Splitting")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        train_size=config['split']['train_size'],
        val_size=config['split']['val_size'],
        test_size=config['split']['test_size'],
        random_state=config['random_seed'],
        stratify=config['split']['stratify']
    )
    
    # Step 5: Handle Class Imbalance
    if config['imbalance']['method'] == 'smote':
        logger.info("\nStep 5: Applying SMOTE")
        X_train, y_train = apply_smote(X_train, y_train, random_state=config['random_seed'])
    
    # Step 6: Train Traditional ML Models
    logger.info("\n" + "="*80)
    logger.info("Step 6: Training Traditional ML Models")
    logger.info("="*80)
    
    ml_results = train_all_ml_models(
        X_train.values, y_train.values,
        X_val.values, y_val.values,
        save_dir=f'results/models/{dataset_type}/ml',
        cv=3 if quick_test else config['evaluation']['cv_folds'],
        random_state=config['random_seed']
    )
    
    # Step 7: Evaluate on Test Set
    logger.info("\n" + "="*80)
    logger.info("Step 7: Final Evaluation on Test Set")
    logger.info("="*80)
    
    test_results = {}
    
    for model_name, result in ml_results.items():
        if 'model' in result:
            logger.info(f"\nEvaluating {model_name} on test set...")
            try:
                y_pred = result['model'].predict(X_test.values)
                y_proba = result['model'].predict_proba(X_test.values)[:, 1] if hasattr(result['model'], 'predict_proba') else None
                
                from evaluation.metrics import calculate_metrics
                test_metrics = calculate_metrics(y_test.values, y_pred, y_proba)
                test_results[model_name] = {'metrics': test_metrics}
                
                logger.info(f"{model_name} Test F1: {test_metrics['f1']:.4f}")
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
    
    # Step 8: Model Comparison
    logger.info("\n" + "="*80)
    logger.info("Step 8: Model Comparison")
    logger.info("="*80)
    
    comparison_df = compare_models(test_results)
    print("\n" + str(comparison_df))
    
    # Save comparison table
    results_dir = Path(f'results/reports/{dataset_type}')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_file = results_dir / 'model_comparison.csv'
    create_comparison_table(comparison_df, save_path=comparison_file)
    
    logger.info(f"\nComparison table saved to {comparison_file}")
    
    logger.info("\n" + "="*80)
    logger.info("Pipeline Completed Successfully!")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Heart Attack Prediction Pipeline (ML Only)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--dataset', type=str, default='ecg', choices=['ecg', 'ppg'], help='Dataset to use')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test with reduced parameters')
    
    args = parser.parse_args()
    
    run_ml_pipeline(
        config_path=args.config,
        dataset_type=args.dataset,
        quick_test=args.quick_test
    )
